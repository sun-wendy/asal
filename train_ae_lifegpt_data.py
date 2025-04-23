#!/usr/bin/env python3
"""
Flexible Auto‑Encoder (AE) trainer for frame data stored in CSV files.

* Encoder downsamples ×4; decoder upsamples ×4 (any `img_size` divisible by 4).
* Optional 1‑ or 3‑channel data (`--channels {1,3}`).
* No output sigmoid – avoids gradient saturation on binary images.
* Static `batch_size` in `sample_batch` fixes JAX trace errors.
* Periodic recon dumps & EMA loss curve.
"""

import argparse, csv, os, pickle
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from flax import linen as nn
import optax

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
#  CSV loader
# -----------------------------------------------------------------------------

def _parse_cell(cell: str, img_size: int, channels: int) -> np.ndarray:
    s = cell.strip()
    if channels == 1:
        arr = np.fromstring(s, sep=" ", dtype=np.float32)
        if arr.size == 0 and "," in s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        if arr.size != img_size * img_size:
            if len(s) == img_size * img_size and set(s) <= {"0", "1"}:
                arr = np.fromiter((float(c) for c in s), dtype=np.float32,
                                   count=img_size * img_size)
    else:  # channels == 3, expect R G B triplets
        arr = np.fromstring(s.replace(",", " "), sep=" ", dtype=np.float32)
    expected = img_size * img_size * channels
    if arr.size != expected:
        raise ValueError(f"expected {expected} values, got {arr.size}")
    return arr


def load_csv_frames(path: str, img_size: int, num_frames: int, channels: int) -> np.ndarray:
    frames: Sequence[np.ndarray] = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if row and row[0].startswith("State"):
                continue
            if len(row) != num_frames:
                raise ValueError(f"{path}: got {len(row)} cells, expected {num_frames}")
            for cell in row:
                arr = _parse_cell(cell, img_size, channels)
                frames.append(arr.reshape(img_size, img_size, channels))
    data = np.stack(frames, axis=0).astype(np.float32)
    print(f"[data] {path}: {data.shape[0]} frames of {img_size}×{img_size}×{channels}")
    return data

# -----------------------------------------------------------------------------
#  Model
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4,4), (2,2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4,4), (2,2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(self.latent_dim)(x)

class Decoder(nn.Module):
    latent_dim: int; img_size: int; channels: int
    @nn.compact
    def __call__(self, z):
        if self.img_size % 4 != 0:
            raise ValueError("img_size must be divisible by 4")
        h = self.img_size // 4
        x = nn.Dense(h*h*64)(z).reshape((-1, h, h, 64))
        x = nn.ConvTranspose(64, (4,4), (2,2), padding="SAME")(x); x = nn.leaky_relu(x)
        x = nn.ConvTranspose(32, (4,4), (2,2), padding="SAME")(x); x = nn.leaky_relu(x)
        return nn.Conv(self.channels, (3,3), padding="SAME")(x)

class AE(nn.Module):
    latent_dim: int; img_size: int; channels: int
    def setup(self):
        self.enc = Encoder(self.latent_dim)
        self.dec = Decoder(self.latent_dim, self.img_size, self.channels)
    def __call__(self, x):
        return self.dec(self.enc(x))

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

mse = lambda x, y: jnp.mean((x - y) ** 2)

@partial(jit, static_argnums=2)  # batch_size static fixes tracing error
def sample_batch(data, key, batch_size):
    idx = random.randint(key, (batch_size,), 0, data.shape[0])
    return data[idx]

# -----------------------------------------------------------------------------
#  Visualisation
# -----------------------------------------------------------------------------

def show_recon(inp, rec, tag="recon", n=5):
    for i in range(min(n, len(inp))):
        fig, ax = plt.subplots(1,2, figsize=(6,3))
        ax[0].imshow(inp[i].squeeze(), cmap="gray" if inp.shape[-1]==1 else None); ax[0].axis("off")
        ax[1].imshow(rec[i].squeeze(), cmap="gray" if rec.shape[-1]==1 else None); ax[1].axis("off")
        plt.tight_layout(); plt.savefig(f"{tag}_{i}.png"); plt.close(fig)

# -----------------------------------------------------------------------------
#  Training loop
# -----------------------------------------------------------------------------

def train_ae(train_csv, val_csv, num_frames, img_size, channels, latent_dim,
             steps, batch_size, lr):

    tr_np = load_csv_frames(train_csv, img_size, num_frames, channels)
    va_np = load_csv_frames(val_csv, img_size, num_frames, channels) if val_csv else None
    tr_ds, va_ds = jnp.array(tr_np), (jnp.array(va_np) if va_np is not None else None)

    model = AE(latent_dim, img_size, channels)
    rng   = random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1,img_size,img_size,channels)))['params']
    opt = optax.adam(lr); opt_state = opt.init(params)

    @jit
    def train_step(p, s, b):
        def loss_fn(pp):
            recon = model.apply({'params': pp}, b)
            return mse(b, recon), recon
        (loss, recon), g = value_and_grad(loss_fn, has_aux=True)(p)
        upd, s = opt.update(g, s, p); p = optax.apply_updates(p, upd)
        return p, s, loss, recon

    @jit
    def eval_step(p, b):
        return mse(b, model.apply({'params': p}, b))

    tr_l, va_l = [], []
    for step in range(1, steps+1):
        rng, k1, k2 = random.split(rng, 3)
        batch = sample_batch(tr_ds, k1, batch_size)
        params, opt_state, loss, recon = train_step(params, opt_state, batch)
        val = eval_step(params, sample_batch(va_ds, k2, batch_size)) if va_ds is not None else jnp.nan
        if step % 500 == 0:
            print(f"[{step}/{steps}] train={float(loss):.4e} val={float(val):.4e}")
        if step % 5000 == 0: show_recon(batch, recon, f"recon{step}")
        tr_l.append(float(loss)); va_l.append(float(val))

    df = pd.DataFrame({'tr':tr_l,'va':va_l}); ema = df.ewm(span=1000).mean()
    plt.plot(df['tr'], alpha=.4, label='raw train'); plt.plot(ema['tr'], label='EMA train')
    if va_ds is not None:
        plt.plot(df['va'], alpha=.4, label='raw val'); plt.plot(ema['va'], label='EMA val')
    plt.legend(); plt.title('AE MSE loss'); plt.savefig('loss_curve.png'); plt.close()

    with open('encoder_params.pkl','wb') as f: pickle.dump(params['enc'], f)
    with open('decoder_params.pkl','wb') as f: pickle.dump(params['dec'], f)
    print('✓ training complete')

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', default='')
    p.add_argument('--num_frames', type=int, default=10)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--channels', type=int, choices=[1,3], default=1)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--total_steps', type=int, default=30_000)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    a = p.parse_args()
    train_ae(a.train_csv, a.val_csv, a.num_frames, a.img_size,
             a.channels, a.latent_dim, a.total_steps,
             a.batch_size, a.learning_rate)
