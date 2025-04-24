#!/usr/bin/env python3
"""
Flexible Variational Auto‑Encoder (VAE) training script for Game‑of‑Life
frames.

✔  Works with **any `img_size` divisible by 4** (16, 32, 64, …).
✔  CSV loader accepts:
      • whitespace‑separated tokens  "0 1 0 …"
      • comma‑separated tokens       "0,1,0 …"
      • contiguous 0/1 strings       "010101…"
✔  Saves encoder / decoder weights as pickles on completion.

Example
-------
    python train_vae_flexible.py \
        --train_csv train.csv \
        --val_csv   val.csv   \
        --img_size  32
"""

import argparse, csv, os, pickle
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad

from flax import linen as nn
import optax

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
#  CSV loader
# -----------------------------------------------------------------------------

def _parse_cell(cell: str, img_size: int) -> np.ndarray:
    """Parse a single CSV cell into a 1‑D NumPy array of length img_size²."""
    s = cell.strip()
    arr = np.fromstring(s, sep=" ", dtype=np.float32)          # whitespace‑sep
    if arr.size == 0 and "," in s:
        arr = np.fromstring(s, sep=",", dtype=np.float32)      # comma‑sep
    if arr.size != img_size * img_size:
        if len(s) == img_size * img_size and set(s) <= {"0", "1"}:
            arr = np.fromiter((float(c) for c in s), dtype=np.float32,
                               count=img_size * img_size)        # contiguous
    if arr.size != img_size * img_size:
        raise ValueError(
            f"Could not parse cell: expected {img_size**2} values, got {arr.size}")
    return arr


def load_csv_frames(csv_file: str, img_size: int, num_frames: int) -> np.ndarray:
    """Load frames from CSV → (N, img_size, img_size, 1) float32 in {0,1}."""
    frames: Sequence[np.ndarray] = []
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("State"):
                continue  # header
            if len(row) != num_frames:
                raise ValueError(f"Row has {len(row)} cells, expected {num_frames}")
            for cell in row:
                arr = _parse_cell(cell, img_size)
                frames.append(arr.reshape(img_size, img_size, 1))
    frames = np.stack(frames, axis=0).astype(np.float32)
    print(f"[data] {csv_file}: {frames.shape[0]} frames of {img_size}×{img_size}")
    return frames

# -----------------------------------------------------------------------------
#  VAE model
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4, 4), (2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        mu     = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class Decoder(nn.Module):
    latent_dim: int
    img_size:  int

    @nn.compact
    def __call__(self, z):
        if self.img_size % 4 != 0:
            raise ValueError("img_size must be divisible by 4 (two stride‑2 convs).")
        h = self.img_size // 4                     # spatial size after encoder
        x = nn.Dense(h * h * 64)(z)
        x = x.reshape((-1, h, h, 64))
        x = nn.ConvTranspose(64, (4, 4), (2, 2), padding="SAME")(x)  # h → 2h
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (4, 4), (2, 2), padding="SAME")(x)  # 2h → 4h
        x = nn.relu(x)
        x = nn.Conv(1, (3, 3), padding="SAME")(x)
        return nn.sigmoid(x)


class VAE(nn.Module):
    latent_dim: int
    img_size:  int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.img_size)

    def __call__(self, x, rng):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, std.shape)
        z   = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar


# -----------------------------------------------------------------------------
#  Loss
# -----------------------------------------------------------------------------

def vae_loss(x, recon, mu, logvar, beta: float = 0.01):
    recon_loss = jnp.mean((x - recon) ** 2)
    kl         = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
    return recon_loss + beta * kl

# -----------------------------------------------------------------------------
#  Visualisation
# -----------------------------------------------------------------------------

def show_recon(inputs, recons, prefix: str, n: int = 5):
    inputs  = np.clip(np.array(inputs),  0.0, 1.0)
    recons  = np.clip(np.array(recons), 0.0, 1.0)
    for i in range(min(n, len(inputs))):
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(inputs[i].squeeze(), cmap="gray");  ax[0].set_title("Input");         ax[0].axis("off")
        ax[1].imshow(recons[i].squeeze(), cmap="gray"); ax[1].set_title("Reconstruction"); ax[1].axis("off")
        plt.tight_layout(); plt.savefig(f"{prefix}_{i}.png"); plt.close(fig)

# -----------------------------------------------------------------------------
#  Training loop
# -----------------------------------------------------------------------------

def train_vae(train_csv: str, val_csv: str, num_frames: int, img_size: int, latent_dim: int,
              total_steps: int, batch_size: int, learning_rate: float):

    train_ds = load_csv_frames(train_csv, img_size, num_frames)
    val_ds   = load_csv_frames(val_csv,   img_size, num_frames) if (val_csv and os.path.exists(val_csv)) else None

    model  = VAE(latent_dim, img_size)
    rng    = random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, img_size, img_size, 1), jnp.float32), rng)['params']
    opt    = optax.adam(learning_rate)
    opt_st = opt.init(params)

    def sample_batch(data, key):
        idx = random.randint(key, (batch_size,), 0, data.shape[0])
        return jnp.array(data[idx])

    @jit
    def train_step(p, opt_s, batch, key):
        def loss_fn(p_):
            recon, mu, logvar = model.apply({'params': p_}, batch, key)
            loss = vae_loss(batch, recon, mu, logvar)
            return loss, recon
        (loss, recon), grads = value_and_grad(loss_fn, has_aux=True)(p)
        updates, opt_s = opt.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss, recon

    @jit
    def eval_step(p, batch, key):
        recon, mu, logvar = model.apply({'params': p}, batch, key)
        return vae_loss(batch, recon, mu, logvar)

    for step in range(1, total_steps + 1):
        rng, key_batch, key_step = random.split(rng, 3)
        batch = sample_batch(train_ds, key_batch)
        params, opt_st, loss, recon = train_step(params, opt_st, batch, key_step)

        if step % 500 == 0:
            log = f"[{step}/{total_steps}] train={loss:.6f}"
            if val_ds is not None:
                rng, key_val = random.split(rng)
                val_loss = eval_step(params, sample_batch(val_ds, key_val), key_val)
                log += f"  val={val_loss:.6f}"
            print(log)
        if step % 5_000 == 0:
            show_recon(batch, recon, prefix=f"recon_step{step}")

    # save weights
    with open("encoder_params.pkl", "wb") as f:
        pickle.dump(params['encoder'], f)
    with open("decoder_params.pkl", "wb") as f:
        pickle.dump(params['decoder'], f)
    print("✓ Training finished – parameters saved.")


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",   required=True)
    ap.add_argument("--val_csv",     default="")
    ap.add_argument("--num_frames",  type=int, default=10)
    ap.add_argument("--img_size",    type=int, default=32)
    ap.add_argument("--latent_dim",  type=int, default=128)
    ap.add_argument("--total_steps", type=int, default=30_000)
    ap.add_argument("--batch_size",  type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    args = ap.parse_args()

    train_vae(args.train_csv, args.val_csv, args.num_frames, args.img_size,
              args.latent_dim, args.total_steps, args.batch_size, args.learning_rate)
