#!/usr/bin/env python3
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import wandb

# ─── Data loader: raw frames [N, T, H, W] ─────────────────────────────────────
def load_frames_from_csv(csv_file: str, img_size: int, num_frames: int) -> np.ndarray:
    import csv
    seqs = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[0].startswith('State'):
                continue
            if len(row) != num_frames:
                raise ValueError(f"Expected {num_frames} columns, got {len(row)}")
            frames = []
            for cell in row:
                s = cell.strip()
                if len(s) != img_size*img_size:
                    raise ValueError(f"Expected cell length {img_size*img_size}, got {len(s)}")
                arr = np.array([float(c) for c in s], dtype=np.float32)
                frames.append(arr.reshape(img_size, img_size))
            seqs.append(np.stack(frames, axis=0))
    return np.stack(seqs, axis=0)

# ─── Residual block that infers its feature‑size from x.shape[-1] ───────────────
class ResBlock(nn.Module):
    @nn.compact
    def __call__(self, x, t_emb):
        C = x.shape[-1]
        h = nn.GroupNorm()(x)
        h = nn.swish(h)
        h = nn.Conv(C, (3,3), padding='SAME')(h)

        scale_shift = nn.Dense(C*2)(t_emb)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        h = h * (1 + scale[:,None,None,:]) + shift[:,None,None,:]

        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        h = nn.Conv(C, (3,3), padding='SAME')(h)
        return x + h

# ─── U‑Net with ConvTranspose upsampling ────────────────────────────────────────
class UNet2D(nn.Module):
    base_channels: int = 64
    depth: int = 3

    @nn.compact
    def __call__(self, x, t):
        # time embedding
        t_emb = nn.Dense(self.base_channels)(
                    nn.swish(
                      nn.Dense(self.base_channels)(
                        sinusoidal_emb(t))))

        # initial conv to base_channels
        h = nn.Conv(self.base_channels, (3,3), padding='SAME')(x)
        skips = []

        # down‑path
        for _ in range(self.depth):
            h = ResBlock()(h, t_emb)
            skips.append(h)
            h = nn.avg_pool(h, window_shape=(2,2), strides=(2,2), padding='VALID')

        # bottleneck
        h = ResBlock()(h, t_emb)

        # up‑path
        for h_skip in reversed(skips):
            # upsample by 2× via transpose‑conv
            h = nn.ConvTranspose(
                    features=h_skip.shape[-1],
                    kernel_size=(4,4),
                    strides=(2,2),
                    padding='SAME')(h)
            # concat skip
            h = jnp.concatenate([h, h_skip], axis=-1)
            h = ResBlock()(h, t_emb)

        # final normalization + output conv
        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        return nn.Conv(1, (3,3), padding='SAME')(h)

# ─── Sinusoidal time embedding ──────────────────────────────────────────────────
def sinusoidal_emb(t, dim=64):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(jnp.array(10000., dtype=jnp.float32)) *
                    (jnp.arange(half, dtype=jnp.float32) / half))
    phases = t[:,None].astype(jnp.float32) * freqs[None,:]
    return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)], axis=-1)

# ─── q_sample and training/eval steps (unchanged) ──────────────────────────────
def q_sample(x0, t, key, alphas_cum):
    noise = jax.random.normal(key, x0.shape)
    a = alphas_cum[t]
    return (
        jnp.sqrt(a)[:,None,None,None] * x0 +
        jnp.sqrt(1.0 - a)[:,None,None,None] * noise
    ), noise

def train_step(state, batch, t, key, alphas_cum, model):
    t = t.astype(jnp.int32)
    def loss_fn(params):
        xk, xk1 = batch; xk1 = xk1[...,None]
        xt, noise = q_sample(xk1, t, key, alphas_cum)
        pred = model.apply({'params':params}, xt, t)
        return jnp.mean((noise - pred)**2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
train_step = jax.jit(train_step, static_argnames=('model',))

def eval_step(params, batch, t, key, alphas_cum, model):
    t = t.astype(jnp.int32)
    xk, xk1 = batch; xk1 = xk1[...,None]
    xt, noise = q_sample(xk1, t, key, alphas_cum)
    pred = model.apply({'params':params}, xt, t)
    return jnp.mean((noise - pred)**2)
eval_step = jax.jit(eval_step, static_argnames=('model',))

def make_pairs(data: np.ndarray) -> np.ndarray:
    pairs = []
    for seq in data:
        for k in range(seq.shape[0]-1):
            pairs.append((seq[k], seq[k+1]))
    return np.stack(pairs, axis=0)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="conway_train.csv")
    parser.add_argument("--val_csv",   default="conway_val.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps",      type=int, default=5000)
    parser.add_argument("--log_every",  type=int, default=100)
    parser.add_argument("--img_size",   type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=20)
    args = parser.parse_args()

    wandb.init(project="gol_diffusion", config=vars(args))

    train_data = load_frames_from_csv(args.train_csv, args.img_size, args.num_frames)
    val_data   = load_frames_from_csv(args.val_csv,   args.img_size, args.num_frames)
    N, T, H, W = train_data.shape
    print(f"Train: {N}×{T} frames of size {H}×{W}; Val: {val_data.shape[0]} trajectories")

    train_pairs = make_pairs(train_data)
    val_pairs   = make_pairs(val_data)
    print(f"Train pairs: {train_pairs.shape[0]}, Val pairs: {val_pairs.shape[0]}")

    model = UNet2D(base_channels=64, depth=3)
    dummy_x = jnp.ones([1, H, W, 1], dtype=jnp.float32)
    dummy_t = jnp.array([0], dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t)['params']
    tx = optax.adam(1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    betas = jnp.array(np.linspace(1e-4, 2e-2, 1000), dtype=jnp.float32)
    alphas_cum = jnp.cumprod(1.0 - betas)

    key = jax.random.PRNGKey(42)
    ema_loss = 0.0; decay = 0.99

    for i in range(1, args.steps+1):
        idx = np.random.randint(0, train_pairs.shape[0], size=(args.batch_size,))
        batch = train_pairs[idx]
        key, sub = jax.random.split(key)
        t = jax.random.randint(sub, (args.batch_size,), 0, betas.shape[0])

        state, loss = train_step(state, (batch[:,0], batch[:,1]), t, sub, alphas_cum, model)
        ema_loss = decay*ema_loss + (1-decay)*float(loss)

        if i % args.log_every == 0:
            val_mse = 0.0
            for _ in range(5):
                idx_v = np.random.randint(0, val_pairs.shape[0], size=(args.batch_size,))
                batch_v = val_pairs[idx_v]
                key, sub = jax.random.split(key)
                t_v = jax.random.randint(sub, (args.batch_size,), 0, betas.shape[0])
                val_mse += float(eval_step(state.params, (batch_v[:,0], batch_v[:,1]),
                                           t_v, sub, alphas_cum, model))
            val_mse /= 5
            print(f"[{i:5d}] train_loss={loss:.6f}, ema={ema_loss:.6f}, val_mse={val_mse:.6f}")
            wandb.log({"train/loss":float(loss),
                       "train/ema_loss":ema_loss,
                       "val/mse":val_mse}, step=i)

    os.makedirs("ckpts", exist_ok=True)
    with open("ckpts/diffusion_pairs_params.pkl","wb") as f:
        import pickle; pickle.dump(state.params, f)
    print("Saved params to ckpts/diffusion_pairs_params.pkl")

if __name__ == "__main__":
    main()
