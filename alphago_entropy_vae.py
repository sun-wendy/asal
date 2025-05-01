#!/usr/bin/env python3
import argparse
import pickle
import math
import random
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze as freeze_dict
from flax.training import train_state
import optax
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from util_gol import frame_to_tokens
from conway_lib import ConwayGame


def compute_reward(arr: np.ndarray,
                             min_period: int = 2,
                             max_shift: int = None,
                             period_weight: float = 1.0
                            ) -> float:
    """
    Reward high translational‐periodicity (“spaceshipness”).

    Args:
      arr: array of shape (T, H, W) of 0/1 frames
      min_period: smallest period P to consider
      max_shift: maximum |u|,|v| to consider (default: H//2)
      period_weight: exponent on best similarity

    Returns:
      best_sim**period_weight, or 0.0 if any frame is trivial.
    """
    T, H, W = arr.shape
    total = H * W

    # zero reward if any frame is all-0 or all-1
    sums = [frame.sum() for frame in arr]
    if any(s==0 for s in sums) or any(s==total for s in sums):
        return 0.0

    if max_shift is None:
        max_shift = H // 2

    first = arr[0]
    best_sim = 0.0

    # for each candidate period
    for P in range(min_period, T):
        frameP = arr[P]
        # for each nonzero shift
        for u in range(-max_shift, max_shift+1):
            for v in range(-max_shift, max_shift+1):
                if u==0 and v==0:
                    continue
                # roll frame P by (u,v)
                rolled = np.roll(np.roll(frameP, shift=u, axis=0),
                                 shift=v, axis=1)
                sim = (first == rolled).mean()
                if sim > best_sim:
                    best_sim = sim

    return best_sim**period_weight

# -----------------------
# Reward + entropy (unchanged)
# -----------------------
def compute_oscillator_reward(arr: np.ndarray,
                   min_period: int = 2,
                   period_weight: float = 1.0) -> float:
    """
    Reward = pure periodicity, with trivial all-dark or all-light patterns zeroed out.

    Args:
      arr: array of shape (T, H, W) of 0/1 frames from ConwayGame simulation
      min_period: minimum P to consider when measuring return-to-self
      period_weight: exponent on best_sim

    Returns:
      best_sim**period_weight, unless the initial frame is trivial (all 0s or all 1s),
      in which case 0.0.
    """
    T, H, W = arr.shape
    total = H * W

    # trivial fixed-point (all dark or all light) gets zero reward
    s = [frame.sum() for frame in arr]
    if any(si == 0 for si in s) or any(si == total for si in s):
        return 0.0

    # find maximum similarity to initial frame at any P ≥ min_period
    first = arr[0]
    best_sim = 0.0
    for P in range(min_period, T):
        sim = (first == arr[P]).mean()
        if sim > best_sim:
            best_sim = sim

    return best_sim**period_weight


def compute_patch_temporal_entropy(arr: np.ndarray, patches_per_dim: int) -> np.ndarray:
    L, H, W = arr.shape
    k = H // patches_per_dim
    entropies = []
    for pi in range(patches_per_dim):
        for pj in range(patches_per_dim):
            patch_seq = arr[:, pi*k:(pi+1)*k, pj*k:(pj+1)*k]
            patterns = patch_seq.reshape(L, k*k)
            uniq, counts = np.unique(patterns, axis=0, return_counts=True)
            probs = counts / counts.sum()
            H_val = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(H_val / (k * k))
    return np.array(entropies)

def decode_latents(z, img_size, patches, model, params, vae, vae_params, μ_vec, σ_vec):
    """
    z: VAE latent vector, shape (latent_dim,)
    μ_vec, σ_vec: arrays of shape (T, C) used for un-normalization
    vae, vae_params: your trained VAE and its params
    model, params: your GPT and its params
    """
    # 1) decode through VAE → recon_norm shape (1, T, C)
    recon_norm = vae.apply({'params': vae_params},z[None], method=LatentVAE.decode)
    recon_norm = np.array(recon_norm)  # (1, T, C)

    # 2) un‐normalize to GPT mid‐latents
    recon_lat = recon_norm * (σ_vec[None] + 1e-6) + μ_vec[None]  # still shape (1, T, C)

    # 3) feed into GPT.decode_mid → logits shape (1, T, token_dim)
    logits = model.apply({'params': params}, recon_lat, method=GPT.decode_mid)[0]  # (T, token_dim)

    # 4) binarize and stitch patches
    bits = (np.array(logits) > 0.0).astype(int)
    patch_size = img_size // patches
    patches_arr = bits.reshape(patches**2, patch_size, patch_size)
    frame = np.zeros((img_size, img_size), dtype=int)
    idx = 0
    for r in range(patches):
        for c in range(patches):
            frame[r*patch_size:(r+1)*patch_size,
                  c*patch_size:(c+1)*patch_size] = patches_arr[idx]
            idx += 1

    return frame


# -----------------------
# GPT definitions (unchanged)
# -----------------------
@dataclass
class GPTConfig:
    img_size: int
    block_size: int
    token_dim: int
    num_tokens: int
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        assert c.n_embd % c.n_head == 0
        self.head_size = c.n_embd // c.n_head
        self.n_head = c.n_head
        self.c_attn = nn.Dense(c.n_embd * 3)
        self.c_proj = nn.Dense(c.n_embd)
        self.attn_dropout = nn.Dropout(c.dropout)
        self.resid_dropout = nn.Dropout(c.dropout)
    def __call__(self, x, *, train: bool):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        k = k.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        v = v.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        idx = jnp.arange(T)
        frame_idx = idx // self.config.num_tokens
        mask = (frame_idx[None, :] <= frame_idx[:, None]).reshape(1,1,T,T)
        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.head_size)
        att = jnp.where(mask, att, float("-inf"))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.transpose(0,2,1,3).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y), deterministic=not train)

class MLPBlock(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        self.c_fc = nn.Dense(4 * c.n_embd)
        self.c_proj = nn.Dense(c.n_embd)
        self.dropout = nn.Dropout(c.dropout)
    def __call__(self, x, *, train: bool):
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x, deterministic=not train)

class Block(nn.Module):
    config: GPTConfig
    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLPBlock(self.config)
    def __call__(self, x, *, train: bool):
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x

class GPT(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        self.token_proj = nn.Dense(c.n_embd)
        self.wpe = nn.Embed(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.h = [Block(c) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5)
        self.head = nn.Dense(c.token_dim)
    def __call__(self, tokens, *, train: bool):
        latents = self.get_mid_latents(tokens, train=train)
        return self.decode_mid(latents, train=train), None
    def get_mid_latents(self, tokens, *, train=False):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        mid = self.config.n_layer // 2
        for blk in self.h[:mid]:
            x = blk(x, train=train)
        return x
    def decode_mid(self, latents, *, train=False):
        x = latents
        mid = self.config.n_layer // 2
        for blk in self.h[mid:]:
            x = blk(x, train=train)
        x = self.ln_f(x)
        return self.head(x)


def encode_frame(frame, img_size, patches, model, params):
    tokens = frame_to_tokens(frame[...,None], (patches, patches))
    lat = model.apply({'params': params}, jnp.array(tokens[None]), method=GPT.get_mid_latents)
    return np.array(lat[0])

# -----------------------
# VAE for mid-layer latents with normalization
# -----------------------
class LatentVAE(nn.Module):
    latent_dim: int
    hidden_dim: int
    T: int
    C: int

    def setup(self):
        self.fc_enc = nn.Dense(self.hidden_dim)
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)
        self.fc_dec_h = nn.Dense(self.hidden_dim)
        self.fc_dec_o = nn.Dense(self.T * self.C)

    def __call__(self, x_norm):
        B, T, C = x_norm.shape
        h = x_norm.reshape(B, -1)
        h = nn.relu(self.fc_enc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        eps = jax.random.normal(self.make_rng('vae'), mu.shape)
        z = mu + jnp.exp(0.5 * logvar) * eps
        h2 = nn.relu(self.fc_dec_h(z))
        recon_norm = self.fc_dec_o(h2).reshape(B, T, C)
        return recon_norm, mu, logvar

    def decode(self, z):
        B = z.shape[0]
        h2 = nn.relu(self.fc_dec_h(z))
        recon_norm = self.fc_dec_o(h2).reshape(B, self.T, self.C)
        return recon_norm

# Loss with KL weight = 1.0
def vae_loss(recon_norm, x_norm, mu, logvar):
    recon_loss = jnp.mean((recon_norm - x_norm)**2)
    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
    return recon_loss + 0.2 * kl, (recon_loss, kl)

# -----------------------
# RL helper classes (unchanged)
# -----------------------
class Policy:
    def __init__(self, latent_shape, sigma_init=1.0, lr=1e-2):
        self.mu = np.zeros(latent_shape, np.float32)
        self.log_sigma = np.log(sigma_init)
        self.lr = lr
    def sample(self, n):
        sigma = np.exp(self.log_sigma)
        eps = np.random.randn(n, *self.mu.shape)
        return self.mu[None] + sigma * eps, eps
    def update(self, eps, adv):
        sigma2 = np.exp(self.log_sigma)**2
        grad = (adv[:,None] * eps / sigma2).mean(axis=0)
        self.mu += self.lr * grad

class ValueNet:
    def __init__(self, latent_shape):
        self.model = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=100, warm_start=True)
    def fit(self, Z, y):
        self.model.fit(Z.reshape(Z.shape[0], -1), y)
    def predict(self, Z):
        return self.model.predict(Z.reshape(Z.shape[0], -1))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buf = []
    def add(self, z, r):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append((z, r))
    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        Z, R = zip(*(self.buf[i] for i in idx))
        return np.stack(Z), np.array(R)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patches", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--patches_per_dim", type=int, default=2)
    p.add_argument("--init_mu_bias", type=float, default=0.0)
    p.add_argument("--vae_latent_dim", type=int, default=64)
    p.add_argument("--vae_hidden_dim", type=int, default=512)
    p.add_argument("--vae_epochs", type=int, default=10)
    p.add_argument("--vae_batch", type=int, default=64)
    args = p.parse_args()

    # load GPT checkpoint
    ckpt = pickle.load(open(args.checkpoint, "rb")); ckpt = unfreeze(ckpt)
    true_block = ckpt["wpe"]["embedding"].shape[0]
    num_tokens = args.patches**2
    token_dim = (args.img_size // args.patches)**2
    config = GPTConfig(args.img_size, true_block, token_dim, num_tokens)
    model = GPT(config)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, true_block, num_tokens))
    model.init(rng, dummy, train=False)
    params = ckpt

    # 1) Gather mid-layer latents
    latents_list = []
    for _ in range(1000):
        frame = (np.random.rand(args.img_size, args.img_size) < 0.5).astype(int)
        lat = encode_frame(frame, args.img_size, args.patches, model, params)
        latents_list.append(lat)
    X = np.stack(latents_list)
    μ_vec = X.mean(axis=0)      # shape (T, C)
    σ_vec = X.std(axis=0)       # shape (T, C)
    X_norm = (X - μ_vec) / (σ_vec + 1e-6)
    # print(f"Latent mean={μ:.4f}, std={σ:.4f}")

    # 2) Train VAE on X_norm
    T, C = X_norm.shape[1], X_norm.shape[2]
    vae = LatentVAE(latent_dim=args.vae_latent_dim, hidden_dim=args.vae_hidden_dim, T=T, C=C)
    dummy_lat = jnp.zeros((1, T, C))
    vae_vars = vae.init({'params': rng, 'vae': rng}, dummy_lat)
    state = train_state.TrainState.create(apply_fn=vae.apply, params=vae_vars['params'], tx=optax.adam(1e-3))

    @jax.jit
    def train_step(state, batch_norm):
        def loss_fn(params):
            recon_norm, mu, logvar = vae.apply({'params': params}, batch_norm, rngs={'vae': rng})
            loss, (r_l, k_l) = vae_loss(recon_norm, batch_norm, mu, logvar)
            return loss, (r_l, k_l)
        grads, (r_l, k_l) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, r_l, k_l

    for epoch in range(args.vae_epochs):
        perm = np.random.permutation(X_norm.shape[0])
        for i in range(X_norm.shape[0] // args.vae_batch):
            batch = jnp.array(X_norm[perm[i*args.vae_batch:(i+1)*args.vae_batch]])
            state, r_l, k_l = train_step(state, batch)
        print(f"VAE epoch {epoch+1} recon_norm={r_l:.6f} kl={k_l:.6f}")
    vae_params = state.params

    # 3) Random baseline (unchanged)
    rand_rews = []
    for _ in range(args.pop):
        frame = (np.random.rand(args.img_size, args.img_size) < 0.5).astype(int)
        game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
        game.grid = (frame == 1)
        seq = game.run(num_iterations=args.steps)
        arr = np.stack([f.astype(int) for f in seq])
        rand_rews.append(-np.min(compute_patch_temporal_entropy(arr, args.patches_per_dim)))
    random_baseline = np.mean(rand_rews)
    print(f"Random baseline = {random_baseline:.4f}")

    # 4) RL in VAE space
    policy = Policy((args.vae_latent_dim,), sigma_init=args.sigma, lr=args.lr)
    policy.mu -= args.init_mu_bias
    value = ValueNet((args.vae_latent_dim,))
    buffer = ReplayBuffer(capacity=args.buffer_size)
    mean_rewards = []

    for gen in range(1, args.iters+1):
        Z, eps = policy.sample(args.pop)
        rewards = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            recon_norm = vae.apply({'params': vae_params}, jnp.array(Z[i][None]), method=LatentVAE.decode, rngs={'vae': rng})
            recon_norm = np.array(recon_norm[0])
            recon_lat = recon_norm * (σ_vec + 1e-6) + μ_vec
            logits = model.apply({'params': params}, recon_lat[None], method=GPT.decode_mid)[0]
            bits = (np.array(logits) > 0.0).astype(int)
            patch_size = args.img_size // args.patches
            patches_arr = bits.reshape(args.patches**2, patch_size, patch_size)
            frame = np.zeros((args.img_size, args.img_size), int)
            idx = 0
            for r in range(args.patches):
                for c in range(args.patches):
                    frame[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = patches_arr[idx]
                    idx += 1
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (frame == 1)
            seq = game.run(num_iterations=args.steps)
            arr = np.stack([f.astype(int) for f in seq])
            rewards[i] = -np.min(compute_reward(arr, args.patches_per_dim))

        mean_rewards.append(rewards.mean())
        for i, r in enumerate(rewards): buffer.add(Z[i], r)
        if len(buffer.buf) >= args.pop:
            Zb, Rb = buffer.sample(args.pop)
            value.fit(Zb, Rb)
        adv = (rewards - value.predict(Z))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy.update(eps, adv)
        print(f"[gen {gen:4d}] mean={rewards.mean():.4f} max={rewards.max():.4f}")

        
        # visualize some rollouts
        n_display = min(10, args.pop)
        histories = []
        for i in range(n_display):
            f0 = decode_latents(
                Z[i],
                args.img_size,
                args.patches,
                model,
                params,
                vae,
                vae_params,
                μ_vec,
                σ_vec
            )
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (f0 == 1)
            hist = game.run(num_iterations=args.steps)
            histories.append([h.astype(int) for h in hist])

        fig, axes = plt.subplots(n_display, args.steps,
                                 figsize=(args.steps*1.5, n_display*1.5))
        for r in range(n_display):
            for c in range(args.steps):
                ax = axes[r, c]
                ax.imshow(histories[r][c], cmap='gray', interpolation='nearest')
                ax.axis('off')
                if r == 0: ax.set_title(f"Step {c}")
        plt.tight_layout()
        plt.savefig(f"alphago_stent_gen_{gen:04d}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()
        

    # 5) Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.iters+1), mean_rewards, label="Mean Reward")
    plt.axhline(random_baseline, linestyle="--", label="Random Baseline")
    plt.xlabel("Generation")
    plt.ylabel("Reward (−min patch entropy)")
    plt.title("RL Reward vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_vs_iteration.png")
    print("Saved reward plot to reward_vs_iteration.png")
