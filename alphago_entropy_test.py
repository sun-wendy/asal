#!/usr/bin/env python3
import argparse
import pickle
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import unfreeze
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from util_gol import frame_to_tokens
from conway_lib import ConwayGame

# ── compute normalized spatio-temporal entropy ─────────────────────────────────
def compute_st_entropy_arr(arr: np.ndarray, patches_per_dim: int) -> float:
    L, H, W = arr.shape
    k = H // patches_per_dim
    blocks = arr.reshape(L, patches_per_dim, k, patches_per_dim, k)
    blocks = blocks.transpose(1,3,0,2,4).reshape(-1, L * k * k)
    uniq, counts = np.unique(blocks, axis=0, return_counts=True)
    probs = counts / counts.sum()
    Hval = -np.sum(probs * np.log2(probs + 1e-12))
    return Hval / (L * k * k)

# ── compute per-patch temporal entropy (to detect oscillators) ─────────────────
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
            H = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(H / (k * k))
    return np.array(entropies)

# ── GPT DEFINITION ─────────────────────────────────────────────────────────────
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
        B, T, d = tokens.shape
        x = self.token_proj(tokens)
        pos = jnp.arange(T)[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for blk in self.h:
            x = blk(x, train=train)
        x = self.ln_f(x)
        return self.head(x), None
    def get_token_latents(self, tokens, *, train=False, layer: Optional[int]=None, apply_ln=True):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for i, blk in enumerate(self.h):
            x = blk(x, train=train)
            if layer is not None and i == layer:
                break
        if apply_ln:
            x = self.ln_f(x)
        return x

# ── Utilities ─────────────────────────────────────────────────────────────
def encode_frame(frame, img_size, patches, model, params):
    tokens = frame_to_tokens(frame[...,None], (patches, patches))
    lat = model.apply({'params': params}, jnp.array(tokens[None]), method=GPT.get_token_latents)
    return np.array(lat[0])

def decode_latents(latents, img_size, patches, model, params):
    x = jnp.array(latents[None])
    logits = model.apply({'params': params}, x, method=lambda m,x,train=False: m.head(m.ln_f(x)), train=False)[0]
    bits = (np.array(logits) > 0.0).astype(int)  # fixed threshold at zero
    patch_size = img_size // patches
    patches_arr = bits.reshape(patches**2, patch_size, patch_size)
    frame = np.zeros((img_size, img_size), int)
    idx = 0
    for i in range(patches):
        for j in range(patches):
            frame[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches_arr[idx]
            idx += 1
    return frame

# ── AlphaGo-style RL components ─────────────────────────────────────────────
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
        grad = (adv[:,None,None] * eps / sigma2).mean(axis=0)
        self.mu += self.lr * grad

class ValueNet:
    def __init__(self, latent_shape):
        flat = latent_shape[0] * latent_shape[1]
        self.model = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=5, warm_start=True)
    def fit(self, Z, y): self.model.fit(Z, y)
    def predict(self, Z): return self.model.predict(Z)

class ReplayBuffer:
    def __init__(self, capacity=10000): self.buf = []
    def add(self, z, r): self.buf.append((z, r))
    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        data = [self.buf[i] for i in idx]
        Z, R = zip(*data)
        return np.stack(Z), np.array(R)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patches", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mcts_sims", type=int, default=20)
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--patches_per_dim", type=int, default=2)
    
    p.add_argument("--init_mu_bias", type=float, default=0.0, help="initial bias to subtract from mu to control sparsity of alive cells")
    args = p.parse_args()

    # load model checkpoint
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

    # initialize RL components
    latent0 = encode_frame((np.random.rand(args.img_size, args.img_size)<0.5).astype(int), args.img_size, args.patches, model, params)
    latent_shape = latent0.shape
    policy = Policy(latent_shape, sigma_init=args.sigma, lr=args.lr)
    # apply initial latent-space bias to mu for sparsity
    policy.mu = policy.mu - args.init_mu_bias  
    value = ValueNet(latent_shape)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    for gen in range(1, args.iters+1):
        Z, eps = policy.sample(args.pop)
        rewards = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            frame = decode_latents(Z[i], args.img_size, args.patches, model, params)
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (frame == 1)
            seq = game.run(num_iterations=args.steps)
            arr = np.stack([f.astype(int) for f in seq])
            patch_ents = compute_patch_temporal_entropy(arr, args.patches_per_dim)
            rewards[i] = -np.min(patch_ents)

        for i, r in enumerate(rewards): buffer.add(Z[i], r)
        if len(buffer.buf) >= args.pop:
            Zb, Rb = buffer.sample(args.pop)
            value.fit(Zb.reshape(args.pop, -1), Rb)

        Z_imp = []
        for z0 in Z:
            best_z, best_v = z0, -1e9
            for _ in range(args.mcts_sims):
                z1 = z0 + np.random.randn(*z0.shape) * args.sigma
                v1 = value.predict(z1.reshape(1, -1))[0]
                if v1 > best_v:
                    best_v, best_z = v1, z1
            Z_imp.append(best_z)
        Z_imp = np.stack(Z_imp)

        Vpred = value.predict(Z.reshape(args.pop, -1))
        adv = rewards - Vpred
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy.update(eps, adv)

        print(f"[gen {gen:4d}] neg-ent mean={rewards.mean():.4f} max={rewards.max():.4f}")

        n_display = min(10, args.pop)
        histories = []
        for i in range(n_display):
            f0 = decode_latents(Z_imp[i], args.img_size, args.patches, model, params)
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (f0 == 1)
            hist = game.run(num_iterations=args.steps)
            histories.append([f0] + [h.astype(int) for h in hist])

        fig, axes = plt.subplots(n_display, args.steps+1,
                                 figsize=((args.steps+1)*1.5, n_display*1.5))
        for r in range(n_display):
            for c in range(args.steps+1):
                ax = axes[r, c]
                ax.imshow(histories[r][c], cmap='gray', interpolation='nearest')
                ax.axis('off')
                if r == 0:
                    ax.set_title(f"Step {c}")
        plt.tight_layout()
        plt.savefig(f"alphago_stent_gen_{gen:04d}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

if __name__ == "__main__":
    main()
