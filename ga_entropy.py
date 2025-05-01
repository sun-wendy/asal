#!/usr/bin/env python3
import os
import argparse
import pickle
import math

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from util_gol import frame_to_tokens   # your tokenizer
from conway_lib import ConwayGame        # your Game of Life class
from dataclasses import dataclass
from typing import Optional

# ─── REWARD FUNCTION ──────────────────────────────────────────────────────────

def compute_reward(arr: np.ndarray,
                   min_period: int = 2,
                   period_weight: float = 1.0) -> float:
    """
    Reward = pure periodicity, with trivial all‑dark or all‑light patterns zeroed out.

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
    if any(si < 10 for si in s) or any(si < 10 for si in s):
        return 0.0

    # find maximum similarity to initial frame at any P ≥ min_period
    best_sim = 0.0
    first = arr[0]
    for P in range(min_period, T):
        sim = (first == arr[P]).mean()
        if sim > best_sim:
            best_sim = sim

    return best_sim**period_weight

# ─── GPT DEFINITION WITH MID-LAYER ACCESS ───────────────────────────────────────

@dataclass
class GPTConfig:
    img_size: int
    block_size: int        # maximum sequence length from checkpoint
    token_dim: int         # vocab or token feature dimension
    num_tokens: int        # patch count per frame
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
    def __call__(self, x, *, train):
        B,T,C = x.shape
        qkv = self.c_attn(x)
        q,k,v = jnp.split(qkv,3,axis=-1)
        q = q.reshape(B,T,self.n_head,self.head_size).transpose(0,2,1,3)
        k = k.reshape(B,T,self.n_head,self.head_size).transpose(0,2,1,3)
        v = v.reshape(B,T,self.n_head,self.head_size).transpose(0,2,1,3)
        idx = jnp.arange(T)
        frame_idx = idx // self.config.num_tokens
        mask = (frame_idx[None,:] <= frame_idx[:,None]).reshape(1,1,T,T)
        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.head_size)
        att = jnp.where(mask, att, float("-inf"))
        att = nn.softmax(att,axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.transpose(0,2,1,3).reshape(B,T,C)
        return self.resid_dropout(self.c_proj(y), deterministic=not train)

class MLPBlock(nn.Module):
    config: GPTConfig
    def setup(self):
        c=self.config
        self.c_fc = nn.Dense(4*c.n_embd)
        self.c_proj = nn.Dense(c.n_embd)
        self.dropout = nn.Dropout(c.dropout)
    def __call__(self,x,*,train):
        x=self.c_fc(x); x=nn.gelu(x); x=self.c_proj(x)
        return self.dropout(x, deterministic=not train)

class Block(nn.Module):
    config: GPTConfig
    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = MLPBlock(self.config)
    def __call__(self,x,*,train):
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
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(c.token_dim)

    def get_mid_latents(self, tokens, *, train=False):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None,:]
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

# ─── ENCODE/DECODE UTILITIES USING MID-LAYER ──────────────────────────────────

def encode_frame(frame, img_size, patches, model, params):
    tokens = frame_to_tokens(frame[...,None], (patches, patches))
    lat = model.apply({'params': params}, jnp.array(tokens[None]), method=GPT.get_mid_latents, train=False)
    return np.array(lat[0])

def decode_latents(z, img_size, patches, model, params):
    logits = model.apply({'params': params}, jnp.array(z[None]), method=GPT.decode_mid, train=False)[0]
    bits = (np.array(logits) > 0).astype(int)
    patch_size = img_size // patches
    patches_arr = bits.reshape(patches**2, patch_size, patch_size)
    frame = np.zeros((img_size, img_size), int)
    idx = 0
    for i in range(patches):
        for j in range(patches):
            frame[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches_arr[idx]
            idx += 1
    return frame

# ─── GA IN LATENT SPACE WITH REWARD & VISUALIZATION ───────────────────────────

def simulate_reward(z, img_size, patches, model, params, steps, toroidal):
    frame = decode_latents(z, img_size, patches, model, params)
    game = ConwayGame(width=img_size, height=img_size, grid_size=1, toroidal=toroidal)
    game.grid = (frame==1)
    seq = game.run(num_iterations=steps)
    arr = np.stack([f.astype(int) for f in seq])
    return compute_reward(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patches", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--child", type=int, default=64)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--toroidal", action="store_true")
    args = p.parse_args()

    ckpt = pickle.load(open(args.checkpoint, "rb"))
    true_block = ckpt["wpe"]["embedding"].shape[0]
    num_tokens = args.patches**2
    token_dim  = (args.img_size//args.patches)**2
    config = GPTConfig(args.img_size, true_block, token_dim, num_tokens)
    model = GPT(config)
    params = ckpt

    # random baseline: reward of random frames
    rand_rews = []
    for _ in range(args.pop):
        z = encode_frame((np.random.rand(args.img_size, args.img_size)<0.5).astype(int),
                         args.img_size, args.patches, model, params)
        rand_rews.append(simulate_reward(z, args.img_size, args.patches, model, params, args.steps, args.toroidal))
    print(f"Random baseline reward: mean={np.mean(rand_rews):.4f}, std={np.std(rand_rews):.4f}")

    # initialize population in latent space
    pop_frames = (np.random.randn(args.pop, args.img_size, args.img_size) < 0.5).astype(int)
    pop_latents = np.stack([encode_frame(f, args.img_size, args.patches, model, params)
                             for f in pop_frames])

    T, n_embd = pop_latents.shape[1:]
    for gen in range(1, args.iters+1):
        rewards = np.array([simulate_reward(z, args.img_size, args.patches, model, params, args.steps, args.toroidal)
                             for z in pop_latents])
        print(f"[gen {gen:4d}] rew min={rewards.min():.4f} mean={rewards.mean():.4f} max={rewards.max():.4f}")

        # visualize top individuals
        top_idx = np.argsort(-rewards)[:8]
        histories = []
        for i in top_idx:
            z = pop_latents[i]
            frame = decode_latents(z, args.img_size, args.patches, model, params)
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=args.toroidal)
            game.grid = (frame==1)
            histories.append(game.run(num_iterations=args.steps))

        fig, axes = plt.subplots(8, args.steps, figsize=(args.steps*1.2, 8*1.2))
        for r, seq in enumerate(histories):
            for c, fr in enumerate(seq):
                ax = axes[r, c]
                ax.imshow(fr, cmap='gray', interpolation='nearest')
                ax.axis('off')
        plt.tight_layout()
        os.makedirs("vis", exist_ok=True)
        plt.savefig(f"vis/ga_rew_gen_{gen:04d}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # select parents by highest reward
        idx = np.argsort(-rewards)
        parents = pop_latents[idx[:args.pop//2]]

        # breed children
        n_mut = args.child//2; n_copy = args.child - n_mut
        muts = []
        for _ in range(n_mut):
            p = parents[np.random.randint(len(parents))].copy()
            tok = np.random.randint(T)
            p[tok] += np.random.randn(n_embd) * args.sigma
            muts.append(p)
        muts = np.stack(muts)
        cp_idx = np.random.randint(len(parents), size=n_copy)
        children = np.concatenate([muts, parents[cp_idx]], axis=0)
        pop_latents = np.concatenate([parents, children], axis=0)

    # final best
    final_best = pop_latents[np.argmax(rewards)]
    print("best final reward:", simulate_reward(final_best, args.img_size, args.patches, model, params, args.steps, args.toroidal))

if __name__=="__main__":
    main()
