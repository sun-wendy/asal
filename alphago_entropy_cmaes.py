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
from flax.core.frozen_dict import unfreeze
import numpy as np
import matplotlib.pyplot as plt
import cma                                # pip install cma

from util_gol import frame_to_tokens
from conway_lib import ConwayGame


def compute_reward(arr: np.ndarray,
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
    best_sim = 0.0
    first = arr[0]
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
            H = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(H / (k * k))
    return np.array(entropies)


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


def decode_latents(z, img_size, patches, model, params):
    logits = model.apply({'params': params}, jnp.array(z[None]), method=GPT.decode_mid)[0]
    bits = (np.array(logits) > 0.0).astype(int)
    patch_size = img_size // patches
    patches_arr = bits.reshape(patches**2, patch_size, patch_size)
    frame = np.zeros((img_size, img_size), int)
    idx = 0
    for i in range(patches):
        for j in range(patches):
            frame[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches_arr[idx]
            idx += 1
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patches", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--patches_per_dim", type=int, default=2)
    p.add_argument("--init_mu_bias", type=float, default=0.0)
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

    # random baseline
    rand_rews = []
    for _ in range(args.pop):
        frame = (np.random.rand(args.img_size, args.img_size) < 0.5).astype(int)
        game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
        game.grid = (frame == 1)
        seq = game.run(num_iterations=args.steps)
        arr = np.stack([f.astype(int) for f in seq])
        rand_rews.append(compute_reward(arr, args.patches_per_dim))
    random_baseline = np.mean(rand_rews)
    print(f"Random baseline = {random_baseline:.4f}")

    # initial latent and flatten
    latent0 = encode_frame((np.random.rand(args.img_size, args.img_size) < 0.5).astype(int),
                            args.img_size, args.patches, model, params)
    x0 = latent0.flatten()

    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {'popsize': args.pop, 'seed': 0})

    mean_rewards = []
    for gen in range(1, args.iters+1):
        sols = es.ask()
        Z = np.stack([np.array(s).reshape(latent0.shape) for s in sols])

        rewards = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            frame = decode_latents(Z[i], args.img_size, args.patches, model, params)
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (frame == 1)
            seq = game.run(num_iterations=args.steps)
            arr = np.stack([f.astype(int) for f in seq])
            rewards[i] = compute_reward(arr, args.patches_per_dim)

        mean_rewards.append(rewards.mean())
        print(f"[gen {gen:4d}] mean={rewards.mean():.4f} max={rewards.max():.4f}")

        # minimize -reward
        es.tell(sols, (-rewards).tolist())
        es.disp()

        if gen % 1 == 0 or gen == 1:
            # pick the top-10 solutions by reward
            top10 = np.argsort(rewards)[-10:]
            # make a 10-row, (steps+1)-col grid
            fig, axes = plt.subplots(
                nrows=10,
                ncols=args.steps+1,
                figsize=((args.steps+1)*1.0, 10*1.0),
                squeeze=False
            )
            for row, i in enumerate(top10):
                Z_i = Z[i].reshape(latent0.shape)
                frame0 = decode_latents(Z_i, args.img_size, args.patches, model, params)
                game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
                game.grid = (frame0 == 1)
                hist = game.run(num_iterations=args.steps)
                # plot each time-step in the row
                for col, frm in enumerate(hist):
                    ax = axes[row, col]
                    ax.imshow(frm, cmap='gray', interpolation='nearest')
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"cmaes_top10_gen_{gen:04d}.png")
            plt.close()

    # final plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.iters+1), mean_rewards, label="Mean Reward")
    plt.axhline(random_baseline, linestyle="--", label="Random Baseline")
    plt.xlabel("Generation")
    plt.ylabel("Reward (−min patch entropy)")
    plt.title("CMA-ES Reward vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cmaes_reward_vs_iteration.png")
    print("Saved reward plot to cmaes_reward_vs_iteration.png")
