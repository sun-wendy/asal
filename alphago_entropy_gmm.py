#!/usr/bin/env python3
import argparse
import pickle
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import unfreeze
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from util_gol import frame_to_tokens
from conway_lib import ConwayGame


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
        # support full forward for init/apply
        mid = self.get_mid_latents(tokens, train=train)
        return self.decode_from_latents(mid, train=train)

    def get_mid_latents(self, tokens, *, train=False, layer: Optional[int]=None):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        mid = self.config.n_layer // 2 if layer is None else layer
        for i, blk in enumerate(self.h):
            x = blk(x, train=train)
            if i == mid:
                break
        return x

    def decode_from_latents(self, latents, *, start_layer=0, train=False):
        x = latents
        for blk in self.h[start_layer:]:
            x = blk(x, train=train)
        x = self.ln_f(x)
        return self.head(x)


# wrappers

def encode_frame(frame, img_size, patches, model, params, layer=None):
    tokens = frame_to_tokens(frame[...,None], (patches, patches))
    lat = model.apply({'params': params}, jnp.array(tokens[None]), method=GPT.get_mid_latents, layer=layer)
    return np.array(lat[0])


def decode_latents(z, img_size, patches, model, params, start_layer=0):
    logits = model.apply({'params': params}, jnp.array(z[None]), method=GPT.decode_from_latents, start_layer=start_layer, train=False)
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
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--gmm_components", type=int, default=5)
    p.add_argument("--gmm_topk", type=int, default=500)
    p.add_argument("--layer", type=int, default=None, help="middle layer index for encode/decode")
    args = p.parse_args()

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

    # warm-up buffer
    buffer_z, buffer_r = [], []
    for _ in range(args.gmm_topk):
        frame = (np.random.rand(args.img_size, args.img_size) < 0.5).astype(int)
        z = encode_frame(frame, args.img_size, args.patches, model, params, layer=args.layer)
        game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
        game.grid = (frame == 1)
        seq = game.run(num_iterations=args.steps)
        arr = np.stack([f.astype(int) for f in seq])
        r = -np.min(compute_patch_temporal_entropy(arr, args.patches))
        buffer_z.append(z)
        buffer_r.append(r)
    buffer_z = np.stack(buffer_z)
    buffer_r = np.array(buffer_r)
    random_baseline = buffer_r.mean()
    print(f"Random baseline = {random_baseline:.4f}")

    mean_rewards = []
    for gen in range(1, args.iters+1):
        top_idx = np.argsort(buffer_r)[-args.gmm_topk:]
        Z_top = buffer_z[top_idx].reshape(len(top_idx), -1)
        gmm = GaussianMixture(n_components=args.gmm_components, covariance_type='full').fit(Z_top)

        samples, _ = gmm.sample(args.pop)
        samples = samples.reshape(args.pop, *buffer_z.shape[1:])
        rewards = np.zeros(args.pop)
        new_z, new_r = [], []
        for i, z in enumerate(samples):
            frame = decode_latents(z, args.img_size, args.patches, model, params, start_layer=args.layer)
            game = ConwayGame(width=args.img_size, height=args.img_size, grid_size=1, toroidal=True)
            game.grid = (frame == 1)
            seq = game.run(num_iterations=args.steps)
            arr = np.stack([f.astype(int) for f in seq])
            r = -np.min(compute_patch_temporal_entropy(arr, args.patches))
            rewards[i] = r
            new_z.append(z)
            new_r.append(r)

        buffer_z = np.concatenate([buffer_z, np.stack(new_z)])
        buffer_r = np.concatenate([buffer_r, np.array(new_r)])
        mean_rewards.append(rewards.mean())

        print(f"[gen {gen:4d}] mean={rewards.mean():.4f} max={rewards.max():.4f}")

        n_display = min(10, args.pop)
        histories = []
        # use samples instead of undefined Z
        for i in range(n_display):
            f0 = decode_latents(samples[i], args.img_size, args.patches, model, params)
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
                if r == 0:
                    ax.set_title(f"Step {c}")
        plt.tight_layout()
        plt.savefig(f"alphago_stent_gen_{gen:04d}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # plot results
    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.iters+1), mean_rewards, label="Mean Reward")
    plt.axhline(random_baseline, linestyle="--", label="Random Baseline")
    plt.xlabel("Generation")
    plt.ylabel("Reward (−min patch entropy)")
    plt.title("GMM RL Reward vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gmm_reward_vs_iteration.png")
    print("Saved reward plot to gmm_reward_vs_iteration.png")
