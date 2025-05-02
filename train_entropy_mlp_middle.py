#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt

from util_gol import frame_to_tokens

# ── compute normalized spatio-temporal entropy ───────────────────────────────────
def compute_st_entropy_arr(arr: np.ndarray, patches_per_dim: int) -> float:
    L, H, W = arr.shape
    k = H // patches_per_dim
    blocks = arr.reshape(L, patches_per_dim, k, patches_per_dim, k)
    blocks = blocks.transpose(1,3,0,2,4).reshape(-1, L*k*k)
    uniq, counts = np.unique(blocks, axis=0, return_counts=True)
    probs = counts / counts.sum()
    Hval = -np.sum(probs * np.log2(probs + 1e-12))
    return Hval / (k*k*L)

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

# ── model definitions with mid-layer extraction ─────────────────────────────────
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
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32).reshape(1,1,T,T)
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
        x = nn.gelu(x, approximate=True)
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
        return x + self.mlp(self.ln_2(x), train=train)

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
        lat = self.get_mid_latents(tokens, train=train)
        return lat, None

    def get_mid_latents(self, tokens, *, train=False):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        mid = self.config.n_layer // 2
        for blk in self.h[:mid]:
            x = blk(x, train=train)
        return x.mean(axis=1)

    def decode_mid(self, latents, *, train=False):
        raise RuntimeError("decode_mid should not be called in surrogate mode")

# ── prepare and cache latents & entropies ────────────────────────────────────────
def prepare_and_cache(
    csv_path: str,
    cache_path: str,
    model: GPT,
    params: dict,
    img_size: int,
    patches_per_dim: int,
    num_frames: int,
    t_skip: int,
):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']
    rows = list(csv.reader(open(csv_path)))[1:]
    n = len(rows)
    seqs = np.array([list("".join(row[:-1])) for row in rows], int)
    seqs = seqs.reshape(n, -1, img_size, img_size)
    seqs = seqs[:, :num_frames]
    if t_skip > 0:
        seqs = seqs[:, ::(t_skip+1)]
    y = np.array([compute_st_entropy_arr(s, patches_per_dim) for s in seqs], dtype=np.float32)
    inits = seqs[:, 0, ...]
    grid = (patches_per_dim, patches_per_dim)
    def extract_latent(frame):
        tokens = frame_to_tokens(frame[..., None], grid)
        lat = model.apply({'params': params}, jnp.array(tokens[None]), method=GPT.get_mid_latents)
        return np.array(lat[0])
    X = np.array([extract_latent(f) for f in inits])
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y

# ── main: train surrogate MLP ────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv',   required=True)
    p.add_argument('--val_csv',     required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--img_size',    type=int, default=16)
    p.add_argument('--patches_per_dim', type=int, default=2)
    p.add_argument('--num_frames',  type=int, default=10)
    p.add_argument('--t_skip',      type=int, default=0)
    p.add_argument('--hidden_layers', nargs='+', type=int, default=[128,64])
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--total_steps', type=int, default=3000)
    p.add_argument('--eval_every',  type=int, default=50)
    p.add_argument('--learning_rate_init', type=float, default=1e-3)
    p.add_argument('--cache_dir',   type=str, default='cache')
    p.add_argument('--output_model', type=str, default='surrogate_mlp.joblib')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    num_eff = math.ceil(args.num_frames / (args.t_skip+1))
    block_size = num_eff * (args.patches_per_dim**2)
    token_dim = (args.img_size // args.patches_per_dim)**2

    # load GPT and params
    config = GPTConfig(args.img_size, block_size, token_dim, args.patches_per_dim**2)
    model = GPT(config)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, block_size, token_dim))
    model.init(rng, dummy, train=False)
    params = pickle.load(open(args.checkpoint, 'rb'))

    # prepare data
    train_cache = os.path.join(args.cache_dir, 'train_mid.npz')
    val_cache   = os.path.join(args.cache_dir, 'val_mid.npz')
    X_train, y_train = prepare_and_cache(
        args.train_csv, train_cache, model, params,
        args.img_size, args.patches_per_dim, args.num_frames, args.t_skip
    )
    X_val, y_val     = prepare_and_cache(
        args.val_csv, val_cache, model, params,
        args.img_size, args.patches_per_dim, args.num_frames, args.t_skip
    )

    # scale features (X)
    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_val   = scaler_X.transform(X_val)

    # scale targets (y) separately for train and val
    scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
    y_train_std    = scaler_y.transform(y_train.reshape(-1,1)).ravel()
    # scaler_y_val   = StandardScaler().fit(y_val.reshape(-1,1))
    y_val_std      = scaler_y.transform(y_val.reshape(-1,1)).ravel()

    print(f"[y_train] min={y_train.min():.6f} max={y_train.max():.6f}, mean={y_train.mean():.6f}, std={y_train.std():.6f}")
    print(f"[y_train_std] min={y_train_std.min():.3f} max={y_train_std.max():.3f}, mean={y_train_std.mean():.3f}, std={y_train_std.std():.3f}")
    print(f"[y_val]   min={y_val.min():.6f} max={y_val.max():.6f}, mean={y_val.mean():.6f}, std={y_val.std():.6f}")
    print(f"[y_val_std]   min={y_val_std.min():.3f} max={y_val_std.max():.3f}, mean={y_val_std.mean():.3f}, std={y_val_std.std():.3f}")

    # build MLP and warm up partial_fit
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu', solver='adam',
        warm_start=False,
        max_iter=1,
        learning_rate_init=args.learning_rate_init,
        learning_rate='constant'
    )

    # initial partial_fit
    mlp.partial_fit(X_train[:args.batch_size], y_train_std[:args.batch_size])

    rng = np.random.default_rng(0)
    n_train = X_train.shape[0]

    train_mse_history = []
    val_mse_history   = []

    # training loop with full-validation evaluation
    for step in range(1, args.total_steps+1):
        idxs = rng.choice(n_train, args.batch_size, replace=False)
        Xb, yb = X_train[idxs], y_train_std[idxs]
        mlp.partial_fit(Xb, yb)

        yb_pred = mlp.predict(Xb)
        train_mse = mean_squared_error(yb, yb_pred)
        train_r2  = r2_score(yb, yb_pred)
        print(f"[Step {step}] train std MSE={train_mse:.4f}, R2={train_r2:.4f}")
        train_mse_history.append(train_mse)

        # evaluate on the entire validation set:
        yv_pred = mlp.predict(X_val)
        mse_all = mean_squared_error(y_val_std, yv_pred)
        r2_all  = r2_score(y_val_std, yv_pred)
        print(f"[Step {step}] full-val std MSE={mse_all:.4f}, R2={r2_all:.4f}")
        val_mse_history.append(mse_all)
    
    df = pd.DataFrame({
        'train': train_mse_history,
        'val':   val_mse_history
    })

    # compute exponential moving‐average with span=1000
    df_smooth = df.ewm(span=100).mean()

    plt.figure()
    # note: df_smooth.index goes from 0…total_steps-1, so +1 to align with steps
    plt.plot(df_smooth.index + 1, df_smooth['train'], label='train MSE (smoothed)')
    plt.plot(df_smooth.index + 1, df_smooth['val'],   label='val MSE (smoothed)')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.title('Smoothed Training & Validation MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('surrogate_mse_smoothed.png')