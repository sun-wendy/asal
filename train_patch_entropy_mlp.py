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
from joblib import dump

from util_gol import frame_to_tokens

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
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        t_idx = jnp.arange(T)
        frame_idx = t_idx // self.config.num_tokens
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32).reshape(1, 1, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(self.head_size))
        att = jnp.where(mask, att, float("-inf"))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)
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
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(c.token_dim)
    def __call__(self, tokens, *, train: bool):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for blk in self.h:
            x = blk(x, train=train)
        x = self.ln_f(x)
        return self.head(x), None
    def get_latent(self, tokens, *, train=False, layer: Optional[int]=None, apply_ln=True):
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
        return jnp.mean(x, axis=1)

# compute per-patch temporal entropy (to detect oscillators)
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

# cache latents & entropies
def prepare_and_cache(
   csv_path: str,
   cache_path: str,
   model: GPT,
   params: dict,
   img_size: int,
   patches_per_dim: int,
   layer_to_extract: Optional[int],
   apply_ln: bool
) -> Tuple[np.ndarray, np.ndarray]:
   if os.path.exists(cache_path):
       data = np.load(cache_path)
       return data['X'], data['y']

   rows = list(csv.reader(open(csv_path)))[1:]
   n = len(rows)
   seqs = np.array([list("".join(row[:-1])) for row in rows], int)
   seqs = seqs.reshape(n, -1, img_size, img_size)
   seqs = seqs[:, :20]

   # compute per-patch-min entropy for each sequence
   ents = []
   for s in seqs:
       arr = np.array(s, dtype=int)
       patch_ents = compute_patch_temporal_entropy(arr, patches_per_dim)
       ents.append(np.min(patch_ents))
   y = np.array(ents, dtype=np.float32)

   # extract latents from initial frame only
   inits = seqs[:,0,...,None]
   grid = (patches_per_dim, patches_per_dim)
   def latent_fn(frame):
       tokens = frame_to_tokens(np.array(frame), grid)
       return model.apply({'params':params}, jnp.array(tokens[None]),
                          method=GPT.get_latent, layer=layer_to_extract, apply_ln=apply_ln)[0]

   X = np.array(jax.vmap(latent_fn)(jnp.array(inits)))
   np.savez_compressed(cache_path, X=X, y=y)
   return X, y

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv',   required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--patches_per_dim', type=int, default=2)
    p.add_argument('--num_frames', type=int, default=10)
    p.add_argument('--t_skip', type=int, default=0)
    p.add_argument('--layer_to_extract', type=int, default=None)
    p.add_argument('--apply_ln', action='store_true')
    p.add_argument('--no_apply_ln', dest='apply_ln', action='store_false')
    p.set_defaults(apply_ln=True)
    p.add_argument('--hidden_layers', nargs='+', type=int, default=[128,64])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--val_batch_size', type=int, default=64)
    p.add_argument('--total_steps', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--learning_rate_init', type=float, default=1e-3)
    p.add_argument('--output_model', type=str, default=None)
    p.add_argument('--cache_dir', type=str, default='cache')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    num_eff = math.ceil(args.num_frames / (args.t_skip+1))
    block = (num_eff - 1) * (args.patches_per_dim**2)
    token_dim = (args.img_size // args.patches_per_dim)**2
    config = GPTConfig(args.img_size, block, token_dim, args.patches_per_dim**2)
    model = GPT(config)
    params = pickle.load(open(args.checkpoint,'rb'))

    X_train, y_train = prepare_and_cache(
        args.train_csv, f"{args.cache_dir}/train.npz",
        model, params, args.img_size, args.patches_per_dim,
        args.layer_to_extract, args.apply_ln)
    X_val,   y_val   = prepare_and_cache(
        args.val_csv,   f"{args.cache_dir}/val.npz",
        model, params, args.img_size, args.patches_per_dim,
        args.layer_to_extract, args.apply_ln)

    mean = y_train.mean()
    std  = y_train.std() if y_train.std()>0 else 1.0
    print(f"Patch-entropy μ={mean:.4f}, σ={std:.4f}")

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu', solver='adam',
        alpha=1e-3, warm_start=True, max_iter=1,
        learning_rate_init=args.learning_rate_init,
        learning_rate='constant'
    )

    rng = np.random.default_rng(0)
    n_train, n_val = X_train.shape[0], X_val.shape[0]

    for step in range(1, args.total_steps+1):
        idxs = rng.choice(n_train, args.batch_size, replace=False)
        Xb, yb = X_train[idxs], y_train[idxs]
        yb_std = (yb - mean) / std
        mlp.fit(Xb, yb_std)
        # Print training stats
        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"[Step {step}] train std MSE={mean_squared_error(yb_std, mlp.predict(Xb)):.3f},"
                  f" R2={r2_score(yb_std, mlp.predict(Xb)):.3f}")

        if step % args.eval_every == 0 or step == args.total_steps:
            vidx = rng.choice(n_val, args.val_batch_size, replace=False)
            Xv, yv = X_val[vidx], y_val[vidx]
            yv_std = (yv - mean) / std
            yv_pred_std = mlp.predict(Xv)
            print(f"[Step {step}] val std MSE={mean_squared_error(yv_std,yv_pred_std):.3f},"
                  f" R2={r2_score(yv_std,yv_pred_std):.3f}")

    if args.output_model:
        dump(mlp, args.output_model)
