import argparse
import csv
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple

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

# prepare and cache dataset once using JAX vmap for latent extraction
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
   # load all initial frames and final counts into numpy arrays
   rows = list(csv.reader(open(csv_path)))
   header, data_rows = rows[0], rows[1:]
   n = len(data_rows)
   # build arrays of shape (n, H, W, 1)
   init_bits = np.array([list(r[0].strip()) for r in data_rows], dtype=int)
   frames = init_bits.reshape(n, img_size, img_size)[..., None]
   finals = np.array([list(r[-2].strip()) for r in data_rows], int).reshape(n, -1)
   counts = finals.sum(axis=1).astype(np.float32)
   # vectorized latent extraction: vmap over first axis
   grid = (patches_per_dim, patches_per_dim)
   def latent_fn(frame):
       tokens = frame_to_tokens(frame, grid)
       return model.apply({'params':params}, jnp.array(tokens[None]),
                          method=GPT.get_latent, layer=layer_to_extract, apply_ln=apply_ln)[0]
   # vmap to get (n, latent_dim)
   latents = jax.vmap(latent_fn)(jnp.array(frames))
#    sigma = 0.1   # try e.g. 0.01–0.1
#    latents = latents + np.random.randn(*latents.shape) * sigma
   X = np.array(latents)
   y = counts
   np.savez_compressed(cache_path, X=X, y=y)
   return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patches_per_dim', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--t_skip', type=int, default=0)
    parser.add_argument('--layer_to_extract', type=int, default=None)
    parser.add_argument('--apply_ln', action='store_true')
    parser.add_argument('--no_apply_ln', dest='apply_ln', action='store_false')
    parser.set_defaults(apply_ln=True)
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[32,16])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--total_steps', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--learning_rate_init', type=float, default=3e-4)
    parser.add_argument('--output_model', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='cache')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    num_eff = math.ceil(args.num_frames / (args.t_skip+1))
    block = (num_eff - 1) * (args.patches_per_dim**2)
    token_dim = (args.img_size // args.patches_per_dim)**2
    config = GPTConfig(args.img_size, block, token_dim, args.patches_per_dim**2)
    model = GPT(config)
    params = pickle.load(open(args.checkpoint,'rb'))

    train_cache = f"{args.cache_dir}/train.npz"
    val_cache   = f"{args.cache_dir}/val.npz"
    X_train, y_train = prepare_and_cache(args.train_csv, train_cache,
                                         model, params,
                                         args.img_size, args.patches_per_dim,
                                         args.layer_to_extract, args.apply_ln)
    X_val, y_val     = prepare_and_cache(args.val_csv, val_cache,
                                         model, params,
                                         args.img_size, args.patches_per_dim,
                                         args.layer_to_extract, args.apply_ln)

    mean = y_train.mean()
    std  = y_train.std() if y_train.std()>0 else 1.0
    print(f"Target μ={mean:.2f}, σ={std:.2f}")

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers), activation='relu', solver='adam',
        warm_start=True, max_iter=1, learning_rate_init=args.learning_rate_init, learning_rate='constant',
        alpha=1e-2
    )
    rng = np.random.default_rng(0)
    n_train, n_val = X_train.shape[0], X_val.shape[0]
    train_loss = []
    val_loss = []
    val_steps = []

    for step in range(1, args.total_steps+1):
        idxs = rng.choice(n_train, args.batch_size, replace=False)
        Xb, yb = X_train[idxs], y_train[idxs]
        # ground-truth distribution of counts
        # print(f"[Step {step}] train gt count: min={yb.min():.0f}, max={yb.max():.0f}, mean={yb.mean():.2f}, std={yb.std():.2f}")
        yb_std = (yb-mean)/std
        # print(f"[Step {step}] train std count: min={yb_std.min():.2f}, max={yb_std.max():.2f}, mean={yb_std.mean():.2f}, std={yb_std.std():.2f}")
        mlp.fit(Xb, yb_std)
        yb_pred_std = mlp.predict(Xb)
        #print(f"[Step {step}] train std pred: min={yb_pred_std.min():.2f}, max={yb_pred_std.max():.2f}, mean={yb_pred_std.mean():.2f}, std={yb_pred_std.std():.2f}")
        mse_s = mean_squared_error(yb_std, yb_pred_std)
        r2_s  = r2_score(yb_std, yb_pred_std)
        train_loss.append(mse_s)
        if step%args.eval_every==0 or step==args.total_steps:
            print(f"[Step {step}] train std MSE={mse_s:.3f}, R2={r2_s:.3f}")
        if step%args.eval_every==0 or step==args.total_steps:
            val_steps.append(step)
            vidx = rng.choice(n_val, args.val_batch_size, replace=False)
            Xv, yv = X_val[vidx], y_val[vidx]
            yv_std = (yv-mean)/std
            yv_pred_std = mlp.predict(Xv)
            mse_v = mean_squared_error(yv_std, yv_pred_std)
            val_loss.append(mse_v)
            print(f"[Step {step}] val std MSE={mean_squared_error(yv_std,yv_pred_std):.3f}, R2={r2_score(yv_std,yv_pred_std):.3f}")

        # plot raw and EMA-smoothed using pandas
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'train': pd.Series(train_loss, index=list(range(1, len(train_loss)+1)))})
    df['val'] = pd.Series(val_loss, index=val_steps)
    df_ewm = df.ewm(span=50, adjust=False).mean()
    plt.figure()
    plt.plot(df['train'], label='train std MSE', alpha=0.3)
    plt.plot(df_ewm['train'], label='train std MSE (EWMA)')
    plt.plot(df['val'], label='val std MSE', alpha=0.3)
    plt.plot(df_ewm['val'], '--', label='val std MSE (EWMA)')
    plt.xlabel('Iteration')
    plt.ylabel('Std-space MSE')
    plt.legend()
    plt.title('Loss & EWMA (span=1000)')
    plt.savefig('train_val_loss.png')

    if args.output_model:
        dump(mlp, args.output_model)
