import argparse
import csv
import math
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
        B, T, d = tokens.shape
        assert d == self.config.token_dim
        x = self.token_proj(tokens)
        pos = jnp.arange(T)[None, :]
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
        if layer is None:
            for blk in self.h: x = blk(x, train=train)
            if apply_ln: x = self.ln_f(x)
        else:
            for i,blk in enumerate(self.h):
                x = blk(x, train=train)
                if i==layer: break
            if apply_ln: x = self.ln_f(x)
        return jnp.mean(x, axis=1)


def extract_latent_for_frame(model, params, frame_tokens, layer_to_extract=None, apply_ln=True):
    tokens = jnp.array(frame_tokens[None, ...])
    latent = model.apply(
        {'params': params},
        tokens,
        method=GPT.get_latent,
        layer=layer_to_extract,
        apply_ln=apply_ln,
        train=False
    )
    return np.array(latent[0])


def load_batch_latent_entropy(
    csv_path: str,
    row_indices: List[int],
    model: GPT,
    params: dict,
    img_size: int,
    patches_per_dim: int,
    layer_to_extract: Optional[int],
    apply_ln: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each selected row index:
      - Compute Shannon entropy H over the entire sequence (all columns)
      - Compute GPT latent only on the initial frame (first column)
    """
    grid = (patches_per_dim, patches_per_dim)
    latents, entropies = [], []
    idx_set = set(row_indices)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        num_cols = len(header)
        for i, row in enumerate(reader):
            if i in idx_set:
                # Entropy over full sequence
                full_bits = "".join(cell.strip() for cell in row)
                p = full_bits.count("1") / len(full_bits)
                H = 0.0 if p in (0.0, 1.0) else -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                entropies.append(H)
                # Latent on initial frame only
                init_bits = row[0].strip()
                frame = np.array(list(init_bits), int).reshape(img_size, img_size)[..., None]
                tokens = frame_to_tokens(frame, grid)
                latents.append(
                    extract_latent_for_frame(model, params, tokens, layer_to_extract, apply_ln)
                )
    return np.stack(latents), np.array(entropies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & eval MLP entropy predictor with batched loading"
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patches_per_dim", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--t_skip", type=int, default=0)
    parser.add_argument("--layer_to_extract", type=int, default=None)
    parser.add_argument("--apply_ln", dest="apply_ln", action="store_true")
    parser.add_argument("--no_apply_ln", dest="apply_ln", action="store_false")
    parser.set_defaults(apply_ln=True)
    parser.add_argument("--hidden_layers", nargs="+", type=int, default=[128,64])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--learning_rate_init", type=float, default=1e-3)
    parser.add_argument("--output_model", type=str, default=None)
    args = parser.parse_args()

    num_eff = math.ceil(args.num_frames / (args.t_skip + 1))
    num_tokens = args.patches_per_dim ** 2
    token_dim = (args.img_size // args.patches_per_dim) ** 2
    block = (num_eff - 1) * num_tokens

    config = GPTConfig(args.img_size, block, token_dim, num_tokens)
    model = GPT(config)

    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    with open(args.train_csv) as f:
        train_size = sum(1 for _ in f) - 1
    with open(args.val_csv) as f:
        val_size = sum(1 for _ in f) - 1

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu", solver="adam",
        warm_start=True, max_iter=1,
        learning_rate_init=args.learning_rate_init,
        learning_rate="constant",
    )

    for step in range(1, args.total_steps + 1):
        train_idx = np.random.choice(train_size, args.batch_size, replace=False)
        Xb, yb = load_batch_latent_entropy(
            args.train_csv, train_idx.tolist(),
            model, params,
            args.img_size, args.patches_per_dim,
            args.layer_to_extract, args.apply_ln
        )
        mlp.fit(Xb, yb)
        train_mse = mean_squared_error(yb, mlp.predict(Xb))
        train_rmse = np.sqrt(train_mse)
        train_baseline = np.var(yb)
        train_r2 = 1 - train_mse / train_baseline if train_baseline > 0 else float('nan')
        print(f"[Step {step}/{args.total_steps}] "
              f"train MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.3f}")

        if step % args.eval_every == 0 or step == args.total_steps:
            val_idx = np.random.choice(val_size, args.val_batch_size, replace=False)
            Xv, yv = load_batch_latent_entropy(
                args.val_csv, val_idx.tolist(),
                model, params,
                args.img_size, args.patches_per_dim,
                args.layer_to_extract, args.apply_ln
            )
            val_mse = mean_squared_error(yv, mlp.predict(Xv))
            val_rmse = np.sqrt(val_mse)
            val_baseline = np.var(yv)
            val_r2 = 1 - val_mse / val_baseline if val_baseline > 0 else float('nan')
            print(f"[Step {step}/{args.total_steps}] "
                  f"val   MSE={val_mse:.6f}, RMSE={val_rmse:.6f}, R2={val_r2:.3f}")
