import os
import argparse
import pickle
import math
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import flax.linen as nn
from flax.core import freeze
from typing import Optional, Tuple
from util_gol import load_dataset_from_csv, tokens_to_frame

# ----------------------------------------------------------------------------------------------------------------------
# Definitions of GPTConfig, CausalSelfAttention, MLP, Block, and GPT (copied from training script),
# including the get_latent method to extract a pooled hidden representation.
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    img_size: int           # image size of each frame (pixels)
    block_size: int         # = (num_eff_frames) * num_tokens
    token_dim: int          # dimension of each token
    num_tokens: int         # number of tokens per frame
    n_layer: int = 12       # number of transformer blocks
    n_head: int = 8         # number of attention heads
    n_embd: int = 256       # transformer embedding dimension
    dropout: float = 0.1    # dropout probability


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        cfg = self.config
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = cfg.n_embd // cfg.n_head
        self.n_head = cfg.n_head
        self.c_attn = nn.Dense(cfg.n_embd * 3)
        self.c_proj = nn.Dense(cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        B, T, C = x.shape
        cfg = self.config
        # project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # reshape for multi-head: (B, n_head, T, head_size)
        q = q.reshape(B, T, cfg.n_head, self.head_size).transpose((0, 2, 1, 3))
        k = k.reshape(B, T, cfg.n_head, self.head_size).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, cfg.n_head, self.head_size).transpose((0, 2, 1, 3))

        # build causal mask on frames
        tokens_per_frame = cfg.num_tokens
        t_idx = jnp.arange(T)
        frame_idx = t_idx // tokens_per_frame
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32)
        mask = mask.reshape(1, 1, T, T)

        # scaled dot-product attention with mask
        att_scores = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(self.head_size))
        att_scores = jnp.where(mask == 1.0, att_scores, float("-inf"))
        att_weights = nn.softmax(att_scores, axis=-1)
        att_weights = self.attn_dropout(att_weights, deterministic=not train)
        y = att_weights @ v  # (B, n_head, T, head_size)
        # merge heads
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y, deterministic=not train)
        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        cfg = self.config
        self.c_fc = nn.Dense(4 * cfg.n_embd)
        self.c_proj = nn.Dense(cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        cfg = self.config
        self.token_proj = nn.Dense(cfg.n_embd)
        self.wpe = nn.Embed(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(cfg.token_dim)

    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        B, T, d = tokens.shape
        assert d == self.config.token_dim, f"Token dim mismatch: got {d}, expected {self.config.token_dim}"
        x = self.token_proj(tokens)                         # (B, T, n_embd)
        pos = jnp.arange(0, T, dtype=jnp.int32)[None, :]    # (1, T)
        pos_emb = self.wpe(pos)                              # (1, T, n_embd)
        x = x + pos_emb
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)                        # (B, T, n_embd)
        x = self.ln_f(x)                                     # final hidden: (B, T, n_embd)
        logits = self.head(x)                                # (B, T, token_dim)
        return logits, None

    def get_latent(
        self,
        tokens: jnp.ndarray,
        *,
        train: bool = False,
        layer: Optional[int] = 6,
        apply_ln: bool = True,
    ) -> jnp.ndarray:
        """
        Return a pooled hidden representation after the final Block (or after `layer` if specified).
        - tokens: shape (1, T, token_dim)
        - Returns: (1, n_embd), i.e. we average across the T positions.
        """
        x = self.token_proj(tokens)                         # (1, T, n_embd)
        T = tokens.shape[1]
        pos = jnp.arange(0, T, dtype=jnp.int32)[None, :]    # (1, T)
        pos_emb = self.wpe(pos)                              # (1, T, n_embd)
        x = x + pos_emb
        x = self.drop(x, deterministic=not train)

        if layer is None:
            for block in self.h:
                x = block(x, train=train)                    # (1, T, n_embd)
        else:
            for i, block in enumerate(self.h):
                x = block(x, train=train)
                if i == layer:
                    break

        if apply_ln:
            x = self.ln_f(x)                                 # (1, T, n_embd)

        # pool across token positions
        latent = jnp.mean(x, axis=1)                         # (1, n_embd)
        return latent                                         

# ----------------------------------------------------------------------------------------------------------------------
# Utilities: compute Shannon entropy, closed‐form ridge regression, and evaluate performance.
# ----------------------------------------------------------------------------------------------------------------------

def compute_shannon_entropy(frames: np.ndarray) -> jnp.ndarray:
    """
    Compute Shannon entropy for each binary frame.
    - frames: shape (N, img_size, img_size), values {0, 1}
    - returns: (N,) array of entropies
    """
    flat = frames.reshape((frames.shape[0], -1))          # (N, img_size * img_size)
    p = jnp.mean(flat, axis=1)                            # (N,) probability of ones
    eps = 1e-8
    p_safe = jnp.clip(p, eps, 1.0 - eps)
    entropy = - (p_safe * jnp.log2(p_safe) + (1 - p_safe) * jnp.log2(1 - p_safe))
    return entropy


def closed_form_ridge(X: jnp.ndarray, y: jnp.ndarray, ridge_lambda: float = 1e-3) -> jnp.ndarray:
    """
    Solve ridge regression: minimize ||X w + b - y||^2 + λ ||w||^2.
    We augment X with a column of ones to learn bias. 
    - X: (N, d), y: (N,)
    - returns w_aug: (d + 1,) where last element is bias.
    """
    N, d = X.shape
    ones = jnp.ones((N, 1))
    X_aug = jnp.concatenate([X, ones], axis=1)            # (N, d+1)

    I = jnp.eye(d + 1)
    I = I.at[-1, -1].set(0.0)                              # do not regularize bias
    XtX = X_aug.T @ X_aug                                  # (d+1, d+1)
    A = XtX + ridge_lambda * I
    b = X_aug.T @ y                                        # (d+1,)
    w_aug = jnp.linalg.solve(A, b)                         # (d+1,)
    return w_aug


def evaluate_ridge_performance(X: jnp.ndarray, y: jnp.ndarray, w_aug: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute MSE and R² on (X, y) given ridge weights w_aug.
    - X: (N, d), y: (N,), w_aug: (d+1,)
    - returns: (mse, r2)
    """
    N, d = X.shape
    ones = jnp.ones((N, 1))
    X_aug = jnp.concatenate([X, ones], axis=1)             # (N, d+1)
    y_pred = X_aug @ w_aug                                 # (N,)
    mse = jnp.mean((y_pred - y) ** 2)
    y_mean = jnp.mean(y)
    ss_tot = jnp.sum((y - y_mean) ** 2)
    ss_res = jnp.sum((y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return float(mse), float(r2)


# ----------------------------------------------------------------------------------------------------------------------
# Main script: load checkpoint + validation dataset, extract latents for frame 0 (respecting t_skip),
# compute entropies for frame dt_probe, fit ridge regression, and report performance.
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear‐probe GPT hidden features for Shannon entropy of future frame.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pickled GPT parameters (state.params) saved during training."
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to the validation CSV file of Game of Life sequences."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=32,
        help="Image size (pixels) of each frame."
    )
    parser.add_argument(
        "--patches_per_dim",
        type=int,
        default=8,
        help="Number of patches per image dimension (e.g. 8 means 8×8 tokens per frame)."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=64,
        help="Total number of frames per sequence in the CSV."
    )
    parser.add_argument(
        "--t_skip",
        type=int,
        default=0,
        help="Skip frequency used in training: 0 means use all frames; 1 means use frames 0,2,4,…"
    )
    parser.add_argument(
        "--dt_probe",
        type=int,
        default=16,
        help="How many steps ahead to compute Shannon entropy (e.g. dt_probe=1 is entropy of frame1)."
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=1e-3,
        help="Regularization weight λ for ridge regression."
    )
    args = parser.parse_args()

    # 1) Load the checkpoint (pickled state.params)
    with open(args.checkpoint, "rb") as f:
        ckpt_params = pickle.load(f)
    params = freeze(ckpt_params)

    # 2) Load the validation dataset
    #    load_dataset_from_csv returns shape (N, num_frames, num_tokens, token_dim)
    val_dataset_np: np.ndarray = load_dataset_from_csv(
        args.val_csv,
        args.img_size,
        128,  # args.num_frames,
        (args.patches_per_dim, args.patches_per_dim),
    )
    val_dataset = jnp.array(val_dataset_np)
    num_sequences, num_frames, num_tokens, token_dim = val_dataset.shape
    print(f"Loaded validation dataset with shape: {val_dataset.shape}")

    # Check dt_probe is valid
    if not (0 <= args.dt_probe < num_frames):
        raise ValueError(f"dt_probe must be between 0 and {num_frames-1}. Got {args.dt_probe}.")

    print("Instantiate GPT model for probing")

    # 3) Reconstruct GPTConfig (must match training)
    #    num_eff_frames = ceil(num_frames / (t_skip+1))
    num_eff_frames = math.ceil(args.num_frames / (args.t_skip + 1))
    block_size = num_eff_frames * (args.patches_per_dim ** 2)
    # Infer n_layer from checkpoint keys: h_0, h_1, …
    n_layer = len([k for k in params.keys() if k.startswith("h_")])
    n_embd = params["token_proj"]["kernel"].shape[-1]
    # In training script, n_head was default 8, so we use 8 here:
    n_head = 8
    gpt_config = GPTConfig(
        img_size=args.img_size,
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=(args.patches_per_dim ** 2),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0  # no dropout during probing
    )

    # 4) Instantiate the GPT model
    model = GPT(gpt_config)

    # 5) Define a JAX‐jitted function to extract hidden latent for a single frame
    @jax.jit
    def extract_latent_single(params: freeze, frame_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Given frame_tokens of shape (num_tokens, token_dim),
        return a pooled latent of shape (n_embd,).
        """
        tokens = frame_tokens[None, ...]  # reshape to (1, T, token_dim)
        latent = model.apply(
            {"params": params},
            tokens,
            method=GPT.get_latent,
            layer=6,        # None => run through all layers
            apply_ln=True,
            train=False
        )  # latent shape: (1, n_embd)
        return latent[0]     # remove batch dim => (n_embd,)

    # 6) Vectorize latent extraction across all sequences for frame 0 (after applying t_skip)
    #    During training with t_skip, the input tokens to GPT were taken as every (t_skip+1)-th frame.
    #    For probing, we feed only the very first frame (frame index 0) as tokens,
    #    but the model’s block_size must match the trained block_size from above.
    #    The token_dim and num_tokens remain the same.
    tokens_first = val_dataset[:, 0, :, :]                          # (N, T, D)
    reps = jax.vmap(lambda tok: extract_latent_single(params, tok))(tokens_first)
    # reps shape: (N, n_embd)

    # 7) Build future frames at dt_probe (after t_skip) → compute entropy
    #    We take the tokens from frame index = dt_probe, then reconstruct binary image via tokens_to_frame.
    tokens_future_np = np.array(val_dataset_np[:, args.dt_probe, :, :])  # shape (N, num_tokens, token_dim)
    frames_future = []
    for i in range(tokens_future_np.shape[0]):
        frame_i = tokens_to_frame(
            tokens_future_np[i], 
            img_size=args.img_size,
            grid_size=(args.patches_per_dim, args.patches_per_dim)
        )  # returns (img_size, img_size) with values 0/1
        frames_future.append(frame_i)
    frames_future = np.stack(frames_future, axis=0)             # (N, img_size, img_size)
    frames_future = jnp.array(frames_future)                    # convert to JAX

    # Compute per‐sequence Shannon entropy
    entropies = compute_shannon_entropy(frames_future)         # (N,)

    # 8) Fit ridge regression: reps (N, d) → entropies (N,)
    X = reps                                                    # (N, d)
    y = entropies                                               # (N,)
    w_aug = closed_form_ridge(X, y, ridge_lambda=args.ridge_lambda)  # (d+1,)

    # 9) Evaluate performance on the same validation set
    mse, r2 = evaluate_ridge_performance(X, y, w_aug)
    print(f"Linear Probe Results (dt_probe={args.dt_probe}, t_skip={args.t_skip}):")
    print(f"  • Ridge λ = {args.ridge_lambda}")
    print(f"  • MSE on validation set: {mse:.6f}")
    print(f"  • R² on validation set: {r2:.6f}")
