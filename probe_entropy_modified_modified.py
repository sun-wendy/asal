import os
import re
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
import matplotlib.pyplot as plt

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
        apply_ln: bool = False,
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
# Main script: iterate over all checkpoints in a directory, parse t_skip and num_frames from filename,
# run for multiple num_frames_cond values, and plot MSE vs checkpoint for each setting.
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate linear-probe performance (MSE & R²) for all GPT checkpoints in a directory, across multiple conditioning frame counts."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the directory containing multiple GPT checkpoint .pkl files."
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
        "--num_frames_cond_list",
        type=int,
        nargs="+",
        required=True,
        help="List of conditioning frame counts to evaluate (e.g., --num_frames_cond_list 1 2 3)."
    )
    parser.add_argument(
        "--dt_probe",
        type=int,
        default=16,
        help="How many steps ahead from the last conditioned frame to compute Shannon entropy."
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=1e-3,
        help="Regularization weight λ for ridge regression."
    )
    parser.add_argument(
        "--layer_to_probe",
        type=int,
        default=6,
        help="Which transformer layer to probe for the latent representation (default: 6)."
    )
    args = parser.parse_args()

    # Compile regex patterns once
    t_skip_pattern = re.compile(r"_tskip(\d+)_")
    frames_pattern = re.compile(r"_frames(\d+)_")

    # Gather sorted list of checkpoint filenames
    ckpt_fnames = sorted([
        fname for fname in os.listdir(args.checkpoint_dir)
        if fname.endswith(".pkl") and t_skip_pattern.search(fname) and frames_pattern.search(fname)
    ])
    if not ckpt_fnames:
        print("No valid checkpoint .pkl files found in the specified directory.")
        exit(1)

    # Load the dataset (with 128 frames fixed, since we'll index into it dynamically)
    val_dataset_np: np.ndarray = load_dataset_from_csv(
        args.val_csv,
        args.img_size,
        256,
        (args.patches_per_dim, args.patches_per_dim),
    )  # shape: (N_sequences, num_frames_total, num_tokens, token_dim)
    val_dataset = jnp.array(val_dataset_np)
    num_sequences, num_frames_total, num_tokens, token_dim = val_dataset.shape
    print(f"Loaded validation dataset shape: {val_dataset.shape}")

    # Prepare a results dictionary: { num_frames_cond: [(fname, mse, r2), ...], ... }
    all_results = {nfc: [] for nfc in args.num_frames_cond_list}

    # 6) Extract latents for all sequences in smaller batches to avoid OOM
    def batched_latents(params, tokens_cond, batch_size=64):
        all_reps = []
        N = tokens_cond.shape[0]
        for i in range(0, N, batch_size):
            chunk = tokens_cond[i : i + batch_size]             # (b, T_cond, token_dim)
            reps_chunk = jax.vmap(lambda tok: extract_latent_single(params, tok))(chunk)
            all_reps.append(reps_chunk)
        return jnp.concatenate(all_reps, axis=0)                # (N, n_embd)

    # Loop over each conditioning frame count
    for nfc in args.num_frames_cond_list:
        print(f"\n=== Evaluating for num_frames_cond = {nfc} ===")
        for fname in ckpt_fnames:
            ckpt_path = os.path.join(args.checkpoint_dir, fname)

            # Extract t_skip and num_frames from filename
            match_skip = t_skip_pattern.search(fname)
            match_frames = frames_pattern.search(fname)
            if not (match_skip and match_frames):
                print(f"Skipping {fname}: filename parsing failed")
                continue

            t_skip = int(match_skip.group(1))
            num_frames = int(match_frames.group(1))
            print(f"\nProcessing {fname} → num_frames={num_frames}, t_skip={t_skip}")

            # Compute number of effective frames
            num_eff_frames = math.ceil(num_frames / (t_skip + 1))
            if nfc > num_eff_frames:
                print(
                    f"  Skipping {fname}: num_frames_cond ({nfc}) > num_eff_frames ({num_eff_frames})"
                )
                continue

            # Determine conditioning frame indices
            frame_indices = [(i * (t_skip + 1)) for i in range(nfc)]
            last_cond_idx = frame_indices[-1]
            if last_cond_idx >= num_frames_total:
                print(
                    f"  Skipping {fname}: last cond idx ({last_cond_idx}) >= total frames ({num_frames_total})"
                )
                continue

            # Determine target frame to probe entropy
            target_frame_idx = last_cond_idx + args.dt_probe
            if target_frame_idx >= num_frames_total:
                print(
                    f"  Skipping {fname}: target frame idx ({target_frame_idx}) >= total frames ({num_frames_total})"
                )
                continue

            print(f"  Conditioning on frames {frame_indices}, probing frame {target_frame_idx}")

            # 1) Load checkpoint parameters
            with open(ckpt_path, "rb") as f:
                ckpt_params = pickle.load(f)
            params = freeze(ckpt_params)

            # 2) Reconstruct GPTConfig
            block_size = num_eff_frames * (args.patches_per_dim ** 2)
            n_layer = len([k for k in params.keys() if k.startswith("h_")])
            n_embd = params["token_proj"]["kernel"].shape[-1]
            n_head = 8  # match training default
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

            # 3) Instantiate GPT model
            model = GPT(gpt_config)

            # 4) JIT‐compiled latent extractor for concatenated conditioning frames
            @jax.jit
            def extract_latent_single(params: freeze, frame_tokens: jnp.ndarray) -> jnp.ndarray:
                """
                Given frame_tokens of shape (T_cond, token_dim),
                return a pooled latent of shape (n_embd,).
                """
                tokens = frame_tokens[None, ...]  # (1, T_cond, token_dim)
                latent = model.apply(
                    {"params": params},
                    tokens,
                    method=GPT.get_latent,
                    layer=args.layer_to_probe,        # probe after layer 6
                    apply_ln=False,
                    train=False
                )  # (1, n_embd)
                return latent[0]  # (n_embd,)

            # 5) Build conditioning tokens: shape (N_sequences, nfc * tokens_per_frame, token_dim)
            tokens_cond_np = val_dataset_np[:, frame_indices, :, :]   # (N, nfc, num_tokens, token_dim)
            N, _, ntok, ndim = tokens_cond_np.shape
            tokens_cond_np = tokens_cond_np.reshape(N, nfc * ntok, ndim)
            tokens_cond = jnp.array(tokens_cond_np)  # (N, T_cond, token_dim)

            # 6) Extract latents for all sequences via vmap
            # reps = jax.vmap(lambda tok: extract_latent_single(params, tok))(tokens_cond)  # (N, n_embd)
            reps = batched_latents(params, tokens_cond, batch_size=64)

            # 7) Build future frames and compute entropy
            tokens_future_np = np.array(val_dataset_np[:, target_frame_idx, :, :])  # (N, num_tokens, token_dim)
            frames_future = []
            for i in range(tokens_future_np.shape[0]):
                frame_i = tokens_to_frame(
                    tokens_future_np[i],
                    img_size=args.img_size,
                    grid_size=(args.patches_per_dim, args.patches_per_dim)
                )
                frames_future.append(frame_i)
            frames_future = np.stack(frames_future, axis=0)  # (N, img_size, img_size)
            frames_future = jnp.array(frames_future)         # (N, img_size, img_size)

            entropies = compute_shannon_entropy(frames_future)  # (N,)

            # 8) Fit ridge regression and evaluate
            X = reps
            y = entropies
            w_aug = closed_form_ridge(X, y, ridge_lambda=args.ridge_lambda)  # (d+1,)
            mse, r2 = evaluate_ridge_performance(X, y, w_aug)

            print(f"  → MSE: {mse:.6f}, R²: {r2:.6f}")
            all_results[nfc].append((fname, mse, r2))

    import re

    # Compile regex pattern for extracting t_skip
    t_skip_pattern = re.compile(r"_tskip(\d+)_")

    # Build and sort list of (fname, t_skip) tuples
    fname_tskip_list = []
    for fname in ckpt_fnames:
        match = t_skip_pattern.search(fname)
        if match:
            t_skip_val = int(match.group(1))
            fname_tskip_list.append((fname, t_skip_val))
    # Sort by t_skip ascending
    fname_tskip_list.sort(key=lambda x: x[1])

    # Extract sorted filenames and corresponding dt labels
    sorted_fnames = [ft[0] for ft in fname_tskip_list]
    dt_values    = [ft[1] + 1 for ft in fname_tskip_list]

    # --- MSE Plot using sorted order and dt labels ---
    # --- Compute baseline MSE (predicting mean entropy) ---
    N, num_frames, num_tokens, token_dim = val_dataset_np.shape
    if not (0 <= args.dt_probe < num_frames):
        raise ValueError(f"dt_probe must be in [0, {num_frames-1}]. Got {args.dt_probe}.")

    frames = []
    for i in range(N):
        tokens_i = val_dataset_np[i, args.dt_probe, :, :]  # (num_tokens, token_dim)
        frame_i = tokens_to_frame(
            tokens_i,
            img_size=args.img_size,
            grid_size=(args.patches_per_dim, args.patches_per_dim),
        )  # (img_size, img_size)
        frames.append(frame_i)
    frames = np.stack(frames, axis=0)  # (N, img_size, img_size)

    # Use the same entropy function (convert to jnp, then back to numpy)
    entropies = compute_shannon_entropy(jnp.array(frames))  # (N,)
    entropies = np.array(entropies)                        # (N,)

    mean_H = entropies.mean()
    baseline_mse = np.mean((entropies - mean_H) ** 2)
    print(f"Baseline MSE (predicting mean entropy={mean_H:.6f}): {baseline_mse:.6f}")

    # --- Combined plot: MSE and R² side by side, using dt_values for x-axis ---
    fig, (ax_mse, ax_r2) = plt.subplots(1, 2, figsize=(14, 6))

    # MSE subplot: plot vs dt_values
    for nfc in args.num_frames_cond_list:
        # build a mapping from filename → mse
        fname_to_mse = {fname: mse for (fname, mse, _) in all_results[nfc]}
        # collect MSEs in the same order as sorted_fnames
        ys = [fname_to_mse.get(fname, np.nan) for fname in sorted_fnames]
        ax_mse.plot(dt_values, ys, marker='o', label=f"{nfc} input frames")

    # draw a vertical line or horizontal line as baseline if desired:
    ax_mse.axhline(y=baseline_mse, color='red', linestyle='--', label='Baseline MSE')

    ax_mse.set_xlabel("dt of World Model", fontsize=14)
    ax_mse.set_ylabel("MSE", fontsize=14)
    ax_mse.set_title(f"Linear-Probe MSE at layer {args.layer_to_probe} for {args.dt_probe}-step ahead")
    ax_mse.set_xticks(dt_values)                      # place ticks at numeric dt
    ax_mse.set_xticklabels([str(d) for d in dt_values], rotation=45, ha='right', fontsize=12)
    ax_mse.legend()

    # R² subplot: same idea
    for nfc in args.num_frames_cond_list:
        fname_to_r2 = {fname: r2 for (fname, _, r2) in all_results[nfc]}
        ys = [fname_to_r2.get(fname, np.nan) for fname in sorted_fnames]
        ax_r2.plot(dt_values, ys, marker='o', label=f"{nfc} input frames")

    ax_r2.set_xlabel("dt of World Model", fontsize=14)
    ax_r2.set_ylabel("R²", fontsize=14)
    ax_r2.set_title(f"Linear-Probe R² at layer {args.layer_to_probe} for {args.dt_probe}-step ahead")
    ax_r2.set_xticks(dt_values)
    ax_r2.set_xticklabels([str(d) for d in dt_values], rotation=45, ha='right', fontsize=12)
    ax_r2.legend()

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/entropy_probe_mse_{args.dt_probe}ahead_layer{args.layer_to_probe}.png")
    print("Saved combined plot to metrics_vs_dt.png")
