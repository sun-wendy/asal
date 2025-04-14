#!/usr/bin/env python3
import os
import pickle
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import freeze
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange
from sklearn.manifold import TSNE
# Optionally, you could import UMAP if desired:
import umap
from umap.umap_ import UMAP

# ----------------- Helper Functions -----------------
def frame_to_tokens(frame: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a binary simulation frame into tokens by rearranging it into patches.
    For grid_size=(p, p), rearrange as: "(H ph) (W pw) D -> (H W) (ph pw D)"
    """
    tokens = rearrange(frame, "(H ph) (W pw) D -> (H W) (ph pw D)",
                        H=grid_size[0], W=grid_size[1])
    tokens = (tokens > 0.5).astype(np.float32)
    return tokens

def tokens_to_frame(tokens: np.ndarray, img_size: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a sequence of tokens back into an image.
    """
    ph = img_size // grid_size[0]
    pw = img_size // grid_size[1]
    frame = rearrange(tokens, "(H W) (ph pw D) -> (H ph) (W pw) D",
                      H=grid_size[0], W=grid_size[1], ph=ph, pw=pw, D=1)
    return frame

def load_dataset(csv_file: str, img_size: int, num_frames: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Load simulation sequences from a CSV file. Each row in the CSV represents a sequence,
    and each column is a frame (timestep) represented as a 1024-character string.
    
    Returns an array of shape:
      (num_sequences, num_frames, num_tokens, token_dim)
    """
    print(f"Loading dataset from {csv_file}")
    sequences = []
    with open(csv_file, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # Skip header rows if present.
            if row[0].strip().startswith("State"):
                continue
            if len(row) != num_frames:
                raise ValueError(f"Expected row to have {num_frames} cells, got {len(row)}")
            seq = []
            for cell in row:
                cell_str = cell.strip()
                arr = np.fromstring(cell_str, sep=" ")
                # If only one number is returned, assume the cell is a contiguous string.
                if arr.size == 1:
                    if len(cell_str) != img_size * img_size:
                        raise ValueError(f"Expected cell string length {img_size*img_size}, got {len(cell_str)}")
                    arr = np.array([float(c) for c in cell_str], dtype=np.float32)
                if arr.size != img_size * img_size:
                    raise ValueError(f"Expected cell to contain {img_size * img_size} values, got {arr.size}")
                state = arr.reshape(img_size, img_size)
                seq.append(state)
            sequences.append(np.stack(seq, axis=0))
    data = np.stack(sequences, axis=0)  # shape: (num_sequences, num_frames, img_size, img_size)
    # Tokenize each frame.
    dataset_tokens = []
    for seq in data:
        seq_tokens = []
        for i in range(num_frames):
            frame = seq[i][..., np.newaxis]  # shape: (img_size, img_size, 1)
            tokens = frame_to_tokens(frame, grid_size)
            seq_tokens.append(tokens)
        dataset_tokens.append(np.stack(seq_tokens, axis=0))
    dataset_tokens = np.stack(dataset_tokens, axis=0)  # shape: (num_sequences, num_frames, num_tokens, token_dim)
    print(f"Dataset shape (sequences, frames, tokens, token_dim): {dataset_tokens.shape}")
    return dataset_tokens

# ----------------- Model Definition -----------------
@dataclass
class GPTConfig:
    img_size: int           # image size of each frame.
    block_size: int         # = (num_frames - 1) * num_tokens
    token_dim: int          # dimension of each token.
    num_tokens: int         # number of tokens per frame.
    n_layer: int = 12       # number of transformer blocks.
    n_head: int = 8         # number of attention heads.
    n_embd: int = 256       # transformer embedding dimension.
    dropout: float = 0.1    # dropout probability.

class CausalSelfAttention(nn.Module):
    config: GPTConfig
    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.c_attn = nn.Dense(config.n_embd * 3)
        self.c_proj = nn.Dense(config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        tokens_per_frame = self.config.num_tokens
        t_idx = jnp.arange(T)
        frame_idx = t_idx // tokens_per_frame
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32)
        mask = mask.reshape(1, 1, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(self.head_size))
        att = jnp.where(mask == 1.0, att, float("-inf"))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y

class MLP(nn.Module):
    config: GPTConfig
    def setup(self):
        config = self.config
        self.c_fc = nn.Dense(4 * config.n_embd)
        self.c_proj = nn.Dense(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
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
        config = self.config
        self.token_proj = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(config.token_dim)
    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        B, T, d = tokens.shape
        assert d == self.config.token_dim, f"Token dim mismatch: got {d}, expected {self.config.token_dim}"
        x = self.token_proj(tokens)
        pos = jnp.arange(0, T, dtype=jnp.int32)[None, :]
        pos_emb = self.wpe(pos)
        x = x + pos_emb
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

    # New method to extract latent representations (post ln_f, averaged over tokens).
    def get_latent(self, tokens: jnp.ndarray, *, train: bool = False):
        x = self.token_proj(tokens)
        pos = jnp.arange(0, tokens.shape[1], dtype=jnp.int32)[None, :]
        pos_emb = self.wpe(pos)
        x = x + pos_emb
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)
        latent = jnp.mean(x, axis=1)
        return latent

# ----------------- Latent Extraction Helper Functions -----------------
def extract_latent_for_frame(model, params, frame_tokens):
    """
    Compute a latent representation for a given frame.
    It passes the frame's tokens through the model up to the final layer normalization,
    then averages over the token dimension.
    """
    tokens = jnp.array(frame_tokens[None, ...])  # shape: (1, num_tokens, token_dim)
    latent = model.apply({'params': params}, tokens, train=False, method=GPT.get_latent)
    return np.array(latent[0])

def extract_latent_for_dataset(model, params, dataset):
    """
    Given a dataset of shape (num_sequences, num_frames, num_tokens, token_dim),
    extracts the latent representation (via get_latent) for each frame.
    Returns:
       latents: array of shape (N, n_embd) where N is total frames.
       labels: list of tuples (sequence index, frame index)
    """
    num_sequences, num_frames, _, _ = dataset.shape
    latent_list = []
    labels = []
    for seq_idx in range(num_sequences):
        for frame_idx in range(num_frames):
            frame_tokens = dataset[seq_idx, frame_idx]  # shape: (num_tokens, token_dim)
            latent = extract_latent_for_frame(model, params, frame_tokens)
            latent_list.append(latent)
            labels.append((seq_idx, frame_idx))
    latents = np.stack(latent_list, axis=0)
    return latents, labels

# ----------------- Main Analysis Script -----------------
def main(args):
    # Define grid and model hyperparameters.
    grid_size = (args.patches_per_dim, args.patches_per_dim)
    token_dim = (args.img_size // args.patches_per_dim) ** 2  # token dimension
    num_tokens = args.patches_per_dim ** 2                    # number of tokens per frame
    block_size = (args.num_frames - 1) * num_tokens           # as used in training

    # Instantiate GPT configuration and model.
    gpt_config = GPTConfig(
        img_size=args.img_size,
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=num_tokens,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    )
    model = GPT(gpt_config)

    # Load saved GPT parameters.
    print(f"Loading saved parameters from {args.params_file}")
    with open(args.params_file, "rb") as f:
        params = pickle.load(f)

    # ----------------- Load Validation Data and Sample Randomly 100 Sequences -----------------
    val_dataset = load_dataset(args.val_csv, args.img_size, args.num_frames, grid_size)
    num_val_sequences = val_dataset.shape[0]
    print(f"Validation dataset shape before sampling: {val_dataset.shape}")
    if num_val_sequences > 100:
        sample_indices = np.random.choice(num_val_sequences, size=500, replace=False)
        val_dataset = val_dataset[sample_indices]
        print(f"Validation dataset shape after sampling 100 sequences: {val_dataset.shape}")

    # Extract latent representations and labels from validation dataset.
    val_latents, val_labels = extract_latent_for_dataset(model, params, val_dataset)
    print(f"Extracted {val_latents.shape[0]} latent representations from validation data.")

    # ----------------- Load Test Data (Use All Sequences) -----------------
    if args.test_csv:
        test_dataset = load_dataset(args.test_csv, args.img_size, args.num_frames, grid_size)
        test_dataset = test_dataset[:5]
        test_latents, test_labels = extract_latent_for_dataset(model, params, test_dataset)
        print(f"Extracted {test_latents.shape[0]} latent representations from test data.")
    else:
        test_latents = np.empty((0, gpt_config.n_embd))
        test_labels = []
    
    # Combine the datasets.
    all_latents = np.concatenate([val_latents, test_latents], axis=0)
    # Create a source label array: 0 for validation, 1 for test.
    source = np.concatenate([np.zeros(len(val_latents)), np.ones(len(test_latents))], axis=0)
    
    # ----------------- Dimensionality Reduction (UMAP) -----------------
    print("Running UMAP on combined latent representations...")
    # reducer = UMAP(n_components=2, random_state=42)
    # all_latent_2d = reducer.fit_transform(all_latents)
    tsne = TSNE(n_components=2, random_state=42)
    all_latent_2d = tsne.fit_transform(all_latents)
    
    # ----------------- Visualization -----------------
    # Set up masks for the two sources.
    val_mask = (source == 0)
    test_mask = (source == 1)
    
    # Use timestep (frame index) coloring for all points.
    all_frame_indices = np.array([lbl[1] for lbl in (val_labels + test_labels)])
    unique_timesteps = np.unique(all_frame_indices)
    cmap = plt.get_cmap('tab10') if len(unique_timesteps) <= 10 else plt.get_cmap('tab20')
    colors = np.array([cmap(idx % cmap.N) for idx in all_frame_indices])
    
    plt.figure(figsize=(10, 8))
    # Plot validation points with small markers.
    plt.scatter(all_latent_2d[val_mask, 0], all_latent_2d[val_mask, 1],
                color=colors[val_mask], s=10, label="Validation")
    # Plot test points with much larger markers.
    plt.scatter(all_latent_2d[test_mask, 0], all_latent_2d[test_mask, 1],
                color=colors[test_mask], s=80, edgecolor='black', linewidth=1.5, label="Test")
    
    plt.title("UMAP Projection of GPT Latent Space\nValidation (small) vs Test (large)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.show()
    print(f"Saved UMAP plot to {args.output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP Analysis of GPT Latent Space with Merged Validation and Test Data")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to the validation CSV file")
    parser.add_argument("--test_csv", type=str, default="",
                        help="Path to the test CSV file (optional)")
    parser.add_argument("--params_file", type=str, default="gpt_params.pkl",
                        help="Path to the saved GPT parameters file")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (in pixels) per frame")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per sequence in the CSV")
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    # Model hyperparameters must match those used in training.
    parser.add_argument("--n_layer", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--output_plot", type=str, default="latent_umap.png",
                        help="Output filename for the UMAP plot")
    args = parser.parse_args()
    main(args)
