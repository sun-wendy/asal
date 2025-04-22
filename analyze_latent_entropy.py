import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from util_gol import frame_to_tokens, tokens_to_frame, load_dataset_from_csv


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

    def get_latent(self, tokens: jnp.ndarray, *, train: bool = False, 
                   layer: Optional[int] = None, apply_ln: bool = True):
        x = self.token_proj(tokens)
        pos = jnp.arange(0, tokens.shape[1], dtype=jnp.int32)[None, :]
        pos_emb = self.wpe(pos)
        x = x + pos_emb
        x = self.drop(x, deterministic=not train)
        if layer is None:
            for block in self.h:
                x = block(x, train=train)
            if apply_ln:
                x = self.ln_f(x)
        else:
            for i, block in enumerate(self.h):
                x = block(x, train=train)
                if i == layer:
                    break
            if apply_ln:
                x = self.ln_f(x)
        latent = jnp.mean(x, axis=1)
        return latent


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

def extract_latent_for_dataset(model, params, dataset, layer_to_extract=None, apply_ln=True):
    num_sequences, num_frames, _, _ = dataset.shape
    latents, labels = [], []
    for seq_idx in range(num_sequences):
        for frame_idx in range(num_frames):
            latents.append(
                extract_latent_for_frame(
                    model, params, dataset[seq_idx, frame_idx],
                    layer_to_extract, apply_ln
                )
            )
            labels.append((seq_idx, frame_idx))
    return np.stack(latents), labels


def compute_frame_entropies(tokens: jnp.ndarray) -> jnp.ndarray:
    """
    tokens: shape [S, F, num_tokens, token_dim], binary 0/1
    returns: entropy array of shape [S*F]
    """
    # mean occupancy per frame
    p = tokens.mean(axis=(2, 3))  # shape [S, F]

    # Shannon entropy per-frame, vectorized
    H = jnp.where(
        (p == 0.0) | (p == 1.0),
        0.0,
        -p * jnp.log2(p) - (1.0 - p) * jnp.log2(1.0 - p)
    )
    return H.reshape(-1)  # flatten to [S*F]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UMAP Analysis of GPT Latent Space, colored by Shannon entropy"
    )
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patches_per_dim", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--t_skip", type=int, default=0)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="gpt_params.pkl")
    parser.add_argument("--analysis_method", type=str, default="umap")
    parser.add_argument("--layer_to_extract", type=int, default=None)
    parser.add_argument("--apply_ln", action="store_true")
    parser.add_argument("--no_apply_ln", dest="apply_ln", action="store_false")
    parser.set_defaults(apply_ln=True)
    args = parser.parse_args()

    grid_size = (args.patches_per_dim, args.patches_per_dim)
    num_tokens = args.patches_per_dim ** 2
    token_dim = (args.img_size // args.patches_per_dim) ** 2
    num_eff = math.ceil(args.num_frames / (args.t_skip + 1))
    block_size = (num_eff - 1) * num_tokens

    print("HERE1")

    # load model
    gpt_config = GPTConfig(
        args.img_size,
        block_size,
        token_dim,
        num_tokens,
        n_layer=(args.layer_to_extract + 1) if args.layer_to_extract is not None else 12,
        n_head=args.patches_per_dim,
        n_embd=token_dim,
        dropout=0.1
    )
    model = GPT(gpt_config)
    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)
    
    print("HERE2")

    # load validation dataset
    val_dataset = load_dataset_from_csv(args.val_csv, args.img_size, args.num_frames, grid_size)
    num_val_sequences = val_dataset.shape[0]
    print(f"Validation dataset shape before sampling: {val_dataset.shape}")
    if num_val_sequences > 100:
        sample_indices = np.random.choice(num_val_sequences, size=500, replace=False)
        val_dataset = val_dataset[sample_indices]
        print(f"Validation dataset shape after sampling 500 sequences: {val_dataset.shape}")
    val_latents, val_labels = extract_latent_for_dataset(
        model, params, val_dataset, 
        layer_to_extract=args.layer_to_extract, 
        apply_ln=args.apply_ln
    )
    print(f"Extracted {val_latents.shape[0]} latent representations from validation data.")

    print("HERE3")

    # compute entropies in one JAX call
    val_entropies = np.array(
        jax.jit(compute_frame_entropies)(jnp.array(val_dataset))
    )

    print("HERE4")

    # optionally load and process test dataset
    if args.test_csv:
        test_dataset = load_dataset_from_csv(
            args.test_csv, args.img_size, args.num_frames, grid_size
        )
        # only take first few sequences if desired, here using all
        test_latents, test_labels = extract_latent_for_dataset(
            model, params, test_dataset,
            layer_to_extract=args.layer_to_extract,
            apply_ln=args.apply_ln
        )
        test_entropies = np.array(
            jax.jit(compute_frame_entropies)(jnp.array(test_dataset))
        )
    else:
        test_latents = np.empty((0, model.config.n_embd))
        test_entropies = np.array([])
        test_labels = []

    print("HERE5")

    # combine everything
    all_latents = np.concatenate([val_latents, test_latents], axis=0)
    all_entropies = np.concatenate([val_entropies, test_entropies], axis=0)
    source = np.concatenate([np.zeros(len(val_latents)), np.ones(len(test_latents))], axis=0)

    print("HERE6")

    # dimensionality reduction
    if args.analysis_method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    elif args.analysis_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    all_2d = reducer.fit_transform(all_latents)

    # plot colored by entropy
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        all_2d[:, 0],
        all_2d[:, 1],
        c=all_entropies,
        cmap="viridis",
        s=20,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Shannon entropy per frame", fontsize=12)
    ax.set_title("GPT Latent Space colored by frame Shannon entropy")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.savefig("latent_entropy_analysis.png", dpi=300)
