import pickle
import argparse
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange
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


def extract_latent_for_frame(model, params, frame_tokens, layer_to_extract: Optional[int] = None, apply_ln: bool = True):
    """
    Compute a latent representation for a given frame.
    It passes the frame's tokens through the model up to the specified transformer block (or full network if None),
    then averages over the token dimension. The apply_ln flag controls whether to apply the final layer norm.
    """
    tokens = jnp.array(frame_tokens[None, ...])  # shape: (1, num_tokens, token_dim)
    latent = model.apply(
        {'params': params}, 
        tokens, 
        train=False, 
        method=GPT.get_latent, 
        layer=layer_to_extract, 
        apply_ln=apply_ln
    )
    return np.array(latent[0])


def extract_latent_for_dataset(model, params, dataset, layer_to_extract: Optional[int] = None, apply_ln: bool = True):
    """
    Given a dataset of shape (num_sequences, num_frames, num_tokens, token_dim),
    extracts the latent representation (via get_latent) for each frame from the specified layer.
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
            latent = extract_latent_for_frame(model, params, frame_tokens, 
                                              layer_to_extract=layer_to_extract, 
                                              apply_ln=apply_ln)
            latent_list.append(latent)
            labels.append((seq_idx, frame_idx))
    latents = np.stack(latent_list, axis=0)
    return latents, labels


def flatten_dataset(dataset):
    """
    Flatten a dataset of shape (num_sequences, num_frames, num_tokens, token_dim)
    into a list of frames.
    """
    num_seq, num_frames, _, _ = dataset.shape
    frames = []
    for seq_idx in range(num_seq):
        for frame_idx in range(num_frames):
            frames.append(dataset[seq_idx, frame_idx])
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UMAP Analysis of GPT Latent Space with Merged Validation and Test Data"
    )
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (pixels) for each frame")
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per simulation sequence in CSV files")
    parser.add_argument("--t_skip", type=int, default=0,
                        help="Skip frequency: 0 uses all frames; 1 uses frames 0, 2, 4, ..., etc.")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to the validation CSV file")
    parser.add_argument("--test_csv", type=str, default="patterns/conway_test_states_32by32_20250414_075008.csv",
                        help="Path to the test CSV file (optional)")
    parser.add_argument("--checkpoint", type=str, default="gpt_params.pkl",
                        help="Path to the saved GPT parameters file")
    # Model hyperparameters must match those used in training
    parser.add_argument("--n_layer", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    # Analysis method and extraction parameters
    parser.add_argument("--analysis_method", type=str, default="umap",
                        help="Analysis method to use: umap, tsne, or pca")
    parser.add_argument("--layer_to_extract", type=int, default=11,
                        help="Transformer layer (0-indexed) to extract latent representations from")
    # Whether to apply an extra layer normalization after extraction
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply_ln", dest="apply_ln", action="store_true",
                       help="Apply final layer norm on the extracted representation")
    group.add_argument("--no_apply_ln", dest="apply_ln", action="store_false",
                       help="Do not apply final layer norm on the extracted representation")
    parser.set_defaults(apply_ln=True)
    args = parser.parse_args()

    grid_size = (args.patches_per_dim, args.patches_per_dim)
    token_dim = (args.img_size // args.patches_per_dim) ** 2  # token dimension
    num_tokens = args.patches_per_dim ** 2                    # number of tokens per frame
    num_eff_frames = math.ceil(args.num_frames / (args.t_skip + 1))
    block_size = (num_eff_frames - 1) * num_tokens

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

    # Load saved GPT parameters
    print(f"Loading saved parameters from {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    # Load validation data & sample 100 sequences, then extract latent representations
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

    if args.test_csv:
        test_dataset = load_dataset_from_csv(args.test_csv, args.img_size, 10, grid_size)
        test_dataset = test_dataset[:5]
        test_latents, test_labels = extract_latent_for_dataset(
            model, params, test_dataset,
            layer_to_extract=args.layer_to_extract,
            apply_ln=args.apply_ln
        )
        print(f"Extracted {test_latents.shape[0]} latent representations from test data.")
    else:
        test_latents = np.empty((0, gpt_config.n_embd))
        test_labels = []

    # Combine latent representations and sources.
    # Combine latent representations and sources.
    all_latents = np.concatenate([val_latents, test_latents], axis=0)
    source = np.concatenate([np.zeros(len(val_latents)), np.ones(len(test_latents))], axis=0)
    # Combine labels: each label is a tuple (sequence index, frame index)
    all_labels = val_labels + test_labels

    print("Running UMAP on combined latent representations...")
    if args.analysis_method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
        all_latent_2d = reducer.fit_transform(all_latents)
    elif args.analysis_method == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        all_latent_2d = tsne.fit_transform(all_latents)
    elif args.analysis_method == "pca":
        pca = PCA(n_components=2, random_state=42)
        all_latent_2d = pca.fit_transform(all_latents)

    val_mask = (source == 0)
    test_mask = (source == 1)

    # Prepare frame index values for validation data.
    val_frame_values = np.array([lbl[1] for lbl in val_labels])
    # Determine min (early) and max (late) frame indices
    min_frame = val_frame_values.min()
    max_frame = val_frame_values.max()

    # Plot the scatter of latent points using the scatter's "c" parameter.
    plt.figure(figsize=(10, 8))
    # Use the "c" parameter to map colors for validation data.
    sc1 = plt.scatter(
        all_latent_2d[val_mask, 0],
        all_latent_2d[val_mask, 1],
        c=val_frame_values,
        cmap=plt.cm.viridis,
        s=10,
        label="Validation data"
    )

    # For test data, check if any exist.
    if len(test_labels) > 0:
        test_frame_values = np.array([lbl[1] for lbl in test_labels])
        plt.scatter(
            all_latent_2d[test_mask, 0],
            all_latent_2d[test_mask, 1],
            c=test_frame_values,
            cmap=plt.cm.viridis,
            s=80, edgecolor='black', linewidth=2.0,
            label="Known periodic patterns",
        )
    else:
        plt.scatter(
            all_latent_2d[test_mask, 0],
            all_latent_2d[test_mask, 1],
            color='grey',
            s=80, edgecolor='black', linewidth=2.0,
            label="Known periodic patterns",
        )

    layer = str(args.layer_to_extract) if args.layer_to_extract is not None else "before final layer norm"
    plt.title(f"Projection of GPT Representation Space ({args.analysis_method}, layer {layer})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="upper right", fontsize=14)

    # Create a colorbar for the validation scatter plot.
    import matplotlib as mpl
    # Create normalization using min and max frame indices.
    norm = mpl.colors.Normalize(vmin=min_frame, vmax=max_frame)
    cbar = plt.colorbar(sc1, ax=plt.gca(), norm=norm)
    cbar.set_label("Timestep", fontsize=14)

    # Set ticks to clearly indicate early vs. late.
    ticks = [min_frame, (min_frame + max_frame) / 2, max_frame]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"Early ({min_frame})", f"{(min_frame + max_frame) / 2:.0f}", f"Late ({max_frame})"], fontsize=14)

    """
    # --- Begin Added Code for Annotating Regions with Example Frames ---
    # Divide the latent space into a grid (3x3) and annotate a cell only if
    # there is at least one datapoint inside the cell.
    grid_n = 3
    x_min, x_max = all_latent_2d[:, 0].min(), all_latent_2d[:, 0].max()
    y_min, y_max = all_latent_2d[:, 1].min(), all_latent_2d[:, 1].max()
    cell_width = (x_max - x_min) / grid_n
    cell_height = (y_max - y_min) / grid_n

    # Flatten the datasets into a list of frames corresponding to the order of latent extraction.
    def flatten_dataset(dataset):
        num_seq, num_frames, _, _ = dataset.shape
        frames = []
        for seq_idx in range(num_seq):
            for frame_idx in range(num_frames):
                frames.append(dataset[seq_idx, frame_idx])
        return frames

    val_frames = flatten_dataset(val_dataset)
    test_frames = flatten_dataset(test_dataset) if len(test_latents) > 0 else []
    all_frames = val_frames + test_frames

    ax = plt.gca()  # get current axes
    for i in range(grid_n):
        for j in range(grid_n):
            cell_x_min = x_min + i * cell_width
            cell_x_max = cell_x_min + cell_width
            cell_y_min = y_min + j * cell_height
            cell_y_max = cell_y_min + cell_height

            # Find indices of latent points within this cell.
            in_cell = np.where(
                (all_latent_2d[:, 0] >= cell_x_min) & (all_latent_2d[:, 0] < cell_x_max) &
                (all_latent_2d[:, 1] >= cell_y_min) & (all_latent_2d[:, 1] < cell_y_max)
            )[0]

            if len(in_cell) == 0:
                continue  # Skip if no datapoint is in the cell.

            # Compute cell center.
            center_x = (cell_x_min + cell_x_max) / 2
            center_y = (cell_y_min + cell_y_max) / 2
            center = np.array([center_x, center_y])

            # Find the point in the cell closest to the center.
            distances = np.linalg.norm(all_latent_2d[in_cell] - center, axis=1)
            idx_in_cell = in_cell[np.argmin(distances)]

            # Retrieve the corresponding frame tokens and convert them to an image.
            frame_tokens = all_frames[idx_in_cell]
            # Pass img_size and grid_size as required.
            img = tokens_to_frame(frame_tokens, args.img_size, grid_size)

            # Create an annotation box without a boundary, with increased zoom and with a grayscale colormap.
            imagebox = OffsetImage(img, zoom=1.5, cmap='gray')
            ab = AnnotationBbox(imagebox, all_latent_2d[idx_in_cell],
                                frameon=False,  # Disable boundary box.
                                pad=0)        # Remove extra padding.
            ax.add_artist(ab)
    # --- End Added Code for Annotating Regions ---
    """
    
    plt.tight_layout()
    plt.savefig(f"latent_{args.analysis_method}.png")
    plt.show()
    print(f"Saved plot to latent_{args.analysis_method}.png")
