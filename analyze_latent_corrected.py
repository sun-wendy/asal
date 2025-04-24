import pickle
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

from util_gol import load_dataset_from_csv, tokens_to_frame

@dataclass
class GPTConfig:
    img_size: int
    block_size: int         # (num_eff_frames - 1) * (num_tokens)
    token_dim: int          # Dimension of each token (e.g., (img_size // patches_per_dim)**2)
    num_tokens: int         # Number of tokens per frame (e.g., patches_per_dim**2)
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


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
        att = jnp.where(mask == 1.0, att, float('-inf'))
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

    def get_hidden(
        self,
        tokens: jnp.ndarray,
        *,
        train: bool = False,
        layer: Optional[int] = None,
        apply_ln: bool = True,
    ) -> jnp.ndarray:
        """
        Return the hidden states after `layer` transformer blocks.
        Pass `layer=None` to run the full stack.
        """
        x = self.token_proj(tokens)                                    # (B,T,n_embd)
        pos = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]    # (1,T)
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)

        # iterate over *self.h*, not self.blocks
        for i, block in enumerate(self.h):
            x = block(x, train=train)
            if layer is not None and i == layer:
                break

        if apply_ln:
            x = self.ln_f(x)
        return x                              # (B,T,n_embd)

# context-aware latent extraction

def extract_latent_with_context(model, params, seq_tokens: np.ndarray,
                                 num_tokens: int, layer: Optional[int], apply_ln: bool) -> np.ndarray:
    hidden = model.apply({'params': params}, jnp.array(seq_tokens[None,...]),
                         train=False, method=GPT.get_hidden,
                         layer=layer, apply_ln=apply_ln)
    hidden = np.array(hidden[0])  # (T_total, n_embd)
    n_frames = hidden.shape[0] // num_tokens
    latents = []
    for i in range(n_frames):
        start, end = i*num_tokens, (i+1)*num_tokens
        latents.append(hidden[start:end].mean(axis=0))
    return np.stack(latents, axis=0)


def extract_dataset_context(model, params, dataset: np.ndarray,
                            num_tokens: int, layer: Optional[int], apply_ln: bool) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    all_lat = []
    labels: List[Tuple[int,int]] = []
    for seq_idx in range(dataset.shape[0]):
        seq = dataset[seq_idx, :-1]
        seq_flat = seq.reshape(-1, seq.shape[-1])
        lat = extract_latent_with_context(model, params, seq_flat,
                                         num_tokens, layer, apply_ln)
        all_lat.append(lat)
        labels += [(seq_idx, f) for f in range(lat.shape[0])]
    return np.vstack(all_lat), labels

# -----------------------------------------------------------------------------#
#                               main                                            #
# -----------------------------------------------------------------------------#
if __name__ == '__main__':
    # ------------------------------- CLI --------------------------------------
    p = argparse.ArgumentParser('Context-aware GPT latent projection')
    p.add_argument('--img_size',        type=int,  default=32)
    p.add_argument('--patches_per_dim', type=int,  default=2)
    p.add_argument('--num_frames',      type=int,  default=10)
    p.add_argument('--t_skip',          type=int,  default=0)
    p.add_argument('--val_csv',         type=str,  required=True)
    p.add_argument('--test_csv',        type=str)
    p.add_argument('--checkpoint',      type=str,  default='gpt_params.pkl')
    p.add_argument('--n_layer',         type=int,  default=12)
    p.add_argument('--n_head',          type=int,  default=8)
    p.add_argument('--n_embd',          type=int,  default=256)
    p.add_argument('--dropout',         type=float, default=0.1)
    p.add_argument('--analysis_method', choices=['umap', 'tsne', 'pca'],
                   default='pca')
    p.add_argument('--layer_to_extract', type=int)
    p.add_argument('--apply_ln',  dest='apply_ln',  action='store_true')
    p.add_argument('--no_apply_ln', dest='apply_ln', action='store_false')
    p.set_defaults(apply_ln=True)
    p.add_argument('--draw_arrows',    dest='draw_arrows',  action='store_true')
    p.add_argument('--no_draw_arrows', dest='draw_arrows',  action='store_false')
    p.set_defaults(draw_arrows=False)
    args = p.parse_args()

    # ---------------------------- build model ---------------------------------
    grid        = (args.patches_per_dim, args.patches_per_dim)
    token_dim   = (args.img_size // args.patches_per_dim) ** 2
    num_tokens  = args.patches_per_dim ** 2
    num_eff     = math.ceil(args.num_frames / (args.t_skip + 1))
    block_size  = (num_eff - 1) * num_tokens

    cfg   = GPTConfig(args.img_size, block_size, token_dim,
                      num_tokens, args.n_layer, args.n_head,
                      args.n_embd, args.dropout)
    model = GPT(cfg)
    with open(args.checkpoint, 'rb') as f:
        params = pickle.load(f)

    # ---------------------------- validation set ------------------------------
    val_ds = load_dataset_from_csv(args.val_csv, args.img_size,
                                   args.num_frames, grid)
    if val_ds.shape[0] > 500:
        rng_fix = np.random.default_rng(0)           # fixed seed
        sel     = rng_fix.choice(val_ds.shape[0], 500, replace=False)
        val_ds  = val_ds[sel]

    val_lat, val_lbl = extract_dataset_context(model, params, val_ds,
                                               num_tokens,
                                               args.layer_to_extract,
                                               args.apply_ln)
    # the largest validation seq-id (-1 if none)
    max_val_id = max((s for s, _ in val_lbl), default=-1)

    # ------------------------------ test set ----------------------------------
    if args.test_csv:
        test_ds  = load_dataset_from_csv(args.test_csv, args.img_size,
                                         args.num_frames, grid)[:5]
        raw_lat, raw_lbl = extract_dataset_context(model, params, test_ds,
                                                   num_tokens,
                                                   args.layer_to_extract,
                                                   args.apply_ln)
        # offset seq-ids so they never collide with validation ids
        offset        = max_val_id + 1
        test_lat      = raw_lat
        test_lbl      = [(s + offset, f) for (s, f) in raw_lbl]
    else:
        test_lat, test_lbl = np.empty((0, args.n_embd)), []

    # ---------------------- dimensionality reduction --------------------------
    all_lat = np.vstack([val_lat, test_lat])
    src     = np.concatenate([np.zeros(len(val_lat)), np.ones(len(test_lat))])
    labels  = val_lbl + test_lbl                       # list[(seq, frame)]

    if args.analysis_method == 'umap':
        proj = UMAP(n_components=2, random_state=0).fit_transform(all_lat)
    elif args.analysis_method == 'tsne':
        proj = TSNE(n_components=2, random_state=0).fit_transform(all_lat)
    else:
        proj = PCA(n_components=2, random_state=0).fit_transform(all_lat)

    # ------------------------------ masks -------------------------------------
    rng = np.random.default_rng(0)
    scatter_val_mask  = (src == 0)   # all validation points
    scatter_test_mask = (src == 1)   # all test points

    if args.draw_arrows:
        val_seq_ids    = sorted({s for s, _ in val_lbl})
        chosen_val_seq = rng.choice(val_seq_ids,
                                    size=min(10, len(val_seq_ids)),
                                    replace=False)
    else:
        chosen_val_seq = []

    # ------------------------------- plot -------------------------------------
    frame_ids = np.array([f for _, f in labels], dtype=np.int32)
    fig, ax   = plt.subplots(figsize=(10, 8))

    sc_val = ax.scatter(proj[scatter_val_mask, 0], proj[scatter_val_mask, 1],
                        c=frame_ids[scatter_val_mask], cmap='viridis', s=10,
                        alpha=0.01 if args.draw_arrows else 1.0,
                        label='validation')
    sc_test = ax.scatter(proj[scatter_test_mask, 0], proj[scatter_test_mask, 1],
                         c=frame_ids[scatter_test_mask], cmap='viridis', s=80,
                         edgecolor='k',
                         alpha=0.01 if args.draw_arrows else 1.0,
                         label='test')

    cbar = fig.colorbar(sc_val, ax=ax); cbar.set_label('timestep')

    # --------------------------- arrow overlay --------------------------------
    if args.draw_arrows:
        from collections import defaultdict
        import matplotlib as mpl
        from matplotlib.lines import Line2D

        cmap_test = mpl.cm.get_cmap('tab20')
        test_seq_ids = sorted({s for s, _ in test_lbl})
        seq2col = {s: cmap_test(i % cmap_test.N) for i, s in enumerate(test_seq_ids)}

        arrow_val = dict(arrowstyle='-|>', mutation_scale=6,
                         lw=.8, color='dimgray')

        seq_pts = defaultdict(list)
        for idx, (s, f) in enumerate(labels):
            seq_pts[s].append((f, idx))

        for s, pts in seq_pts.items():
            pts.sort(key=lambda t: t[0]); idx = [i for _, i in pts]
            if (s in test_seq_ids) or (s in chosen_val_seq):
                if s in test_seq_ids:
                    col  = seq2col[s]
                    line_kw  = dict(color=col, lw=.8, zorder=2)
                    arrow_kw = dict(arrow_val, color=col)
                else:
                    line_kw  = dict(color='lightgray', lw=.6, zorder=2)
                    arrow_kw = arrow_val

                ax.plot(proj[idx, 0], proj[idx, 1], **line_kw)
                for i1, i2 in zip(idx[:-1], idx[1:]):
                    ax.annotate('', xy=proj[i2], xytext=proj[i1],
                                arrowprops=arrow_kw)

        test_names = ["glider", "lwss", "mwss", "hwss"]
        handles = [Line2D([], [], color='lightgray', lw=1,
                           label='random val seq')] + \
                  [Line2D([], [], color=seq2col[s], lw=1,
                           label=f'pattern {name}') for (s, name) in zip(test_seq_ids, test_names)]
        ax.legend(handles=handles, fontsize='small', ncol=2)

    # ------------------------------ cosmetics ---------------------------------
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    layer_desc = args.layer_to_extract if args.layer_to_extract is not None else 'last'
    ax.set_title(f'GPT latent space (layer={layer_desc}, '
                 f'{args.analysis_method.upper()})')
    plt.tight_layout()
    plt.savefig(f'latent_space_{layer_desc}_{args.analysis_method}.png')
