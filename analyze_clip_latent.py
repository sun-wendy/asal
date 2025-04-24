#!/usr/bin/env python3
"""
Vector-ised CLIP latent extraction + 2-D projection plot
-------------------------------------------------------

* Uses **JAX vmap** to batch tokens→images→CLIP embeddings (no Python loops).
* Limits to the first `--max_frames` (default 120) of every sequence.
* Randomly subsamples **≤ 500 validation sequences** for speed / clarity.
"""

import argparse, numpy as np, matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from einops import rearrange
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

import foundation_models
from util_gol import tokens_to_frame, load_dataset_from_csv


# -----------------------------------------------------------------------------#
#  Batched CLIP helper (monkey-patched onto the model)                          #
# -----------------------------------------------------------------------------#
def _embed_imgs(self, imgs_bhwc):
    """ imgs_bhwc (B,H,W,C) → (B,D) L2-normalised CLIP embeddings """
    B, H, W, C = imgs_bhwc.shape
    if (H, W) != (224, 224):
        imgs_bhwc = jax.image.resize(imgs_bhwc, (B, 224, 224, C), method='bilinear')
    imgs_bchw = rearrange((imgs_bhwc - self.img_mean) / self.img_std, 'b h w c -> b c h w')
    z = self.clip_model.get_image_features(imgs_bchw)
    return z / jnp.linalg.norm(z, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------#
#  Vectorised latent extraction                                                #
# -----------------------------------------------------------------------------#
def extract_clip_latents_vmap(clip, dataset, img_size, grid, max_frames=120, batch_size=256):
    """
    dataset : (N_seq, num_frames, H, W) integer tokens
    Returns  : latents (N_seq*max_frames,D), labels, frames
    """
    n_seq, n_frames, H, W = dataset.shape
    K = min(max_frames, n_frames)
    print("HERE2")
    # tokens → RGB frames
    to_rgb   = lambda t: tokens_to_frame(t, img_size, grid)
    tok_flat = dataset[:, :K].reshape(-1, H, W)
    frames   = jax.vmap(to_rgb)(tok_flat)                                      # (N*K,H,W,3)
    print("HERE3")
    # batched CLIP embed
    lat_chunks = [np.array(clip.embed_imgs(frames[i:i+batch_size]))
                  for i in range(0, frames.shape[0], batch_size)]
    latents = np.concatenate(lat_chunks, axis=0)
    print("HERE4")
    # labels
    seq_ids   = np.repeat(np.arange(n_seq), K)
    frame_ids = np.tile(np.arange(K), n_seq)
    labels    = list(zip(seq_ids.tolist(), frame_ids.tolist()))
    return latents, labels, frames


# -----------------------------------------------------------------------------#
#  main                                                                         #
# -----------------------------------------------------------------------------#
def main():
    # -------------------------- CLI ------------------------------------------
    ap = argparse.ArgumentParser("CLIP latent space visualiser")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--patches_per_dim", type=int, default=2)
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv")
    ap.add_argument("--analysis_method", choices=["umap", "tsne", "pca"], default="umap")
    ap.add_argument("--draw_arrows", action="store_true")
    args = ap.parse_args()

    grid = (args.patches_per_dim, args.patches_per_dim)

    # -------- CLIP model with batched embed ----------------------------------
    clip = foundation_models.create_foundation_model("clip")
    clip.embed_imgs = _embed_imgs.__get__(clip)

    # -------- load & (optionally) subsample validation -----------------------
    val_tok = load_dataset_from_csv(args.val_csv, args.img_size, args.num_frames, grid)
    print("HERE1")
    if val_tok.shape[0] > 500:                     # ≤ 500 validation sequences
        rng_fix = np.random.default_rng(0)           # fixed seed
        sel     = rng_fix.choice(val_tok.shape[0], 500, replace=False)
        val_tok  = val_tok[sel]
    val_lat, val_lbl, _ = extract_clip_latents_vmap(
        clip, val_tok, args.img_size, grid, max_frames=args.max_frames)
    max_val_id = max((s for s, _ in val_lbl), default=-1)

    # -------- test set (optional) -------------------------------------------
    if args.test_csv:
        test_tok = load_dataset_from_csv(args.test_csv, args.img_size, args.num_frames, grid)
        raw_lat, raw_lbl, _ = extract_clip_latents_vmap(
            clip, test_tok, args.img_size, grid, max_frames=args.max_frames)
        offset   = max_val_id + 1
        test_lat = raw_lat
        test_lbl = [(s + offset, f) for (s, f) in raw_lbl]
    else:
        test_lat, test_lbl = np.empty((0, val_lat.shape[1])), []

    # -------- concatenate & project -----------------------------------------
    all_lat = np.vstack([val_lat, test_lat])
    src     = np.concatenate([np.zeros(len(val_lat)), np.ones(len(test_lat))])
    labels  = val_lbl + test_lbl

    if args.analysis_method == "umap":
        proj = UMAP(n_components=2, random_state=0).fit_transform(all_lat)
    elif args.analysis_method == "tsne":
        proj = TSNE(n_components=2, random_state=0).fit_transform(all_lat)
    else:
        proj = PCA(n_components=2, random_state=0).fit_transform(all_lat)

    # -------- masks & colours ----------------------------------------------
    scatter_val = src == 0
    scatter_test = src == 1

    rng = np.random.default_rng(0)
    chosen_val_seq = (rng.choice(sorted({s for s, _ in val_lbl}), size=min(10, len(set(s for s,_ in val_lbl))), replace=False)
                      if args.draw_arrows else [])

    import matplotlib as mpl
    cmap_test = mpl.cm.get_cmap("tab20")
    test_seq_ids = sorted({s for s, _ in test_lbl})
    seq2col = {s: cmap_test(i % cmap_test.N) for i, s in enumerate(test_seq_ids)}

    # -------------------- plotting ------------------------------------------
    frame_ids = np.array([f for _, f in labels], dtype=int)
    fig, ax = plt.subplots(figsize=(10, 8))

    sc_val = ax.scatter(proj[scatter_val, 0], proj[scatter_val, 1],
                        c=frame_ids[scatter_val], cmap='viridis', s=12,
                        alpha=0.01 if args.draw_arrows else 1.0, label="validation")
    sc_tst = ax.scatter(proj[scatter_test, 0], proj[scatter_test, 1],
                        c=frame_ids[scatter_test], cmap='viridis', s=60,
                        edgecolor='k',
                        alpha=0.01 if args.draw_arrows else 1.0, label="test")
    cb = plt.colorbar(sc_val, ax=ax); cb.set_label("timestep")

    # -------------------- arrows --------------------------------------------
    if args.draw_arrows:
        from collections import defaultdict
        from matplotlib.lines import Line2D
        arrow_val = dict(arrowstyle='-|>', shrinkA=0, shrinkB=0,
                         mutation_scale=6, lw=.8, color='dimgray')

        seq_pts = defaultdict(list)
        for idx, (s, f) in enumerate(labels):
            seq_pts[s].append((f, idx))

        for s, pts in seq_pts.items():
            pts.sort(key=lambda t: t[0]); idxs = [i for _, i in pts]
            if (s in test_seq_ids) or (s in chosen_val_seq):
                if s in test_seq_ids:
                    col, line_kw, arrow_kw = seq2col[s], dict(color=seq2col[s], lw=.8, zorder=2), dict(arrow_val, color=seq2col[s])
                else:
                    col, line_kw, arrow_kw = 'lightgray', dict(color='lightgray', lw=.6, zorder=2), arrow_val
                ax.plot(proj[idxs, 0], proj[idxs, 1], **line_kw)
                for i1, i2 in zip(idxs[:-1], idxs[1:]):
                    ax.annotate("", xy=proj[i2], xytext=proj[i1], arrowprops=arrow_kw)

        test_names = ["glider", "lwss", "mwss", "hwss"]
        handles = [Line2D([], [], color='lightgray', lw=1, label='random val seq')] + \
                  [Line2D([], [], color=seq2col[s], lw=1, label=f'pattern {name}') for (s, name) in zip(test_seq_ids, test_names)]
        ax.legend(handles=handles, fontsize='small', ncol=2)

    # -------------------- finish --------------------------------------------
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    ax.set_title(f"CLIP latent space ({args.analysis_method.upper()})")
    plt.tight_layout()
    plt.savefig(f"clip_latent_{args.analysis_method}.png")


if __name__ == "__main__":
    main()
