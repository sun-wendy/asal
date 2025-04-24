#!/usr/bin/env python3
"""
AE latent extraction + 2‑D projection plot (CLIP‑style)
------------------------------------------------------

This mirrors the original *clip_latent_vis.py* workflow **line‑for‑line** so you
can swap `clip` ↔︎ `ae` without touching your analysis scripts:

* JAX‑vectorised encoder forward pass, chunked to avoid OOM.
* Same CLI flags / defaults (including `--draw_arrows` behaviour).
* Picks 10 random validation sequences to highlight with grey arrows; colours
  test‑set patterns with **tab20** just like before.
"""

import argparse, numpy as np, matplotlib.pyplot as plt, pickle, csv, os
import jax, jax.numpy as jnp
from jax import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import flax.linen as nn

# -----------------------------------------------------------------------------#
#  Data loader (frames already in CSV)                                          #
# -----------------------------------------------------------------------------#

def _parse_cell(cell: str, img_size: int, channels: int):
    s = cell.strip()
    if channels == 1:
        arr = np.fromstring(s, sep=" ", dtype=np.float32)
        if arr.size == 0 and "," in s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        if arr.size != img_size * img_size and len(s) == img_size * img_size:
            arr = np.fromiter((float(c) for c in s), dtype=np.float32,
                               count=img_size * img_size)
    else:
        arr = np.fromstring(s.replace(",", " "), sep=" ", dtype=np.float32)
    if arr.size != img_size * img_size * channels:
        raise ValueError("bad cell: got %d vals, expected %d" % (arr.size, img_size*img_size*channels))
    return arr


def load_dataset_from_csv(path, img_size, num_frames, channels):
    seqs = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if row and row[0].startswith("State"):  # skip optional header
                continue
            if len(row) != num_frames:
                raise ValueError(f"{path}: need {num_frames} cols, got {len(row)}")
            frames = [_parse_cell(c, img_size, channels).reshape(img_size, img_size, channels)
                      for c in row]
            seqs.append(np.stack(frames, axis=0))
    data = np.stack(seqs, axis=0).astype(np.float32)
    print(f"[data] {path}: {data.shape[0]} seq × {num_frames} frames ({img_size}²×{channels})")
    return data

# -----------------------------------------------------------------------------#
#  Encoder definition (match training script)                                   #
# -----------------------------------------------------------------------------#

class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4,4), (2,2), padding="SAME")(x); x = nn.relu(x)
        x = nn.Conv(64, (4,4), (2,2), padding="SAME")(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(self.latent_dim)(x)

# -----------------------------------------------------------------------------#
#  Vectorised latent extraction                                                 #
# -----------------------------------------------------------------------------#

def extract_ae_latents_vmap(encode_fn, dataset, max_frames=120, batch_size=120):
    """dataset : (N_seq, num_frames, H, W, C)"""
    n_seq, n_frames, H, W, C = dataset.shape
    K = min(max_frames, n_frames)
    flat = dataset[:, :K].reshape(-1, H, W, C)                 # (N*K, H, W, C)
    lat_chunks = [jax.device_get(encode_fn(flat[i:i+batch_size]))
                  for i in range(0, flat.shape[0], batch_size)]
    latents = np.concatenate(lat_chunks, axis=0)
    seq_ids   = np.repeat(np.arange(n_seq), K)
    frame_ids = np.tile(np.arange(K), n_seq)
    labels    = list(zip(seq_ids.tolist(), frame_ids.tolist()))
    return latents, labels, flat

# -----------------------------------------------------------------------------#
#  main                                                                         #
# -----------------------------------------------------------------------------#

def main():
    ap = argparse.ArgumentParser("AE latent space visualiser (CLIP‑style)")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--channels", type=int, choices=[1,3], default=1)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--encoder_pkl", default="encoder_params.pkl")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv")
    ap.add_argument("--analysis_method", choices=["umap","tsne","pca"], default="umap")
    ap.add_argument("--draw_arrows", action="store_true")
    args = ap.parse_args()

    # ---------------- load datasets ---------------------------------------
    val_ds = load_dataset_from_csv(args.val_csv, args.img_size, args.num_frames, args.channels)
    if val_ds.shape[0] > 500:
        rng_fix = np.random.default_rng(0)           # fixed seed
        sel     = rng_fix.choice(val_ds.shape[0], 500, replace=False)
        val_ds  = val_ds[sel]

    test_ds = load_dataset_from_csv(args.test_csv, args.img_size, args.num_frames, args.channels) if args.test_csv else None

    # ---------------- encoder ---------------------------------------------
    with open(args.encoder_pkl, "rb") as f:
        enc_params = pickle.load(f)
    encoder = Encoder(args.latent_dim)
    encode_fn = jax.jit(lambda x: encoder.apply({'params': enc_params}, x))

    # ---------------- latents ---------------------------------------------
    val_lat, val_lbl, _ = extract_ae_latents_vmap(encode_fn, val_ds, args.max_frames)
    max_val_id = max((s for s,_ in val_lbl), default=-1)

    if test_ds is not None:
        raw_lat, raw_lbl, _ = extract_ae_latents_vmap(encode_fn, test_ds, args.max_frames)
        offset   = max_val_id + 1
        test_lat = raw_lat
        test_lbl = [(s+offset, f) for (s,f) in raw_lbl]
    else:
        test_lat, test_lbl = np.empty((0, val_lat.shape[1])), []

    # ---------------- concatenate & project -------------------------------
    all_lat = np.vstack([val_lat, test_lat])
    src     = np.concatenate([np.zeros(len(val_lat)), np.ones(len(test_lat))])
    labels  = val_lbl + test_lbl

    if args.analysis_method == "umap":
        proj = UMAP(n_components=2, random_state=0).fit_transform(all_lat)
    elif args.analysis_method == "tsne":
        proj = TSNE(n_components=2, random_state=0).fit_transform(all_lat)
    else:
        proj = PCA(n_components=2, random_state=0).fit_transform(all_lat)

    # ---------------- plotting (identical style to CLIP script) -----------
    scatter_val  = src == 0
    scatter_test = src == 1
    frame_ids = np.array([f for _,f in labels], dtype=int)

    fig, ax = plt.subplots(figsize=(10,8))
    sc_val = ax.scatter(proj[scatter_val,0], proj[scatter_val,1],
                        c=frame_ids[scatter_val], cmap='viridis', s=12,
                        alpha=0.01 if args.draw_arrows else 1.0, label='validation')
    sc_tst = ax.scatter(proj[scatter_test,0], proj[scatter_test,1],
                        c=frame_ids[scatter_test], cmap='viridis', s=60,
                        edgecolor='k', alpha=0.01 if args.draw_arrows else 1.0,
                        label='test')
    cb = plt.colorbar(sc_val, ax=ax); cb.set_label('timestep')

    # ---------------- arrows & legend -------------------------------------
    if args.draw_arrows:
        from collections import defaultdict
        from matplotlib.lines import Line2D
        rng = np.random.default_rng(0)
        chosen_val_seq = rng.choice(sorted({s for s,_ in val_lbl}), size=min(10, len(set(s for s,_ in val_lbl))), replace=False)

        import matplotlib as mpl
        cmap_test = mpl.cm.get_cmap('tab20')
        test_seq_ids = sorted({s for s,_ in test_lbl})
        seq2col = {s: cmap_test(i % cmap_test.N) for i,s in enumerate(test_seq_ids)}

        arrow_val = dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, mutation_scale=6,
                         lw=.8, color='dimgray')
        seq_pts = defaultdict(list)
        for idx,(s,f) in enumerate(labels):
            seq_pts[s].append((f, idx))
        for s, pts in seq_pts.items():
            pts.sort(key=lambda t:t[0]); idxs=[i for _,i in pts]
            if (s in test_seq_ids) or (s in chosen_val_seq):
                if s in test_seq_ids:
                    col = seq2col[s]
                    line_kw, arrow_kw = dict(color=col, lw=.8, zorder=2), dict(arrow_val, color=col)
                else:
                    line_kw, arrow_kw = dict(color='lightgray', lw=.6, zorder=2), arrow_val
                ax.plot(proj[idxs,0], proj[idxs,1], **line_kw)
                for i1,i2 in zip(idxs[:-1], idxs[1:]):
                    ax.annotate('', xy=proj[i2], xytext=proj[i1], arrowprops=arrow_kw)

        test_names = ["glider", "lwss", "mwss", "hwss"]
        pattern_names = [f'pattern {name}' for name in test_names]
        handles = [Line2D([],[],color='lightgray', lw=1, label='random val seq')] + \
                  [Line2D([],[],color=seq2col[s], lw=1, label=name) for s,name in zip(test_seq_ids, pattern_names)]
        ax.legend(handles=handles, fontsize='small', ncol=2)

    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.set_title(f"AE latent space ({args.analysis_method.upper()})")
    plt.tight_layout(); plt.savefig(f"ae_latent_{args.analysis_method}.png")
    print('✓ saved ae_latent_%s.png' % args.analysis_method)


if __name__ == "__main__":
    main()
