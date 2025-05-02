#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# ── compute normalized spatio-temporal entropy ───────────────────────────────────
def compute_st_entropy_arr(arr: np.ndarray, patches_per_dim: int) -> float:
    L, H, W = arr.shape
    k = H // patches_per_dim
    blocks = arr.reshape(L, patches_per_dim, k, patches_per_dim, k)
    blocks = blocks.transpose(1,3,0,2,4).reshape(-1, L*k*k)
    uniq, counts = np.unique(blocks, axis=0, return_counts=True)
    probs = counts / counts.sum()
    Hval = -np.sum(probs * np.log2(probs + 1e-12))
    return Hval / (k*k*L)

def compute_patch_temporal_entropy(arr: np.ndarray, patches_per_dim: int) -> np.ndarray:
    L, H, W = arr.shape
    k = H // patches_per_dim
    entropies = []
    for pi in range(patches_per_dim):
        for pj in range(patches_per_dim):
            patch_seq = arr[:, pi*k:(pi+1)*k, pj*k:(pj+1)*k]
            patterns = patch_seq.reshape(L, k*k)
            uniq, counts = np.unique(patterns, axis=0, return_counts=True)
            probs = counts / counts.sum()
            H = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(H / (k * k))
    return np.array(entropies)

# ── prepare and cache raw-frame features & entropies ────────────────────────────
def prepare_and_cache(
    csv_path: str,
    cache_path: str,
    img_size: int,
    patches_per_dim: int,
    num_frames: int,
    t_skip: int,
):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']

    # load sequences from CSV (each row: flattened frames, last entry ignored)
    rows = list(csv.reader(open(csv_path)))[1:]
    n = len(rows)
    seqs = np.array([list("".join(row[:-1])) for row in rows], int)
    seqs = seqs.reshape(n, -1, img_size, img_size)
    seqs = seqs[:, :num_frames]
    if t_skip > 0:
        seqs = seqs[:, ::(t_skip+1)]

    y = np.array([compute_st_entropy_arr(s, patches_per_dim) for s in seqs], dtype=np.float32)

    # take only the initial frame, flatten it to a vector
    inits = seqs[:, 0, ...]                              # shape (n, H, W)
    X = inits.reshape(n, img_size * img_size)           # shape (n, H*W)

    # cache for next time
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y

# ── main: train surrogate MLP from raw frames ───────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv',   required=True)
    p.add_argument('--val_csv',     required=True)
    p.add_argument('--img_size',    type=int, default=16)
    p.add_argument('--patches_per_dim', type=int, default=2)
    p.add_argument('--num_frames',  type=int, default=10)
    p.add_argument('--t_skip',      type=int, default=0)
    p.add_argument('--hidden_layers', nargs='+', type=int, default=[128,64])
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--total_steps', type=int, default=3000)
    p.add_argument('--learning_rate_init', type=float, default=1e-3)
    p.add_argument('--cache_dir',   type=str, default='cache')
    p.add_argument('--output_model', type=str, default='surrogate_mlp.joblib')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # prepare data (raw‐frame features)
    train_cache = os.path.join(args.cache_dir, 'train_raw.npz')
    val_cache   = os.path.join(args.cache_dir, 'val_raw.npz')
    X_train, y_train = prepare_and_cache(
        args.train_csv, train_cache,
        args.img_size, args.patches_per_dim,
        args.num_frames, args.t_skip
    )
    X_val, y_val     = prepare_and_cache(
        args.val_csv, val_cache,
        args.img_size, args.patches_per_dim,
        args.num_frames, args.t_skip
    )

    # scale features (X)
    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_val   = scaler_X.transform(X_val)

    y_train = y_train.reshape(-1, 1)
    y_val   = y_val.reshape(-1, 1)

    # scale targets (y) as 2D for multi-output
    scaler_y = StandardScaler().fit(y_train)    # fit on train only
    y_train_std = scaler_y.transform(y_train)   # shape (N, P²)
    y_val_std   = scaler_y.transform(y_val)     # shape (N, P²)

    print(f"[y_train] min={y_train.min():.6f} max={y_train.max():.6f}")
    print(f"[y_train_std] min={y_train_std.min():.3f} max={y_train_std.max():.3f}")
    print(f"[y_val]   min={y_val.min():.6f} max={y_val.max():.6f}")
    print(f"[y_val_std]   min={y_val_std.min():.3f} max={y_val_std.max():.3f}")

    # build MLP and warm up partial_fit
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu', solver='adam',
        warm_start=False,
        max_iter=1,
        learning_rate_init=args.learning_rate_init,
        learning_rate='constant'
    )

    # initial partial_fit
    mlp.partial_fit(X_train[:args.batch_size], y_train_std[:args.batch_size])

    rng = np.random.default_rng(0)
    n_train = X_train.shape[0]

    train_mse_history = []
    val_mse_history   = []

    # training loop with full‐validation evaluation
    for step in range(1, args.total_steps+1):
        idxs = rng.choice(n_train, args.batch_size, replace=False)
        Xb, yb = X_train[idxs], y_train_std[idxs]
        mlp.partial_fit(Xb, yb)

        yb_pred = mlp.predict(Xb)
        train_mse = mean_squared_error(yb, yb_pred)
        train_r2  = r2_score(yb, yb_pred, multioutput='uniform_average')
        print(f"[Step {step}] train std MSE={train_mse:.4f}, R2={train_r2:.4f}")
        train_mse_history.append(train_mse)

        # evaluate on the entire validation set
        yv_pred = mlp.predict(X_val)                # shape (N, P²)
        mse_all = mean_squared_error(y_val_std, yv_pred)
        r2_all  = r2_score(y_val_std, yv_pred, multioutput='uniform_average')
        print(f"[Step {step}] full-val std MSE={mse_all:.4f}, R2={r2_all:.4f}")
        val_mse_history.append(mse_all)

    df = pd.DataFrame({
        'train': train_mse_history,
        'val':   val_mse_history
    })

    # compute exponential moving‐average with span=1000
    df_smooth = df.ewm(span=100).mean()

    plt.figure()
    # note: df_smooth.index goes from 0…total_steps-1, so +1 to align with steps
    plt.plot(df_smooth.index + 1, df_smooth['train'], label='train MSE (smoothed)')
    plt.plot(df_smooth.index + 1, df_smooth['val'],   label='val MSE (smoothed)')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.title('Smoothed Training & Validation MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('surrogate_baseline_mse_smoothed.png')
