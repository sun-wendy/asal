#!/usr/bin/env python3
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from collections import deque
from conway_lib import ConwayGame

# ── compute per-patch temporal entropy (to detect oscillators) ─────────────────
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

# ── RL components: policy over entire frame ────────────────────────────────────
class Policy:
    def __init__(self, grid_shape, sigma, lr):
        # grid_shape: (H, W)
        self.mu = np.zeros(grid_shape, dtype=np.float32)
        self.log_sigma = np.log(sigma)
        self.lr = lr

    def sample(self, n):
        sigma = np.exp(self.log_sigma)
        eps = np.random.randn(n, *self.mu.shape)
        return self.mu[None] + sigma * eps, eps

    def update(self, eps, adv):
        sigma2 = np.exp(self.log_sigma)**2
        # eps.shape = (pop, H, W)
        grad = (adv[:, None, None] * eps / sigma2).mean(axis=0)
        self.mu += self.lr * grad

class ValueNet:
    def __init__(self, img_size):
        # input dim = img_size * img_size
        self.model = MLPRegressor(hidden_layer_sizes=(256,128),
                                  max_iter=10, warm_start=True)

    def fit(self, Z, y):
        # Z: (n_samples, img_size*img_size)
        self.model.fit(Z, y)

    def predict(self, Z):
        # Z: (n_queries, img_size*img_size)
        return self.model.predict(Z)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # stores (flattened_grid, reward)
        self.buf = deque(maxlen=capacity)

    def add(self, z, r):
        self.buf.append((z, r))

    def sample(self, n):
        data = random.sample(self.buf, n)
        Z, R = zip(*data)
        return np.stack(Z), np.array(R)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mcts_sims", type=int, default=20)
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--patches_per_dim", type=int, default=2)
    args = p.parse_args()

    # initialize policy over full grid
    grid_shape = (args.img_size, args.img_size)
    policy = Policy(grid_shape, sigma=args.sigma, lr=args.lr)
    value  = ValueNet(args.img_size)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    P = args.patches_per_dim

    for gen in range(1, args.iters+1):
        # 1) sample continuous grids and threshold → binary grids
        Z, eps = policy.sample(args.pop)            # (pop, H, W)
        grids = (Z > 0.0).astype(int)               # (pop, H, W)

        # 2) compute rewards: negative min patch-temporal entropy
        rewards = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            game = ConwayGame(width=args.img_size, height=args.img_size,
                               grid_size=1, toroidal=True)
            game.grid = grids[i].astype(bool)
            seq = game.run(num_iterations=args.steps)
            arr = np.stack([f.astype(int) for f in seq])
            patch_ents = compute_patch_temporal_entropy(arr, P)
            rewards[i] = -np.min(patch_ents)

        # 3) train critic on full-frame examples
        # store flattened grid
        for i in range(args.pop):
            buffer.add(grids[i].flatten(), rewards[i])

        if len(buffer.buf) >= args.pop:
            Zb, Rb = buffer.sample(args.pop)
            value.fit(Zb, Rb)

        # 4) MCTS-style improvement on full-grid
        Z_imp = []
        for z0 in Z:
            best_z, best_v = z0, -1e9
            for _ in range(args.mcts_sims):
                z1 = z0 + np.random.randn(*z0.shape) * np.exp(policy.log_sigma)
                grid1 = (z1 > 0.0).astype(int).flatten()[None]
                v1 = value.predict(grid1)[0]
                if v1 > best_v:
                    best_v, best_z = v1, z1
            Z_imp.append(best_z)
        Z_imp = np.stack(Z_imp)

        # 5) policy update using advantage
        Vpred = np.zeros(args.pop, dtype=np.float32)
        for i, z0 in enumerate(Z):
            Vpred[i] = value.predict(((z0>0.0).astype(int).flatten())[None])[0]
        adv = rewards - Vpred
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy.update(eps, adv)

        print(f"[gen {gen:4d}] reward mean={rewards.mean():.4f} max={rewards.max():.4f}")

        # 6) visualize first 10 improved
        if gen % 50 == 0:
            n_display = min(10, args.pop)
            fig, axes = plt.subplots(n_display, args.steps+1,
                                     figsize=((args.steps+1)*1.5, n_display*1.5))
            for i in range(n_display):
                grid = (Z_imp[i] > 0.0).astype(int)
                game = ConwayGame(width=args.img_size, height=args.img_size,
                                   grid_size=1, toroidal=True)
                game.grid = grid.astype(bool)
                seq = game.run(num_iterations=args.steps)
                frames = [grid] + [h.astype(int) for h in seq]
                for c, frame in enumerate(frames):
                    ax = axes[i, c]
                    ax.imshow(frame, cmap="gray", interpolation="nearest")
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"Step {c}")
            plt.tight_layout()
            plt.savefig(f"frame_search_gen_{gen:04d}.png", 
                        bbox_inches="tight", pad_inches=0.1)
            plt.close()

if __name__ == "__main__":
    main()
