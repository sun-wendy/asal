import argparse
import numpy as np
from util_gol import load_dataset_from_csv, tokens_to_frame

def compute_shannon_entropy_np(frames: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy for each binary frame.
    - frames: shape (N, img_size, img_size), values {0,1}
    - returns: (N,) array of entropies
    """
    flat = frames.reshape((frames.shape[0], -1))         # (N, img_size*img_size)
    p = np.mean(flat, axis=1)                            # (N,) probability of 1
    eps = 1e-8
    p_safe = np.clip(p, eps, 1.0 - eps)
    entropy = - (p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))
    return entropy

def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline MSE: predict mean entropy of frame at dt_probe"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to the validation CSV file of Game of Life sequences.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=32,
        help="Image size (pixels) of each frame.",
    )
    parser.add_argument(
        "--patches_per_dim",
        type=int,
        default=8,
        help="Number of patches per image dimension (e.g. 8 means 8×8 tokens per frame).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=128,
        help="Total number of frames per sequence in the CSV.",
    )
    parser.add_argument(
        "--dt_probe",
        type=int,
        default=16,
        help="Index of frame whose entropy you want to predict (0-based).",
    )
    args = parser.parse_args()

    # 1) Load validation dataset: shape (N, num_frames, num_tokens, token_dim)
    dataset_np = load_dataset_from_csv(
        args.val_csv,
        args.img_size,
        args.num_frames,
        (args.patches_per_dim, args.patches_per_dim),
    )
    N, num_frames, num_tokens, token_dim = dataset_np.shape

    if not (0 <= args.dt_probe < num_frames):
        raise ValueError(f"dt_probe must be in [0, {num_frames-1}]. Got {args.dt_probe}.")

    # 2) Reconstruct each frame at dt_probe as a binary image
    frames = []
    for i in range(N):
        tokens_i = dataset_np[i, args.dt_probe, :, :]  # (num_tokens, token_dim)
        frame_i = tokens_to_frame(
            tokens_i,
            img_size=args.img_size,
            grid_size=(args.patches_per_dim, args.patches_per_dim),
        )  # returns (img_size, img_size) binary array
        frames.append(frame_i)
    frames = np.stack(frames, axis=0)  # (N, img_size, img_size)

    # 3) Compute entropy vector
    entropies = compute_shannon_entropy_np(frames)  # (N,)

    # 4) Compute baseline MSE: predict the mean ENTROPY for all examples
    mean_H = entropies.mean()
    baseline_mse = np.mean((entropies - mean_H) ** 2)

    print(f"Baseline MSE (predicting mean entropy={mean_H:.6f}): {baseline_mse:.6f}")


if __name__ == "__main__":
    main()
