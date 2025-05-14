import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from einops import rearrange
import csv

import substrates
from rollout import rollout_simulation


def frame_to_tokens(frame: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a simulation frame (H x W x D) into tokens by rearranging it into patches.
    For example, if grid_size=(p, p), the rearrangement is:
         "(H ph) (W pw) D -> (H W) (ph pw D)"
    The Game of Life frame is assumed binary (0=black, 1=white); we threshold the patch values.
    """
    tokens = rearrange(frame, "(H ph) (W pw) D -> (H W) (ph pw D)",
                        H=grid_size[0], W=grid_size[1])
    tokens = (tokens > 0.5).astype(np.float32)
    return tokens  # shape: (p*p, token_dim)

def tokens_to_frame(tokens: np.ndarray, img_size: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a sequence of tokens back into an image.
    Inverse operation of frame_to_tokens. For grid_size=(p, p),
    tokens of shape (p*p, token_dim) are rearranged into an image.
    """
    ph = img_size // grid_size[0]
    pw = img_size // grid_size[1]
    frame = rearrange(tokens, "(H W) (ph pw D) -> (H ph) (W pw) D",
                      H=grid_size[0], W=grid_size[1], ph=ph, pw=pw, D=1)
    return frame


def load_dataset_from_csv(csv_file: str, img_size: int, num_frames: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Load simulation sequences from a CSV file where each row has 'num_frames' cells.
    Each cell is a string representing a state for a 32×32 grid.
    Instead of expecting 1024 space-separated numbers, we assume that each cell is a
    1024-character string (one character per pixel).
    
    Returns an array of shape:
      (num_sequences, num_frames, num_tokens, token_dim)
    """
    print(f"Loading dataset from {csv_file}")
    sequences = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # Skip header rows if present
            if row[0].strip().startswith("State"):
                continue
            if len(row) != num_frames:
                raise ValueError(f"Expected row to have {num_frames} cells, got {len(row)}")
            seq = []
            for cell in row:
                cell_str = cell.strip()
                arr = np.fromstring(cell_str, sep=" ")
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
    
    # Tokenize each frame
    dataset_tokens = []
    for seq in data:
        seq_tokens = []
        for i in range(num_frames):
            frame = seq[i][..., np.newaxis]  # shape: (img_size, img_size, 1)
            tokens = frame_to_tokens(frame, grid_size)
            seq_tokens.append(tokens)
        dataset_tokens.append(np.stack(seq_tokens, axis=0))
    dataset_tokens = np.stack(dataset_tokens, axis=0)  # shape: (num_sequences, num_frames, num_tokens, token_dim)
    return dataset_tokens


def generate_token_dataset(rng, substrate, num_rollouts: int, rollout_steps: int,
                           img_size: int, grid_size: Tuple[int, int], t_skip: int = 0) -> np.ndarray:
    """
    Generate a batch of simulation rollouts.
    Each rollout produces a sequence of frames (binary images) from which tokens are computed.
    After tokenizing the full simulation (of length rollout_steps), only every (t_skip+1)-th frame is retained.
    
    Returns an array of shape:
        (num_rollouts, rollout_steps_eff, num_tokens, token_dim)
    where:
      - rollout_steps_eff = rollout_steps // (t_skip+1)
      - num_tokens = grid_size[0]*grid_size[1]
      - token_dim = (img_size//grid_size[0])^2.
    """
    rollout_rngs = jax.random.split(rng, num_rollouts)
    param_shape = substrate.default_params(jax.random.PRNGKey(0)).shape
    flat_params = jnp.full(param_shape, 6152)

    def rollout_fn(rng_i):
        result = rollout_simulation(
            rng_i, params=flat_params, substrate=substrate, fm=None,
            rollout_steps=rollout_steps, time_sampling='video',
            img_size=img_size, return_state=False
        )
        video = np.array(result['rgb'])  # shape: (rollout_steps, img_size, img_size, 3)
        gray_video = video[..., :1]       # use one channel; shape: (rollout_steps, img_size, img_size, 1)
        tokens_seq = np.array([frame_to_tokens(frame, grid_size) for frame in gray_video])
        # Sub-sample the frames by taking every (t_skip+1)-th frame.
        if t_skip > 0:
            tokens_seq = tokens_seq[::(t_skip + 1)]
        return tokens_seq

    sequences = [rollout_fn(rng_i) for rng_i in rollout_rngs]
    sequences = np.stack(sequences, axis=0)
    return sequences  # shape: (num_rollouts, rollout_steps_eff, num_tokens, token_dim)
