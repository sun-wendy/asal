import numpy as np
from typing import Tuple
from einops import rearrange

import jax
import jax.numpy as jnp
import torch
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


def generate_token_dataset_torch(rng: torch.Generator,
                                 substrate,
                                 num_rollouts: int,
                                 rollout_steps: int,
                                 img_size: int,
                                 grid_size: Tuple[int, int],
                                 t_skip: int = 0) -> torch.Tensor:
    """
    Generate a batch of simulation rollouts using PyTorch.
    Each rollout produces a sequence of frames (binary images) from which tokens are computed.
    After tokenizing the full simulation (of length rollout_steps), only every (t_skip+1)-th frame is retained.

    Returns a tensor of shape:
        (num_rollouts, rollout_steps_eff, num_tokens, token_dim)
    where:
      - rollout_steps_eff = rollout_steps // (t_skip+1)
      - num_tokens = grid_size[0]*grid_size[1]
      - token_dim = (img_size//grid_size[0])^2.
    """
    # Generate integer seeds for each rollout using torch.randint.
    rollout_seeds = torch.randint(0, 2**31, (num_rollouts,), generator=rng).tolist()

    # Obtain parameter shape from the substrate.
    # Adjust this call if substrate.default_params now works with torch.
    param_shape = substrate.default_params(torch.tensor(0)).shape
    flat_params = torch.full(param_shape, 6152)

    def rollout_fn(seed_i):
        # Here, we assume that rollout_simulation has been ported to work with torch seeds.
        result = rollout_simulation(
            seed_i, params=flat_params, substrate=substrate, fm=None,
            rollout_steps=rollout_steps, time_sampling='video',
            img_size=img_size, return_state=False
        )
        # Assume result['rgb'] is a tensor (or convertible to one) of shape:
        # (rollout_steps, img_size, img_size, 3)
        video = result['rgb']
        # Use one channel only (e.g. grayscale).
        gray_video = video[..., :1]  # shape: (rollout_steps, img_size, img_size, 1)

        # Tokenize each frame by calling your frame_to_tokens function.
        # We assume frame_to_tokens returns a tensor of shape (num_tokens, token_dim)
        tokens_seq = torch.stack([frame_to_tokens(frame, grid_size) for frame in gray_video])
        # Sub-sample the frames by taking every (t_skip+1)-th frame.
        if t_skip > 0:
            tokens_seq = tokens_seq[::(t_skip + 1)]
        return tokens_seq

    # Run the rollout for each generated seed.
    sequences = [rollout_fn(seed) for seed in rollout_seeds]
    # Stack into one tensor: shape will be
    # (num_rollouts, rollout_steps_eff, num_tokens, token_dim)
    sequences = torch.stack(sequences, dim=0)
    return sequences
