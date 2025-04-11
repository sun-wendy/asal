import math
import os
import pickle
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax import traverse_util
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.core.frozen_dict import freeze
from einops import rearrange  # for simple rearrangement operations

import substrates
from rollout import rollout_simulation


# =============================================
#  Data Generation and Tokenization Functions
# =============================================

def frame_to_tokens(frame: np.ndarray, grid_size=(2, 2)) -> np.ndarray:
    """
    Convert a simulation frame (H x W x D) into tokens by rearranging it into patches.
    For example, for grid_size=(2,2), we rearrange the frame:
    
       "(H ph) (W pw) D -> (H W) (ph pw D)"
    
    For Game of Life the frame is assumed to be binary (0=black, 1=white). We threshold
    the patch values to yield binary tokens.
    """
    # Rearrange the image into patches.
    tokens = rearrange(frame, "(H ph) (W pw) D -> (H W) (ph pw D)", H=grid_size[0], W=grid_size[1])
    # Ensure binary outputs (0 or 1).
    tokens = (tokens > 0.5).astype(np.float32)
    return tokens  # shape: (grid_size[0]*grid_size[1], token_dim)

def tokens_to_frame(tokens: np.ndarray, img_size=64, grid_size=(2, 2)) -> np.ndarray:
    """
    Convert a sequence of tokens back into an image.
    Inverse operation of frame_to_tokens. For grid_size=(2,2),
    tokens of shape (4, token_dim) are rearranged back into a frame.
    """
    ph = img_size // grid_size[0]
    pw = img_size // grid_size[1]
    frame = rearrange(tokens, "(H W) (ph pw D) -> (H ph) (W pw) D", 
                        H=grid_size[0], W=grid_size[1], ph=ph, pw=pw, D=1)
    return frame

def generate_token_dataset(rng, substrate, num_rollouts=32, rollout_steps=256, img_size=64):
    """
    Generate a batch of simulation rollouts on the fly.
    Each rollout produces a sequence of frames (binary images) from which we compute tokens.
    Returns an array of shape (num_rollouts, rollout_steps, num_tokens, token_dim),
    where num_tokens is determined by grid_size (here 2x2=4) and token_dim = (img_size//2)^2.
    """
    rollout_rngs = jax.random.split(rng, num_rollouts)
    # Create dummy substrate parameters.
    param_shape = substrate.default_params(jax.random.PRNGKey(0)).shape
    flat_params = jnp.full(param_shape, 6152)

    def rollout_fn(rng_i):
        result = rollout_simulation(
            rng_i, params=flat_params, substrate=substrate, fm=None,
            rollout_steps=rollout_steps, time_sampling='video',
            img_size=img_size, return_state=False
        )
        # result['rgb'] is an array of shape (rollout_steps, img_size, img_size, 3)
        video = np.array(result['rgb'])
        # Use one channel (assume binary image).
        gray_video = video[..., :1]  # shape: (rollout_steps, img_size, img_size, 1)
        tokens_seq = np.array([frame_to_tokens(frame, grid_size=(2,2)) for frame in gray_video])
        return tokens_seq  # shape: (rollout_steps, 4, token_dim)

    sequences = [rollout_fn(rng_i) for rng_i in rollout_rngs]
    sequences = np.stack(sequences, axis=0)  # shape: (num_rollouts, rollout_steps, 4, token_dim)
    return sequences


# =============================================
#  VisionGPT Model (Adapted from nanoGPT)
# =============================================

@dataclass
class GPTConfig:
    block_size: int = 1024        # maximum sequence length (rollout_steps * num_tokens)
    token_dim: int = 1024         # dimension of each token; for 64x64 image with 2x2 grid: (32*32*1)=1024
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 4             # number of attention heads (must divide n_embd)
    n_embd: int = 128           # transformer embedding dimension
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.c_attn = nn.Dense(config.n_embd * 3)  # combined Q, K, V
        self.c_proj = nn.Dense(config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # x: (B, T, n_embd)
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3*n_embd)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)  # (B, n_head, T, head_size)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        # Build block-causal mask: tokens within the same frame (4 tokens) can attend fully.
        t_idx = jnp.arange(T)
        frame_idx = t_idx // 4  # each frame contributes 4 tokens
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32)
        mask = mask.reshape(1, 1, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(self.head_size))
        att = jnp.where(mask == 1.0, att, float('-inf'))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v  # (B, n_head, T, head_size)
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
        # Input projection: maps each token vector (token_dim) to transformer embedding space.
        self.token_proj = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)  # positional embeddings
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        # Output head: project back to token_dim (each entry becomes a logit for binary classification).
        self.head = nn.Dense(config.token_dim)

    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Expects tokens of shape (B, L, token_dim), where L = (# frames * 4).
        """
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
        logits = self.head(x)  # shape: (B, L, token_dim)
        return logits, None

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        def get_optimizer(decay):
            return optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1],
                               weight_decay=decay)
        
        def partition_fn(path, x):
            if path[-1] in ('bias', 'scale', 'embedding'):
                return 'no_decay'
            elif path[-1] == 'kernel':
                return 'decay'
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        partition_optimizers = {
            'decay': get_optimizer(weight_decay),
            'no_decay': get_optimizer(0.0)
        }
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)
        return tx

    def create_state(
        self, learning_rate, weight_decay, beta1, beta2, 
        decay_lr=None, warmup_iters=None, lr_decay_iters=None, min_lr=None,
        params=None,
        **kwargs
    ):
        if params is None:
            variables = self.init(jax.random.PRNGKey(0), jnp.ones((1, 1, self.config.token_dim)), train=False)
            params = variables['params']
        params = freeze(params)
        
        if decay_lr:
            assert warmup_iters is not None and lr_decay_iters is not None and min_lr is not None
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=learning_rate,
                warmup_steps=warmup_iters, decay_steps=lr_decay_iters,
                end_value=min_lr,
            )
        else:
            lr_schedule = learning_rate

        tx = self.configure_optimizers(
            params, weight_decay=weight_decay, learning_rate=lr_schedule,
            betas=(beta1, beta2))
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)


# =============================================
#  Evaluation Function for VisionGPT
# =============================================

def evaluate_rollout_vgpt(model, state, substrate, rng, rollout_steps, img_size):
    """
    Evaluate VisionGPT by:
      - Generating a ground truth simulation via rollout_simulation.
      - Tokenizing the ground truth simulation.
      - Using the first frame as prompt and auto-regressively generating the rest.
      - Saving the ground truth and generated frames as images and generating a side-by-side video.
    """
    # --- Ground truth simulation ---
    flat_params = jnp.full(substrate.default_params(jax.random.PRNGKey(0)).shape, 6152)
    result = rollout_simulation(
        rng, params=flat_params, substrate=substrate, fm=None,
        rollout_steps=rollout_steps, time_sampling='video',
        img_size=img_size, return_state=False
    )
    video = np.array(result['rgb'])  # shape: (rollout_steps, img_size, img_size, 3)
    gray_video = video[..., :1]        # shape: (rollout_steps, img_size, img_size, 1)
    ground_truth_tokens = np.array([frame_to_tokens(frame, grid_size=(2,2))
                                      for frame in gray_video])
    # ground_truth_tokens: (rollout_steps, 4, token_dim)

    # --- Model Generation ---
    prompt = ground_truth_tokens[0:1]  # Use the first frame as prompt, shape: (1, 4, token_dim)
    gen_seq = [prompt[0]]  # initialize with the first frame (shape: (4, token_dim))
    num_frames = 1
    while num_frames < rollout_steps:
        print(f"Generating frame {num_frames} of {rollout_steps}")
        inp = np.stack(gen_seq, axis=0)   # shape: (num_frames, 4, token_dim)
        inp = inp.reshape(1, -1, model.config.token_dim)
        logits, _ = model.apply({'params': state.params}, inp, train=False)
        logits = logits.reshape(num_frames, 4, model.config.token_dim)
        last_frame_logits = logits[-1]
        next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
        gen_seq.append(np.array(next_frame_tokens))
        num_frames += 1
    generated_tokens = np.stack(gen_seq, axis=0)  # shape: (rollout_steps, 4, token_dim)

    # --- Save Images and Generate Video ---
    gt_folder = "vgpt_eval_groundtruth"
    gen_folder = "vgpt_eval_generated"
    side_folder = "vgpt_eval_sidebyside"
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(gen_folder, exist_ok=True)
    os.makedirs(side_folder, exist_ok=True)
    for i in range(rollout_steps):
        # Convert 1-channel frames to 3 channels for saving.
        gt_frame = np.repeat(tokens_to_frame(ground_truth_tokens[i], img_size=img_size, grid_size=(2,2)), 3, axis=-1)
        gen_frame = np.repeat(tokens_to_frame(generated_tokens[i], img_size=img_size, grid_size=(2,2)), 3, axis=-1)
        imageio.imwrite(os.path.join(gt_folder, f"frame_{i:04d}.png"),
                        (gt_frame * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(gen_folder, f"frame_{i:04d}.png"),
                        (gen_frame * 255).astype(np.uint8))
        canvas = np.concatenate([gt_frame, gen_frame], axis=1)  # side-by-side
        imageio.imwrite(os.path.join(side_folder, f"frame_{i:04d}.png"),
                        (canvas * 255).astype(np.uint8))
    video_path = "vgpt_eval_sidebyside.mp4"
    frame_files = sorted([os.path.join(side_folder, f) for f in os.listdir(side_folder)
                           if f.endswith(".png")])
    with imageio.get_writer(video_path, fps=20) as writer:
        for frame_path in frame_files:
            img = imageio.imread(frame_path)
            writer.append_data(img)
    print(f"[Eval] Side-by-side video saved to {video_path}")


# =============================================
#  Training Pipeline for VisionGPT
# =============================================

def train_visiongpt():
    # Hyperparameters and training settings
    rng = jax.random.PRNGKey(42)
    total_steps = 10000        # adjust as needed
    batch_size = 32
    rollout_steps = 256        # number of frames per rollout
    img_size = 64
    eval_every = 1

    # For a 64x64 image split into 2x2 patches, each token has dimension (32*32*1)=1024.
    token_dim = (img_size // 2) * (img_size // 2) * 1

    # GPT configuration.
    # For next-frame prediction: use frames 0...N-2 as inputs and predict frames 1...N-1.
    # Hence, block_size = (rollout_steps - 1) * 4 tokens.
    gpt_config = GPTConfig(
        block_size=(rollout_steps - 1) * 4,
        token_dim=token_dim,
        n_layer=6,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = GPT(gpt_config)
    state = model.create_state(
        learning_rate=1e-3, weight_decay=1e-2, beta1=0.9, beta2=0.95,
        params=None
    )

    # Create substrate.
    substrate = substrates.create_substrate("gol")
    substrate = substrates.FlattenSubstrateParameters(substrate)

    @jax.jit
    def train_step(state, tokens_batch, dropout_rng):
        def loss_fn(params):
            # tokens_batch: (B, num_frames, 4, token_dim)
            B, num_frames, num_tokens, token_dim = tokens_batch.shape
            inputs = tokens_batch[:, :num_frames - 1, :, :]   # frames 0 to N-2
            targets = tokens_batch[:, 1:, :, :]              # frames 1 to N-1
            inputs = inputs.reshape(B, -1, token_dim)   # shape: (B, (num_frames-1)*4, token_dim)
            targets = targets.reshape(B, -1, token_dim)   # shape: (B, (num_frames-1)*4, token_dim)
            logits, _ = model.apply({'params': params}, inputs, train=True, rngs={'dropout': dropout_rng})
            loss = optax.sigmoid_binary_cross_entropy(logits, targets).mean()
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    train_losses = []
    for step in range(1, total_steps + 1):
        rng, rollout_rng = jax.random.split(rng)
        dataset = generate_token_dataset(rollout_rng, substrate,
                                         num_rollouts=32, rollout_steps=rollout_steps, img_size=img_size)
        indices = np.random.choice(dataset.shape[0], batch_size, replace=False)
        tokens_batch = dataset[indices]  # shape: (batch_size, rollout_steps, 4, token_dim)
        
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, tokens_batch, dropout_rng)
        train_losses.append(float(loss))
        print(f"[Step {step}] Loss: {loss:.6f}")
        
        if step % eval_every == 0:
            evaluate_rollout_vgpt(model, state, substrate, rng, 64, img_size)
    
    # Plot training loss.
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss", alpha=0.8)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VisionGPT Training Loss")
    plt.savefig("visiongpt_loss_curve.png")
    plt.close()
    print("[Done] Training loss curve saved to visiongpt_loss_curve.png")
    
    # Save final model parameters.
    with open("visiongpt_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("[Done] Final model parameters saved to visiongpt_params.pkl")


if __name__ == "__main__":
    train_visiongpt()
