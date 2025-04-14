import math
import os
import pickle
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import argparse
import wandb

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax import traverse_util
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.core.frozen_dict import freeze
from einops import rearrange

import substrates
from rollout import rollout_simulation
from util_gol import frame_to_tokens, tokens_to_frame, generate_token_dataset


# Updated GPT model configuration.
@dataclass
class GPTConfig:
    img_size: int           # Image size of each frame.
    block_size: int         # = (rollout_eff - 1) * (num_tokens)
    token_dim: int          # Dimension of each token (e.g., (img_size // patches_per_dim)**2)
    num_tokens: int         # Number of tokens per frame (e.g., patches_per_dim**2)
    n_layer: int = 4       # Number of transformer blocks.
    n_head: int = 4         # Number of attention heads (n_embd must be divisible by n_head).
    n_embd: int = 128       # Transformer embedding dimension.
    dropout: float = 0.0    # Dropout probability.


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.c_attn = nn.Dense(config.n_embd * 3)  # combined Q, K, V.
        self.c_proj = nn.Dense(config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # x: (B, T, n_embd); T is the total number of tokens in the flattened input.
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3*n_embd)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        # --- Block-causal mask ---
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
        # Input projection: map each token vector (of dimension token_dim) to model embedding.
        self.token_proj = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)  # Positional embeddings.
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        # Output head: project from n_embd back to token_dim.
        self.head = nn.Dense(config.token_dim)

    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Expects tokens of shape (B, L, token_dim), where L = (# frames * num_tokens).
        In our next-frame prediction, the input (after flattening) represents frames 0 ... N-2,
        and the target is frames 1 ... N.
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
        logits = self.head(x)
        return logits, None

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                weight_decay=decay
            )
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

    def create_state(self, learning_rate, weight_decay, beta1, beta2,
                     decay_lr=None, warmup_iters=None, lr_decay_iters=None, min_lr=None,
                     params=None, **kwargs):
        if params is None:
            # Initialize with a dummy input of shape (1, 1, token_dim) (a single frame).
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
        tx = self.configure_optimizers(params, weight_decay=weight_decay, learning_rate=lr_schedule, betas=(beta1, beta2))
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def generate_rollout(self, key, params, initial_tokens: jnp.ndarray, num_new_frames: int) -> jnp.ndarray:
        """
        Generate additional frames autoregressively in a frame-by-frame manner.
        Given an initial rollout (shape: (num_frames, num_tokens, token_dim)),
        the model receives all previous frames and predicts the tokens of the next frame.
        """
        seq = initial_tokens.copy()  # shape: (num_frames, num_tokens, token_dim)
        num_frames = seq.shape[0]
        for _ in range(num_new_frames):
            inp = seq.reshape(1, -1, self.config.token_dim)
            logits, _ = self.apply({'params': params}, inp, train=False)
            logits = logits.reshape(num_frames, -1, self.config.token_dim)
            last_frame_logits = logits[-1]
            next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
            next_frame_tokens = np.array(next_frame_tokens)
            seq = np.concatenate([seq, next_frame_tokens[None, :]], axis=0)
            num_frames += 1
        return seq


def evaluate_training_performance(model, state, tokens_batch, img_size, grid_size: Tuple[int, int], step: int):
    """
    Evaluate the GPT model on the training data.

    Uses the first sequence in tokens_batch as ground truth. The model is given the first frame
    as a prompt and then autoregressively generates tokens, which are compared to the ground truth.
    Visualizations are saved as images and a side-by-side video is created.
    """
    # Use the first training sequence as ground truth.
    ground_truth_tokens = tokens_batch[0]   # shape: (rollout_eff, num_tokens, token_dim)
    rollout_eff = ground_truth_tokens.shape[0]
    
    # --- Model Generation on Training Data ---
    prompt = ground_truth_tokens[0:1]  # use the first frame as prompt.
    gen_seq = [prompt[0]]              # initialize with the first frame.
    num_frames = 1
    while num_frames < rollout_eff:
        inp = np.stack(gen_seq, axis=0)  # shape: (num_frames, num_tokens, token_dim)
        inp = inp.reshape(1, -1, model.config.token_dim)
        logits, _ = model.apply({'params': state.params}, inp, train=False)
        logits = logits.reshape(num_frames, -1, model.config.token_dim)
        last_frame_logits = logits[-1]
        next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
        gen_seq.append(np.array(next_frame_tokens))
        num_frames += 1
    generated_tokens = np.stack(gen_seq, axis=0)
    
    # --- Compute Accuracy ---
    accuracy = np.mean(generated_tokens == ground_truth_tokens) * 100.0
    print(f"[Train Eval] Step {step} Accuracy on Training Data: {accuracy:.2f}%")
    wandb.log({"train_eval_accuracy": accuracy, "eval_step": step})
    
    # --- Save Visualization Images ---
    gt_folder = "vgpt_train_eval_groundtruth"
    gen_folder = "vgpt_train_eval_generated"
    side_folder = "vgpt_train_eval_sidebyside"
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(gen_folder, exist_ok=True)
    os.makedirs(side_folder, exist_ok=True)
    for i in range(rollout_eff):
        gt_frame = np.repeat(tokens_to_frame(ground_truth_tokens[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
        gen_frame = np.repeat(tokens_to_frame(generated_tokens[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
        imageio.imwrite(os.path.join(gt_folder, f"frame_{i:04d}.png"), (gt_frame * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(gen_folder, f"frame_{i:04d}.png"), (gen_frame * 255).astype(np.uint8))
        canvas = np.concatenate([gt_frame, gen_frame], axis=1)
        imageio.imwrite(os.path.join(side_folder, f"frame_{i:04d}.png"), (canvas * 255).astype(np.uint8))
    video_path = "vgpt_train_eval.mp4"
    frame_files = sorted([os.path.join(side_folder, f) for f in os.listdir(side_folder) if f.endswith(".png")])
    with imageio.get_writer(video_path, fps=20) as writer:
        for frame_path in frame_files:
            img = imageio.imread(frame_path)
            writer.append_data(img)
    print(f"[Train Eval] Side-by-side video saved to {video_path}")
    wandb.log({
        "train_eval_video": wandb.Video(video_path, fps=20, format="mp4"),
        "train_eval_step": step
    })


def train_gpt(batch_size: int = 32, rollout_steps: int = 256, train_steps: int = 3000,
              eval_every: int = 200, patches_per_dim: int = 2, t_skip: int = 0, seed: int = 42,
              img_size: int = 4):
    """
    Train the GPT-style vision model using a single, fixed batch of data for both training and evaluation.
    1. Next-frame prediction: each token in a frame is used to predict the corresponding token in the next frame.
    2. Autoregressive conditioning: the input is constructed by flattening past frames.
    """
    wandb.init(project="gol_world_model",
               name=f"gpt_patches{patches_per_dim}_t{t_skip}_batch{batch_size}_seed{seed}",
               config={"patches_per_dim": patches_per_dim,
                       "t_skip": t_skip,
                       "rollout_steps": rollout_steps,
                       "train_steps": train_steps,
                       "batch_size": batch_size,
                       "seed": seed,
                       "img_size": img_size})
    
    rng = jax.random.PRNGKey(seed)
    
    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2        # Dimension of each token.
    num_tokens = patches_per_dim ** 2                      # Number of tokens per frame.
    rollout_eff = rollout_steps // (t_skip + 1)
    block_size = (rollout_eff - 1) * num_tokens            # For the input sequence.
    print(f"Training with patches_per_dim = {patches_per_dim}, grid_size = {grid_size}, token_dim = {token_dim}, "
          f"t_skip = {t_skip}, effective rollout_steps = {rollout_eff}, block_size = {block_size}, img_size = {img_size}")
    
    # Create the GPT configuration.
    gpt_config = GPTConfig(
        img_size=img_size,
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=num_tokens,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0
    )
    
    model = GPT(gpt_config)
    state = model.create_state(
        learning_rate=1e-3, weight_decay=1e-2, beta1=0.9, beta2=0.95, params=None
    )
    
    substrate = substrates.create_substrate("gol")
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    # --- Generate the fixed dataset once for training and evaluation ---
    rng, rollout_rng = jax.random.split(rng)
    dataset = generate_token_dataset(rollout_rng, substrate,
                                     num_rollouts=32, rollout_steps=rollout_steps,
                                     img_size=img_size, grid_size=grid_size, t_skip=t_skip)
    indices = np.random.choice(dataset.shape[0], batch_size, replace=False)
    tokens_batch = dataset[indices]  # Shape: (B, rollout_eff, num_tokens, token_dim)
    
    @jax.jit
    def train_step(state, tokens_batch, dropout_rng):
        def loss_fn(params):
            B, num_frames, num_tokens, token_dim = tokens_batch.shape
            # --- Next-Frame Prediction Objective ---
            inputs = tokens_batch[:, :num_frames - 1, :, :]
            targets = tokens_batch[:, 1:, :, :]
            inputs = inputs.reshape(B, -1, token_dim)
            targets = targets.reshape(B, -1, token_dim)
            logits, _ = model.apply({'params': params}, inputs, train=True, rngs={'dropout': dropout_rng})
            loss = optax.sigmoid_binary_cross_entropy(logits, targets).mean()
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    train_losses = []
    for step in range(1, train_steps + 1):
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, tokens_batch, dropout_rng)
        train_losses.append(float(loss))
        wandb.log({"step": step, "train_loss": float(loss)})
        print(f"[Step {step}] Loss: {loss:.6f}")
        
        # Evaluate on the training data every eval_every steps.
        if step % eval_every == 0:
            evaluate_training_performance(model, state, tokens_batch, img_size, grid_size, step)
    
    with open("gpt_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("[Done] Final model parameters saved to gpt_params.pkl")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--t_skip", type=int, default=0,
                        help="Skip frequency: 0 uses all frames; 1 uses frames 0,2,4,..., etc.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--img_size", type=int, default=16,
                        help="Image size (pixels) for each frame")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_gpt(batch_size=args.batch_size, 
              rollout_steps=args.rollout_steps, 
              train_steps=args.train_steps, 
              eval_every=args.eval_every,
              patches_per_dim=args.patches_per_dim, 
              t_skip=args.t_skip,
              seed=args.seed,
              img_size=args.img_size)
