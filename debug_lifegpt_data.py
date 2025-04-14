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
import csv
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

# ----------------- Helper Functions -----------------
def frame_to_tokens(frame: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a simulation frame (H x W x D) into tokens by rearranging it into patches.
    For grid_size=(p, p), rearrange as: "(H ph) (W pw) D -> (H W) (ph pw D)"
    The frame is assumed binary.
    """
    tokens = rearrange(frame, "(H ph) (W pw) D -> (H W) (ph pw D)",
                        H=grid_size[0], W=grid_size[1])
    tokens = (tokens > 0.5).astype(np.float32)
    return tokens

def tokens_to_frame(tokens: np.ndarray, img_size: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a sequence of tokens back into an image.
    """
    ph = img_size // grid_size[0]
    pw = img_size // grid_size[1]
    frame = rearrange(tokens, "(H W) (ph pw D) -> (H ph) (W pw) D",
                      H=grid_size[0], W=grid_size[1], ph=ph, pw=pw, D=1)
    return frame

def load_dataset(csv_file: str, img_size: int, num_frames: int, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Load simulation sequences from a CSV file where each row has 'num_frames' cells.
    Each cell is a string representing a state for a 32×32 grid.
    Here, instead of expecting 1024 space-separated numbers, we assume that each cell is a
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
                # Try to parse the cell as space-separated numbers.
                arr = np.fromstring(cell_str, sep=" ")
                # If only one value is returned, assume the cell is a contiguous string of 1024 characters.
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
    
    # Tokenize each frame.
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

# ----------------- Evaluation Function -----------------
def evaluate_model(model, state, val_dataset: np.ndarray, img_size: int, grid_size: Tuple[int, int], step: int):
    """
    Evaluate the GPT model on the entire validation dataset.
    Computes average accuracy over randomly sampled validation sequences,
    and then randomly selects one validation sample (using a time-based seed)
    to re-run autoregressive generation for visualization.
    """
    total_accuracy = 0.0
    num_val = val_dataset.shape[0]
    
    # Use a new RNG for accuracy sampling.
    rng_acc = np.random.default_rng() 

    # Loop over a fixed number of random validation sequences for accuracy computation.
    num_eval_samples = 10  
    for i in range(num_eval_samples):
        idx = rng_acc.integers(0, num_val)
        print(f"[Eval] Processing validation sequence {idx} of {num_val}")
        val_sequence = val_dataset[idx]  # shape: (num_frames, num_tokens, token_dim)
        num_frames, _, _ = val_sequence.shape
        prompt = val_sequence[0:1]
        pred_seq = [prompt[0]]
        num_pred = 1
        while num_pred < num_frames:
            inp = np.stack(pred_seq, axis=0)
            inp = inp.reshape(1, -1, model.config.token_dim)
            logits, _ = model.apply({'params': state.params}, inp, train=False)
            logits = logits.reshape(num_pred, -1, model.config.token_dim)
            last_frame_logits = logits[-1]
            next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
            pred_seq.append(np.array(next_frame_tokens))
            num_pred += 1

        generated_sequence = np.stack(pred_seq, axis=0)
        accuracy = np.mean(generated_sequence == val_sequence) * 100.0
        total_accuracy += accuracy

    avg_accuracy = total_accuracy / num_eval_samples
    print(f"[Eval] Step {step} Overall Evaluation Accuracy: {avg_accuracy:.2f}%")
    wandb.log({"eval_accuracy": avg_accuracy, "eval_step": step})

    # --- Visualization ---
    # Use an RNG seeded with the current time for a non-deterministic sample choice.
    vis_rng = np.random.default_rng()
    random_idx = vis_rng.integers(0, num_val)
    vis_sequence = val_dataset[random_idx]
    num_frames, _, _ = vis_sequence.shape
    prompt = vis_sequence[0:1]
    pred_seq = [prompt[0]]
    num_pred = 1
    while num_pred < num_frames:
        inp = np.stack(pred_seq, axis=0)
        inp = inp.reshape(1, -1, model.config.token_dim)
        logits, _ = model.apply({'params': state.params}, inp, train=False)
        logits = logits.reshape(num_pred, -1, model.config.token_dim)
        last_frame_logits = logits[-1]
        next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
        pred_seq.append(np.array(next_frame_tokens))
        num_pred += 1
    generated_vis = np.stack(pred_seq, axis=0)

    gt_folder = "eval_groundtruth"
    gen_folder = "eval_generated"
    side_folder = "eval_sidebyside"
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(gen_folder, exist_ok=True)
    os.makedirs(side_folder, exist_ok=True)
    for i in range(num_frames):
        gt_frame = np.repeat(tokens_to_frame(vis_sequence[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
        gen_frame = np.repeat(tokens_to_frame(generated_vis[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
        imageio.imwrite(os.path.join(gt_folder, f"frame_{i:04d}.png"), (gt_frame * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(gen_folder, f"frame_{i:04d}.png"), (gen_frame * 255).astype(np.uint8))
        canvas = np.concatenate([gt_frame, gen_frame], axis=1)
        imageio.imwrite(os.path.join(side_folder, f"frame_{i:04d}.png"), (canvas * 255).astype(np.uint8))
    video_path = "eval_video.mp4"
    frame_files = sorted([os.path.join(side_folder, f) for f in os.listdir(side_folder) if f.endswith(".png")])
    with imageio.get_writer(video_path, fps=20) as writer:
        for frame_path in frame_files:
            img = imageio.imread(frame_path)
            writer.append_data(img)
    print(f"[Eval] Side-by-side video saved to {video_path}")
    wandb.log({"eval_video": wandb.Video(video_path, fps=20, format="mp4"), "eval_step": step})

# ----------------- Model Definition -----------------
@dataclass
class GPTConfig:
    img_size: int           # Image size of each frame.
    block_size: int         # = (num_frames - 1) * (num_tokens)
    token_dim: int          # Dimension of each token (e.g., (img_size // patches_per_dim)**2)
    num_tokens: int         # Number of tokens per frame (e.g., patches_per_dim**2)
    n_layer: int = 12       # Number of transformer blocks.
    n_head: int = 8         # Number of attention heads.
    n_embd: int = 256       # Transformer embedding dimension.
    dropout: float = 0.1    # Dropout probability.

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

# ----------------- Training Function -----------------
def train_gpt(batch_size: int = 32, train_steps: int = 3000,
              eval_every: int = 200, patches_per_dim: int = 2,
              img_size: int = 32, num_frames: int = 10, 
              train_csv: str = "conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
              val_csv: str = "your_validation_file.csv"):
    """
    Training routine using CSV file data.
    Assumes each simulation sequence in the CSV has num_frames frames of size (img_size x img_size).
    """
    wandb.init(project="gol_world_model",
               name=f"gpt_patches{patches_per_dim}_batch{batch_size}",
               config={"patches_per_dim": patches_per_dim,
                       "train_steps": train_steps,
                       "batch_size": batch_size,
                       "img_size": img_size,
                       "num_frames": num_frames})
    
    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2  # dimension of each token
    num_tokens = patches_per_dim ** 2                # number of tokens per frame
    block_size = (num_frames - 1) * num_tokens        # for next-frame prediction
    
    print(f"Training with patches_per_dim = {patches_per_dim}, grid_size = {grid_size}, token_dim = {token_dim}, "
          f"num_frames = {num_frames}, block_size = {block_size}, img_size = {img_size}")
    
    gpt_config = GPTConfig(
        img_size=img_size,
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=num_tokens,
        n_layer=12,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )
    
    model = GPT(gpt_config)
    state = model.create_state(
        learning_rate=1e-3, weight_decay=1e-2, beta1=0.9, beta2=0.95, params=None
    )
    
    train_dataset = load_dataset(train_csv, img_size, num_frames, grid_size)
    val_dataset = load_dataset(val_csv, img_size, num_frames, grid_size) if os.path.exists(val_csv) else None
    num_train = train_dataset.shape[0]
    print(f"Loaded {num_train} training sequences.")
    
    @jax.jit
    def train_step(state, tokens_batch, dropout_rng):
        def loss_fn(params):
            B, num_frames, num_tokens, token_dim = tokens_batch.shape
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
    rng = jax.random.PRNGKey(int(time.time()))
    for step in range(1, train_steps + 1):
        batch_indices = np.random.choice(num_train, batch_size, replace=False)
        tokens_batch = train_dataset[batch_indices]
    
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, tokens_batch, dropout_rng)
        train_losses.append(float(loss))
        wandb.log({"step": step, "train_loss": float(loss)})
        print(f"[Step {step}] Loss: {loss:.6f}")
        
        # Evaluate on the full validation dataset
        if step % eval_every == 0 and val_dataset is not None:
            evaluate_model(model, state, val_dataset, img_size, grid_size, step)
    
    with open("gpt_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("[Done] Final model parameters saved to gpt_params.pkl")
    wandb.finish()

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=50000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (pixels) for each frame")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per simulation sequence in the CSV files")
    parser.add_argument("--train_csv", type=str, default="conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
                        help="Path to the validation CSV file")
    args = parser.parse_args()

    # Do not set a fixed global seed.
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    train_gpt(batch_size=args.batch_size, 
              train_steps=args.train_steps, 
              eval_every=args.eval_every,
              patches_per_dim=args.patches_per_dim, 
              img_size=args.img_size,
              num_frames=args.num_frames,
              train_csv=args.train_csv,
              val_csv=args.val_csv)
