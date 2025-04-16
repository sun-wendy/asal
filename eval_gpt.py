import math
import os
import io
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.core.frozen_dict import freeze

from util_gol import frame_to_tokens, tokens_to_frame, load_dataset_from_csv


@dataclass
class GPTConfig:
    img_size: int
    block_size: int         # (num_eff_frames - 1) * (num_tokens)
    token_dim: int          # Dimension of each token (e.g., (img_size // patches_per_dim)**2)
    num_tokens: int         # Number of tokens per frame (e.g., patches_per_dim**2)
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


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


def evaluate(model, state, val_dataset: np.ndarray, img_size: int, grid_size: Tuple[int, int], step: int, t_skip: int = 0, vis_all: bool = False):
    """
    Evaluate the GPT model on the validation dataset
    """
    total_accuracy = 0.0
    num_val = val_dataset.shape[0]
    
    rng_acc = np.random.default_rng() 

    # Loop over a fixed number of random validation sequences for accuracy computation
    num_eval_samples = 1000 if not vis_all else val_dataset.shape[0]
    for i in range(num_eval_samples):
        idx = rng_acc.integers(0, num_val)
        print(f"[Eval] Processing validation sequence {idx} of {num_val}")
        val_sequence = val_dataset[idx]  # shape: (num_frames, num_tokens, token_dim)
        if t_skip > 0:
            val_sequence = val_sequence[::(t_skip + 1)]
        num_eff_frames, _, _ = val_sequence.shape
        prompt = val_sequence[0:1]
        pred_seq = [prompt[0]]
        num_pred = 1
        while num_pred < num_eff_frames:
            inp = np.stack(pred_seq, axis=0)  # Autoregressive generation
            inp = inp.reshape(1, -1, model.config.token_dim)
            logits, _ = model.apply({'params': state.params}, inp, train=False)
            logits = logits.reshape(num_pred, -1, model.config.token_dim)
            last_frame_logits = logits[-1]
            next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
            pred_seq.append(np.array(next_frame_tokens))
            num_pred += 1

        generated_sequence = np.stack(pred_seq[1:], axis=0)
        accuracy = np.mean(generated_sequence == val_sequence[1:]) * 100.0
        total_accuracy += accuracy

    avg_accuracy = total_accuracy / num_eval_samples
    print(f"[Eval] Step {step} Overall Evaluation Accuracy: {avg_accuracy:.2f}%")

    # Visualization
    if vis_all:
        for idx in range(num_eval_samples):
            vis_sequence = val_dataset[idx]
            if t_skip > 0:
                vis_sequence = vis_sequence[::(t_skip + 1)]
            num_eff_frames, _, _ = vis_sequence.shape
            prompt = vis_sequence[0:1]
            pred_seq = [prompt[0]]
            num_pred = 1
            while num_pred < num_eff_frames:
                inp = np.stack(pred_seq, axis=0)
                inp = inp.reshape(1, -1, model.config.token_dim)
                logits, _ = model.apply({'params': state.params}, inp, train=False)
                logits = logits.reshape(num_pred, -1, model.config.token_dim)
                last_frame_logits = logits[-1]
                next_frame_tokens = (last_frame_logits > 0).astype(np.float32)
                pred_seq.append(np.array(next_frame_tokens))
                num_pred += 1
            generated_vis = np.stack(pred_seq, axis=0)

            gt_folder = "eval/ground_truth"
            gen_folder = "eval/generated"
            side_folder = "eval/side_by_side"
            os.makedirs(gt_folder, exist_ok=True)
            os.makedirs(gen_folder, exist_ok=True)
            os.makedirs(side_folder, exist_ok=True)
            
            for i in range(num_eff_frames):
                gt_frame = np.repeat(tokens_to_frame(vis_sequence[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
                gen_frame = np.repeat(tokens_to_frame(generated_vis[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
                imageio.imwrite(os.path.join(gt_folder, f"frame_{i:04d}.png"), (gt_frame * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(gen_folder, f"frame_{i:04d}.png"), (gen_frame * 255).astype(np.uint8))
                # Create discrepancy panel
                dark_blue = np.array([0.0, 0.0, 139/255.0])
                yellow = np.array([1.0, 1.0, 0.0])
                discrepancy = np.ones_like(gt_frame) * dark_blue
                diff_mask = (gt_frame[..., 0] != gen_frame[..., 0])
                discrepancy[diff_mask] = yellow
                
                # Create a three-panel figure with labels.
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(gt_frame)
                axes[0].set_title("Ground Truth")
                axes[0].axis("off")
                axes[1].imshow(gen_frame)
                axes[1].set_title("Model Generation")
                axes[1].axis("off")
                axes[2].imshow(discrepancy)
                axes[2].set_title("Discrepancy")
                axes[2].axis("off")
                plt.tight_layout()
                
                # Save the composite side-by-side image to file.
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                comp_img = np.array(Image.open(buf))
                buf.close()
                plt.close(fig)
                
                imageio.imwrite(os.path.join(side_folder, f"frame_{i:04d}.png"), comp_img)
            
            frame_files = sorted([os.path.join(side_folder, f) for f in os.listdir(side_folder) if f.endswith(".png")])
            frames = [imageio.imread(frame_path) for frame_path in frame_files]
            video_path = f"eval/side_by_side_{idx}.mp4"
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame_path in frame_files:
                    img = imageio.imread(frame_path)
                    writer.append_data(img)
            print(f"[Eval] Side-by-side video saved to {video_path}")
            gif_path = f"eval/side_by_side_{idx}.gif"
            imageio.mimsave(gif_path, frames, fps=20)
            print(f"[Eval] Side-by-side GIF saved to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=50000)
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (pixels) for each frame")
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per simulation sequence in CSV files")
    parser.add_argument("--t_skip", type=int, default=0,
                        help="Skip frequency: 0 uses all frames; 1 uses frames 0, 2, 4, ..., etc.")
    parser.add_argument("--val_csv", type=str, default="conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
                        help="Path to validation CSV file")
    parser.add_argument("--pattern_csv", type=str, default="patterns/conway_test_states_32by32_20250414_075008.csv",
                        help="Path to pattern CSV file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    grid_size = (args.patches_per_dim, args.patches_per_dim)
    token_dim = (args.img_size // args.patches_per_dim) ** 2  # token dimension
    num_tokens = args.patches_per_dim ** 2                    # number of tokens per frame
    num_eff_frames = math.ceil(args.num_frames / (args.t_skip + 1))
    block_size = (num_eff_frames - 1) * num_tokens           # as used in training

    # Instantiate GPT configuration and model.
    gpt_config = GPTConfig(
        img_size=args.img_size,
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=num_tokens,
        n_layer=12,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )
    model = GPT(gpt_config)

    print(f"Loading checkpoint parameters from {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        loaded_params = pickle.load(f)
    
    dummy_tx = optax.adam(learning_rate=0.0)
    state = train_state.TrainState.create(apply_fn=model.apply, params=freeze(loaded_params), tx=dummy_tx)

    val_dataset = load_dataset_from_csv(args.val_csv, args.img_size, args.num_frames, grid_size)
    print(f"Loaded validation dataset of shape: {val_dataset.shape}")
    pattern_dataset = load_dataset_from_csv(args.pattern_csv, args.img_size, 10, grid_size)
    print(f"Loaded pattern dataset of shape: {pattern_dataset.shape}")

    evaluate(model, state, val_dataset, img_size=args.img_size, grid_size=grid_size, step=0, t_skip=args.t_skip, vis_all=False)
    evaluate(model, state, pattern_dataset, img_size=args.img_size, grid_size=grid_size, step=0, t_skip=args.t_skip, vis_all=True)
