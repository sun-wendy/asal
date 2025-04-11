import math
import os
import pickle
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import random
import argparse
import wandb

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.core.frozen_dict import freeze
from einops import rearrange

import substrates
from rollout import rollout_simulation
from util_gol import frame_to_tokens, tokens_to_frame, generate_token_dataset


# Use a special value for masked tokens that is outside the normal range (0 or 1).
MASK_VALUE = -1.0


# BERT model
@dataclass
class BERTConfig:
    block_size: int         # = rollout_steps * num_tokens
    token_dim: int          # = (img_size // patches_per_dim)^2
    num_tokens: int         # = patches_per_dim^2
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class FullAttention(nn.Module):
    config: BERTConfig

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
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1,2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1,2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1,2)
        # Full bidirectional attention: use a mask of ones.
        mask = jnp.ones((T, T), dtype=jnp.float32).reshape(1, 1, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(self.head_size))
        att = jnp.where(mask == 1.0, att, float('-inf'))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.swapaxes(1,2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y


class MLP(nn.Module):
    config: BERTConfig

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
    config: BERTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = FullAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x


class BERT(nn.Module):
    config: BERTConfig

    def setup(self):
        config = self.config
        self.token_proj = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(config.token_dim)

    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Expects tokens of shape (B, L, token_dim) where L = rollout_steps * num_tokens.
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
            return optax.adamw(learning_rate=learning_rate, 
                               b1=betas[0], b2=betas[1],
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
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters,
                decay_steps=lr_decay_iters,
                end_value=min_lr
            )
        else:
            lr_schedule = learning_rate
        tx = self.configure_optimizers(params, weight_decay=weight_decay,
                                       learning_rate=lr_schedule,
                                       betas=(beta1, beta2))
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)


# Training pipeline
def train_bert(batch_size: int = 32, rollout_steps: int = 256, train_steps: int = 3000, eval_every: int = 200, patches_per_dim: int = 2, mask_rate: float = 0.15, seed: int = 42):
    """
    Train a BERT-style vision model with a masked prediction objective.
    Here we mimic actual BERT training by replacing a percentage of tokens with a special mask token.
    In this example, when a token is masked, we replace its value with a constant MASK_VALUE (set to -1.0)
    and only compute the BCE loss on the masked positions.
    """
    wandb.init(project="gol_world_model", name=f"bert_patches{patches_per_dim}_mask{mask_rate}_batch{batch_size}_seed{seed}", 
               config={"patches_per_dim": patches_per_dim,
                       "mask_rate": mask_rate,
                       "rollout_steps": rollout_steps,
                       "train_steps": train_steps,
                       "batch_size": batch_size,
                       "seed": seed})
    
    rng = jax.random.PRNGKey(seed)
    img_size = 64

    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2
    num_tokens = patches_per_dim ** 2
    block_size = rollout_steps * num_tokens  # full sequence length
    print(f"Using grid_size = {grid_size}, token_dim = {token_dim}, block_size = {block_size}")

    config = BERTConfig(
        block_size=block_size,
        token_dim=token_dim,
        num_tokens=num_tokens,
        n_layer=6,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = BERT(config)
    state = model.create_state(
        learning_rate=1e-3, weight_decay=1e-2, beta1=0.9, beta2=0.95, params=None
    )

    substrate = substrates.create_substrate("gol")
    substrate = substrates.FlattenSubstrateParameters(substrate)

    @jax.jit
    def train_step(state, tokens_batch, dropout_rng):
        """
        tokens_batch: shape (B, rollout_steps, num_tokens, token_dim)
        We flatten it to (B, L, token_dim), with L = rollout_steps * num_tokens.
        For BERT-style training, we randomly select a subset of positions to mask.
        At masked positions, we replace the original token with the MASK_VALUE.
        The loss (BCE) is computed only on the masked positions.
        """
        def loss_fn(params):
            B, R, N, D = tokens_batch.shape
            x_orig = tokens_batch.reshape(B, -1, D)  # shape: (B, L, token_dim)
            mask = jax.random.bernoulli(dropout_rng, p=mask_rate, shape=x_orig.shape[:-1])
            mask = mask[..., None].astype(jnp.float32)  # shape: (B, L, 1)
            # Replace masked positions with MASK_VALUE
            x_masked = jnp.where(mask == 1.0, MASK_VALUE, x_orig)
            logits, _ = model.apply({'params': params}, x_masked, train=True, rngs={'dropout': dropout_rng})
            loss_all = optax.sigmoid_binary_cross_entropy(logits, x_orig)
            loss_masked = (loss_all * mask).mean()
            return loss_masked
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    train_losses = []
    for step in range(1, train_steps + 1):
        rng, rollout_rng = jax.random.split(rng)
        dataset = generate_token_dataset(rollout_rng, substrate,
                                         num_rollouts=32, rollout_steps=rollout_steps,
                                         img_size=img_size, grid_size=grid_size)
        indices = np.random.choice(dataset.shape[0], batch_size, replace=False)
        tokens_batch = dataset[indices]  # shape: (B, rollout_steps, num_tokens, token_dim)
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, tokens_batch, dropout_rng)
        train_losses.append(float(loss))
        wandb.log({"step": step, "train_loss": float(loss)})
        print(f"[Step {step}] Loss: {loss:.6f}")
        
        if step % eval_every == 0:
            # --- Evaluation: reconstruct masked tokens in a ground truth simulation ---
            flat_params = jnp.full(substrate.default_params(jax.random.PRNGKey(0)).shape, 6152)
            result = rollout_simulation(
                rng, params=flat_params, substrate=substrate, fm=None,
                rollout_steps=64, time_sampling='video',
                img_size=img_size, return_state=False
            )
            video = np.array(result['rgb'])
            gray_video = video[..., :1]
            tokens_full = np.array([frame_to_tokens(frame, grid_size) for frame in gray_video])
            x_orig = tokens_full.reshape(1, -1, token_dim)
            mask = jax.random.bernoulli(dropout_rng, p=mask_rate, shape=x_orig.shape[:-1])
            mask = mask[..., None].astype(jnp.float32)
            x_masked = jnp.where(mask == 1.0, MASK_VALUE, x_orig)
            logits, _ = model.apply({'params': state.params}, x_masked, train=False)
            pred = (logits > 0).astype(jnp.float32)
            recon = jnp.where(mask == 1.0, pred, x_orig)
            recon_seq = np.array(recon).reshape(64, num_tokens, token_dim)
            
            gt_folder = "bert_eval_groundtruth"
            rec_folder = "bert_eval_reconstruction"
            side_folder = "bert_eval_sidebyside"
            os.makedirs(gt_folder, exist_ok=True)
            os.makedirs(rec_folder, exist_ok=True)
            os.makedirs(side_folder, exist_ok=True)
            for i in range(64):
                gt_frame = np.repeat(tokens_to_frame(tokens_full[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
                rec_frame = np.repeat(tokens_to_frame(recon_seq[i], img_size=img_size, grid_size=grid_size), 3, axis=-1)
                imageio.imwrite(os.path.join(gt_folder, f"frame_{i:04d}.png"), (gt_frame * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(rec_folder, f"frame_{i:04d}.png"), (rec_frame * 255).astype(np.uint8))
                canvas = np.concatenate([gt_frame, rec_frame], axis=1)
                imageio.imwrite(os.path.join(side_folder, f"frame_{i:04d}.png"), (canvas * 255).astype(np.uint8))
            video_path = "bert_eval.mp4"
            frame_files = sorted([os.path.join(side_folder, f) for f in os.listdir(side_folder) if f.endswith(".png")])
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame_path in frame_files:
                    img = imageio.imread(frame_path)
                    writer.append_data(img)
            print(f"[Eval] Side-by-side video saved to {video_path}")
            wandb.log({
                "eval_video": wandb.Video(video_path, fps=20, format="mp4"),
                "eval_step": step
            })
    
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss", alpha=0.8)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("BERT-Style Vision Model Training Loss")
    plt.savefig("bert_loss_curve.png")
    plt.close()
    print("[Done] Training loss curve saved to bert_loss_curve.png")
    
    with open(f"bert_params_patches{patches_per_dim}.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print(f"[Done] Final model parameters saved to bert_params_patches{patches_per_dim}.pkl")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--mask_rate", type=float, default=0.5,
                        help="Mask rate for BERT objective")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_bert(batch_size=args.batch_size, 
               rollout_steps=args.rollout_steps, 
               train_steps=args.train_steps, 
               eval_every=args.eval_every,
               patches_per_dim=args.patches_per_dim,
               mask_rate=args.mask_rate,
               seed=args.seed)
