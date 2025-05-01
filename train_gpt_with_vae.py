import math
import os
import io
import pickle
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import wandb
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.core.frozen_dict import freeze as freeze_dict

from util_gol import frame_to_tokens, tokens_to_frame, load_dataset_from_csv


@dataclass
class GPTConfig:
    img_size: int
    block_size: int         # (num_eff_frames) * (num_tokens)
    token_dim: int          # Dimension of each token
    num_tokens: int         # Number of tokens per frame
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        c = self.config
        assert c.n_embd % c.n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = c.n_embd // c.n_head
        self.n_head = c.n_head
        self.c_attn = nn.Dense(c.n_embd * 3)
        self.c_proj = nn.Dense(c.n_embd)
        self.attn_dropout = nn.Dropout(c.dropout)
        self.resid_dropout = nn.Dropout(c.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        t_idx = jnp.arange(T)
        frame_idx = t_idx // self.config.num_tokens
        mask = (frame_idx[None, :] <= frame_idx[:, None]).astype(jnp.float32).reshape(1, 1, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(self.head_size))
        att = jnp.where(mask, att, float('-inf'))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y), deterministic=not train)


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        c = self.config
        self.c_fc = nn.Dense(4 * c.n_embd)
        self.c_proj = nn.Dense(c.n_embd)
        self.dropout = nn.Dropout(c.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        return self.dropout(x, deterministic=not train)


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), train=train)
        return x + self.mlp(self.ln_2(x), train=train)


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        c = self.config
        self.token_proj = nn.Dense(c.n_embd)
        self.wpe = nn.Embed(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.h = [Block(c) for _ in range(c.n_layer)]
        # VAE bottleneck layers
        self.vae_mu = nn.Dense(c.n_embd)
        self.vae_logvar = nn.Dense(c.n_embd)
        self.vae_proj = nn.Dense(c.n_embd)
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(c.token_dim)

    def __call__(self, tokens: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B, T, d = tokens.shape
        assert d == self.config.token_dim
        x = self.token_proj(tokens)
        pos = jnp.arange(T)[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)

        # first half blocks
        mid = len(self.h) // 2
        for block in self.h[:mid]:
            x = block(x, train=train)

        # VAE bottleneck: compute latent
        mu = self.vae_mu(x)  # Dense applied per (B,T,D) → (B,T,D)
        logvar = self.vae_logvar(x)
        logvar = jnp.clip(logvar, a_min=None, a_max=10.0)
        eps = jax.random.normal(self.make_rng('vae'), mu.shape)
        z = mu + jnp.exp(0.5 * logvar) * eps    # (B,T,D)
        proj_z = self.vae_proj(z)
        x_pre  = x
        x = proj_z + self.wpe(pos)  # + x

        # second half blocks
        for block in self.h[mid:]:
            x = block(x, train=train)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, (mu, logvar, x_pre, proj_z)

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        def get_opt(decay):
            return optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1], weight_decay=decay)
        def part_fn(path, _):
            return 'no_decay' if path[-1] in ('bias','scale','embedding') else 'decay'
        tx = optax.multi_transform(
            {'decay': get_opt(weight_decay), 'no_decay': get_opt(0.0)},
            freeze_dict(path_aware_map(part_fn, params))
        )
        return tx  # optax.chain(optax.clip_by_global_norm(1.0), tx)

    def create_state(self, learning_rate, weight_decay, beta1, beta2, decay_lr=None, warmup_iters=None, lr_decay_iters=None, min_lr=None, params=None):
        if params is None:
            vars = self.init(jax.random.PRNGKey(0), jnp.ones((1,1,self.config.token_dim)), train=False)
            params = vars['params']
        params = freeze_dict(params)
        lr = (
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters or 0,
                decay_steps=lr_decay_iters or 1,
                end_value=min_lr or 0.0
            )
            if decay_lr else learning_rate
        )
        tx = self.configure_optimizers(params, weight_decay, lr, (beta1, beta2))
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)


def evaluate(model, state, val_dataset: np.ndarray, img_size: int, grid_size: Tuple[int, int], step: int, t_skip: int = 0):
    total_accuracy = 0.0
    num_val = val_dataset.shape[0]
    rng = jax.random.PRNGKey(int(time.time()))
    rng_acc = np.random.default_rng() 

    num_eval_samples = 50
    for i in range(num_eval_samples):
        idx = rng_acc.integers(0, num_val)
        print(f"[Eval] Processing validation sequence {idx} of {num_val}")
        val_sequence = val_dataset[idx]
        if t_skip > 0:
            val_sequence = val_sequence[::(t_skip + 1)]
        val_sequence = val_sequence[:20]
        prompt = val_sequence[0:1]
        pred_seq = [prompt[0]]
        num_pred = 1
        while num_pred < 20:
            inp = np.stack(pred_seq, axis=0).reshape(1, -1, model.config.token_dim)
            rng, vae_rng = jax.random.split(rng)
            logits, _ = model.apply(
                {'params': state.params}, inp,
                train=False, rngs={'vae': vae_rng}
            )
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
    wandb.log({"eval_accuracy": avg_accuracy, "eval_step": step})

    """
    # Visualization
    vis_rng = np.random.default_rng()
    random_idx = vis_rng.integers(0, num_val)
    vis_sequence = val_dataset[random_idx]
    if t_skip > 0:
        vis_sequence = vis_sequence[::(t_skip + 1)]
    num_eff_frames, _, _ = vis_sequence.shape
    vis_sequence = vis_sequence[:10]
    prompt = vis_sequence[0:1]
    pred_seq = [prompt[0]]
    num_pred = 1
    while num_pred < 20:
        inp = np.stack(pred_seq, axis=0).reshape(1, -1, model.config.token_dim)
        rng, vae_rng = jax.random.split(rng)
        logits, _ = model.apply(
            {'params': state.params}, inp,
            train=False, rngs={'vae': vae_rng}
        )
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
    
    for i in range(20):
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
    video_path = "eval/side_by_side.mp4"
    with imageio.get_writer(video_path, fps=20) as writer:
        for frame_path in frame_files:
            img = imageio.imread(frame_path)
            writer.append_data(img)
    print(f"[Eval] Side-by-side video saved to {video_path}")
    gif_path = "eval/side_by_side.gif"
    imageio.mimsave(gif_path, frames, fps=20)
    print(f"[Eval] Side-by-side GIF saved to {gif_path}")
    wandb.log({"eval_gif": wandb.Video(gif_path, fps=20, format="gif"), "eval_step": step})
    """


def train_gpt(batch_size: int = 32, train_steps: int = 3000, eval_every: int = 200, img_size: int = 32, patches_per_dim: int = 2,
              num_frames: int = 10, t_skip: int = 0,
              train_csv: str = "conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
              val_csv: str = "conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",):
    wandb.init(project="gol_world_model",
               name=f"gpt_vae_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}",
               config={"batch_size": batch_size,
                       "train_steps": train_steps,
                       "img_size": img_size,
                       "patches_per_dim": patches_per_dim,
                       "num_frames": num_frames,
                       "t_skip": t_skip})
    
    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2
    num_tokens = patches_per_dim ** 2
    num_eff_frames = math.ceil(num_frames / (t_skip + 1))
    block_size = num_eff_frames * num_tokens
    
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
    
    train_dataset = load_dataset_from_csv(train_csv, img_size, num_frames, grid_size)
    val_dataset = load_dataset_from_csv(val_csv, img_size, num_frames, grid_size) if os.path.exists(val_csv) else None
    num_train = train_dataset.shape[0]
    
    @jax.jit
    def train_step(state, tokens_batch, step_rng):
        # split into separate rngs for dropout vs VAE
        dropout_rng, vae_rng = jax.random.split(step_rng, 2)
        def loss_fn(params):
            B, num_eff_frames, num_tokens, token_dim = tokens_batch.shape
            inputs = tokens_batch[:, :num_eff_frames - 1, :, :].reshape(B, -1, token_dim)
            targets = tokens_batch[:, 1:, :, :].reshape(B, -1, token_dim)
            logits, (mu, logvar, x_pre, proj_z) = model.apply(
                {'params': params},
                inputs,
                train=True,
                rngs={'dropout': dropout_rng, 'vae': vae_rng}
            )
            pred_loss = optax.sigmoid_binary_cross_entropy(logits, targets).mean()
            kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
            recon_loss = jnp.mean((proj_z - x_pre)**2)
            return pred_loss + 10.0 * kl_loss + 10.0 * recon_loss, (pred_loss, kl_loss, recon_loss)

        (loss, (pred_l, kl_l, recon_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, pred_l, kl_l, recon_l

    rng = jax.random.PRNGKey(int(time.time()))
    for step in range(1, train_steps + 1):
        batch_indices = np.random.choice(num_train, batch_size, replace=False)
        tokens_batch = train_dataset[batch_indices]
        if t_skip > 0:
            tokens_batch = tokens_batch[:, ::(t_skip + 1), :, :]
        rng, step_rng = jax.random.split(rng)
        state, loss, pl, kl, rl = train_step(state, tokens_batch, step_rng)
        wandb.log({"step": step, "train_loss": float(loss)})
        print(f"[Step {step}] Loss: {loss:.6f}")
        # print(f"  Pred Loss: {pl:.6f}, KL Loss: {kl:.6f}, Recon Loss: {rl:.6f}")
        
        if step % eval_every == 0 and val_dataset is not None:
            evaluate(model, state, val_dataset, img_size, grid_size, step, t_skip=t_skip)
    
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/gpt_vae_params_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(state.params, f)
    print(f"[Done] Model parameters saved to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (pixels) for each frame")
    parser.add_argument("--patches_per_dim", type=int, default=2,
                        help="Number of patches per image dimension (e.g. 2 means 2x2 grid)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per simulation sequence in CSV files")
    parser.add_argument("--t_skip", type=int, default=0,
                        help="Skip frequency: 0 uses all frames; 1 uses frames 0, 2, 4, ..., etc.")
    parser.add_argument("--train_csv", type=str, default="conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
                        help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
                        help="Path to validation CSV file")
    args = parser.parse_args()

    train_gpt(batch_size=args.batch_size, 
              train_steps=args.train_steps, 
              eval_every=args.eval_every, 
              img_size=args.img_size, 
              patches_per_dim=args.patches_per_dim, 
              num_frames=args.num_frames, 
              t_skip=args.t_skip, 
              train_csv=args.train_csv, 
              val_csv=args.val_csv)
