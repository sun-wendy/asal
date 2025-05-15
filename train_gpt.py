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
from flax.core.frozen_dict import freeze
import random
from sklearn.metrics import precision_score, recall_score, roc_auc_score

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
        att = jnp.where(mask == 1.0, att, -1e9)
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


def evaluate(model, state, val_dataset: np.ndarray, img_size: int, grid_size: Tuple[int, int], step: int, t_skip: int = 0):
    """
    Evaluate the GPT model on the validation dataset
    """
    # Evaluation using teacher-forcing
    num_val = val_dataset.shape[0]
    num_eval = 100
    rng = np.random.default_rng()
    # sample a batch of sequences
    idxs = rng.choice(num_val, size=num_eval, replace=False)
    batch = val_dataset[idxs]               # shape: (B, T, num_tokens, token_dim)
    if t_skip > 0:
        batch = batch[:, ::(t_skip+1), :, :]
    # split into inputs (frames 0…T-2) and targets (1…T-1)
    inputs = batch[:, :-1, :, :]            # shape: (B, T-1, N, D)
    targets = batch[:, 1:, :, :]            # shape: (B, T-1, N, D)
    B, t_minus1, N, D = inputs.shape
    # flatten time+tokens into sequence dimension
    inputs = inputs.reshape(B, -1, D)       # (B, seq_len, D)
    targets = targets.reshape(B, -1, D)     # (B, seq_len, D)
    # single forward pass
    logits, _ = model.apply(
        {'params': state.params},
        inputs,
        train=False
    )                                        # (B, seq_len, D)
    preds = (logits > 0).astype(np.float32)

    probs = jax.nn.sigmoid(logits)
    y_true = targets.reshape(-1)
    y_prob = probs.reshape(-1)
    y_pred = preds.reshape(-1)
    # Compute metrics
    accuracy  = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    # AUROC requires at least one positive and one negative in y_true
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')
    # balanced accuracy
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    bal_acc = 0.5 * (tp/(tp+fn+1e-8) + tn/(tn+fp+1e-8))
    # accuracy = (preds == targets).mean() * 100.0
    print(f"[Eval] Step {step} — Acc: {accuracy*100:.2f}%, Prec: {precision:.3f}, Rec: {recall:.3f}, AUROC: {auroc:.3f}")
    wandb.log({
        "eval_accuracy":    float(accuracy),
        "eval_precision":   float(precision),
        "eval_recall":      float(recall),
        "eval_auroc":       float(auroc),
        "eval_balanced_acc": float(bal_acc),
        "eval_step":        step,
    })

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
    while num_pred < 2:
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
    
    for i in range(2):
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


def train_gpt(batch_size: int = 32, train_steps: int = 3000, eval_every: int = 200, img_size: int = 32, patches_per_dim: int = 2,
              num_frames: int = 10, t_skip: int = 0, loss_beta: float = 2.0,
              train_csv: str = "conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
              val_csv: str = "conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
              seed: int = 42):
    """
    Training routine using CSV file data.
    Assumes each simulation sequence in the CSV has num_frames frames of size (img_size x img_size).
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    wandb.init(project="gol_world_model",
               name=f"gpt_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}",
               config={"batch_size": batch_size,
                       "train_steps": train_steps,
                       "img_size": img_size,
                       "patches_per_dim": patches_per_dim,
                       "num_frames": num_frames,
                       "t_skip": t_skip,
                       "loss_beta": loss_beta})
    
    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2
    num_tokens = patches_per_dim ** 2
    num_eff_frames = math.ceil(num_frames / (t_skip + 1))
    block_size = num_eff_frames * num_tokens
    
    print(f"Training with batch size {batch_size}, train steps {train_steps}, eval every {eval_every}, img size {img_size}, patches per dim {patches_per_dim}, num frames {num_frames} (num effective frames {num_eff_frames}), t_skip {t_skip}, block size {block_size}, token dim {token_dim}, num tokens {num_tokens}, loss beta {loss_beta}")
    
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
    print(f"Loaded {num_train} training sequences.")
    
    @jax.jit
    def train_step(state, tokens_batch, dropout_rng, beta):
        def loss_fn(params):
            B, num_eff_frames, num_tokens, token_dim = tokens_batch.shape
            inputs = tokens_batch[:, :-1, :, :]
            targets = tokens_batch[:, 1:, :, :]
            inputs = inputs.reshape(B, -1, token_dim)
            targets = targets.reshape(B, -1, token_dim)
            logits, _ = model.apply({'params': params}, inputs, train=True, rngs={'dropout': dropout_rng})
            bce_per_elem = optax.sigmoid_binary_cross_entropy(logits, targets)
            # weight = jnp.ones_like(targets)
            # weight = weight.at[targets == 1].set(5.0)
            weight = jnp.where(targets == 1.0, beta, 1.0)
            return (weight * bce_per_elem).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    train_losses = []
    # rng = jax.random.PRNGKey(int(time.time()))
    for step in range(1, train_steps + 1):
        batch_indices = np.random.choice(num_train, batch_size, replace=False)
        tokens_batch = train_dataset[batch_indices]
        if t_skip > 0:
            tokens_batch = tokens_batch[:, ::(t_skip + 1), :, :]
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, tokens_batch, dropout_rng, loss_beta)
        train_losses.append(float(loss))
        wandb.log({"step": step, "train_loss": float(loss)})
        print(f"[Step {step}] Loss: {loss:.6f}")
        
        if step % eval_every == 0 and val_dataset is not None:
            evaluate(model, state, val_dataset, img_size, grid_size, step, t_skip=t_skip)
    
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/gpt_params_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print(f"[Done] Model parameters saved to checkpoints/gpt_params_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}.pkl")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=50000)
    parser.add_argument("--eval_every", type=int, default=1000)
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--loss_beta", type=float, default=2.0,
                        help="Loss weight for alive cells")
    args = parser.parse_args()

    train_gpt(batch_size=args.batch_size, 
              train_steps=args.train_steps, 
              eval_every=args.eval_every, 
              img_size=args.img_size, 
              patches_per_dim=args.patches_per_dim, 
              num_frames=args.num_frames, 
              t_skip=args.t_skip, 
              loss_beta=args.loss_beta,
              train_csv=args.train_csv, 
              val_csv=args.val_csv,
              seed=args.seed)
