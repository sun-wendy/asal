import math
import os
import io
import pickle
import time
import numpy as np
import imageio.v2 as imageio
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
    block_size: int
    token_dim: int
    num_tokens: int
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.c_attn = nn.Dense(config.n_embd * 3)
        self.c_proj = nn.Dense(config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jnp.ndarray, *, train: bool):
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

    def __call__(self, x: jnp.ndarray, *, train: bool):
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

    def __call__(self, x: jnp.ndarray, *, train: bool):
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

    def __call__(self, tokens: jnp.ndarray, *, train: bool):
        B, T, d = tokens.shape
        assert d == self.config.token_dim
        x = self.token_proj(tokens)
        pos = jnp.arange(T)[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)
        logits = self.head(x)
        preds = jax.nn.sigmoid(logits)   # continuous grayscale
        return preds, None

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


def evaluate(model, state, val_dataset: np.ndarray, img_size: int, grid_size: Tuple[int,int], step: int, t_skip: int=0):
    num_val = val_dataset.shape[0]
    num_eval = 1000
    batch_size = 100
    rng = np.random.default_rng()
    idxs = rng.choice(num_val, size=num_eval, replace=False)

    all_true = []
    all_pred = []

    for i in range(0, num_eval, batch_size):
        sub = idxs[i:i+batch_size]
        batch = val_dataset[sub]
        if t_skip>0:
            batch = batch[:, ::(t_skip+1),:,:]
        inp = batch[:,:-1,:,:].reshape(len(sub),-1, batch.shape[-1])
        tgt = batch[:,1:,:,:].reshape(len(sub),-1, batch.shape[-1])
        preds, _ = model.apply({'params': state.params}, inp, train=False)
        all_true.append(tgt.reshape(-1))
        all_pred.append(np.array(preds).reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Regression metric
    mse = np.mean((y_pred - y_true)**2)

    # Classification metrics via threshold=0.5
    y_bin = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall    = recall_score(y_true, y_bin, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = float('nan')
    tp = np.sum((y_bin==1)&(y_true==1))
    tn = np.sum((y_bin==0)&(y_true==0))
    fp = np.sum((y_bin==1)&(y_true==0))
    fn = np.sum((y_bin==0)&(y_true==1))
    bal_acc = 0.5*(tp/(tp+fn+1e-8) + tn/(tn+fp+1e-8))

    print(f"[Eval] Step {step} — MSE: {mse:.6f}, Prec: {precision:.3f}, Rec: {recall:.3f}, AUROC: {auroc:.3f}, BalAcc: {bal_acc:.3f}")
    wandb.log({
        "eval_mse": mse,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_auroc": auroc,
        "eval_balanced_acc": bal_acc,
        "eval_step": step,
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
        preds, _ = model.apply({'params': state.params}, inp, train=False)
        preds = preds.reshape(num_pred, -1, model.config.token_dim)
        last_frame = preds[-1]
        pred_seq.append(np.array(last_frame))
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
        # dark_blue = np.array([0.0, 0.0, 139/255.0])
        # yellow = np.array([1.0, 1.0, 0.0])
        # discrepancy = np.ones_like(gt_frame) * dark_blue
        # diff_mask = (gt_frame[..., 0] != gen_frame[..., 0])
        # discrepancy[diff_mask] = yellow
        abs_err = np.abs(gt_frame[...,0] - gen_frame[...,0])
        discrepancy = np.repeat(abs_err[:, :, None], 3, axis=-1)
        
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


def train_gpt(batch_size: int=32, train_steps: int=3000, eval_every: int=200,
              img_size: int=32, patches_per_dim: int=2, num_frames: int=10,
              t_skip: int=0, loss_beta: float = 2.0, train_csv: str="...", val_csv: str="...", seed: int=42):

    random.seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    wandb.init(project="gol_sim",
               name=f"gpt_mse_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}_seed{seed}",
               config={"batch_size": batch_size,
                       "train_steps": train_steps,
                       "img_size": img_size,
                       "patches_per_dim": patches_per_dim,
                       "num_frames": num_frames,
                       "t_skip": t_skip,
                       "loss_beta": loss_beta,
                       "seed": seed})

    grid_size = (patches_per_dim, patches_per_dim)
    token_dim = (img_size // patches_per_dim) ** 2
    num_tokens = patches_per_dim ** 2
    num_eff_frames = math.ceil(num_frames / (t_skip + 1))
    block_size = num_eff_frames * num_tokens

    print(f"Training (MSE) with batch size {batch_size}, train steps {train_steps}, eval every {eval_every}, img size {img_size}, patches per dim {patches_per_dim}, num frames {num_frames} (num effective frames {num_eff_frames}), t_skip {t_skip}, block size {block_size}, token dim {token_dim}, num tokens {num_tokens}, loss beta {loss_beta}, seed {seed}")
    
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
    def train_step(state, batch, dropout_rng, beta):
        def loss_fn(params):
            B, ef, nt, td = batch.shape
            inp = batch[:,:-1,:,:].reshape(B, -1, td)
            tgt = batch[:,1:,:,:].reshape(B, -1, td)
            preds, _ = model.apply({'params': params}, inp, train=True, rngs={'dropout': dropout_rng})
            weight = jnp.where(tgt == 1.0, beta, 1.0)
            sq_err = (preds - tgt)**2
            weighted_sq_err = weight * sq_err
            return weighted_sq_err.mean()
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
    with open(f"checkpoints/gpt_params_mse_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}_seed{seed}.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print(f"[Done] Model parameters saved to checkpoints/gpt_params_mse_bsz{batch_size}_trainsteps{train_steps}_img{img_size}_patches{patches_per_dim}_frames{num_frames}_tskip{t_skip}_beta{loss_beta}_seed{seed}.pkl")
    wandb.finish()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=50000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size (pixels) for each frame")
    parser.add_argument("--patches_per_dim", type=int, default=8,
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
