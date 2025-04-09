import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import imageio
import time

import substrates
from rollout import rollout_simulation


# === VAE Definition ===
class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar

class Decoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(8 * 8 * 64)(z)
        x = x.reshape((-1, 8, 8, 64))
        x = nn.ConvTranspose(64, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(3, (4, 4), strides=(2, 2))(x)
        x = nn.sigmoid(x)
        return x


# === Load Trained Encoder & Decoder ===
with open("encoder_params.pkl", "rb") as f:
    encoder_params = pickle.load(f)
with open("decoder_params.pkl", "rb") as f:
    decoder_params = pickle.load(f)

latent_dim = 128
encoder_model = Encoder(latent_dim)
decoder_model = Decoder(latent_dim)


# === Dynamics MLP ===
class DynamicsModel(nn.Module):
    latent_dim: int
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.latent_dim)(z)
        return z


# === Dataset Generator ===
def generate_full_dataset(rng, substrate, num_rollouts=128, rollout_steps=256, img_size=64):
    rollout_rngs = jax.random.split(rng, num_rollouts)
    param_shape = substrate.default_params(jax.random.PRNGKey(0)).shape
    flat_params = jnp.full(param_shape, 6152)

    def rollout_fn(rng_i):
        result = rollout_simulation(rng_i, params=flat_params, substrate=substrate, fm=None,
                                    rollout_steps=rollout_steps, time_sampling='video',
                                    img_size=img_size, return_state=False)
        return result['rgb']

    vids = jax.vmap(rollout_fn)(rollout_rngs)
    vids = vids.reshape(-1, img_size, img_size, 3)
    return vids


# === Log for 2-step comparison ===
def show_dynamics_log(x_t, x_next, recon_t, recon_pred_next, prefix="dynamics_log", n=5):
    x_t = jnp.clip(x_t, 0.0, 1.0)
    x_next = jnp.clip(x_next, 0.0, 1.0)
    recon_t = jnp.clip(recon_t, 0.0, 1.0)
    recon_pred_next = jnp.clip(recon_pred_next, 0.0, 1.0)

    os.makedirs("dynamics_logs", exist_ok=True)

    for i in range(n):
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        ax[0, 0].imshow(np.array(x_t[i]))
        ax[0, 0].set_title("x_t")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(np.array(x_next[i]))
        ax[0, 1].set_title("x_{t+1}")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(np.array(recon_t[i]))
        ax[1, 0].set_title("Decoder(z_t)")
        ax[1, 0].axis('off')

        ax[1, 1].imshow(np.array(recon_pred_next[i]))
        ax[1, 1].set_title("Decoder(f(z_t))")
        ax[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"dynamics_logs/{prefix}_{i}.png")
        plt.close(fig)


# === Long Rollout Evaluation ===
def make_video_from_folder(frame_dir, video_path, fps=20, ext=".png"):
    """Stitch sorted frames from a folder into a video."""
    frame_files = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(ext)
    ])
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame_path in frame_files:
            img = imageio.imread(frame_path)
            writer.append_data(img)
    print(f"[Eval] Video saved to {video_path}")

def evaluate_rollout(dynamics_params, substrate, rollout_rng,
                     encoder_params, decoder_params,
                     save_dir="eval_videos", latent_dim=128):

    os.makedirs(save_dir, exist_ok=True)
    true_frames_dir = os.path.join(save_dir, "true_frames")
    pred_frames_dir = os.path.join(save_dir, "pred_frames")
    combined_frames_dir = os.path.join(save_dir, "side_by_side")
    os.makedirs(true_frames_dir, exist_ok=True)
    os.makedirs(pred_frames_dir, exist_ok=True)
    os.makedirs(combined_frames_dir, exist_ok=True)

    # === Step 1: Generate true simulation frames ===
    frames = generate_full_dataset(rollout_rng, substrate, num_rollouts=1, rollout_steps=256)
    frames = jnp.array(frames.reshape(256, 64, 64, 3))  # raw RGB simulation frames

    # === Step 2: Encode only the first frame ===
    mu, _ = encoder_model.apply({'params': encoder_params}, frames)
    z_seq = [mu[0]]

    # === Step 3: Predict latent sequence using dynamics model ===
    for _ in range(255):
        next_z = DynamicsModel(latent_dim).apply({'params': dynamics_params}, z_seq[-1][None])[0]
        z_seq.append(next_z)
    z_seq = jnp.stack(z_seq, axis=0)

    # === Step 4: Decode predicted latents ===
    recon_pred = decoder_model.apply({'params': decoder_params}, z_seq)

    # === Step 5: Save individual frames and side-by-side canvases ===
    for i in range(256):
        x_true = np.clip(np.array(frames[i]), 0.0, 1.0)
        x_pred = np.clip(np.array(recon_pred[i]), 0.0, 1.0)

        true_path = os.path.join(true_frames_dir, f"frame_{i:04d}.png")
        pred_path = os.path.join(pred_frames_dir, f"frame_{i:04d}.png")
        canvas_path = os.path.join(combined_frames_dir, f"frame_{i:04d}.png")

        imageio.imwrite(true_path, (x_true * 255).astype(np.uint8))
        imageio.imwrite(pred_path, (x_pred * 255).astype(np.uint8))

        canvas = np.concatenate([x_true, x_pred], axis=1)  # shape: (64, 128, 3)
        imageio.imwrite(canvas_path, (canvas * 255).astype(np.uint8))

    # === Step 6: Make video ===
    video_path = os.path.join(save_dir, "rollout_comparison.mp4")
    make_video_from_folder(combined_frames_dir, video_path, fps=20)

    print(f"[Eval] Saved all frames and video to {save_dir}")


# === Training ===
def train_dynamics_model():
    rng = jax.random.PRNGKey(123)
    learning_rate = 1e-3
    batch_size = 2048
    total_steps = 30000
    eval_every = 1000
    img_size = 64

    substrate = substrates.create_substrate("gol")
    substrate = substrates.FlattenSubstrateParameters(substrate)

    dynamics = DynamicsModel(latent_dim)
    params = dynamics.init(rng, jnp.ones((1, latent_dim)))['params']
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    train_losses = []

    @jit
    def encode_batch(x, encoder_params):
        mu, _ = encoder_model.apply({'params': encoder_params}, x)
        return mu

    @jit
    def decode_batch(z, decoder_params):
        return decoder_model.apply({'params': decoder_params}, z)

    @jit
    def train_step(params, opt_state, z_t, z_next):
        def loss_fn(p):
            z_pred = DynamicsModel(latent_dim).apply({'params': p}, z_t)
            loss = jnp.mean((z_pred - z_next) ** 2)
            return loss
        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for step in range(1, total_steps + 1):
        rng, rollout_rng = jax.random.split(rng)
        dataset = generate_full_dataset(rollout_rng, substrate, num_rollouts=128, rollout_steps=256, img_size=img_size)

        # Sample consecutive pairs
        frame_rng, rng = jax.random.split(rng)
        indices = jax.random.randint(frame_rng, (batch_size,), 0, dataset.shape[0] - 1)
        x_t = dataset[indices]
        x_next = dataset[indices + 1]

        z_t = encode_batch(x_t, encoder_params)
        z_next = encode_batch(x_next, encoder_params)

        params, opt_state, loss = train_step(params, opt_state, z_t, z_next)
        train_losses.append(float(loss))

        if step % 100 == 0:
            z_pred_next = DynamicsModel(latent_dim).apply({'params': params}, z_t)
            recon_t = decode_batch(z_t, decoder_params)
            recon_pred_next = decode_batch(z_pred_next, decoder_params)
            print(f"[Step {step}] Train Loss: {loss:.6f}")
        
        if step % 1000 == 0:
            show_dynamics_log(x_t, x_next, recon_t, recon_pred_next, prefix=f"step_{step}")

        if step % eval_every == 0:
            eval_rng, rng = jax.random.split(rng)
            evaluate_rollout(params, substrate, eval_rng, encoder_params, decoder_params)
    
    # === Plot loss curves using exponential moving average ===
    import pandas as pd
    df = pd.DataFrame({'train': train_losses})
    ema = df.ewm(span=1000).mean()
    plt.figure()
    plt.plot(df['train'], label='Raw Train Loss', alpha=0.5)
    plt.plot(ema['train'], label='EMA Train Loss')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Dynamics Model Training Loss")
    plt.savefig("dynamics_loss_curve.png")
    plt.close()
    print("[Done] Loss curve saved to dynamics_loss_curve.png")

    # === Save final model ===
    with open("dynamics_params.pkl", "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    train_dynamics_model()
