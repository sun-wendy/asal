import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import csv
import argparse
import os
import imageio
import time

# ----------------- CSV Dataset Loader -----------------
def load_csv_frames(csv_file: str, img_size: int, num_frames: int) -> np.ndarray:
    """
    Load individual simulation state frames from a CSV file.
    Each row in the CSV should have num_frames cells, where each cell is a 1024-character
    (for img_size=32) string or a space-separated list of numbers.
    
    Returns an array of shape (N, img_size, img_size, 1), where
      N = (num_sequences * num_frames)
    """
    frames = []
    print(f"Loading frames from {csv_file}")
    with open(csv_file, newline='') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # Skip header rows if present
            if row[0].strip().startswith("State"):
                continue
            if len(row) != num_frames:
                raise ValueError(f"Expected row to have {num_frames} cells, got {len(row)}")
            for cell in row:
                cell_str = cell.strip()
                arr = np.fromstring(cell_str, sep=" ")
                # If only one value is returned, assume the cell is a contiguous string.
                if arr.size == 1:
                    if len(cell_str) != img_size * img_size:
                        raise ValueError(f"Expected cell string length {img_size*img_size}, got {len(cell_str)}")
                    arr = np.array([float(c) for c in cell_str], dtype=np.float32)
                if arr.size != img_size * img_size:
                    raise ValueError(f"Expected cell to contain {img_size * img_size} values, got {arr.size}")
                frame = arr.reshape(img_size, img_size)
                frames.append(frame[..., np.newaxis])    # add a channel dimension
    frames = np.stack(frames, axis=0)  # shape: (N, img_size, img_size, 1)
    print(f"Loaded {frames.shape[0]} frames.")
    return frames

# ----------------- VAE Definition (for grayscale images) -----------------
class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        # x expected shape: (B, H, W, 1)
        x = nn.Conv(32, (4, 4), strides=(2, 2), padding="SAME")(x)  # (B, 16, 16, 32)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding="SAME")(x)  # (B, 8, 8, 64)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten; expected shape is (B, 4096)
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar

class Decoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, z):
        # Decoder for grayscale images.
        x = nn.Dense(8 * 8 * 64)(z)
        x = x.reshape((-1, 8, 8, 64))
        x = nn.ConvTranspose(64, (4, 4), strides=(2, 2), padding="SAME")(x)  # (B, 16, 16, 64)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (4, 4), strides=(2, 2), padding="SAME")(x)  # (B, 32, 32, 32)
        x = nn.relu(x)
        # Final layer: project to 1 channel.
        x = nn.Conv(1, (3, 3), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x

class VAE(nn.Module):
    latent_dim: int
    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)
    def __call__(self, x, rng):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, std.shape)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar

def compute_vae_loss(x, recon, mu, logvar):
    recon_loss = jnp.mean((x - recon)**2)
    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
    return recon_loss + 0.01 * kl

# ----------------- Dynamics Model Definition -----------------
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

# ----------------- Load Pretrained VAE Parameters -----------------
with open("encoder_params.pkl", "rb") as f:
    encoder_params = pickle.load(f)
with open("decoder_params.pkl", "rb") as f:
    decoder_params = pickle.load(f)

latent_dim = 128
encoder_model = Encoder(latent_dim)
decoder_model = Decoder(latent_dim)

# ----------------- Dynamics Logging -----------------
def show_dynamics_log(x_t, x_next, recon_t, recon_pred_next, prefix="dynamics_log", n=5):
    x_t = jnp.clip(x_t, 0.0, 1.0)
    x_next = jnp.clip(x_next, 0.0, 1.0)
    recon_t = jnp.clip(recon_t, 0.0, 1.0)
    recon_pred_next = jnp.clip(recon_pred_next, 0.0, 1.0)
    os.makedirs("dynamics_logs", exist_ok=True)
    for i in range(n):
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        # Convert one-channel images to three channels for visualization.
        x_t_rgb = np.repeat(np.array(x_t[i].squeeze())[..., np.newaxis], 3, axis=-1)
        x_next_rgb = np.repeat(np.array(x_next[i].squeeze())[..., np.newaxis], 3, axis=-1)
        recon_t_rgb = np.repeat(np.array(recon_t[i].squeeze())[..., np.newaxis], 3, axis=-1)
        recon_pred_rgb = np.repeat(np.array(recon_pred_next[i].squeeze())[..., np.newaxis], 3, axis=-1)
        ax[0, 0].imshow(x_t_rgb)
        ax[0, 0].set_title("x_t")
        ax[0, 0].axis("off")
        ax[0, 1].imshow(x_next_rgb)
        ax[0, 1].set_title("x_{t+1}")
        ax[0, 1].axis("off")
        ax[1, 0].imshow(recon_t_rgb)
        ax[1, 0].set_title("Decoder(z_t)")
        ax[1, 0].axis("off")
        ax[1, 1].imshow(recon_pred_rgb)
        ax[1, 1].set_title("Decoder(f(z_t))")
        ax[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(f"dynamics_logs/{prefix}_{i}.png")
        plt.close(fig)

# ----------------- Save GIF from Frames -----------------
def make_gif_from_folder(frame_dir, gif_path, fps=20, ext=".png"):
    """Stitch sorted frames from a folder into a GIF."""
    frame_files = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(ext)
    ])
    frames = []
    for frame_path in frame_files:
        img = imageio.imread(frame_path)
        frames.append(img)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[Eval] GIF saved to {gif_path}")

# ----------------- Evaluation using Validation CSV -----------------
def evaluate_dynamics(val_frames, dynamics_params, encoder_params, decoder_params, latent_dim, num_frames: int, save_dir="eval_videos"):
    """
    Evaluate the dynamics model on a single randomly sampled simulation sequence.
    Expects val_frames to have shape (N, img_size, img_size, 1), where N = (num_sequences * num_frames).
    This function randomly selects one sequence (of length num_frames) for evaluation.
    """
    os.makedirs(save_dir, exist_ok=True)
    true_frames_dir = os.path.join(save_dir, "true_frames")
    pred_frames_dir = os.path.join(save_dir, "pred_frames")
    combined_frames_dir = os.path.join(save_dir, "side_by_side")
    os.makedirs(true_frames_dir, exist_ok=True)
    os.makedirs(pred_frames_dir, exist_ok=True)
    os.makedirs(combined_frames_dir, exist_ok=True)

    total_frames = val_frames.shape[0]
    num_sequences = total_frames // num_frames
    # Randomly select one simulation sequence.
    seq_index = np.random.randint(0, num_sequences)
    start_idx = seq_index * num_frames
    sim_sequence = val_frames[start_idx : start_idx + num_frames]  # shape: (num_frames, H, W, 1)

    # Encode only the first frame for dynamics.
    first_frame = sim_sequence[0]  # shape: (H, W, 1)
    mu, _ = encoder_model.apply({'params': encoder_params}, first_frame[None])
    z_seq = [mu[0]]
    # Generate latent sequence using the dynamics model.
    for _ in range(num_frames - 1):
        next_z = DynamicsModel(latent_dim).apply({'params': dynamics_params}, z_seq[-1][None])[0]
        z_seq.append(next_z)
    z_seq = jnp.stack(z_seq, axis=0)
    # Decode predicted latents.
    recon_pred = decoder_model.apply({'params': decoder_params}, z_seq)

    # For saving, convert one-channel images to three channels.
    for i in range(num_frames):
        x_true = np.array(sim_sequence[i])
        x_pred = np.array(recon_pred[i])
        x_true_rgb = np.repeat(x_true, 3, axis=-1)
        x_pred_rgb = np.repeat(x_pred, 3, axis=-1)
        true_path = os.path.join(true_frames_dir, f"frame_{i:04d}.png")
        pred_path = os.path.join(pred_frames_dir, f"frame_{i:04d}.png")
        canvas_path = os.path.join(combined_frames_dir, f"frame_{i:04d}.png")
        imageio.imwrite(true_path, (x_true_rgb * 255).astype(np.uint8))
        imageio.imwrite(pred_path, (x_pred_rgb * 255).astype(np.uint8))
        canvas = np.concatenate([x_true_rgb, x_pred_rgb], axis=1)
        imageio.imwrite(canvas_path, (canvas * 255).astype(np.uint8))
    gif_path = os.path.join(save_dir, "dynamics_evaluation.gif")
    make_gif_from_folder(combined_frames_dir, gif_path, fps=20)
    print(f"[Eval] Saved evaluation GIF to {gif_path}")

# ----------------- Training Dynamics Model (Using CSV data) -----------------
def train_dynamics_model(train_csv: str, val_csv: str, num_frames: int, img_size: int):
    rng = jax.random.PRNGKey(123)
    learning_rate = 1e-3
    batch_size = 2048
    total_steps = 300000
    eval_every = 10000

    # Load training and validation datasets from CSV.
    train_frames = load_csv_frames(train_csv, img_size, num_frames)
    val_frames = load_csv_frames(val_csv, img_size, num_frames)
    # Assume the training CSV contains multiple sequences.
    num_sequences = train_frames.shape[0] // num_frames
    train_data = train_frames.reshape(num_sequences, num_frames, img_size, img_size, 1)
    
    # Initialize dynamics model.
    dynamics = DynamicsModel(latent_dim)
    params = dynamics.init(rng, jnp.ones((1, latent_dim)))['params']
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)
    train_losses = []

    # Helpers to encode and decode batches.
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
            loss = jnp.mean((z_pred - z_next)**2)
            return loss
        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for step in range(1, total_steps + 1):
        # Randomly select a sequence and time step.
        s_indices = np.random.randint(0, num_sequences, size=batch_size)
        t_indices = np.random.randint(0, num_frames - 1, size=batch_size)
        x_t = train_data[s_indices, t_indices]      # shape: (batch_size, img_size, img_size, 1)
        x_next = train_data[s_indices, t_indices + 1] # shape: (batch_size, img_size, img_size, 1)
        x_t = jnp.array(x_t)
        x_next = jnp.array(x_next)
        # Encode both frames.
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
            evaluate_dynamics(val_frames, params, encoder_params, decoder_params, latent_dim, num_frames)

    # === Plot Loss Curve ===
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

    # === Save final dynamics model parameters ===
    with open("dynamics_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("[Done] Dynamics model parameters saved to dynamics_params.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
                        help="Path to the validation CSV file")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per sequence in the CSV files")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size for each simulation frame")
    args = parser.parse_args()

    train_dynamics_model(args.train_csv, args.val_csv, args.num_frames, args.img_size)
