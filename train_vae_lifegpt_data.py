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

# ----------------- CSV Dataset Loader -----------------
def load_csv_frames(csv_file: str, img_size: int, num_frames: int) -> np.ndarray:
    """
    Load individual simulation state frames from a CSV file.
    Each row in the CSV should have num_frames cells, where each cell is a 1024-character
    (for img_size=32) string or a space-separated list of numbers.
    
    Returns:
      A numpy array of shape (N, img_size, img_size, 1), where N = (num_sequences * num_frames)
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
                # If only one value is returned, assume the cell is a contiguous string of characters
                if arr.size == 1:
                    if len(cell_str) != img_size * img_size:
                        raise ValueError(f"Expected cell string length {img_size*img_size}, got {len(cell_str)}")
                    arr = np.array([float(c) for c in cell_str], dtype=np.float32)
                if arr.size != img_size * img_size:
                    raise ValueError(f"Expected cell to contain {img_size * img_size} values, got {arr.size}")
                frame = arr.reshape(img_size, img_size)  # shape: (img_size, img_size)
                frames.append(frame[..., np.newaxis])    # add a channel dimension
    frames = np.stack(frames, axis=0)  # shape: (N, img_size, img_size, 1)
    print(f"Loaded {frames.shape[0]} frames.")
    return frames

# ----------------- VAE Definition -----------------
class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        # x assumed to be (B, img_size, img_size, 1)
        x = nn.Conv(32, (4, 4), strides=(2, 2), padding='SAME')(x)  # (B, 16, 16, 32)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding='SAME')(x)  # (B, 8, 8, 64)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar

class Decoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, z):
        # First, project the latent vector and reshape to a spatial feature map.
        x = nn.Dense(8 * 8 * 64)(z)
        x = x.reshape((-1, 8, 8, 64))
        x = nn.ConvTranspose(64, (4, 4), strides=(2, 2), padding='SAME')(x)  # (B, 16, 16, 64)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (4, 4), strides=(2, 2), padding='SAME')(x)  # (B, 32, 32, 32)
        x = nn.relu(x)
        # Final layer to project to 1 channel.
        x = nn.Conv(1, (3, 3), padding='SAME')(x)
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

# ----------------- Loss Function -----------------
def compute_vae_loss(x, recon, mu, logvar):
    recon_loss = jnp.mean((x - recon) ** 2)
    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
    return recon_loss + 0.01 * kl

# ----------------- Training Loop -----------------
def train_vae(train_csv: str, val_csv: str, num_frames: int, img_size: int, latent_dim: int,
              total_steps: int, batch_size: int, learning_rate: float):
    # Load training and validation frames.
    train_dataset = load_csv_frames(train_csv, img_size, num_frames)
    if os.path.exists(val_csv):
        val_dataset = load_csv_frames(val_csv, img_size, num_frames)
    else:
        val_dataset = None

    num_train = train_dataset.shape[0]
    print(f"Training on {num_train} frames.")
    if val_dataset is not None:
        print(f"Validation on {val_dataset.shape[0]} frames.")

    # Initialize the model and optimizer.
    vae = VAE(latent_dim)
    rng = random.PRNGKey(0)
    dummy_input = jnp.ones((1, img_size, img_size, 1), dtype=jnp.float32)
    params = vae.init(rng, dummy_input, rng)['params']
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    # Define a helper function to sample a random batch.
    def get_batch(data, rng, batch_size):
        indices = jax.random.randint(rng, (batch_size,), 0, data.shape[0])
        return data[indices]

    @jit
    def train_step(params, opt_state, batch, step_rng):
        def loss_fn(p):
            recon, mu, logvar = VAE(latent_dim).apply({'params': p}, batch, step_rng)
            loss = compute_vae_loss(batch, recon, mu, logvar)
            return loss, recon
        (loss, recon), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, recon

    @jit
    def eval_step(params, batch, rng):
        recon, mu, logvar = VAE(latent_dim).apply({'params': params}, batch, rng)
        loss = compute_vae_loss(batch, recon, mu, logvar)
        return loss, recon

    train_losses = []
    val_losses = []

    # Training loop
    for step in range(1, total_steps + 1):
        rng, step_rng, batch_rng = random.split(rng, 3)
        batch = get_batch(train_dataset, batch_rng, batch_size)
        params, opt_state, loss, recon = train_step(params, opt_state, batch, step_rng)
        train_losses.append(float(loss))

        # Every 100 steps, evaluate on validation data (if provided) and visualize reconstructions.
        if step % 100 == 0:
            if val_dataset is not None:
                val_batch = get_batch(val_dataset, batch_rng, batch_size)
                val_loss, val_recon = eval_step(params, val_batch, step_rng)
                val_losses.append(float(val_loss))
                print(f"[Step {step}] Train Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")
                show_recon(batch, recon, prefix="train_recon")
                show_recon(val_batch, val_recon, prefix="val_recon")
            else:
                print(f"[Step {step}] Train Loss: {loss:.6f}")

    # === Plot loss curves ===
    # df = pd.DataFrame({'train': train_losses, 'val': pd.Series(val_losses)})
    # ema = df.ewm(span=1000).mean()
    # plt.plot(df['train'], label='Raw Train Loss', alpha=0.5)
    # if not val_losses == []:
    #     plt.plot(df['val'], label='Raw Val Loss', alpha=0.5)
    #     plt.plot(ema['val'], label='EMA Val Loss')
    # plt.plot(ema['train'], label='EMA Train Loss')
    # plt.xlabel("Training Step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title("VAE Loss Curve")
    # plt.savefig("vae_loss_curve.png")
    # plt.close()

    # === Save encoder and decoder parameters ===
    with open("encoder_params.pkl", "wb") as f:
        pickle.dump(params['encoder'], f)
    with open("decoder_params.pkl", "wb") as f:
        pickle.dump(params['decoder'], f)
    print("Training complete. Parameters saved.")

# ----------------- Visualization Helper -----------------
def show_recon(input_batch, recon_batch, prefix="recon", n=5):
    input_batch = jnp.clip(input_batch, 0.0, 1.0)
    recon_batch = jnp.clip(recon_batch, 0.0, 1.0)
    for i in range(n):
        fig, ax = plt.subplots(1, 2, figsize=(6,3))
        ax[0].imshow(np.array(input_batch[i].squeeze()), cmap='gray')
        ax[0].set_title("Input")
        ax[0].axis('off')
        ax[1].imshow(np.array(recon_batch[i].squeeze()), cmap='gray')
        ax[1].set_title("Reconstruction")
        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"{prefix}_{i}.png")
        plt.close(fig)

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="conway_states_0_1_1000by32by32by10_toroidal_20240711_151806.csv",
                        help="Path to the validation CSV file (optional)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames per simulation sequence in the CSV files")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Image size for each simulation state")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent dimension for the VAE")
    parser.add_argument("--total_steps", type=int, default=30000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    args = parser.parse_args()

    train_vae(train_csv=args.train_csv,
              val_csv=args.val_csv,
              num_frames=args.num_frames,
              img_size=args.img_size,
              latent_dim=args.latent_dim,
              total_steps=args.total_steps,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate)
