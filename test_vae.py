import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from flax.core import freeze
import imageio
import os

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


# === Visualize ===
def show_recon(input_batch, recon_batch, prefix="recon", n=5):
    input_batch = jnp.clip(input_batch, 0.0, 1.0)
    recon_batch = jnp.clip(recon_batch, 0.0, 1.0)
    for i in range(n):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.array(input_batch[i]))
        ax[0].set_title("Input")
        ax[0].axis('off')
        ax[1].imshow(np.array(recon_batch[i]))
        ax[1].set_title("Reconstruction")
        ax[1].axis('off')
        plt.savefig(f"{prefix}_{i}.png")
        plt.close(fig)


# === Sample from Gaussian latent space ===
def sample_from_gaussian_latent(decoder, decoder_params, latent_dim=128, n=5, prefix="gaussian_sample"):
    import time
    rng = jax.random.PRNGKey(int(time.time()))  # or use time-based for variability
    z = jax.random.normal(rng, shape=(n, latent_dim))

    recon = decoder.apply({'params': freeze(decoder_params)}, z)
    recon = jnp.clip(recon, 0.0, 1.0)

    for i in range(n):
        fig, ax = plt.subplots()
        ax.imshow(np.array(recon[i]))
        ax.set_title("Sampled Decode")
        ax.axis('off')
        plt.savefig(f"{prefix}_{i}.png")
        plt.close(fig)

# Recreate model
latent_dim = 128  # or whatever value you used
vae = VAE(latent_dim)

# Load saved params
with open("encoder_params.pkl", "rb") as f:
    encoder_params = pickle.load(f)

with open("decoder_params.pkl", "rb") as f:
    decoder_params = pickle.load(f)

# sample_from_gaussian_latent(decoder, decoder_params, latent_dim=latent_dim, n=5)


# === Visualization ===
def show_recon_frame(input_img, recon_img, path):
    input_img = jnp.clip(input_img, 0.0, 1.0)
    recon_img = jnp.clip(recon_img, 0.0, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].imshow(np.array(input_img))
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(np.array(recon_img))
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")
    plt.savefig(path)
    plt.close(fig)

# === Run a single simulation ===
def generate_single_rollout(rng, substrate, rollout_steps=256, img_size=64):
    params = jnp.full(substrate.default_params(rng).shape, 6152)
    result = rollout_simulation(rng, params=params, substrate=substrate, fm=None,
                                 rollout_steps=rollout_steps, time_sampling='video',
                                 img_size=img_size, return_state=False)
    return result['rgb']  # shape: (256, H, W, 3)

import time
rng = jax.random.PRNGKey(int(time.time()))
substrate = substrates.create_substrate("gol")
substrate = substrates.FlattenSubstrateParameters(substrate)

frames = generate_single_rollout(rng, substrate)  # (256, 64, 64, 3)
frames = jnp.array(frames)
vae_params = {'encoder': encoder_params, 'decoder': decoder_params}
recon, _, _ = vae.apply({'params': vae_params}, frames, rng)

# === Save individual frames ===
output_folder = "simulation_recon_frames"
os.makedirs(output_folder, exist_ok=True)

for i in range(frames.shape[0]):
    img_path = os.path.join(output_folder, f"frame_{i:04d}.png")
    show_recon_frame(frames[i], recon[i], img_path)

print(f"Saved 256 recon frames to {output_folder}/")

# === Create video ===
video_path = "simulation_recon.mp4"
with imageio.get_writer(video_path, fps=20) as writer:
    for i in range(256):
        img = imageio.imread(os.path.join(output_folder, f"frame_{i:04d}.png"))
        writer.append_data(img)

print(f"Video saved to {video_path}")

