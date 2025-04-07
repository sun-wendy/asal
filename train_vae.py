import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# === Loss Function ===
def compute_vae_loss(x, recon, mu, logvar):
    # recon_loss = jnp.mean((x - recon) ** 2)
    # kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
    bce = -jnp.mean(x * jnp.log(recon + 1e-6) + (1 - x) * jnp.log(1 - recon + 1e-6))
    return bce  # recon_loss + 0.0 * kl


# === Generate a new dataset of frames per step ===
def generate_full_dataset(rng, substrate, num_rollouts=512, rollout_steps=256, img_size=64):
    rollout_rngs = jax.random.split(rng, num_rollouts)
    params = jax.vmap(substrate.default_params)(rollout_rngs)

    def rollout_fn(rng_i, param_i):
        result = rollout_simulation(rng_i, param_i, substrate=substrate, fm=None,
                                    rollout_steps=rollout_steps, time_sampling='video',
                                    img_size=img_size, return_state=False)
        return result['rgb']

    vids = jax.vmap(rollout_fn)(rollout_rngs, params)
    vids = vids.reshape(-1, img_size, img_size, 3)
    return vids

# === Sample batch from full dataset ===
def get_batch_from_dataset(dataset, rng, batch_size):
    indices = jax.random.randint(rng, (batch_size,), 0, dataset.shape[0])
    return dataset[indices]


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


# === Evaluation Step ===
@jit
def eval_step(params, batch, rng):
    recon, mu, logvar = VAE(latent_dim).apply({'params': params}, batch, rng)
    loss = compute_vae_loss(batch, recon, mu, logvar)
    return loss, recon

# === Training Loop ===
def train_vae():
    global latent_dim
    latent_dim = 128
    learning_rate = 1e-3
    batch_size = 1024
    img_size = 64
    total_steps = 30000

    rng = jax.random.PRNGKey(0)
    substrate = substrates.create_substrate("gol")
    substrate = substrates.FlattenSubstrateParameters(substrate)

    # Init model and optimizer
    vae = VAE(latent_dim)
    dummy_input = jnp.ones((1, img_size, img_size, 3), dtype=jnp.float32)
    params = vae.init(rng, dummy_input, rng)['params']
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    # Create fixed test dataset
    test_rng = random.PRNGKey(42)
    test_dataset = generate_full_dataset(test_rng, substrate, num_rollouts=128, rollout_steps=256, img_size=img_size)

    train_losses = []
    test_losses = []

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

    for step in range(1, total_steps + 1):
        rng, step_rng = random.split(rng)
        dataset_rng, step_rng = random.split(step_rng)
        dataset = generate_full_dataset(dataset_rng, substrate, num_rollouts=128, rollout_steps=256, img_size=img_size)
        batch = get_batch_from_dataset(dataset, step_rng, batch_size)

        params, opt_state, loss, recon = train_step(params, opt_state, batch, step_rng)

        test_batch = get_batch_from_dataset(test_dataset, step_rng, batch_size)
        test_loss, test_recon = eval_step(params, test_batch, step_rng)

        train_losses.append(float(loss))
        test_losses.append(float(test_loss))

        if step % 100 == 0:
            print(f"[Step {step}] Train Loss: {loss:.6f} | Test Loss: {test_loss:.6f}")
            show_recon(batch, recon, prefix="train_recon")
            show_recon(test_batch, test_recon, prefix="test_recon")

    # === Plot loss curves ===
    df = pd.DataFrame({'train': train_losses, 'test': test_losses})
    ema = df.ewm(span=1000).mean()
    plt.plot(ema['train'], label='EMA Train Loss')
    plt.plot(ema['test'], label='EMA Test Loss')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Exponential Moving Average of Loss")
    plt.savefig("smoothed_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    train_vae()
