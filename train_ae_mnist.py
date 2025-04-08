import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist


# === AE Definition ===
class Encoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        z = nn.Dense(self.latent_dim)(x)
        return z

class Decoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(7 * 7 * 64)(z)
        x = x.reshape((-1, 7, 7, 64))
        x = nn.ConvTranspose(64, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x)
        x = nn.ConvTranspose(32, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x)
        x = nn.ConvTranspose(1, (4, 4), strides=(1, 1), padding='SAME')(x)
        return x

class AE(nn.Module):
    latent_dim: int
    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)
    def __call__(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# === Loss Function ===
def compute_ae_loss(x, recon):
    recon_loss = jnp.mean((x - recon) ** 2)
    return recon_loss

# === Load MNIST Dataset ===
def load_mnist():
    (train_x, _), (test_x, _) = mnist.load_data()
    train_x = train_x.astype(np.float32) / 255.0
    test_x = test_x.astype(np.float32) / 255.0
    train_x = np.expand_dims(train_x, -1)  # Shape: (N, 28, 28, 1)
    test_x = np.expand_dims(test_x, -1)
    return train_x, test_x

# === Sample batch ===
def get_batch(dataset, indices):
    return dataset[indices]

# === Visualize ===
def show_recon(input_batch, recon_batch, prefix="recon", n=5):
    input_batch = jnp.clip(input_batch, 0.0, 1.0)
    recon_batch = jnp.clip(recon_batch, 0.0, 1.0)
    for i in range(n):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.squeeze(input_batch[i]), cmap='gray')
        ax[0].set_title("Input")
        ax[0].axis('off')
        ax[1].imshow(np.squeeze(recon_batch[i]), cmap='gray')
        ax[1].set_title("Reconstruction")
        ax[1].axis('off')
        plt.savefig(f"{prefix}_{i}.png")
        plt.close(fig)

# === Evaluation Step ===
@jit
def eval_step(params, batch):
    recon = AE(latent_dim).apply({'params': params}, batch)
    loss = compute_ae_loss(batch, recon)
    return loss, recon

# === Training Loop ===
def train_ae():
    global latent_dim
    latent_dim = 32
    learning_rate = 1e-3
    batch_size = 128
    total_steps = 10000

    rng = jax.random.PRNGKey(0)
    train_x, test_x = load_mnist()
    train_x = jnp.array(train_x)
    test_x = jnp.array(test_x)

    # Init model and optimizer
    ae = AE(latent_dim)
    dummy_input = jnp.ones((1, 28, 28, 1), dtype=jnp.float32)
    params = ae.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    train_losses = []
    test_losses = []

    # Shuffle training set once to avoid overlap with test data
    train_rng, rng = jax.random.split(rng)
    train_perm = jax.random.permutation(train_rng, train_x.shape[0])
    train_x = train_x[train_perm]

    # Training epoch-style index
    train_index = 0

    @jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            recon = AE(latent_dim).apply({'params': p}, batch)
            loss = compute_ae_loss(batch, recon)
            return loss, recon
        (loss, recon), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, recon

    for step in range(1, total_steps + 1):
        # Epoch-style non-repeating batching
        if train_index + batch_size > train_x.shape[0]:
            # Reshuffle and reset
            shuffle_rng, rng = random.split(rng)
            train_perm = jax.random.permutation(shuffle_rng, train_x.shape[0])
            train_x = train_x[train_perm]
            train_index = 0

        batch = train_x[train_index:train_index + batch_size]
        train_index += batch_size

        rng, step_rng = random.split(rng)
        test_indices = jax.random.randint(step_rng, (batch_size,), 0, test_x.shape[0])
        test_batch = get_batch(test_x, test_indices)

        params, opt_state, loss, recon = train_step(params, opt_state, batch)
        test_loss, test_recon = eval_step(params, test_batch)

        train_losses.append(float(loss))
        test_losses.append(float(test_loss))

        if step % 100 == 0:
            print(f"[Step {step}] Train Loss: {loss:.6f} | Test Loss: {test_loss:.6f}")
            show_recon(batch, recon, prefix="train_recon")
            show_recon(test_batch, test_recon, prefix="test_recon")

    # === Plot loss curves ===
    df = pd.DataFrame({'train': train_losses, 'test': test_losses})
    ema = df.ewm(span=100).mean()
    plt.plot(df['train'], label='Raw Train Loss', alpha=0.5)
    plt.plot(df['test'], label='Raw Test Loss', alpha=0.5)
    plt.plot(ema['train'], label='EMA Train Loss')
    plt.plot(ema['test'], label='EMA Test Loss')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve with EMA and Raw Loss")
    plt.savefig("mnist_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    train_ae()
