import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from flax import linen as nn
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd


from substrates import create_substrate
from foundation_models import create_foundation_model
from rollout import rollout_simulation


class MLPRegressor(nn.Module):
    hidden_dims: list
    output_dim: int
    dropout_rate: float = 0.0
    train: bool = True

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not self.train)
        x = nn.Dense(self.output_dim)(x)
        return x


class TrainState(train_state.TrainState):
    pass


@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        preds = state.apply_fn(params, x_batch)
        return jnp.mean((preds - y_batch) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, x_batch, y_batch):
    preds = state.apply_fn(state.params, x_batch)
    return jnp.mean((preds - y_batch) ** 2)


def generate_batch(substrate, rollout_fn, rng, batch_size, param_dim):
    rng, param_rng, sim_rng = jax.random.split(rng, 3)
    params = jax.random.normal(param_rng, (batch_size, param_dim))

    def single_rollout(sim_rng, param):
        return rollout_fn(sim_rng, param)['z']

    sim_rngs = jax.random.split(sim_rng, batch_size)
    clip_vectors = jax.vmap(single_rollout)(sim_rngs, params)
    return rng, params, clip_vectors


def train_model(substrate_name, hidden_dims, lr, steps, batch_size, seed):
    rng = PRNGKey(seed)

    substrate = create_substrate(substrate_name)
    fm = create_foundation_model('clip')
    rollout_fn = partial(
        rollout_simulation,
        substrate=substrate,
        fm=fm,
        rollout_steps=substrate.rollout_steps,
        time_sampling='final',
        img_size=224,
        return_state=False
    )
    rollout_fn = jax.jit(rollout_fn)

    rng, init_rng = split(rng)
    param_dim = substrate.default_params(init_rng).shape[0]

    rng, norm_rng = split(rng)
    _, x_norm, y_norm = generate_batch(substrate, rollout_fn, norm_rng, 256, param_dim)
    clip_mean = jnp.mean(y_norm, axis=0)
    clip_std = jnp.std(y_norm, axis=0) + 1e-6

    model = MLPRegressor(hidden_dims=hidden_dims, output_dim=clip_mean.shape[0])
    sample_input = jnp.ones((1, param_dim))
    params = model.init(init_rng, sample_input)
    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses = []
    test_losses = []

    for step in range(steps):
        rng, train_rng, test_rng = jax.random.split(rng, 3)

        train_rng, x_batch, y_batch = generate_batch(substrate, rollout_fn, train_rng, batch_size, param_dim)
        y_batch = (y_batch - clip_mean) / clip_std
        state, train_loss = train_step(state, x_batch, y_batch)
        train_losses.append(float(train_loss))

        test_rng, x_test, y_test = generate_batch(substrate, rollout_fn, test_rng, batch_size, param_dim)
        y_test = (y_test - clip_mean) / clip_std
        test_loss = float(eval_step(state, x_test, y_test))
        test_losses.append(test_loss)

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{steps}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return state, model, train_losses, test_losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--substrate_name', type=str, default='lenia')
    parser.add_argument('--hidden_dims', type=str, default='512,512,512,512,512,512,512,512')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--save_plot', type=str, default='loss_plot.png')
    args = parser.parse_args()

    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

    print("Training model with dynamic sampling and normalized CLIP targets...")
    state, model, train_losses, test_losses = train_model(
        args.substrate_name,
        hidden_dims=hidden_dims,
        lr=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=42
    )

    # Apply exponential moving average (EMA) smoothing
    train_smooth = pd.DataFrame(train_losses).ewm(span=1000).mean()
    test_smooth = pd.DataFrame(test_losses).ewm(span=1000).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(train_smooth, label='Train Loss (EMA)')
    plt.plot(test_smooth, label='Test Loss (EMA)')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("Smoothed Train/Test Loss with Dynamic Sampling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Loss plot saved to {args.save_plot}")
