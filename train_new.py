import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class MLPRegressor(nn.Module):
    hidden_dims: list
    output_dim: int
    dropout_rate: float = 0.0
    train: bool = True

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.zeros
            )(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not self.train)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros
        )(x)
        return x

class TrainState(train_state.TrainState):
    pass

def load_dataset(path, name):
    with open(f"{path}/{name}_data.pkl", "rb") as f:
        return pickle.load(f)

def interpolate_batch(x_batch, y_batch):
    n = x_batch.shape[0]
    idx1, idx2 = jnp.triu_indices(n, k=1)
    x1, x2 = x_batch[idx1], x_batch[idx2]
    y1, y2 = y_batch[idx1], y_batch[idx2]
    x_interp = (x1 + x2) / 2.0
    y_interp = (y1 + y2) / 2.0
    return x_interp, y_interp

@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        preds = state.apply_fn(params, x_batch)
        x_interp, y_interp = interpolate_batch(x_batch, y_batch)
        preds_interp = state.apply_fn(params, x_interp)
        loss = jnp.mean((preds - y_batch) ** 2)
        interp_loss = jnp.mean((preds_interp - y_interp) ** 2)
        return (loss + interp_loss) / 2.0

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, x_batch, y_batch):
    preds = state.apply_fn(state.params, x_batch)
    x_interp, y_interp = interpolate_batch(x_batch, y_batch)
    preds_interp = state.apply_fn(state.params, x_interp)
    loss = jnp.mean((preds - y_batch) ** 2)
    interp_loss = jnp.mean((preds_interp - y_interp) ** 2)
    return (loss + interp_loss) / 2.0

def cosine_similarity(a, b):
    a_norm = jnp.linalg.norm(a, axis=1, keepdims=True)
    b_norm = jnp.linalg.norm(b, axis=1, keepdims=True)
    dot_product = jnp.sum(a * b, axis=1)
    similarity = dot_product / (a_norm.squeeze() * b_norm.squeeze())
    return similarity

def train_model(dataset, hidden_dims, lr, steps, batch_size, seed):
    model = MLPRegressor(hidden_dims=hidden_dims, output_dim=dataset['clip_dim'])
    rng = jax.random.PRNGKey(seed)
    sample_input = jnp.array(dataset['train']['params'][0:1])
    params = model.init(rng, sample_input)
    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_X = jnp.array(dataset['train']['params'])
    train_y = jnp.array(dataset['train']['final_vectors'])
    test_X = jnp.array(dataset['test']['params'])
    test_y = jnp.array(dataset['test']['final_vectors'])

    train_losses = []
    test_losses = []

    for step in range(steps):
        rng, step_rng = jax.random.split(rng)
        train_idx = jax.random.choice(step_rng, len(train_X), shape=(batch_size,), replace=False)
        test_idx = jax.random.choice(step_rng, len(test_X), shape=(batch_size,), replace=False)

        x_train = train_X[train_idx]
        y_train = train_y[train_idx]
        x_test = test_X[test_idx]
        y_test = test_y[test_idx]

        state, train_loss = train_step(state, x_train, y_train)
        test_loss = eval_step(state, x_test, y_test)

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{steps}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return state, model, train_losses, test_losses

def evaluate_model(state, dataset):
    test_X = jnp.array(dataset['test']['params'])
    test_y = jnp.array(dataset['test']['final_vectors'])
    predictions = state.apply_fn(state.params, test_X)

    mse = jnp.mean((predictions - test_y) ** 2)
    similarities = cosine_similarity(predictions, test_y)
    avg_similarity = jnp.mean(similarities)

    print(f"\nFinal Test MSE: {mse:.4f}")
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

    return {
        'mse': float(mse),
        'cosine_similarity': float(avg_similarity)
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--substrate_name', type=str, default='lenia')
    parser.add_argument('--hidden_dims', type=str, default='512,512,512')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--save_plot', type=str, default='loss_plot.png')
    args = parser.parse_args()

    dataset = {
        'train': load_dataset(args.dataset_dir, f"{args.substrate_name}_train"),
        'test': load_dataset(args.dataset_dir, f"{args.substrate_name}_test"),
    }
    dataset['clip_dim'] = dataset['train']['final_vectors'].shape[1]
    dataset['params_dim'] = dataset['train']['params'].shape[1]

    clip_train = dataset['train']['final_vectors']
    clip_mean = clip_train.mean(axis=0)
    clip_std = clip_train.std(axis=0) + 1e-6

    dataset['train']['final_vectors'] = (clip_train - clip_mean) / clip_std
    dataset['test']['final_vectors'] = (dataset['test']['final_vectors'] - clip_mean) / clip_std

    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

    print("\nTraining model with interpolation on both train and test...")
    state, model, train_losses, test_losses = train_model(
        dataset,
        hidden_dims=hidden_dims,
        lr=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=42
    )

    print("\nEvaluating final model...")
    results = evaluate_model(state, dataset)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss (with Interpolation)")
    plt.title("Train & Test Loss during Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Loss curve saved to {args.save_plot}")
