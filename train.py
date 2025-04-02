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
    train: bool = True  # For dropout

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

# Define a training state
class TrainState(train_state.TrainState):
    pass

# Utility to load dataset
def load_dataset(path, name):
    with open(f"{path}/{name}_data.pkl", "rb") as f:
        return pickle.load(f)

@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        preds = state.apply_fn(params, x_batch)
        return jnp.mean((preds - y_batch) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch, labels):
    preds = state.apply_fn(state.params, batch)
    return jnp.mean((preds - labels) ** 2)

def cosine_similarity(a, b):
    a_norm = jnp.linalg.norm(a, axis=1, keepdims=True)
    b_norm = jnp.linalg.norm(b, axis=1, keepdims=True)
    dot_product = jnp.sum(a * b, axis=1)
    similarity = dot_product / (a_norm.squeeze() * b_norm.squeeze())
    return similarity

def train_model(dataset, hidden_dims=[256, 256], lr=1e-3, epochs=100, batch_size=32, seed=42):
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

    for epoch in range(epochs):
        rng, step_rng = jax.random.split(rng)
        perm = jax.random.permutation(step_rng, len(train_X))
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]

        epoch_losses = []
        for i in range(0, len(train_X_shuffled), batch_size):
            x_batch = train_X_shuffled[i:i+batch_size]
            y_batch = train_y_shuffled[i:i+batch_size]
            state, loss = train_step(state, x_batch, y_batch)
            epoch_losses.append(loss)

        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))

        test_loss = float(eval_step(state, test_X, test_y))
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

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
    parser.add_argument('--hidden_dims', type=str, default='512,512,512,512,512,512,512,512')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_plot', type=str, default='loss_plot.png')
    args = parser.parse_args()

    # Load dataset
    dataset = {
        'train': load_dataset(args.dataset_dir, f"{args.substrate_name}_train"),
        'test': load_dataset(args.dataset_dir, f"{args.substrate_name}_test"),
    }
    dataset['clip_dim'] = dataset['train']['final_vectors'].shape[1]
    dataset['params_dim'] = dataset['train']['params'].shape[1]

    # Normalize CLIP targets
    clip_train = dataset['train']['final_vectors']
    clip_mean = clip_train.mean(axis=0)
    clip_std = clip_train.std(axis=0) + 1e-6  # avoid divide-by-zero

    dataset['train']['final_vectors'] = (clip_train - clip_mean) / clip_std
    dataset['test']['final_vectors'] = (dataset['test']['final_vectors'] - clip_mean) / clip_std

    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

    print(f"Dataset loaded:")
    print(f"  Training samples: {len(dataset['train']['params'])}")
    print(f"  Test samples: {len(dataset['test']['params'])}")
    print(f"  Parameter dimension: {dataset['params_dim']}")
    print(f"  CLIP vector dimension: {dataset['clip_dim']}")
    print(f"  Hidden dimensions: {hidden_dims}")

    print("\nTraining model...")
    state, model, train_losses, test_losses = train_model(
        dataset,
        hidden_dims=hidden_dims,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\nEvaluating model...")
    results = evaluate_model(state, dataset)

    # Plot training and test loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train/Test Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Loss curve saved to {args.save_plot}")
