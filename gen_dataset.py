import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os
import pickle
from tqdm import tqdm

from substrates import create_substrate
from foundation_models import create_foundation_model
from rollout import rollout_simulation


def build_clip_dataset(params_path='illumination_lenia.npz',
                       substrate_name='lenia',
                       save_dir='dataset',
                       time_sampling='final',
                       img_size=224,
                       seed=42):
    """
    Given a .npz file of substrate parameters, generate a dataset of
    (params, final CLIP vector) pairs and save to a .pkl file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load parameters
    raw_data = np.load(params_path)
    all_params = raw_data['params']
    print(f"Loaded {len(all_params)} parameter sets from {params_path}")

    # Create substrate and foundation model
    fm = create_foundation_model('clip')
    substrate = create_substrate(substrate_name)

    # Setup rollout function
    rollout_fn = partial(
        rollout_simulation,
        substrate=substrate,
        fm=fm,
        rollout_steps=substrate.rollout_steps,
        time_sampling=time_sampling,
        img_size=img_size,
        return_state=False
    )
    rollout_fn = jax.jit(rollout_fn)

    # Generate dataset
    master_rng = jax.random.PRNGKey(seed)

    clip_vectors = []
    param_list = []

    for i, param_vec in enumerate(tqdm(all_params)):
        master_rng, sim_rng = jax.random.split(master_rng)
        try:
            # reshape param back to JAX form
            param_jax = jnp.array(param_vec)
            rollout_result = rollout_fn(sim_rng, param_jax)
            z = np.array(rollout_result['z'])  # final CLIP embedding
            clip_vectors.append(z)
            param_list.append(param_vec)
        except Exception as e:
            print(f"Error on index {i}: {e}")
            continue

    # Stack and save
    params_array = np.stack(param_list, axis=0)
    clip_array = np.stack(clip_vectors, axis=0)

    print(f"Generated {len(params_array)} valid (param, clip) pairs")

    # Train/test split
    split_idx = int(len(params_array) * 0.8)
    train_data = {
        'params': params_array[:split_idx],
        'final_vectors': clip_array[:split_idx],
    }
    test_data = {
        'params': params_array[split_idx:],
        'final_vectors': clip_array[split_idx:],
    }

    # Save to disk
    train_path = os.path.join(save_dir, f'{substrate_name}_train_data.pkl')
    test_path = os.path.join(save_dir, f'{substrate_name}_test_data.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Saved train to {train_path}")
    print(f"Saved test to {test_path}")


if __name__ == "__main__":
    build_clip_dataset(params_path="illumination_lenia.npz")
