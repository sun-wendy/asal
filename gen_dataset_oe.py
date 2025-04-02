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


def calc_open_endedness_score(z):
    """
    Calculates the open-endedness score from ASAL.
    The returned score should be minimized.
    z: jnp.ndarray of shape (T, D)
    """
    kernel = (z @ z.T)
    kernel = jnp.tril(kernel, k=-1)
    return kernel.max(axis=-1).mean()


def build_oe_score_dataset(params_path='illumination_lenia.npz',
                           substrate_name='lenia',
                           save_dir='dataset',
                           time_sampling=50,
                           img_size=224,
                           seed=42):
    """
    Given a .npz file of substrate parameters, generate a dataset of
    (params, open-endedness score) pairs and save to .pkl files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load params
    raw_data = np.load(params_path)
    all_params = raw_data['params']
    print(f"Loaded {len(all_params)} parameter sets from {params_path}")

    # Init substrate and CLIP model
    fm = create_foundation_model('clip')
    substrate = create_substrate(substrate_name)

    # Setup rollout fn
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

    master_rng = jax.random.PRNGKey(seed)
    param_list = []
    oe_score_list = []

    print(f"Generating OE scores for {len(all_params)} samples...")
    for i, param_vec in enumerate(tqdm(all_params)):
        master_rng, sim_rng = jax.random.split(master_rng)
        try:
            param_jax = jnp.array(param_vec)
            rollout_result = rollout_fn(sim_rng, param_jax)
            z = rollout_result['z']  # (T, D)
            score = float(calc_open_endedness_score(z))
            param_list.append(np.array(param_vec))
            oe_score_list.append(score)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Stack data
    params_array = np.stack(param_list, axis=0)
    oe_array = np.array(oe_score_list)

    print(f"Generated {len(params_array)} valid (param, OE score) pairs")

    # Split into train/test
    split_idx = int(0.8 * len(params_array))
    train_data = {
        'params': params_array[:split_idx],
        'oe_scores': oe_array[:split_idx]
    }
    test_data = {
        'params': params_array[split_idx:],
        'oe_scores': oe_array[split_idx:]
    }

    # Save
    train_path = os.path.join(save_dir, f'{substrate_name}_train_oe_score_data.pkl')
    test_path = os.path.join(save_dir, f'{substrate_name}_test_oe_score_data.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Saved training data to {train_path}")
    print(f"Saved test data to {test_path}")
    print(f"  Train size: {len(train_data['params'])}")
    print(f"  Test size: {len(test_data['params'])}")


if __name__ == "__main__":
    build_oe_score_dataset(params_path="illumination_lenia.npz")
