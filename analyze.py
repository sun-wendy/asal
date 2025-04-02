import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Import the necessary modules
from substrates import create_substrate
from foundation_models import create_foundation_model
from rollout import rollout_simulation

def show_final_images(substrate_name='lenia', num_samples=16, rows=4, dataset_path=None):
    """
    Show a simple grid of final simulation images.
    
    Parameters:
    -----------
    substrate_name : str
        Name of the substrate
    num_samples : int
        Number of samples to show
    rows : int
        Number of rows in the grid
    dataset_path : str
        Path to the dataset file
    """
    # Calculate columns
    cols = (num_samples + rows - 1) // rows
    
    # Initialize substrate and foundation model
    fm = create_foundation_model('clip')
    substrate = create_substrate(substrate_name)
    
    # Configure rollout function
    get_final_state = partial(
        rollout_simulation, 
        substrate=substrate, 
        fm=fm, 
        rollout_steps=substrate.rollout_steps,
        time_sampling='final', 
        img_size=224,
        return_state=False
    )
    
    # JIT compile
    get_final_state_jit = jax.jit(get_final_state)
    
    # Load dataset
    if dataset_path is None:
        dataset_path = f'dataset/{substrate_name}_test_data.pkl'
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get parameters
    params_array = data['params']
    
    # Sample indices
    if len(params_array) > num_samples:
        indices = np.random.choice(len(params_array), num_samples, replace=False)
    else:
        indices = range(len(params_array))
        num_samples = len(indices)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    # Generate and plot images
    for i, idx in enumerate(indices):
        print("here")
        if i >= len(axes):
            break
            
        # Get parameters
        params = jnp.array(params_array[idx])
        
        # Run simulation to get final state
        rng = PRNGKey(42 + idx)
        final_data = get_final_state_jit(rng, params)
        
        # Plot image
        axes[i].imshow(final_data['rgb'])
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{substrate_name}_final_images.png')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--substrate', type=str, default='lenia')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default=None)
    
    args = parser.parse_args()
    
    show_final_images(
        substrate_name=args.substrate,
        num_samples=args.samples,
        rows=args.rows,
        dataset_path=args.dataset_path
    )