import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial
import jax
import jax.numpy as jnp
from jax.random import split, normal, PRNGKey
import numpy as np
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='boids', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--prompts", type=str, default="three cells;a red crowd;a green organism", help="prompts to optimize for")
group.add_argument("--init_iters", type=int, default=1000, help="number of random initialization iterations (G)")
group.add_argument("--time_sampling", type=int, default=32, help="number of frames to sample")

group = parser.add_argument_group("optimization")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--total_iters", type=int, default=10000, help="total number of iterations (I)")
group.add_argument("--bs", type=int, default=1, help="number of init states to average over")
group.add_argument("--n_iters", type=int, default=None, help="alias for total_iters")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    
    # Handle n_iters alias for total_iters
    if args.n_iters is not None:
        args.total_iters = args.n_iters
    
    return args


class MapElites:
    """MAP-Elites algorithm following pseudocode exactly."""
    def __init__(self, solution_dim):
        self.solution_dim = solution_dim
        # Empty maps P and X
        self.P = {}  # Performance map
        self.X = {}  # Solution map
    
    def random_solution(self, rng):
        """Generate random solution"""
        return normal(rng, (self.solution_dim,))
    
    def random_selection(self, rng):
        """Randomly select elite from map X"""
        if not self.X:  # If map is empty
            return self.random_solution(rng)
        # Choose random key from X
        keys = list(self.X.keys())
        idx = jax.random.randint(rng, (), 0, len(keys))
        return self.X[keys[idx]]


@jax.jit
def random_variation(rng, x, sigma):
    """Create randomly modified copy of x via mutation (jitted)."""
    return x + normal(rng, x.shape) * sigma


def main(args):
    # Parse the user prompts
    prompts = args.prompts.split(";")
    n_prompts = len(prompts)
    print(f"Optimizing for {n_prompts} prompts: {prompts}")
    
    # Create the foundation model and substrate
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    # A partial of the rollout to fix substrate, foundation model, etc.
    # So the only dynamic arguments to this function are (rng, x).
    rollout_fn = partial(
        rollout_simulation, 
        s0=None, 
        substrate=substrate, 
        fm=fm, 
        rollout_steps=args.rollout_steps, 
        time_sampling=(args.time_sampling, True),  # Sample frames evenly
        img_size=224, 
        return_state=False
    )

    # JIT the rollout to avoid retracing on every call
    rollout_jit = jax.jit(rollout_fn)

    # Precompute text embeddings for prompts
    z_prompts = fm.embed_txt(prompts)  # shape: (P, embed_dim)

    @jax.jit
    def feature_descriptor(rng, x):
        """
        Simulate candidate solution x (jitted) and return a discrete descriptor:
        'which prompt does it best match?'
        """
        rollout_data = rollout_jit(rng, x)  
        final_embedding = rollout_data['z'][-1]  # shape: (embed_dim,)
        # Similarities to each prompt
        similarities = jnp.dot(final_embedding, z_prompts.T)  # shape: (P,)
        return jnp.argmax(similarities)  # int index of best matching prompt
    
    @jax.jit
    def performance(rng, x, prompt_idx):
        """
        Simulate candidate solution x (jitted) and return its performance with
        respect to a *particular* prompt.
        """
        rollout_data = rollout_jit(rng, x)
        final_embedding = rollout_data['z'][-1]
        return jnp.dot(final_embedding, z_prompts[prompt_idx])

    # Initialize MAP-Elites
    map_elites = MapElites(substrate.n_params)
    rng = PRNGKey(args.seed)
    data = []

    pbar = tqdm(range(args.total_iters))
    for iter_i in pbar:
        rng, step_rng = split(rng)
        
        # Generate solution
        if iter_i < args.init_iters:  
            # Initialization phase: generate random solutions
            rng, _rng = split(step_rng)
            x_prime = map_elites.random_solution(_rng)
        else:
            # Variation phase: pick from map and mutate
            rng, _rng1, _rng2 = split(step_rng, 3)
            x = map_elites.random_selection(_rng1)
            x_prime = random_variation(_rng2, x, args.sigma)
        
        # Evaluate solution
        rng, _rng1, _rng2 = split(rng, 3)
        b_prime = feature_descriptor(_rng1, x_prime)  # which prompt is best matched
        p_prime = performance(_rng2, x_prime, b_prime)  # performance for that prompt

        # If the cell for that feature descriptor is empty or improved, store new elite
        b_prime_int = int(b_prime)  # convert from JAX int to Python int
        if (b_prime_int not in map_elites.P) or (map_elites.P[b_prime_int] < p_prime):
            map_elites.P[b_prime_int] = p_prime
            map_elites.X[b_prime_int] = x_prime
        
        # Bookkeeping for logging
        di = {
            'best_fitness': jnp.array([
                map_elites.P.get(i, float('-inf')) for i in range(n_prompts)
            ]),
            'total_solutions': len(map_elites.X),
        }
        data.append(di)
        pbar.set_postfix(avg_best_fitness=float(di['best_fitness'].mean()))
        
        # Save data periodically
        if (args.save_dir is not None 
            and (iter_i % (args.total_iters // 10) == 0 or iter_i == args.total_iters - 1)):
            # Convert all data so far into numpy arrays
            data_save = jax.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            
            # Save best solutions and their fitness for each prompt
            best = jax.tree_map(
                lambda x: np.array(x),
                (
                    {
                        i: map_elites.X.get(i, jnp.zeros(substrate.n_params))
                        for i in range(n_prompts)
                    },
                    {
                        i: map_elites.P.get(i, float('-inf'))
                        for i in range(n_prompts)
                    }
                )
            )
            util.save_pkl(args.save_dir, "best", best)

    return map_elites.X, map_elites.P


if __name__ == '__main__':
    args = parse_args()
    solutions, performances = main(args)
