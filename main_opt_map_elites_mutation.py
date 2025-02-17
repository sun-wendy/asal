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
group.add_argument("--substrate", type=str, default='lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--prompts", type=str, default="noun_list.txt", help="prompt file")
group.add_argument("--init_iters", type=int, default=1000, help="number of random initialization iterations (G)")
group.add_argument("--time_sampling", type=int, default=32, help="number of frames to sample")

group = parser.add_argument_group("optimization")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--total_iters", type=int, default=10000, help="total number of iterations (I)")
group.add_argument("--pop_size", type=int, default=16, help="number of new solutions each iteration")
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
    """
    MAP-Elites array-based storage.
    p_best: shape [n_prompts], best performance per prompt (initially -inf)
    x_best: shape [n_prompts, solution_dim], best solution per prompt
    """
    def __init__(self, n_prompts, solution_dim):
        self.n_prompts = n_prompts
        self.solution_dim = solution_dim
        self.p_best = jnp.full((n_prompts,), -jnp.inf)
        self.x_best = jnp.zeros((n_prompts, solution_dim))

    def random_solution(self, rng):
        """Generate a random solution."""
        return normal(rng, (self.solution_dim,))

    def random_selection(self, rng):
        """Randomly select an elite from among the filled bins, if any exist."""
        # Indices where niche is non-empty
        filled_indices = jnp.where(self.p_best > -jnp.inf, size=self.n_prompts)[0]
        filled_count = filled_indices.shape[0]

        def no_filled():
            # If no filled niche, return a random solution.
            return self.random_solution(rng)

        def some_filled():
            idx = jax.random.choice(rng, filled_indices)
            return self.x_best[idx]

        return jax.lax.cond(filled_count == 0, no_filled, some_filled)

# Standard Gaussian mutation function (if needed elsewhere)
@jax.jit
def random_variation(rng, x, sigma):
    """Create a randomly modified copy of x."""
    return x + normal(rng, x.shape) * sigma

def diversity_mutation(rng, map_elites, sigma):
    """
    Mutation procedure that probabilistically applies crossover between two elites,
    then adds Gaussian noise. With probability 0.5, two elites are chosen and combined;
    otherwise, a single elite is perturbed.
    """
    # Split rng for different operations.
    rng_choice, rng_parent1, rng_parent2, rng_noise = split(rng, 4)
    # Decide if we do crossover or just mutation (50% chance)
    do_crossover = jax.random.uniform(rng_choice) < 0.5

    def crossover(_):
        parent1 = map_elites.random_selection(rng_parent1)
        parent2 = map_elites.random_selection(rng_parent2)
        # Random interpolation weight between 0 and 1.
        alpha = jax.random.uniform(rng_choice, shape=())
        return alpha * parent1 + (1 - alpha) * parent2

    def simple_mutation(_):
        return map_elites.random_selection(rng_parent1)

    # Use crossover if selected, otherwise simple mutation.
    child = jax.lax.cond(do_crossover, crossover, simple_mutation, operand=None)
    # Add Gaussian noise to encourage further exploration.
    return child + normal(rng_noise, child.shape) * sigma


def main(args):
    # Parse user prompts
    prompts = util.read_prompt_file(args.prompts)
    n_prompts = len(prompts)
    print(f"Optimizing for {n_prompts} prompts: {prompts}")
    
    # Create foundation model and substrate
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    # Jit-compiled rollout function
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
    rollout_jit = jax.jit(rollout_fn)

    # Precompute text embeddings.
    z_prompts = fm.embed_txt(prompts)  # shape: (n_prompts, embed_dim)

    @jax.jit
    def performance(rng, x, prompt_idx):
        """Roll out x and compute alignment with a particular prompt."""
        rollout_data = rollout_jit(rng, x)
        final_embedding = rollout_data['z'][-1]
        return jnp.dot(final_embedding, z_prompts[prompt_idx])

    # Initialize MAP-Elites archive.
    map_elites = MapElites(n_prompts, substrate.n_params)
    rng = PRNGKey(args.seed)

    data = []
    pbar = tqdm(range(args.total_iters))

    for iter_i in pbar:
        rng, step_rng = split(rng)

        # Produce a batch of new solutions of size pop_size.
        if iter_i < args.init_iters:
            # Initialization phase: sample random solutions.
            def sample_new(rng_input):
                return map_elites.random_solution(rng_input)
        else:
            # Variation phase: use diversity-aware mutation with crossover.
            def sample_new(rng_input):
                return diversity_mutation(rng_input, map_elites, args.sigma)

        rng, rng_pop = split(rng)
        rng_batch = jax.random.split(rng_pop, args.pop_size)
        pop_solutions = jax.vmap(sample_new)(rng_batch)  # shape: [pop_size, solution_dim]

        # Evaluate the entire batch on all prompts.
        rng, eval_rng = split(rng)

        def eval_one_candidate(x_candidate):
            return jax.vmap(lambda idx: performance(eval_rng, x_candidate, idx))(
                jnp.arange(n_prompts)
            )
        p_all = jax.vmap(eval_one_candidate)(pop_solutions)  # shape: [pop_size, n_prompts]

        # Find the best solution for each prompt.
        best_idx_for_prompt = jnp.argmax(p_all, axis=0)  # [n_prompts]
        best_val_for_prompt = jnp.max(p_all, axis=0)       # [n_prompts]
        # Gather the solutions that produce these best values.
        occupant_new = pop_solutions[best_idx_for_prompt, :]  # [n_prompts, solution_dim]
        # Check which prompts get improved.
        better_mask = best_val_for_prompt > map_elites.p_best
        # Update p_best & x_best.
        map_elites.p_best = jnp.where(better_mask, best_val_for_prompt, map_elites.p_best)
        map_elites.x_best = jnp.where(better_mask[:, None], occupant_new, map_elites.x_best)

        # Log data.
        di = {
            'best_fitness': map_elites.p_best,
            'total_solutions': jnp.sum(map_elites.p_best > -jnp.inf)
        }
        data.append(di)
        pbar.set_postfix(avg_best_fitness=float(di['best_fitness'].mean()))

        # Save data periodically.
        if (args.save_dir is not None 
            and (iter_i % (args.total_iters // 10) == 0 or iter_i == args.total_iters - 1)):
            data_save = jax.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = (np.array(map_elites.x_best), np.array(map_elites.p_best))
            util.save_pkl(args.save_dir, "best", best)

    return map_elites.x_best, map_elites.p_best


if __name__ == '__main__':
    args = parse_args()
    solutions, performances = main(args)
