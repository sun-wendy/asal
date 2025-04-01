import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation_scale
import asal_metrics
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='boids', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, leave None for the default of the substrate")


group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use (don't touch this)")

group = parser.add_argument_group("optimization")
group.add_argument("--k_nbrs", type=int, default=2, help="k_neighbors for nearest neighbor calculation (2 is best)")
group.add_argument("--n_child", type=int, default=32, help="number of children to generate")
group.add_argument("--pop_size", type=int, default=256, help="population size for the genetic algorithm")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--time_sampling", type=int, default=50, help="number of time steps to sample from the rollout")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    print(args)

    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps
    rollout_fn_ = partial(rollout_simulation_scale, s0=None, substrate=substrate, fm=fm, rollout_steps=args.rollout_steps, time_sampling=args.time_sampling, img_size=224, return_state=False)
    
    # Modified rollout function to calculate and include open-endedness score
    def _rollout_with_oe(rng, p):
        result = rollout_fn_(rng, p)
        oe_score = asal_metrics.calc_open_endedness_score(result['z'])
        return dict(params=p, oe_score=oe_score, **result)
    
    rollout_fn = jax.jit(_rollout_with_oe)

    rng = jax.random.PRNGKey(args.seed)

    rng, _rng = split(rng)
    params_init = 0.*jax.random.normal(_rng, (args.pop_size, substrate.n_params))
    pop = [rollout_fn(_rng, p) for p in tqdm(params_init)]
    pop = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *pop)

    @jax.jit
    def do_iter(pop, rng): # do one iteration of the optimization
        rng, _rng = split(rng)
        idx_p = jax.random.randint(_rng, (args.n_child, ), minval=0, maxval=args.pop_size) # randomly sample parent indices
        params_parent = pop['params'][idx_p]  # bs D
        rng, _rng1, _rng2 = split(rng, 3)
        noise = jax.random.normal(_rng1, (args.n_child, substrate.n_params))
        params_children = params_parent + args.sigma * noise  # mutate parents to get children

        rng, _rng = split(rng)
        children = jax.vmap(rollout_fn)(split(_rng, args.n_child), params_children) # rollout the children params to their latent representations

        pop = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *[pop, children]) # concat them all together into one big pool

        X = pop['z']
        X_last = X[:, -1, :] # (pop_size+bs) D
        D = -X_last@X_last.T # (pop_size+bs) (pop_size+bs) # calculate the negative similarity between all pairs of latent representations
        D = D.at[jnp.arange(args.pop_size+args.n_child), jnp.arange(args.pop_size+args.n_child)].set(jnp.inf) # set diagonal to inf

        # Find 32 disjoint closest pairs based on the distance matrix D
        # For each pair, we'll remove the one with higher open-endedness score (lower diversity)
        
        num_pairs_to_find = args.n_child  # We need 32 pairs
        n_total = args.pop_size + args.n_child
        
        # Initialize mask to track which individuals have been paired
        mask = jnp.ones((n_total, n_total), dtype=bool)
        # Set diagonal to False (can't pair with self)
        mask = mask.at[jnp.arange(n_total), jnp.arange(n_total)].set(False)
        
        # Initialize arrays to store pairs
        closest_i = jnp.zeros(num_pairs_to_find, dtype=jnp.int32)
        closest_j = jnp.zeros(num_pairs_to_find, dtype=jnp.int32)
        
        def find_closest_pair(carry, idx):
            D_masked, mask, closest_i, closest_j = carry
            
            # Find indices of closest pair in the masked distance matrix
            flat_idx = jnp.argmin(jnp.where(mask, D_masked, jnp.inf))
            i, j = jnp.unravel_index(flat_idx, D_masked.shape)
            
            # Store the pair
            closest_i = closest_i.at[idx].set(i)
            closest_j = closest_j.at[idx].set(j)
            
            # Update mask to exclude these indices in future iterations
            # Neither i nor j can be used in future pairs
            mask_i = mask.at[:, i].set(False)  # Can't pair with i
            mask_j = mask_i.at[:, j].set(False)  # Can't pair with j
            mask_updated = mask_j.at[i, :].set(False).at[j, :].set(False)  # i and j can't be paired with anything
            
            return (D_masked, mask_updated, closest_i, closest_j), None
        
        # Iteratively find the closest pairs while ensuring they're disjoint
        (_, _, closest_i, closest_j), _ = jax.lax.scan(
            find_closest_pair, 
            (D, mask, closest_i, closest_j), 
            jnp.arange(num_pairs_to_find)
        )
        
        # Compare open-endedness scores to decide which one to remove from each pair
        oe_scores = pop['oe_score']
        
        # Initialize array to track individuals to kill
        to_kill = jnp.zeros(args.n_child, dtype=jnp.int32)
        
        def process_pair(carry, pair_idx):
            to_kill, idx = carry
            i, j = closest_i[pair_idx], closest_j[pair_idx]
            
            # Compare open-endedness scores and remove the one with higher score
            # (higher score means less open-ended in this implementation)
            to_remove = jnp.where(oe_scores[i] > oe_scores[j], i, j)
            to_kill = to_kill.at[idx].set(to_remove)
            
            return (to_kill, idx + 1), None
        
        (to_kill, _), _ = jax.lax.scan(process_pair, (to_kill, 0), jnp.arange(num_pairs_to_find))
        
        # Keep everyone except those in to_kill
        to_keep = jnp.setdiff1d(jnp.arange(args.pop_size+args.n_child), to_kill, assume_unique=True, size=args.pop_size)

        pop = jax.tree.map(lambda x: x[to_keep], pop) # these are the ones that survived
        D = D[to_keep, :][:, to_keep]

        # Calculate metrics
        illumination_loss = asal_metrics.calc_illumination_score(pop['z'][:, -1, :]) # calculate the illumination score
        oe_score = jnp.mean(pop['oe_score'])  # Use cached scores
        return pop, dict(illumination_loss=illumination_loss, oe_score=oe_score)

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        pop, di = do_iter(pop, rng)

        data.append(di)
        pbar.set_postfix(iloss=di['illumination_loss'].item(), oe_score=di['oe_score'].item())
        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1): # save data every 10% of the run
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)

            # print(jax.tree_map(lambda x: x.shape, pop))
            util.save_pkl(args.save_dir, "pop", jax.tree.map(lambda x: np.array(x), pop))
            
if __name__ == '__main__':
    main(parse_args())
