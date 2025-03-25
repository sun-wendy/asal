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

        n_child_diversity = args.n_child // 2
        n_child_oe = args.n_child - n_child_diversity

        to_kill = jnp.zeros(args.n_child, dtype=int) # indices of pop to kill

        def kill_least(carry, _): # loop through and kill the individual which is "least novel" from the rest
            D, to_kill, i = carry

            tki = D.sort(axis=-1)[:, :args.k_nbrs].mean(axis=-1).argmin()
            D = D.at[:, tki].set(jnp.inf)
            D = D.at[tki, :].set(jnp.inf)
            to_kill = to_kill.at[i].set(tki)

            return (D, to_kill, i+1), None

        (D, to_kill_diversity, _), _ = jax.lax.scan(kill_least, (D, to_kill, 0), None, length=n_child_diversity) # do this loop {bs} times
        print("Killed least diverse")

        # Use cached open-endedness scores instead of recalculating
        open_endedness_scores = pop['oe_score']
        mask = jnp.ones(args.pop_size+args.n_child, dtype=jnp.bool_)
        mask = mask.at[to_kill_diversity].set(False)
        masked_scores = jnp.where(mask, open_endedness_scores, -jnp.inf)
        to_kill_openended = jnp.zeros(n_child_oe, dtype=jnp.int32)

        def kill_least_openended(carry, _):
            scores, killed, i = carry
            
            # Find individual with worst open-endedness score (highest value)
            tki = jnp.argmax(scores)
            scores = scores.at[tki].set(-jnp.inf)  # Mark as killed
            killed = killed.at[i].set(tki)
            
            return (scores, killed, i+1), None
        
        (_, to_kill_openended, _), _ = jax.lax.scan(kill_least_openended, (masked_scores, to_kill_openended, 0), None, length=n_child_oe)
        print("Killed least openended")

        to_kill_all = jnp.concatenate([to_kill_diversity, to_kill_openended])
        to_keep = jnp.setdiff1d(jnp.arange(args.pop_size+args.n_child), to_kill_all, assume_unique=True, size=args.pop_size)

        pop = jax.tree.map(lambda x: x[to_keep], pop) # these are the ones that survived
        D = D[to_keep, :][:, to_keep]

        # loss = -D.min(axis=-1).mean()
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
