import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from einop import einop

from clip import CLIP
from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")

group.add_argument("--noun_file", type=str, default=None, help="path of noun file")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")
group.add_argument("--prompt", type=str, default="an image of a {}", help="prompt for CLIP")
# group.add_argument("--replace_only_one_niche", type=lambda x: x=='True', default=False, help="replace only the primary niche")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=1000000, help="")
group.add_argument("--pop_size", type=int, default=None, help="")
group.add_argument("--n_mutations", type=int, default=16, help="number of mutations to do")
group.add_argument("--mutation", type=str, default="gaussian", help="type of mutation to do")
group.add_argument("--sigma", type=float, default=0.5, help="mutation strength")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    print(args)
    with open(args.noun_file, 'r') as f:
        nouns = f.read().strip().split('\n')
    if args.pop_size is None:
        args.pop_size = len(nouns)

    nouns = nouns[:args.pop_size]
    nouns = [args.prompt.format(noun) for noun in nouns]

    clip = CLIP()
    z_txt = clip.embed_txt(nouns)

    # cppn = CPPN(n_layers=8, d_hidden=8, nonlin='tanh', hsv=True)
    cppn = CPPN(n_layers=4, d_hidden=16, nonlin='tanh', hsv=True)
    cppn = FlattenCPPNParameters(cppn)

    rng = jax.random.PRNGKey(args.seed)

    def get_pheno(params):
        img = cppn.generate_image(params)
        z_img = clip.embed_img(img)
        return dict(params=params, z_img=z_img)
    
    def mutate_fn(rng, params):
        if args.mutation == 'gaussian':
            noise = jax.random.normal(rng, params.shape)
            return params + noise * args.sigma
        elif args.mutation == 'sparse':
            rng, _rng = split(rng)
            mask = jax.random.uniform(rng, params.shape) < args.sigma
            noise = jax.random.normal(_rng, params.shape)
            return noise * mask + params * (1-mask)
        else:
            raise NotImplementedError

    def place_in_archive(archive, pheno): # place pheno into all niches that it beats
        z_img = pheno['z_img']
        scores = z_img@archive['z_txt'].T # (N, )
        replace = scores > archive['quality'] # (N, )

        # if args.replace_only_one_niche:
        #     nid = jnp.argmax(scores)
        #     replace_single = jax.nn.one_hot(nid, args.pop_size, dtype=jnp.bool_)
        #     replace = replace & replace_single

        pheno = jax.tree.map(lambda x: einop(x, "... -> 1 ..."), pheno)
        replace_fn = lambda x, y: jnp.where(replace.reshape((args.pop_size, )+(1,)*(x.ndim-1)), x, y)
        archive['pheno'] = jax.tree.map(replace_fn, pheno, archive['pheno'])
        archive['quality'] = jnp.where(replace, scores, archive['quality'])
        data = dict(n_transfers=replace.sum())
        return archive, data

    @jax.jit
    def do_iter(archive, rng):
        p = (archive['quality'] > -jnp.inf).astype(jnp.float32)
        p = p / p.sum()
        rng, _rng = split(rng)

        idx_parents = jax.random.choice(_rng, args.pop_size, shape=(args.n_mutations, ), replace=True, p=p)
        params_parents = archive['pheno']['params'][idx_parents]
        params_children = jax.vmap(mutate_fn)(split(rng, args.n_mutations), params_parents)
        phenos_children = jax.vmap(get_pheno)(params_children)
        archive, data = jax.lax.scan(place_in_archive, archive, phenos_children)

        data = jax.tree.map(lambda x: x.sum(axis=0), data)
        # data['nid_parent'] = idx_parents
        # quality = archive['quality'] * jnp.isfinite(archive['quality'])
        # data['percent_archive_filled'] = (quality > 0).mean()
        # data['avg_quality'] = quality.sum()/(quality>0).sum()
        data['avg_quality'] = archive['quality'].mean()
        return archive, data


    params_init = jnp.zeros((cppn.n_params, ))
    params_init = jax.vmap(mutate_fn, in_axes=(0, None))(split(rng, args.pop_size), params_init)
    scan_fn = lambda _, p: (None, get_pheno(p))
    _, phenos_init = jax.lax.scan(scan_fn, None, params_init)
    quality_init = (phenos_init['z_img'] * z_txt).sum(axis=-1)

    # pheno = get_pheno(jax.random.normal(rng, (cppn.n_params, )))
    archive = dict(
        z_txt=z_txt,
        # pheno=jax.tree.map(lambda x: einop(x, "... -> N ...", N=args.pop_size), pheno),
        # quality=jnp.full((args.pop_size,), -jnp.inf),
        pheno=phenos_init,
        quality=quality_init,
    )
    print('archive shape: ', jax.tree.map(lambda x: x.shape, archive))

    data = []
    args.n_iters = args.n_iters//100
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        archive, di = jax.lax.scan(do_iter, archive, split(_rng, 100))
        data.append(di)

        if i_iter % (10) == 0:
            pbar.set_postfix(n_transfers=di['n_transfers'].mean().item(), avg_quality=di['avg_quality'].mean().item())

        if args.save_dir and (i_iter % (args.n_iters//5) == 0 or i_iter == args.n_iters-1):
            util.save_pkl(args.save_dir, 'archive', jax.tree.map(lambda x: np.array(x), archive))
            util.save_pkl(args.save_dir, 'data', jax.tree.map(lambda *x: np.array(jnp.concatenate(x, axis=0)), *data))

if __name__ == '__main__':
    main(parse_args())
