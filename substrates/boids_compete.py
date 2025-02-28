import jax
import jax.numpy as jnp
from jax.random import split

import numpy as np
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial

import flax.linen as nn
from einops import rearrange, reduce, repeat

import evosax

# from . import FlattenSimulationParameters
from .boids import Boids

"""
This is the code for simulating the competition between different boids species as shown in https://sakana.ai/asal/.
"""
class BoidsCompeting():
    def __init__(self, n_boids, k_sims, space_size=1., init_dist='random', colors='bb3e03-0a9396-001219-e9d8a6-9b2226-94d2bd-ee9b00-ca6702-005f73-ae2012'):
        self.n_boids = n_boids
        self.k_sims = k_sims
        self.space_size = space_size
        assert self.n_boids % self.k_sims == 0
        self.model_boids = FlattenSimulationParameters(Boids(n_boids=n_boids, space_size=space_size, red_boid=False))

        self.init_dist = init_dist

        assert self.k_sims <= len(colors.split('-'))
        self.species_colors = jnp.array([mcolors.to_rgb(f"#{c}") for c in colors.split('-')])[:self.k_sims]

    def default_params(self, rng):
        return jax.vmap(self.model_boids.default_params, in_axes=(0,))(split(rng, self.k_sims))

    def init_state(self, rng, params):
        if self.init_dist == 'random':
            _rng1, _rng2 = split(rng, 2)
            x = jax.random.uniform(_rng1, (self.n_boids, 2), minval=0., maxval=self.space_size)
            v = jax.random.normal(_rng2, (self.n_boids, 2))
            v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
            return dict(x=x, v=v)
        elif self.init_dist == 'grid':
            _rng1, _rng2 = split(rng, 2)
            x = jax.random.uniform(_rng1, (self.n_boids, 2), minval=0., maxval=1.)
            v = jax.random.normal(_rng2, (self.n_boids, 2))
            v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

            offset = jnp.arange(self.k_sims)
            offset = repeat(offset, "K -> (K N)", N=self.n_boids//self.k_sims)
            gridlen = int(np.ceil(np.sqrt(self.k_sims)))
            offx, offy = offset%gridlen, offset//gridlen
            offset = jnp.stack([offy, offx], axis=-1)
            x = x + offset
            return dict(x=x, v=v)
        else:
            raise NotImplementedError

    def step_state(self, rng, state, params):
        state = jax.vmap(self.model_boids.step_state, in_axes=(None, None, 0))(rng, state, params)
        idx = jnp.arange(self.k_sims)
        idx = repeat(idx, 'k -> (k p)', p=self.n_boids//self.k_sims)
        state = jax.tree.map(lambda x: x[idx, jnp.arange(self.n_boids)], state)
        return state
    
    def render_state(self, state, params, img_size=256):
        # params = jax.tree.map(lambda x: x[0], params)
        # return self.model_boids.render_state(state, params, img_size)

        state = jax.tree.map(lambda x: rearrange(x, "(K N) ... -> K N ...", K=self.k_sims), state)
        img = jax.vmap(self.model_boids.render_state, in_axes=(0, 0, None))(state, params, img_size)
        img = 1.-img

        img = img * (1.-self.species_colors[:, None, None, :])
        img = 1.- img
        img = img.min(axis=0)
        return img


class FlattenSimulationParameters():
    def __init__(self, sim):
        self.sim = sim
        self.param_reshaper = evosax.ParameterReshaper(self.sim.default_params(jax.random.PRNGKey(0)))
        self.n_params = self.param_reshaper.total_params

    def default_params(self, rng):
        params = self.sim.default_params(rng)
        return self.param_reshaper.flatten_single(params)
    
    def init_state(self, rng, params):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.init_state(rng, params)
    
    def step_state(self, rng, state, params):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.step_state(rng, state, params)
    
    def render_state(self, state, params, img_size=None):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.render_state(state, params, img_size)
