import jax
import evosax

from .boids import Boids
from .lenia import Lenia
from .plife import ParticleLife
from .plife_plus import ParticleLifePlus
from .plenia import ParticleLenia
from .dnca import DNCA
from .nca import NCA
from .gol import GameOfLife
from .boids_compete import BoidsCompeting
from .gol_compete import GameOfLifeCompeting


def create_substrate(substrate_name):
    """
    Create the substrate given a substrate name.
    The substrate parameterizes the space of simulations to search over.
    It has the following methods attached to it:
        - substrate.default_params(rng) to sample random parameters
        - substrate.init_state(rng, params) to sample the state from the initial state distribution
        - substrate.step_state(rng, state, params) to step the state forward one timestep
    
    Possible substrate names:
        - 'boids': Boids
        - 'lenia': Lenia
        - 'plife': ParticleLife
        - 'plife_plus': ParticleLifePlus
        - 'plenia': ParticleLenia
        - 'dnca': DNCA
        - 'nca_d1': NCA with d_state=1
        - 'nca_d3': NCA with d_state=3
        - 'gol': GameOfLife
    """
    rollout_steps = 1000
    if substrate_name=='boids':
        substrate = Boids(n_boids=128, n_nbrs=16, visual_range=0.1, speed=0.5, controller='network', dt=0.01, bird_render_size=0.015, bird_render_sharpness=40.)
    elif substrate_name=='lenia':
        substrate = Lenia(grid_size=128, center_phenotype=True, phenotype_size=64, start_pattern="5N7KKM", clip1=1.)
        rollout_steps = 256
    elif substrate_name=='plife':
        substrate = ParticleLife(n_particles=5000, n_colors=6, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
    elif substrate_name=='plife_plus':
        substrate = ParticleLifePlus(n_particles=1000, dt=0.02, render_radius=0.04, sharpness=30., background_color='black')
    elif substrate_name=='plenia':
        substrate = ParticleLenia(n_particles=200, dt=0.1)
    elif substrate_name=='dnca':
        substrate = DNCA(grid_size=128, d_state=8, n_groups=1, identity_bias=0., temperature=1e-3)
    elif substrate_name=='nca_d1':
        substrate = NCA(grid_size=128, d_state=1, p_drop=0.5, dt=0.1)
    elif substrate_name=='nca_d3':
        substrate = NCA(grid_size=128, d_state=3, p_drop=0.5, dt=0.1)
    elif substrate_name=='gol':
        substrate = GameOfLife(grid_size=64)
        rollout_steps = 1024
    elif substrate_name=='boids_compete':
        substrate = BoidsCompeting(n_boids=128, k_sims=8, space_size=3., init_dist='random', colors='bb3e03-0a9396-001219-e9d8a6-9b2226-94d2bd-ee9b00-ca6702-005f73-ae2012')
        rollout_steps = 2048
    elif substrate_name=='gol_compete':
        substrate = GameOfLifeCompeting(k_sims=9, grid_size=64)
        rollout_steps = 1024
    else:
        raise ValueError(f"Unknown substrate name: {substrate_name}")
    substrate.name = substrate_name
    substrate.rollout_steps = rollout_steps
    return substrate

class FlattenSubstrateParameters():
    def __init__(self, substrate):
        self.substrate = substrate
        self.param_reshaper = evosax.ParameterReshaper(self.substrate.default_params(jax.random.PRNGKey(0)))
        self.n_params = self.param_reshaper.total_params

    def default_params(self, rng):
        params = self.substrate.default_params(rng)
        return self.param_reshaper.flatten_single(params)
    
    def init_state(self, rng, params):
        params = self.param_reshaper.reshape_single(params)
        return self.substrate.init_state(rng, params)
    
    def step_state(self, rng, state, params):
        params = self.param_reshaper.reshape_single(params)
        return self.substrate.step_state(rng, state, params)
    
    def render_state(self, state, params, img_size=None):
        params = self.param_reshaper.reshape_single(params)
        return self.substrate.render_state(state, params, img_size)
    
    def __getattr__(self, name):
        return getattr(self.substrate, name)