import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial
import flax.linen as nn


class PLifeNetwork(nn.Module):
    @nn.compact
    def __call__(self, c1, c2):  # c1, c2 are (n_colors,)
        d, = c1.shape
        c = jnp.concatenate([c1, c2], axis=-1)
        x = nn.Dense(features=8)(c)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1 + d)(x)
        alpha, dc1 = x[:1], x[1:]
        alpha = jax.nn.tanh(alpha) * 1.5
        dc1 = jax.nn.tanh(dc1)
        return alpha, dc1


class ParticleLifePlusOpt():
    def __init__(self, n_particles=5000, n_colors=6, n_dims=2, x_dist_bins=7,
                 beta=0.3, alpha=0., mass=0.1,
                 dt=0.002, half_life=0.04, rmax=0.1,
                 render_radius=1e-2, sharpness=20.,
                 update_colors=True, world_size=1.,
                 color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00',
                 background_color='black'):
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_dims = n_dims
        assert n_dims == 2, 'only 2d supported for now'
        # x_dist_bins is used as a minimum grid resolution.
        self.x_dist_bins = x_dist_bins  
        self.plife_net = PLifeNetwork()

        self.render_radius = render_radius
        self.sharpness = sharpness
        self.update_colors = update_colors
        self.world_size = world_size
        self.color_palette = color_palette
        self.background_color = background_color

        self.fixed_params = dict(
            beta=jnp.array(beta, dtype=jnp.float32),
            alpha=None,
            mass=jnp.array(mass, dtype=jnp.float32),
            dt=jnp.array(dt, dtype=jnp.float32),
            half_life=jnp.array(half_life, dtype=jnp.float32),
            rmax=jnp.array(rmax, dtype=jnp.float32),
        )
        # Note: n_cells will be computed per step after adjusting grid resolution.

    def default_params(self, rng):
        alpha = self.plife_net.init(rng, 
                                    jnp.zeros((self.n_colors,), dtype=jnp.float32), 
                                    jnp.ones((self.n_colors,), dtype=jnp.float32))
        return dict(alpha=alpha)
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)
        c = jax.random.normal(_rng1, (self.n_particles, self.n_colors), dtype=jnp.float32)
        c = c / jnp.linalg.norm(c, axis=-1, keepdims=True)
        x = jax.random.uniform(_rng2, (self.n_particles, self.n_dims), minval=0., maxval=1., dtype=jnp.float32)
        v = jnp.zeros((self.n_particles, self.n_dims), dtype=jnp.float32)
        return dict(c=c, x=x, v=v)
    
    # Helper: compute linear cell index for a 2D position in [0,1]^2.
    def get_cell_index(self, pos, bins):
        cell = jnp.floor(pos * bins).astype(jnp.int32) % bins
        return cell[0] * bins + cell[1]
    
    # ------------------------------------------------------------------
    # Adaptive spatial hashing step_state:
    # We adjust the grid resolution so that the average occupancy is near a target.
    # ------------------------------------------------------------------
    def step_state(self, rng, state, params):
        x, v, c = state['x'], state['v'], state['c']
        dt = self.fixed_params['dt']
        mass = self.fixed_params['mass']
        half_life = self.fixed_params['half_life']
        beta = self.fixed_params['beta']
        rmax = self.fixed_params['rmax']

        # Set a target occupancy (number of particles per cell)
        target_occupancy = 20
        # Compute number of bins per axis so that average occupancy ~ target_occupancy.
        bins_desired = math.ceil(math.sqrt(self.n_particles / target_occupancy))
        # Ensure at least the minimum resolution is used.
        bins = max(self.x_dist_bins, bins_desired)
        n_cells = bins * bins
        cell_size = 1.0 / bins

        # 1. Compute cell indices for each particle (using the adaptive grid).
        cell_idx = jax.vmap(lambda pos: self.get_cell_index(pos, bins))(x)  # shape (n_particles,)
        
        # 2. Sort particles by cell index.
        sort_perm = jnp.argsort(cell_idx)
        x_sorted = x[sort_perm]
        c_sorted = c[sort_perm]
        cell_idx_sorted = cell_idx[sort_perm]
        
        # 3. For each cell (0..n_cells-1), get start/end indices via searchsorted.
        cell_numbers = jnp.arange(n_cells, dtype=jnp.int32)
        cell_start = jnp.searchsorted(cell_idx_sorted, cell_numbers, side='left')
        cell_end = jnp.searchsorted(cell_idx_sorted, cell_numbers, side='right')
        
        # 4. Compute forces by only considering particles in neighbor cells.
        neighbor_offsets = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ], dtype=jnp.int32)
        
        def particle_force(i):
            pos1 = x[i]
            col1 = c[i]
            # Compute particle's cell coordinate.
            cell_coord = jnp.floor(pos1 * bins).astype(jnp.int32) % bins
            f_total = jnp.zeros_like(pos1)
            dcol_total = jnp.zeros_like(col1)
            
            def neighbor_body(j, acc):
                f_acc, dcol_acc = acc
                offset = neighbor_offsets[j]
                neighbor_coord = (cell_coord + offset) % bins
                neighbor_lin = neighbor_coord[0] * bins + neighbor_coord[1]
                start = cell_start[neighbor_lin]
                end = cell_end[neighbor_lin]
                candidate_count = end - start
                
                def candidate_body(k, inner_acc):
                    f_inner, dcol_inner = inner_acc
                    idx_candidate = sort_perm[start + k]
                    def compute_interaction():
                        pos2 = x[idx_candidate]
                        col2 = c[idx_candidate]
                        r = pos2 - pos1
                        r = jax.lax.select(r > 0.5, r - 1.0,
                              jax.lax.select(r < -0.5, r + 1.0, r))
                        rlen = jnp.linalg.norm(r)
                        def interact():
                            alpha, dc = self.plife_net.apply(params['alpha'], col1, col2)
                            rdir = r / (rlen + 1e-8)
                            flen = rmax * (
                                (rlen / beta - 1.0) * jnp.where(rlen < beta, 1.0, 0.0) +
                                alpha * (1 - jnp.abs(2 * rlen - 1 - beta) / (1 - beta)) *
                                jnp.where((rlen >= beta) & (rlen < 1.0), 1.0, 0.0)
                            )
                            return rdir * flen, dc * jax.nn.relu(1.0 - rlen / rmax)
                        return jax.lax.cond(rlen < rmax,
                                            lambda _: interact(),
                                            lambda _: (jnp.zeros_like(pos1), jnp.zeros_like(col1)),
                                            operand=None)

                    def no_interaction():
                        return (jnp.zeros_like(pos1), jnp.zeros_like(col1))
                    return jax.lax.cond(jnp.equal(idx_candidate, i),
                                        lambda _: (jnp.zeros_like(pos1), jnp.zeros_like(col1)),
                                        lambda _: compute_interaction(),
                                        operand=None,
                                        )
                f_neighbor, dcol_neighbor = jax.lax.fori_loop(0, candidate_count, candidate_body,
                                                              (jnp.zeros_like(pos1), jnp.zeros_like(col1)))
                return (f_acc + f_neighbor, dcol_acc + dcol_neighbor)
            
            f_total, dcol_total = jax.lax.fori_loop(0, neighbor_offsets.shape[0], neighbor_body,
                                                    (f_total, dcol_total))
            return f_total, dcol_total
        
        forces, dc1 = jax.vmap(particle_force)(jnp.arange(self.n_particles))
        acc = forces / mass
        
        # Update velocities and positions.
        mu = (0.5) ** (dt / half_life)
        v = mu * v + acc * dt
        x = (x + v * dt) % 1.0
        
        # Update colors if enabled.
        if self.update_colors:
            c = c + dc1 * dt
            c = c / jnp.linalg.norm(c, axis=-1, keepdims=True)
        
        return dict(c=c, x=x, v=v)
    
    def render_state(self, state, params, img_size=256):
        background_color = jnp.array(mcolors.to_rgb(self.background_color), dtype=jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)
        
        render_radius = self.render_radius
        sharpness = self.sharpness / render_radius
        
        x, c = state['x'], state['c'][:, :3]
        c = (c + 1.) / 2.
        mass = jnp.ones((self.n_particles,), dtype=jnp.float32) * self.fixed_params['mass']
        
        xgrid = ygrid = jnp.linspace(0, 1, img_size, dtype=jnp.float32)
        xgrid, ygrid = jnp.meshgrid(xgrid, ygrid, indexing='ij')
        
        def render_circle(img, circle_data):
            x_pt, y_pt, radius, color = circle_data
            d2 = (x_pt - xgrid)**2 + (y_pt - ygrid)**2
            d = jnp.sqrt(d2)
            coeff = 1. - (1. / (1. + jnp.exp(-sharpness * (d - radius))))
            img = coeff[:, :, None] * color + (1 - coeff)[:, :, None] * img
            return img, None
        
        radius = jnp.sqrt(mass) * render_radius
        img, _ = jax.lax.scan(render_circle, img, (*x.T, radius, c))
        return img
