"""
import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial
import flax.linen as nn


# ------------------------------------------------------------------
# PLifeNetwork remains unchanged
# ------------------------------------------------------------------
class PLifeNetwork(nn.Module):
    @nn.compact
    def __call__(self, c1, c2): # D, D
        d, = c1.shape
        c = jnp.concatenate([c1, c2], axis=-1)
        x = nn.Dense(features=8)(c)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1+d)(x)
        alpha, dc1 = x[:1], x[1:]
        alpha = jax.nn.tanh(alpha) * 1.5
        dc1 = jax.nn.tanh(dc1)
        return alpha, dc1

# ------------------------------------------------------------------
# ParticleLifePlus with spatial hashing via cell–linked list
# ------------------------------------------------------------------
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
        self.x_dist_bins = x_dist_bins  # grid resolution along each axis
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
        # total number of cells = x_dist_bins^2
        self.n_cells = self.x_dist_bins * self.x_dist_bins

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
    
    # Helper: compute linear cell index given a 2D position in [0,1]^2.
    def get_cell_index(self, pos):
        # Compute integer cell coordinates in [0, x_dist_bins-1]
        cell = jnp.floor(pos * self.x_dist_bins).astype(jnp.int32) % self.x_dist_bins
        # Linear index: row-major order.
        return cell[0] * self.x_dist_bins + cell[1]
    
    # ------------------------------------------------------------------
    # The new step_state uses spatial hashing by grouping particles into cells.
    # We first compute each particle’s cell index, then sort the particles by cell.
    # For each cell, we use jnp.searchsorted to get the start/end indices in the sorted array.
    # Then, for each particle, we only compute interactions from particles in its own cell
    # and in the eight neighboring cells (with periodic boundaries).
    # ------------------------------------------------------------------
    def step_state(self, rng, state, params):
        x, v, c = state['x'], state['v'], state['c']
        dt = self.fixed_params['dt']
        mass = self.fixed_params['mass']
        half_life = self.fixed_params['half_life']
        beta = self.fixed_params['beta']
        rmax = self.fixed_params['rmax']
        
        # ---------------------------
        # 1. Compute cell indices for all particles.
        # ---------------------------
        # Each particle’s cell index is computed as a linear index in [0, n_cells-1].
        cell_idx = jax.vmap(self.get_cell_index)(x)  # shape (n_particles,)
        
        # ---------------------------
        # 2. Sort particles by cell index.
        # ---------------------------
        sort_perm = jnp.argsort(cell_idx)
        x_sorted = x[sort_perm]
        c_sorted = c[sort_perm]
        cell_idx_sorted = cell_idx[sort_perm]
        
        # ---------------------------
        # 3. For each cell (0 .. n_cells-1), determine start and end indices in the sorted arrays.
        #    (We use jnp.searchsorted, which is fully vectorized.)
        # ---------------------------
        cell_numbers = jnp.arange(self.n_cells, dtype=jnp.int32)
        cell_start = jnp.searchsorted(cell_idx_sorted, cell_numbers, side='left')
        cell_end = jnp.searchsorted(cell_idx_sorted, cell_numbers, side='right')
        # cell_start, cell_end: shape (n_cells,)
        
        # ---------------------------
        # 4. For each particle, sum interactions only from particles in neighbor cells.
        # ---------------------------
        neighbor_offsets = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ], dtype=jnp.int32)
        
        # For a given particle (with original index i), we want to:
        #   a. Compute its cell coordinate (cx, cy).
        #   b. For each neighbor offset, compute neighbor cell coordinates (with periodic wrap).
        #   c. Get the linear neighbor cell index.
        #   d. Use the precomputed cell_start and cell_end arrays to know which indices in the sorted list
        #      belong to that cell.
        #   e. Compute interactions between the particle and each candidate in that cell.
        
        # Define a function to compute force contributions for a single particle i.
        def compute_particle(i):
            pos = x[i]
            col = c[i]
            # Compute the cell coordinate (as integers) for this particle.
            cell_coord = jnp.floor(pos * self.x_dist_bins).astype(jnp.int32) % self.x_dist_bins
            # Initialize accumulators for force and color change.
            f_total = jnp.zeros_like(pos)
            dc_total = jnp.zeros_like(col)
            
            # Loop over the 9 neighbor offsets.
            def body_offset(j, acc):
                f_acc, dc_acc = acc
                offset = neighbor_offsets[j]
                # Neighbor cell coordinates (with wrap–around)
                neighbor_coord = (cell_coord + offset) % self.x_dist_bins
                neighbor_lin = neighbor_coord[0] * self.x_dist_bins + neighbor_coord[1]
                # Get start and end indices for the neighbor cell.
                start = cell_start[neighbor_lin]
                end = cell_end[neighbor_lin]
                # For each candidate in [start, end), compute interaction.
                def body_candidate(k, acc2):
                    f2, dc2 = acc2
                    # Get the original index for the candidate.
                    cand_idx = sort_perm[start + k]
                    # Avoid self–interaction.
                    def compute_interaction():
                        pos2 = x[cand_idx]
                        col2 = c[cand_idx]
                        # Compute difference with periodic (circular) boundary.
                        r = pos2 - pos
                        r = jax.lax.select(r > 0.5, r - 1.0,
                              jax.lax.select(r < -0.5, r + 1.0, r))
                        rlen = jnp.linalg.norm(r)
                        def interact():
                            alpha, dc = self.plife_net.apply(params['alpha'], col, col2)
                            rdir = r / (rlen + 1e-8)
                            flen = rmax * (
                                (rlen / beta - 1.0) * jnp.where(rlen < beta,
                                                              1.0, 0.0) +
                                alpha * (1 - jnp.abs(2 * rlen - 1 - beta) / (1 - beta)) *
                                jnp.where((rlen >= beta) & (rlen < 1.0), 1.0, 0.0)
                            )
                            f_val = rdir * flen
                            dc_val = dc * jax.nn.relu(1.0 - rlen / rmax)
                            return f_val, dc_val
                        # Only compute if within interaction radius.
                        return jax.lax.cond(rlen < rmax, interact,
                                            lambda: (jnp.zeros_like(pos), jnp.zeros_like(col)))
                    def no_interaction():
                        return (jnp.zeros_like(pos), jnp.zeros_like(col))
                    # Skip if candidate is the same as self.
                    return jax.lax.cond(jnp.equal(cand_idx, i),
                                        no_interaction,
                                        compute_interaction)
                n_candidates = end - start
                f_offset, dc_offset = jax.lax.fori_loop(0, n_candidates, body_candidate, 
                                                        (jnp.zeros_like(pos), jnp.zeros_like(col)))
                return (f_acc + f_offset, dc_acc + dc_offset)
            
            f_total, dc_total = jax.lax.fori_loop(0, neighbor_offsets.shape[0], body_offset, 
                                                  (f_total, dc_total))
            return f_total, dc_total
        
        # Vectorize over all particles.
        f_all, dc_all = jax.vmap(compute_particle)(jnp.arange(self.n_particles))
        # f_all: (n_particles, n_dims); dc_all: (n_particles, n_colors)
        
        # Compute acceleration and update velocity and position.
        acc = f_all / mass
        mu = (0.5) ** (dt / half_life)
        v = mu * v + acc * dt
        x = (x + v * dt) % 1.0
        
        # Update colors if needed.
        if self.update_colors:
            c = c + dc_all * dt
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
            d2 = (x_pt - xgrid) ** 2 + (y_pt - ygrid) ** 2
            d = jnp.sqrt(d2)
            coeff = 1. - (1. / (1. + jnp.exp(-sharpness * (d - radius))))
            img = coeff[:, :, None] * color + (1 - coeff)[:, :, None] * img
            return img, None
        
        radius = jnp.sqrt(mass) * render_radius
        img, _ = jax.lax.scan(render_circle, img, (*x.T, radius, c))
        return img


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
        self.x_dist_bins = x_dist_bins
        self.plife_net = PLifeNetwork()

        self.render_radius = render_radius
        self.sharpness = sharpness
        self.update_colors = update_colors
        self.world_size = world_size
        self.color_palette = color_palette
        self.background_color = background_color

        self.fixed_params = dict(
            beta=jnp.array(beta),
            alpha=None,
            mass=jnp.array(mass),
            dt=jnp.array(dt),
            half_life=jnp.array(half_life),
            rmax=jnp.array(rmax),
        )

    def default_params(self, rng):
        alpha = self.plife_net.init(rng, jnp.zeros((self.n_colors, )), jnp.ones((self.n_colors, )))
        return dict(alpha=alpha)
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)

        c = jax.random.normal(_rng1, (self.n_particles, self.n_colors))
        c = c / jnp.linalg.norm(c, axis=-1, keepdims=True)

        x = jax.random.uniform(_rng2, (self.n_particles, self.n_dims), minval=0., maxval=1.)
        v = jnp.zeros((self.n_particles, self.n_dims))
        return dict(c=c, x=x, v=v)
    
    def step_state(self, rng, state, params):
        x, v, c = state['x'], state['v'], state['c']

        mass = self.fixed_params['mass']
        half_life = self.fixed_params['half_life']
        dt = self.fixed_params['dt']
        beta = self.fixed_params['beta']
        rmax = self.fixed_params['rmax']

        # -------------------------------------------------------------------
        # Define the force functions and the spatial hashing routine.
        # -------------------------------------------------------------------
        def force_graph(r, alpha, beta):
            first = r / beta - 1.
            second = alpha * (1 - jnp.abs(2 * r - 1 - beta) / (1 - beta))
            cond_first = (r < beta)
            cond_second = (r >= beta) & (r < 1)
            return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))

        def calc_force(pos1, pos2, col1, col2):
            # Compute difference vector with periodic boundary (torus)
            r_vec = pos2 - pos1
            r_vec = jnp.where(r_vec > 0.5, r_vec - 1., jnp.where(r_vec < -0.5, r_vec + 1., r_vec))
            # Get dynamics parameters from the network
            alpha, dcol = self.plife_net.apply(params['alpha'], col1, col2)
            rlen = jnp.linalg.norm(r_vec)
            rdir = r_vec / (rlen + 1e-8)
            flen = rmax * force_graph(rlen / rmax, alpha, beta)
            force = rdir * flen
            # The color update decays with distance
            dcol = dcol * jax.nn.relu(1. - rlen / rmax)
            return force, dcol

        # This function uses spatial hashing: We bin particles into a fixed grid of x_dist_bins×x_dist_bins.
        # Then for each particle we only consider candidates in its own and neighboring cells.
        def spatial_forces(x, c):
            n_particles = x.shape[0]
            bins = self.x_dist_bins  # number of cells per dimension
            cell_size = 1.0 / bins

            # Compute cell coordinates (as ints) for each particle.
            cell_coords = (jnp.floor(x * bins) % bins).astype(jnp.int32)  # shape (n_particles, 2)
            cell_idx = cell_coords[:, 0] * bins + cell_coords[:, 1]         # shape (n_particles,)

            # Sort particles by cell index.
            sort_perm = jnp.argsort(cell_idx)
            sorted_cell_idx = cell_idx[sort_perm]

            # Define the 9 neighbor offsets.
            neighbor_offsets = jnp.array([[-1, -1], [-1, 0], [-1, 1],
                                          [0, -1],  [0, 0],  [0, 1],
                                          [1, -1],  [1, 0],  [1, 1]], dtype=jnp.int32)

            # For each particle (indexed by its original index) compute the total force and color update.
            def particle_force(i):
                pos1 = x[i]
                col1 = c[i]
                ci, cj = cell_coords[i]
                f_total = jnp.zeros_like(pos1)
                dcol_total = jnp.zeros_like(col1)

                # Loop over each neighbor cell.
                def neighbor_body(neighbor_idx, acc):
                    f_acc, dcol_acc = acc
                    offset = neighbor_offsets[neighbor_idx]
                    # Wrap-around for periodic boundary:
                    ni = (ci + offset[0]) % bins
                    nj = (cj + offset[1]) % bins
                    neighbor_cell = ni * bins + nj

                    # Use binary search (jnp.searchsorted) over the sorted cell indices.
                    start = jnp.searchsorted(sorted_cell_idx, neighbor_cell, side='left')
                    end = jnp.searchsorted(sorted_cell_idx, neighbor_cell, side='right')
                    candidate_count = end - start

                    # For the candidates in this neighbor cell, accumulate contributions.
                    def candidate_body(j, inner_acc):
                        f_inner, dcol_inner = inner_acc
                        # Get candidate index from sorted order.
                        idx_candidate = sort_perm[start + j]
                        # Skip self-interaction.
                        def compute_interaction():
                            pos2 = x[idx_candidate]
                            col2 = c[idx_candidate]
                            force, dcol = calc_force(pos1, pos2, col1, col2)
                            return force, dcol
                        force, dcol = jax.lax.cond(idx_candidate == i,
                                                   lambda: (jnp.zeros_like(pos1), jnp.zeros_like(col1)),
                                                   lambda: compute_interaction())
                        return (f_inner + force, dcol_inner + dcol)
                    f_neighbor, dcol_neighbor = jax.lax.fori_loop(0, candidate_count, candidate_body,
                                                                  (jnp.zeros_like(pos1), jnp.zeros_like(col1)))
                    return (f_acc + f_neighbor, dcol_acc + dcol_neighbor)

                f_total, dcol_total = jax.lax.fori_loop(0, neighbor_offsets.shape[0], neighbor_body,
                                                        (f_total, dcol_total))
                return f_total, dcol_total

            forces, dc1 = jax.vmap(particle_force)(jnp.arange(n_particles))
            return forces, dc1

        # Compute forces using spatial hashing.
        forces, dc1 = spatial_forces(x, c)
        acc = forces / mass

        # Update velocities and positions.
        mu = (0.5) ** (dt / half_life)
        v = mu * v + acc * dt
        x = (x + v * dt) % 1.0  # circular boundary

        # Update colors if enabled.
        if self.update_colors:
            c = c + dc1 * dt
            c = c / jnp.linalg.norm(c, axis=-1, keepdims=True)
        return dict(c=c, x=x, v=v)
    
    def render_state(self, state, params, img_size=256):
        background_color = jnp.array(mcolors.to_rgb(self.background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)

        render_radius = self.render_radius
        sharpness = self.sharpness / render_radius

        x, c = state['x'], state['c'][:, :3]
        c = (c + 1.) / 2.
        mass = jnp.ones((self.n_particles, )) * self.fixed_params['mass']

        xgrid = ygrid = jnp.linspace(0, 1, img_size)
        xgrid, ygrid = jnp.meshgrid(xgrid, ygrid, indexing='ij')

        def render_circle(img, circle_data):
            x0, y0, radius, color = circle_data
            d2 = (x0 - xgrid)**2 + (y0 - ygrid)**2
            d = jnp.sqrt(d2)
            coeff = 1. - (1. / (1. + jnp.exp(-sharpness * (d - radius))))
            img = coeff[:, :, None] * color + (1 - coeff)[:, :, None] * img
            return img, None

        radius = jnp.sqrt(mass) * render_radius
        img, _ = jax.lax.scan(render_circle, img, (*x.T, radius, c))
        return img

"""

import math
import jax
import jax.numpy as jnp
from jax.random import split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from einops import repeat, rearrange
from functools import partial
import flax.linen as nn

# ------------------------------------------------------------------
# PLifeNetwork remains unchanged
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# ParticleLifePlusOpt with adaptive grid resolution for spatial hashing
# ------------------------------------------------------------------
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

