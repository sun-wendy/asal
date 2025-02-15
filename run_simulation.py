import jax
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import imageio
import numpy as np


if __name__ == "__main__":
    fm = foundation_models.create_foundation_model('clip')
    substrate = substrates.create_substrate('boids')
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=substrate.rollout_steps, time_sampling='video', img_size=224, return_state=False) # create the rollout function
    rollout_fn = jax.jit(rollout_fn) # jit for speed
    # now you can use rollout_fn as you need...
    rng = jax.random.PRNGKey(0)
    params = substrate.default_params(rng) # sample random parameters
    rollout_data = rollout_fn(rng, params)
    rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
    z = rollout_data['z'] # shape: (8, 512)
    oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()
    
    video_frames = (np.array(rollout_data['rgb']) * 255).astype(np.uint8)
    imageio.mimsave('video.mp4', video_frames, fps=30)
