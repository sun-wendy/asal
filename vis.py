# import os, sys, glob, pickle
# from functools import partial

# import jax
# import jax.numpy as jnp
# from jax.random import split
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# from tqdm.auto import tqdm
# from einops import rearrange, reduce, repeat

# import substrates
# import foundation_models
# from rollout import rollout_simulation
# import asal_metrics
# import util


# if __name__ == "__main__":
#     save_dir = "./data/lenia"
#     data = util.load_pkl(save_dir, "data") # load optimization data
#     params, best_loss = util.load_pkl(save_dir, "best") # load the best parameters found

#     # fm = foundation_models.create_foundation_model('clip') # we don't need the foundation model for just the rollout currently
#     substrate = substrates.create_substrate('lenia') # create the substrate
#     substrate = substrates.FlattenSubstrateParameters(substrate) # useful wrapper to flatten the substrate parameters

#     rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None, rollout_steps=substrate.rollout_steps, time_sampling="video", img_size=224, return_state=False)
#     rollout_fn = jax.jit(rollout_fn)

#     rng = jax.random.PRNGKey(0)
#     rollout_data = rollout_fn(rng, params) # rollout the simulation using this rng seed and simulation parameters

#     # Plot optimization loss
#     plt.figure(figsize=(20, 5))
#     plt.plot(data['best_loss'])
#     plt.xlabel("Iterations", fontsize=20); plt.ylabel("Loss", fontsize=20); plt.title("Optimization", fontsize=25)
#     plt.xticks(fontsize=15); plt.yticks(fontsize=15)
#     plt.text(0.8, 0.8, f"Best Loss: {best_loss:.3f}", color='darkgreen', fontsize=20, transform=plt.gca().transAxes)
#     plt.show()
#     plt.savefig(os.path.join(save_dir, "loss.png"))

#     # Plot rollout
#     plt.figure(figsize=(20, 6))
#     img = np.array(rollout_data['rgb'])
#     img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=0.5)
#     img = rearrange(img, "T H W D -> H (T W) D")
#     img = np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5)
#     plt.imshow(img)
#     plt.xticks(np.arange(0, 8)*228+228//2, np.arange(0, 8)*substrate.rollout_steps//8, fontsize=15); plt.yticks([], fontsize=15)
#     plt.title("Visualizing the Simulation Rollout", fontsize=25)
#     plt.xlabel("Simulation Timestep", fontsize=20)
#     plt.show()
#     plt.savefig(os.path.join(save_dir, "rollout.png"))

#     # Create video
#     video_frames = (np.array(rollout_data['rgb']) * 255).astype(np.uint8)
#     os.makedirs(os.path.join(f"{save_dir}/videos"), exist_ok=True)
#     imageio.mimsave(os.path.join(f"{save_dir}/videos", "video.mp4"), video_frames, fps=30)


import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from functools import partial
import imageio

import substrates
from rollout import rollout_simulation
import util

# Path where MAP-Elites results are saved
save_dir = "./data/lenia_entropy"

data = util.load_pkl(save_dir, "data")  # Optimization data (list or dict of per-iteration logs)
# 'solutions': {prompt_index: best parameters}
# 'performances': {prompt_index: best fitness score}
solutions, performances = util.load_pkl(save_dir, "best")

# Create the substrate and flatten parameters
substrate = substrates.create_substrate('lenia')
substrate = substrates.FlattenSubstrateParameters(substrate)

# Jitted rollout function, with no foundation model required for simple visualization
rollout_fn = partial(
    rollout_simulation,
    s0=None,
    substrate=substrate,
    fm=None,  # no text/image embedding needed here
    rollout_steps=substrate.rollout_steps,
    time_sampling="video",
    img_size=224,
    return_state=False
)
rollout_fn = jax.jit(rollout_fn)
rng = jax.random.PRNGKey(0)

# Plot best fitness vs. iteration, for each prompt
best_fitness_array = data['best_fitness']  # shape: [Iters, N_Prompts]
plt.figure(figsize=(20, 5))
num_prompts = best_fitness_array.shape[1]
for prompt_idx in range(num_prompts):
    plt.plot(best_fitness_array[:, prompt_idx], label=f"Prompt {prompt_idx}")

plt.xlabel("Iterations", fontsize=20)
plt.ylabel("Fitness", fontsize=20)
plt.title("MAP-Elites: Best Fitness per Prompt over Time", fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fitness.png"))
plt.show()

# For each prompt, visualize the best solution
prompts = util.read_prompt_file(os.path.join(save_dir, "../../noun_list.txt"))
for prompt_idx in range(solutions.shape[0]):
    x_best = solutions[prompt_idx]  # Flattened parameters for the best solution for this prompt
    p_best = performances[prompt_idx]
    rollout_data = rollout_fn(rng, x_best)
    rgb = rollout_data['rgb']  # shape: [T, H, W, 3], e.g. 8 frames

    # Create a video from frames
    video_frames = (np.array(rgb) * 255).astype(np.uint8)
    os.makedirs(os.path.join(f"{save_dir}/videos"), exist_ok=True)
    imageio.mimsave(os.path.join(f"{save_dir}/videos", f"{prompts[prompt_idx]}.mp4"), video_frames, fps=30)
