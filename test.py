# import os, sys, glob, pickle
# from functools import partial

# import jax
# import jax.numpy as jnp
# from jax.random import split
# import numpy as np
# import matplotlib.pyplot as plt
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

#     rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None, rollout_steps=substrate.rollout_steps, time_sampling=8, img_size=224, return_state=False)
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
#     plt.savefig(os.path.join("loss.png"))

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
#     plt.savefig(os.path.join("rollout.png"))



# import os
# import jax
# import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt
# from einops import rearrange
# from functools import partial

# import substrates
# from rollout import rollout_simulation
# import util

# # Path where MAP-Elites results are saved
# save_dir = "./data/lenia"

# # Load optimization data (list or dict of per-iteration logs)
# data = util.load_pkl(save_dir, "data")
# # 'solutions' is a dict mapping prompt_index -> best parameters
# # 'performances' is a dict mapping prompt_index -> best fitness score
# solutions, performances = util.load_pkl(save_dir, "best")

# # Create the substrate and flatten parameters
# substrate = substrates.create_substrate('lenia')
# substrate = substrates.FlattenSubstrateParameters(substrate)

# # Jitted rollout function, with no foundation model required for simple visualization
# rollout_fn = partial(
#     rollout_simulation,
#     s0=None,
#     substrate=substrate,
#     fm=None,  # no text/image embedding needed here
#     rollout_steps=substrate.rollout_steps,
#     time_sampling=8,
#     img_size=224,
#     return_state=False
# )
# rollout_fn = jax.jit(rollout_fn)

# rng = jax.random.PRNGKey(0)

# # ------------------------------------------------------------------
# # 1. Plot Best Fitness vs Iteration, for each prompt
# #    (assuming data['best_fitness'] is shape [num_iterations, num_prompts])
# # ------------------------------------------------------------------
# best_fitness_array = data['best_fitness']  # e.g. shape [Iters, N_Prompts]

# plt.figure(figsize=(20, 5))
# num_prompts = best_fitness_array.shape[1]
# for prompt_idx in range(num_prompts):
#     plt.plot(best_fitness_array[:, prompt_idx], label=f"Prompt {prompt_idx}")

# plt.xlabel("Iterations", fontsize=20)
# plt.ylabel("Fitness", fontsize=20)
# plt.title("MAP-Elites: Best Fitness per Prompt over Time", fontsize=25)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "fitness.png"))
# plt.show()


# # ------------------------------------------------------------------
# # 2. Visualize the rollout of the best solution found for each prompt
# # ------------------------------------------------------------------
# for prompt_idx, x_best in solutions.items():
#     # x_best is the flattened parameters for the best solution for this prompt
#     rollout_data = rollout_fn(rng, x_best)

#     # Extract the frames
#     img = np.array(rollout_data['rgb'])  # shape: [T, H, W, 3]

#     # Pad images a bit for nice borders
#     img = np.pad(img, ((0,0), (2,2), (2,2), (0,0)), constant_values=0.5)
#     # Reshape to a single wide image: [H, T*W, 3]
#     img = rearrange(img, "T H W C -> H (T W) C")
#     # Add another small border
#     img = np.pad(img, ((2,2), (2,2), (0,0)), constant_values=0.5)

#     plt.figure(figsize=(20, 6))
#     plt.imshow(img)
#     # Example x-ticks: show the approximate simulation timesteps for each frame
#     n_frames = rollout_data['rgb'].shape[0]  # should match 8
#     px_per_frame = rollout_data['rgb'].shape[2] + 4  # original W plus padding
#     x_positions = np.arange(n_frames)*px_per_frame + px_per_frame//2

#     plt.xticks(x_positions,
#                np.arange(n_frames)*(substrate.rollout_steps//n_frames),
#                fontsize=15)
#     plt.yticks([], fontsize=15)
#     plt.title(f"Best Solution Rollout for Prompt {prompt_idx}", fontsize=25)
#     plt.xlabel("Simulation Timestep", fontsize=20)

#     best_fitness_for_prompt = performances[prompt_idx]
#     plt.text(0.8, 0.8, 
#              f"Best Fitness: {best_fitness_for_prompt:.3f}",
#              color='darkgreen', fontsize=16,
#              transform=plt.gca().transAxes)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"rollout_prompt_{prompt_idx}.png"))
#     plt.show()


import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from functools import partial
import imageio  # <-- For writing video

import substrates
from rollout import rollout_simulation
import util

# Path where MAP-Elites results are saved
save_dir = "./data/lenia"

# Load optimization data (list or dict of per-iteration logs)
data = util.load_pkl(save_dir, "data")
# 'solutions' is a dict mapping prompt_index -> best parameters
# 'performances' is a dict mapping prompt_index -> best fitness score
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

# 1. Plot Best Fitness vs Iteration, for each prompt
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

# 2. For each prompt, visualize the best solution
for prompt_idx, x_best in solutions.items():
    # x_best is the flattened parameters for the best solution for this prompt
    rollout_data = rollout_fn(rng, x_best)
    rgb = rollout_data['rgb']  # shape: [T, H, W, 3], e.g. 8 frames

    # ------------------------------------------------------------------
    # (A) Create a VIDEO from the frames
    # ------------------------------------------------------------------
    # Rescale frames to uint8
    video_frames = (np.array(rgb) * 255).astype(np.uint8)
    # Save as an MP4. (Or specify .gif for an animated GIF)
    video_path = os.path.join(save_dir, f"rollout_prompt_{prompt_idx}.mp4")
    imageio.mimsave(video_path, video_frames, fps=30)
    print(f"Saved video for prompt {prompt_idx} to: {video_path}")

    # ------------------------------------------------------------------
    # (B) Optionally, also create a single wide PNG
    # ------------------------------------------------------------------
    img = np.array(rgb)  # shape: [T, H, W, 3]
    # Pad images for nice borders
    img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=0.5)
    # Merge frames into one wide strip
    img = rearrange(img, "T H W C -> H (T W) C")
    img = np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5)

    plt.figure(figsize=(20, 6))
    plt.imshow(img)
    n_frames = rgb.shape[0]  # 8
    px_per_frame = rgb.shape[2] + 4  # original width + padding
    x_positions = np.arange(n_frames)*px_per_frame + px_per_frame//2

    plt.xticks(
        x_positions,
        np.arange(n_frames)*(substrate.rollout_steps//n_frames),
        fontsize=15
    )
    plt.yticks([], fontsize=15)
    plt.title(f"Best Solution Rollout for Prompt {prompt_idx}", fontsize=25)
    plt.xlabel("Simulation Timestep", fontsize=20)

    best_fitness_for_prompt = performances[prompt_idx]
    plt.text(
        0.8, 0.8, 
        f"Best Fitness: {best_fitness_for_prompt:.3f}",
        color='darkgreen', fontsize=16,
        transform=plt.gca().transAxes
    )

    plt.tight_layout()
    png_path = os.path.join(save_dir, f"rollout_prompt_{prompt_idx}.png")
    plt.savefig(png_path)
    plt.show()
    print(f"Saved PNG for prompt {prompt_idx} to: {png_path}")
