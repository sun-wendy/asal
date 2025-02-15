import os, sys, glob, pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util


if __name__ == "__main__":
    save_dir = "./data/open_endedness_3"
    data = util.load_pkl(save_dir, "data") # load optimization data
    params, best_loss = util.load_pkl(save_dir, "best") # load the best parameters found

    # fm = foundation_models.create_foundation_model('clip') # we don't need the foundation model for just the rollout currently
    substrate = substrates.create_substrate('plife') # create the substrate
    substrate = substrates.FlattenSubstrateParameters(substrate) # useful wrapper to flatten the substrate parameters

    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None, rollout_steps=substrate.rollout_steps, time_sampling=8, img_size=224, return_state=False)
    rollout_fn = jax.jit(rollout_fn)

    rng = jax.random.PRNGKey(0)
    rollout_data = rollout_fn(rng, params) # rollout the simulation using this rng seed and simulation parameters

    # Plot optimization loss
    plt.figure(figsize=(20, 5))
    plt.plot(data['best_loss'])
    plt.xlabel("Iterations", fontsize=20); plt.ylabel("Loss", fontsize=20); plt.title("Optimization", fontsize=25)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.text(0.8, 0.8, f"Best Loss: {best_loss:.3f}", color='darkgreen', fontsize=20, transform=plt.gca().transAxes)
    plt.show()
    plt.savefig(os.path.join("loss.png"))

    # Plot rollout
    plt.figure(figsize=(20, 6))
    img = np.array(rollout_data['rgb'])
    img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=0.5)
    img = rearrange(img, "T H W D -> H (T W) D")
    img = np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5)
    plt.imshow(img)
    plt.xticks(np.arange(0, 8)*228+228//2, np.arange(0, 8)*substrate.rollout_steps//8, fontsize=15); plt.yticks([], fontsize=15)
    plt.title("Visualizing the Simulation Rollout", fontsize=25)
    plt.xlabel("Simulation Timestep", fontsize=20)
    plt.show()
    plt.savefig(os.path.join("rollout.png"))
