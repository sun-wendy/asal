from PIL import Image
import numpy as np
import argparse
import jax
import jax.numpy as jnp
from einops import rearrange
import matplotlib.pyplot as plt
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import util
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data", help="path to save results to")
    parser.add_argument("--substrate", type=str, default="plife_plus", help="substrate to use")
    parser.add_argument("--foundation_model", type=str, default="clip", help="foundation model to use")
    parser.add_argument("--grid_rows", type=int, default=16, help="number of rows in visualization grid")
    parser.add_argument("--grid_cols", type=int, default=32, help="number of columns in visualization grid")
    parser.add_argument("--cache_images", action="store_true", help="cache simulated images to disk")
    args = parser.parse_args()

    # Load population data (contains the parameters for each individual)
    pop = util.load_pkl(args.save_dir, 'pop')
    data = util.load_pkl(args.save_dir, 'data')
    
    # Check the structure of the pop dictionary
    print("Population dictionary keys:", list(pop.keys()))
    
    # Plot the loss over iterations
    plt.figure(figsize=(10, 6))
    iterations = np.arange(len(data['illumination_loss']))
    plt.plot(iterations, data['illumination_loss'])
    plt.xlabel('Iterations')
    plt.ylabel('Illumination Score')
    plt.title('Illumination Score vs Iterations')
    plt.grid(True)
    plt.savefig(f"{args.save_dir}/illumination_score.png", dpi=300)
    plt.close()

    # Plot the open endedness score over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, data['oe_score'])
    plt.xlabel('Iterations')
    plt.ylabel('Open Endedness Score')
    plt.title('Open Endedness Score vs Iterations')
    plt.grid(True)
    plt.savefig(f"{args.save_dir}/open_endedness_score.png", dpi=300)
    plt.close()
    
    # Set up the simulation components
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    # Create the rollout function - we're just interested in the final state image
    rollout_fn = partial(
        rollout_simulation, 
        s0=None, 
        substrate=substrate, 
        fm=fm, 
        rollout_steps=getattr(substrate, 'rollout_steps', 256), 
        time_sampling='final',  # Only the final frame
        img_size=224, 
        return_state=False
    )
    rollout_fn = jax.jit(rollout_fn)  # JIT for speed
    
    # Calculate number of images to generate
    num_images = min(args.grid_rows * args.grid_cols, len(pop['params']))
    print(f"Generating {num_images} images from parameters...")
    
    # Check if cached images exist
    cache_dir = os.path.join(args.save_dir, "image_cache")
    cache_file = os.path.join(cache_dir, "simulated_images.npy")
    
    if args.cache_images and os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}")
        images = np.load(cache_file)
        images = images[:num_images]
    else:
        # Initialize RNG
        rng = jax.random.PRNGKey(0)
        
        # Run simulations to get final images for the first num_images parameters
        images = []
        for i in tqdm(range(num_images)):
            rng, key = jax.random.split(rng)
            params = pop['params'][i]
            rollout_data = rollout_fn(key, params)
            rgb = rollout_data['rgb']  # Shape: (224, 224, 3)
            images.append(rgb)
        
        images = np.array(images)
        
        # Cache the images if requested
        if args.cache_images:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, images)
            print(f"Cached images saved to {cache_file}")
    
    print(f"Image array shape: {images.shape}")
    
    # Convert images to uint8 if they're in [0,1] range
    if images.dtype == np.float32 or images.dtype == np.float64:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
    
    # Calculate appropriate grid dimensions based on actual number of images
    actual_images = len(images)
    
    if actual_images < args.grid_rows * args.grid_cols:
        # If we have fewer images than the requested grid size, adjust the grid
        if args.grid_cols == args.grid_rows:
            # For square grid targets, calculate the closest square
            grid_side = int(np.ceil(np.sqrt(actual_images)))
            grid_rows = grid_side
            grid_cols = grid_side
        else:
            # Try to maintain the aspect ratio of the original request
            aspect_ratio = args.grid_cols / args.grid_rows
            grid_rows = int(np.ceil(np.sqrt(actual_images / aspect_ratio)))
            grid_cols = int(np.ceil(actual_images / grid_rows))
            
        print(f"Adjusting grid to {grid_rows}x{grid_cols} for {actual_images} images")
    else:
        grid_rows = args.grid_rows
        grid_cols = args.grid_cols
    
    # Pad the images array if necessary to match the grid size
    total_cells = grid_rows * grid_cols
    if actual_images < total_cells:
        # Create blank images (white) for padding
        blank_image = np.ones((images.shape[1], images.shape[2], images.shape[3]), dtype=images.dtype)
        if images.dtype != np.uint8 and images.max() <= 1.0:
            # For float images [0,1]
            blank_image = blank_image * 1.0
        else:
            # For uint8 images [0,255]
            blank_image = blank_image * 255
            
        padding = np.array([blank_image] * (total_cells - actual_images))
        images = np.concatenate([images, padding], axis=0)
    
    # Reshape into grid
    grid = rearrange(images, "(r c) h w d -> (r h) (c w) d", r=grid_rows, c=grid_cols)
    
    # Save the visualization
    Image.fromarray(grid).save(f"{args.save_dir}/illuminate_grid.png")
    print(f"Visualization saved to {args.save_dir}/illuminate_grid.png")

    # Create a more detailed visualization with index numbers
    padding = 30  # pixels of padding at the top for text
    padded_height = images.shape[1] + padding
    padded_width = images.shape[2]
    
    # Create a blank canvas for the padded grid
    padded_grid = np.ones(
        (grid_rows * padded_height, grid_cols * padded_width, 3), 
        dtype=np.uint8
    ) * 255
    
    # Place images in the padded grid with proper index calculation
    for i in range(actual_images):  # Only place actual images, not padding
        row = i // grid_cols  # Integer division to get row index
        col = i % grid_cols   # Modulo to get column index
        
        # Calculate coordinates in the output grid
        y_start = row * padded_height + padding
        x_start = col * padded_width
        
        # Copy the image into position
        padded_grid[
            y_start:y_start + images.shape[1],
            x_start:x_start + images.shape[2]
        ] = images[i]
    
    # Convert to PIL Image for drawing text
    padded_grid_pil = Image.fromarray(padded_grid)
    
    # Use PIL's ImageDraw to add index numbers
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(padded_grid_pil)
    
    try:
        # Try to load a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    # Add index numbers to each image
    for i in range(actual_images):  # Only add numbers to actual images
        row = i // grid_cols
        col = i % grid_cols
        
        x = col * padded_width + 5
        y = row * padded_height + 15
        draw.text((x, y), f"{i}", fill=(0, 0, 0), font=font)
    
    # Save the detailed visualization
    padded_grid_pil.save(f"{args.save_dir}/illuminate_grid_numbered.png")
    print(f"Numbered visualization saved to {args.save_dir}/illuminate_grid_numbered.png")
