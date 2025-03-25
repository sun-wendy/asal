import os
import numpy as np
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import imageio
import util
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data", help="path to save results to")
    parser.add_argument("--substrate", type=str, default="plife_plus", help="substrate to use")
    parser.add_argument("--foundation_model", type=str, default="clip", help="foundation model to use")
    parser.add_argument("--num_videos", type=int, default=10, help="number of videos to generate")
    parser.add_argument("--selection", type=str, default="random", 
                        choices=["best", "diverse", "random", "indices"],
                        help="how to select individuals (best=highest quality, diverse=maximally different, random=random selection)")
    parser.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7,8,9", 
                        help="comma-separated list of indices to use (only for --selection=indices)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second for videos")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="directory to save videos (defaults to {save_dir}/videos)")
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.save_dir, "videos")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load population data
    pop = util.load_pkl(args.save_dir, 'pop')
    data = util.load_pkl(args.save_dir, 'data')
    
    print("Population dictionary keys:", list(pop.keys()))
    print(f"Population size: {len(pop['params'])}")
    
    # Select individuals based on the selection strategy
    if args.selection == "indices":
        try:
            selected_indices = [int(idx) for idx in args.indices.split(",")]
            if len(selected_indices) < args.num_videos:
                print(f"Warning: Only {len(selected_indices)} indices provided, but {args.num_videos} videos requested.")
                args.num_videos = len(selected_indices)
            elif len(selected_indices) > args.num_videos:
                print(f"Warning: {len(selected_indices)} indices provided, but only {args.num_videos} videos requested. Using first {args.num_videos} indices.")
                selected_indices = selected_indices[:args.num_videos]
        except ValueError:
            print("Error parsing indices. Using random selection instead.")
            args.selection = "random"
    
    if args.selection == "best":
        # Find individuals with the lowest loss values across iterations
        # Use the final loss values since they should be the best
        final_loss = data['loss'][-1] if isinstance(data['loss'], np.ndarray) else data['loss']
        selected_indices = np.argsort(final_loss)[:args.num_videos]
        print(f"Selected {args.num_videos} individuals with the lowest loss values")
    
    elif args.selection == "diverse":
        # Select diverse individuals based on latent space embeddings
        if 'z' in pop:
            z = pop['z']
            # Start with the individual that has the lowest loss
            final_loss = data['loss'][-1] if isinstance(data['loss'], np.ndarray) else data['loss']
            selected_indices = [np.argmin(final_loss)]
            
            # Then greedily add the most dissimilar individuals
            remaining = list(range(len(pop['params'])))
            remaining.remove(selected_indices[0])
            
            for _ in range(args.num_videos - 1):
                if not remaining:
                    break
                    
                # For each remaining individual, compute minimum distance to already selected
                min_distances = []
                for idx in remaining:
                    # Compute cosine similarity
                    similarities = z[idx].dot(z[selected_indices].T)
                    # Convert to distance (1 - similarity)
                    distances = 1 - similarities
                    min_distances.append(distances.min())
                
                # Select the individual with the maximum minimum distance
                next_idx = remaining[np.argmax(min_distances)]
                selected_indices.append(next_idx)
                remaining.remove(next_idx)
            
            print(f"Selected {len(selected_indices)} diverse individuals based on latent embeddings")
        else:
            print("Cannot select diverse individuals without latent embeddings. Using random selection.")
            args.selection = "random"
    
    if args.selection == "random":
        # Randomly select individuals
        selected_indices = np.random.choice(len(pop['params']), args.num_videos, replace=False)
        print(f"Randomly selected {args.num_videos} individuals")
    
    # Set up the simulation components
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    # Create the rollout function - now we want to capture the entire video
    rollout_fn = partial(
        rollout_simulation, 
        s0=None, 
        substrate=substrate, 
        fm=fm, 
        rollout_steps=getattr(substrate, 'rollout_steps', 256), 
        time_sampling='video',  # Get all frames
        img_size=224, 
        return_state=False
    )
    rollout_fn = jax.jit(rollout_fn)  # JIT for speed
    
    # Initialize RNG
    rng = jax.random.PRNGKey(0)
    
    # Generate videos for selected individuals
    print(f"Generating {len(selected_indices)} videos...")
    for i, idx in enumerate(tqdm(selected_indices)):
        # Get the parameters for this individual
        params = pop['params'][idx]
        
        # Run the simulation to get all frames
        rng, key = jax.random.split(rng)
        rollout_data = rollout_fn(key, params)
        
        # Extract video frames
        frames = rollout_data['rgb']  # Shape: (time_steps, 224, 224, 3)
        
        # Convert to uint8 for video
        video_frames = (np.array(frames) * 255).astype(np.uint8)
        
        # Calculate any metrics of interest
        z = rollout_data['z']  # Shape: (time_steps, embedding_dim)
        try:
            oe_score = asal_metrics.calc_open_endedness_score(z)
            score_txt = f"_oe{oe_score:.2f}"
        except:
            score_txt = ""
        
        # Save video
        video_path = os.path.join(args.output_dir, f"simulation_{idx}{score_txt}.mp4")
        imageio.mimsave(video_path, video_frames, fps=args.fps)
        print(f"Saved video for individual {idx} to {video_path}")
        
        # Optional: save a thumbnail image of the first, middle and last frame
        first_frame = video_frames[0]
        middle_frame = video_frames[len(video_frames)//2]
        last_frame = video_frames[-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(first_frame)
        axes[0].set_title('First Frame')
        axes[0].axis('off')
        
        axes[1].imshow(middle_frame)
        axes[1].set_title('Middle Frame')
        axes[1].axis('off')
        
        axes[2].imshow(last_frame)
        axes[2].set_title('Last Frame')
        axes[2].axis('off')
        
        fig.suptitle(f"Individual {idx}{score_txt}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"thumbnail_{idx}{score_txt}.png"), dpi=150)
        plt.close()
    
    # Create an HTML gallery for easy viewing
    create_html_gallery(args.output_dir, selected_indices, score_txt)
    
    print(f"Done! Generated {len(selected_indices)} videos in {args.output_dir}")

def create_html_gallery(output_dir, indices, score_txt=""):
    """Create an HTML gallery of videos for easy viewing"""
    html_path = os.path.join(output_dir, "gallery.html")
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Gallery</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
                .item { border: 1px solid #ddd; border-radius: 5px; padding: 10px; }
                video { width: 100%; }
                h2 { margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Simulation Gallery</h1>
            <div class="gallery">
        """)
        
        for idx in indices:
            video_path = f"simulation_{idx}{score_txt}.mp4"
            thumbnail_path = f"thumbnail_{idx}{score_txt}.png"
            
            if os.path.exists(os.path.join(output_dir, video_path)):
                f.write(f"""
                <div class="item">
                    <h2>Individual #{idx}</h2>
                    <video controls loop>
                        <source src="{video_path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <img src="{thumbnail_path}" alt="Thumbnail" style="width: 100%; margin-top: 10px;">
                </div>
                """)
        
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    print(f"Created HTML gallery at {html_path}")

if __name__ == "__main__":
    main()
