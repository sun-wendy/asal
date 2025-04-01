import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle
import os
import argparse
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from PIL import Image
import jax
import jax.numpy as jnp
from functools import partial
from jax.random import split
import substrates
import foundation_models
from rollout import rollout_simulation
import util
from tqdm import tqdm

def select_representative_samples(embeddings, n_samples=12, method='kmeans', random_seed=42):
    """
    Select representative samples from the embedding space.
    
    Args:
        embeddings: UMAP embeddings of shape (n_samples, 2)
        n_samples: Number of representative samples to select
        method: Method to use for selection ('kmeans', 'uniform', or 'random')
        random_seed: Random seed for reproducibility
        
    Returns:
        indices: Indices of selected samples
    """
    if method == 'kmeans':
        # Use K-means to find cluster centers
        kmeans = KMeans(n_clusters=n_samples, random_state=random_seed, n_init=10)
        kmeans.fit(embeddings)
        
        # Find the closest point to each cluster center
        closest_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(embeddings - center, axis=1)
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        return np.array(closest_indices)
    
    elif method == 'uniform':
        # Select samples uniformly across the embedding space
        x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
        
        # Create a grid of points
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        
        selected_indices = []
        for i in range(min(n_samples, grid_size * grid_size)):
            x_idx = i % grid_size
            y_idx = i // grid_size
            
            # Find the closest point to this grid location
            grid_point = np.array([x_grid[x_idx], y_grid[y_idx]])
            distances = np.linalg.norm(embeddings - grid_point, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if this point is already selected
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
        
        return np.array(selected_indices)
    
    else:
        # Random selection
        np.random.seed(random_seed)
        return np.random.choice(len(embeddings), n_samples, replace=False)

def generate_images(pop, indices, substrate_name, foundation_model_name):
    """
    Generate images for the selected indices using the simulation.
    
    Args:
        pop: Population data dictionary containing parameters
        indices: Indices of samples to generate images for
        substrate_name: Name of the substrate to use
        foundation_model_name: Name of the foundation model to use
        
    Returns:
        images: Array of generated images
    """
    # Set up the simulation components
    fm = foundation_models.create_foundation_model(foundation_model_name)
    substrate = substrates.create_substrate(substrate_name)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    # Create the rollout function
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
    
    # Initialize RNG
    rng = jax.random.PRNGKey(0)
    
    # Run simulations to get images for the selected parameters
    images = []
    for i in tqdm(range(len(indices)), desc="Generating images"):
        rng, key = jax.random.split(rng)
        params = pop['params'][indices[i]]
        rollout_data = rollout_fn(key, params)
        rgb = rollout_data['rgb']  # Shape: (224, 224, 3)
        images.append(rgb)
    
    images = np.array(images)
    
    # Convert images to uint8 if they're in [0,1] range
    if images.dtype == np.float32 or images.dtype == np.float64:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
    
    return images

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualization with example CA images")
    
    # Input/output parameters
    parser.add_argument("--save_dir", type=str, default="data", help="path to save results to")
    parser.add_argument("--output_filename", type=str, default="clip_umap_visualization.png", 
                        help="filename for the output visualization")
    
    # Substrate and model parameters
    parser.add_argument("--substrate", type=str, default="plife_plus", help="substrate to use")
    parser.add_argument("--foundation_model", type=str, default="clip", help="foundation model to use")
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=15, help="n_neighbors parameter for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1, help="min_dist parameter for UMAP")
    parser.add_argument("--use_cached_umap", action="store_true", help="use cached UMAP embeddings if available")
    
    # Sampling parameters
    parser.add_argument("--sample_size", type=int, default=None, help="number of samples to use (None for all)")
    parser.add_argument("--n_samples", type=int, default=12, help="number of representative samples to show")
    parser.add_argument("--sample_method", type=str, default="kmeans", choices=["kmeans", "uniform", "random"], 
                        help="method to select samples")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    
    # Visualization parameters
    parser.add_argument("--figsize", type=float, nargs=2, default=[16, 14], help="figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--sample_size_inches", type=float, default=1.5, help="size of sample images in inches")
    parser.add_argument("--marker_size", type=int, default=10, help="size of scatter points")
    parser.add_argument("--marker_alpha", type=float, default=0.7, help="transparency of scatter points")
    parser.add_argument("--colormap", type=str, default="plasma", help="matplotlib colormap to use")
    parser.add_argument("--add_arrows", action="store_true", help="add arrows pointing to samples")
    parser.add_argument("--add_legend", action="store_true", help="add legend for rule types")
    parser.add_argument("--bg_color", type=str, default=None, help="background color (e.g., 'black', '#FFFFFF')")
    parser.add_argument("--axis_limits", type=float, nargs=4, default=None, 
                        help="axis limits [xmin, xmax, ymin, ymax]")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.save_dir}...")
    
    # Load population data
    pop = util.load_pkl(args.save_dir, 'pop')
    print("Population keys:", list(pop.keys()))
    
    # Extract the CLIP embeddings and open-endedness scores
    if 'z' in pop:
        clip_embeddings = pop['z']
        print(f"CLIP embeddings shape: {clip_embeddings.shape}")
        
        # For UMAP, we'll use the last time step if there are multiple
        if len(clip_embeddings.shape) > 2:
            clip_embeddings = clip_embeddings[:, -1, :]
            print(f"Using last time step embeddings: {clip_embeddings.shape}")
    else:
        raise ValueError("CLIP embeddings ('z') not found in population data")
    
    if 'oe_score' in pop:
        oe_scores = pop['oe_score']
        print(f"Open-endedness scores shape: {oe_scores.shape}")
        print(f"OE score range: {oe_scores.min()} to {oe_scores.max()}")
    else:
        print("Warning: Open-endedness scores ('oe_score') not found, using default values")
        oe_scores = np.zeros(len(clip_embeddings))
    
    # Subsample if needed
    indices = None
    if args.sample_size is not None and args.sample_size < len(clip_embeddings):
        np.random.seed(args.random_seed)
        indices = np.random.choice(len(clip_embeddings), args.sample_size, replace=False)
        clip_embeddings = clip_embeddings[indices]
        oe_scores = oe_scores[indices]
        print(f"Subsampled to {args.sample_size} examples")
    
    # Normalize the OE scores for coloring (0 to 1 range)
    scaler = MinMaxScaler()
    oe_scores_normalized = scaler.fit_transform(oe_scores.reshape(-1, 1)).flatten()
    
    # Check if we have cached UMAP embeddings
    umap_cache_path = os.path.join(args.save_dir, "umap_embeddings.npy")
    if args.use_cached_umap and os.path.exists(umap_cache_path):
        print(f"Loading cached UMAP embeddings from {umap_cache_path}")
        embedding = np.load(umap_cache_path)
        # If we subsampled, we need to filter the embeddings
        if indices is not None:
            embedding = embedding[indices]
    else:
        print("Performing UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            random_state=args.random_seed
        )
        
        # Fit and transform the embeddings
        embedding = reducer.fit_transform(clip_embeddings)
        print(f"UMAP embedding shape: {embedding.shape}")
        
        # Save the embeddings for future use
        np.save(umap_cache_path, embedding)
        print(f"UMAP embeddings saved to {umap_cache_path}")
    
    # Select representative samples
    selected_indices = select_representative_samples(
        embedding, 
        n_samples=args.n_samples, 
        method=args.sample_method,
        random_seed=args.random_seed
    )
    print(f"Selected {len(selected_indices)} representative samples")
    
    # Create a mapping from local to global indices if we subsampled
    if indices is not None:
        global_selected_indices = indices[selected_indices]
    else:
        global_selected_indices = selected_indices
    
    # Generate images for selected samples
    print("Generating images for selected samples...")
    sample_images = generate_images(
        pop, 
        global_selected_indices, 
        args.substrate, 
        args.foundation_model
    )
    
    # Create the plot
    plt.figure(figsize=tuple(args.figsize))
    
    # Set background color if specified
    if args.bg_color:
        plt.gca().set_facecolor(args.bg_color)
        plt.gcf().set_facecolor(args.bg_color)
    
    # Create the scatter plot
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=oe_scores_normalized,
        cmap=args.colormap, 
        s=args.marker_size,
        alpha=args.marker_alpha
    )
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Open-Endedness Score', rotation=270, labelpad=20)
    
    # Add labels at min and max
    cbar.ax.text(0, -0.05, 'low', transform=cbar.ax.transAxes, ha='center')
    cbar.ax.text(0, 1.05, 'high', transform=cbar.ax.transAxes, ha='center')
    
    # Add a legend for rule types if requested
    if args.add_legend:
        rule_types = [
            "B3/S23",
            "B0136/S034678",
            "B013456/S123",
            "B01356/S0123",
            "B012345/S02356",
            "B0145/S01234"
        ]
        rule_colors = [
            "royalblue",
            "firebrick",
            "forestgreen",
            "khaki",
            "turquoise",
            "saddlebrown"
        ]
        
        legend_elements = []
        for color, rule in zip(rule_colors, rule_types):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, label=rule))
        
        plt.legend(handles=legend_elements, loc='upper left', title="Rule Types")
    
    # Add sample images to the plot
    for i, (idx, img) in enumerate(zip(selected_indices, sample_images)):
        # Create an OffsetImage
        imagebox = OffsetImage(img, zoom=args.sample_size_inches)
        
        # Create an AnnotationBbox
        ab = AnnotationBbox(
            imagebox, 
            (embedding[idx, 0], embedding[idx, 1]),
            xycoords='data',
            frameon=True,
            pad=0.1,
            boxcoords="offset points"
        )
        
        # Add the annotation to the plot
        plt.gca().add_artist(ab)
        
        # Add an arrow if requested
        if args.add_arrows:
            # Calculate arrow direction (away from the center)
            x_start = embedding[idx, 0]
            y_start = embedding[idx, 1]
            x_end = x_start + 0.5
            y_end = y_start + 0.5
            
            plt.arrow(
                x_end, y_end,  # Start point
                x_start - x_end, y_start - y_end,  # Direction vector
                head_width=0.2,
                head_length=0.3,
                fc='black',
                ec='black',
                length_includes_head=True
            )
    
    # Set labels and title
    plt.title('Map of All Life-Like CAs', fontsize=24)
    plt.xlabel('CLIP UMAP 1', fontsize=18)
    plt.ylabel('CLIP UMAP 2', fontsize=18)
    
    # Set axis limits if specified
    if args.axis_limits:
        plt.xlim(args.axis_limits[0], args.axis_limits[1])
        plt.ylim(args.axis_limits[2], args.axis_limits[3])
    
    # Save the figure
    output_path = os.path.join(args.save_dir, args.output_filename)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Save the selected indices
    np.save(os.path.join(args.save_dir, "selected_sample_indices.npy"), global_selected_indices)
    print(f"Selected sample indices saved to {os.path.join(args.save_dir, 'selected_sample_indices.npy')}")
    
    # Optionally save the generated images
    sample_images_path = os.path.join(args.save_dir, "sample_images.npy")
    np.save(sample_images_path, sample_images)
    print(f"Sample images saved to {sample_images_path}")
    
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
