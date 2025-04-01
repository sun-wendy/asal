import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle
import os
import argparse
import umap
from sklearn.preprocessing import MinMaxScaler

def load_pkl(save_dir, name):
    with open(os.path.join(save_dir, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data", help="path to load data from and save results to")
    parser.add_argument("--n_neighbors", type=int, default=15, help="n_neighbors parameter for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1, help="min_dist parameter for UMAP")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 10], help="figure size (width, height)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--sample_size", type=int, default=None, help="number of samples to use (None for all)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    
    print(f"Loading data from {args.save_dir}...")
    
    # Load population data
    pop = load_pkl(args.save_dir, 'pop')
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
    if args.sample_size is not None and args.sample_size < len(clip_embeddings):
        np.random.seed(args.random_seed)
        indices = np.random.choice(len(clip_embeddings), args.sample_size, replace=False)
        clip_embeddings = clip_embeddings[indices]
        oe_scores = oe_scores[indices]
        print(f"Subsampled to {args.sample_size} examples")
    
    # Normalize the OE scores for coloring (0 to 1 range)
    scaler = MinMaxScaler()
    oe_scores_normalized = scaler.fit_transform(oe_scores.reshape(-1, 1)).flatten()
    
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
    
    # Plot the UMAP projection
    plt.figure(figsize=tuple(args.figsize))
    
    # Create a colormap similar to the one in your example
    cmap = plt.cm.plasma
    
    # Create the scatter plot
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=oe_scores_normalized,
        cmap=cmap, 
        s=10,  # Marker size
        alpha=0.7  # Transparency
    )
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Open-Endedness Score', rotation=270, labelpad=20)
    
    # Add labels at min and max
    cbar.ax.text(0, -0.05, 'low', transform=cbar.ax.transAxes, ha='center')
    cbar.ax.text(0, 1.05, 'high', transform=cbar.ax.transAxes, ha='center')
    
    # Set labels and title
    plt.title('Map of All Life-Like CAs', fontsize=24)
    plt.xlabel('CLIP UMAP 1', fontsize=18)
    plt.ylabel('CLIP UMAP 2', fontsize=18)
    
    # Save the figure
    output_path = os.path.join(args.save_dir, "clip_umap_visualization.png")
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Optional: Save the UMAP embeddings for future use
    np.save(os.path.join(args.save_dir, "umap_embeddings.npy"), embedding)
    print(f"UMAP embeddings saved to {os.path.join(args.save_dir, 'umap_embeddings.npy')}")

if __name__ == "__main__":
    main()
