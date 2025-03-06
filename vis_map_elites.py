from PIL import Image
import cv2
import numpy as np
import argparse
import jax
import jax.numpy as jnp
from einops import rearrange
import matplotlib.pyplot as plt
import substrates
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data", help="path to save results to")
    parser.add_argument("--substrate", type=str, default="lenia", help="substrate to use")
    parser.add_argument("--noun_file", type=str, default="noun_list.txt", help="path of noun file")
    args = parser.parse_args()
    save_dir = args.save_dir

    # Load nouns WITHOUT applying prompt template (just use raw nouns)
    with open(args.noun_file, 'r') as f:
        nouns = f.read().strip().split('\n')

    # Create substrate
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    # Load archive data
    archive = util.load_pkl(save_dir, 'archive')
    data = util.load_pkl(save_dir, 'data')

    # Create plots for quality and transfers
    plt.figure(figsize=(16, 6))
    
    # 1. Quality vs iteration
    plt.subplot(1, 2, 1)
    iterations = np.arange(len(data['avg_quality'])) * 100  # Multiply by 100 since each data point represents 100 iterations
    plt.plot(iterations, data['avg_quality'])
    plt.xlabel('Iterations')
    plt.ylabel('Average Quality')
    plt.title('Quality vs Iterations')
    plt.grid(True)
    
    # 2. Number of transfers vs iteration
    plt.subplot(1, 2, 2)
    cumulative_transfers = np.cumsum(data['n_transfers'])
    plt.plot(iterations, cumulative_transfers)
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Number of Transfers')
    plt.title('Cumulative Transfers vs Iterations')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_plot.png", dpi=300)
    plt.close()
    

    # Initialize rng
    rng = jax.random.PRNGKey(0)
    
    # # Define function to render each parameter set
    # def render_fn(params):
    #     state = substrate.init_state(rng, params)
    #     img = substrate.render_state(state, params)
    #     return img
    
    # # Use vmap to render all images
    # imgs = jax.vmap(render_fn)(archive['pheno']['params'])
    # imgs = np.array(imgs)

    imgs = np.array(archive['pheno']['img'])
    print(imgs.shape)
    
    # Determine dimensions
    num_images = min(6800, len(imgs))
    cols = 100
    rows = (num_images + cols - 1) // cols
    
    # Truncate nouns to match number of images
    nouns = nouns[:num_images]
    
    # Create poster using the original layout approach
    poster = imgs[:num_images]
    poster = np.pad(poster, ((0, 0), (35, 10), (5, 5), (0, 0)), constant_values=1.)
    poster = rearrange(poster, "(R C) H W D -> (R H) (C W) D", R=rows, C=cols)
    
    img_height = imgs.shape[1]
    img_width = imgs.shape[2]
    
    for i in range(num_images):
        y, x = divmod(i, cols)
        x_pos = x * (img_width + 10)
        y_pos = y * (img_height + 45)
        
        # Display just the noun, not the full prompt
        if i < len(nouns):
            txt = nouns[i]
            if len(txt) > 15:  # Truncate long text
                txt = txt[:12] + "..."
            
            cv2.putText(poster, txt, (x_pos + 5, y_pos + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Score display
        txt = f"{archive['quality'][i].item():.3f}"
        cv2.putText(poster, txt, (x_pos + 5, y_pos + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    Image.fromarray((poster * 255).astype('uint8')).save(f"{save_dir}/poster.png")
