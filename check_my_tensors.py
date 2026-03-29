import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch 
import matplotlib.colors as mcolors

cmap = mcolors.ListedColormap(['black', 'gray', 'blue', 'red', 'yellow'])

sample_token = "356d81f38dd9473ba590f39e266f54e5"

def visualize_processed_data(prediction = None, sample_token = sample_token, input_dir='./processed_data/input_bev', gt_dir='./processed_data/ground_truth', ):
    # 1. Load the files
    x_path = Path(input_dir) / f"{sample_token}_x.npy"
    y_path = Path(gt_dir) / f"{sample_token}_y.npy"
    
    x_data = np.load(x_path) # (200, 200, 3)
    y_data = np.load(y_path) # (200, 200)
    
    # 2. Create a plot with 4 subplots
    n_cols = 6 if prediction is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 6))
    
    # Channel 0: Height
    axes[0].imshow(x_data[:, :, 0], cmap='viridis')
    axes[0].set_title("Input: Height Channel")
    
    # Channel 1: Intensity
    axes[1].imshow(x_data[:, :, 1], cmap='magma')
    axes[1].set_title("Input: Intensity Channel")
    
    # Channel 2: Density
    axes[2].imshow(x_data[:, :, 2], cmap='inferno')
    axes[2].set_title("Input: Density Channel")
    
    
    # Channel 3: Roughness 
    # On a good road, this should be mostly DARK (near 0)
    # Curbs, cars, and debris should be BRIGHT (High variance)
    axes[3].imshow(x_data[:, :, 3], cmap='hot')
    axes[3].set_title("Channel 3: Surface Roughness")
    
    # Ground Truth: Classes
    # We use 'tab10' colormap to clearly see different integer IDs
    axes[4].imshow(y_data, cmap='tab10', vmin=0, vmax=3)
    axes[4].set_title("Ground Truth: Class IDs")

    if prediction is not None:
        
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        
        # Ensure it's 2D (H, W)
        if prediction.ndim == 3:
            prediction = prediction.squeeze(0)

        
        
        axes[5].imshow(prediction, cmap='tab10', vmin=0, vmax=10)
        axes[5].set_title("Prediction")
    
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# Run it for one of your ready tokens
if __name__ == "__main__":
    visualize_processed_data("356d81f38dd9473ba590f39e266f54e5")