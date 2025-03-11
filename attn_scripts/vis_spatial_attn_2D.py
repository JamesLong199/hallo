import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import glob
import argparse
from PIL import Image, ImageOps
import matplotlib.cm as cm
from tqdm import tqdm

def overlay_attention_on_image(file_path, image_path, save_dir='attention_overlays', pixel_indices=None):
    """
    Visualize the attention map for specified pixels overlaid on the grayscale source image.
    Normalizes A and B attention maps jointly for direct comparison of attention strengths.
    
    Args:
        file_path: Path to the saved attention dictionary
        image_path: Path to the source image
        save_dir: Directory to save visualizations
        pixel_indices: List of (y, x) tuples for pixels to visualize. If None, only center pixel is used.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the saved dictionary
    print(f"Loading attention data from {file_path}")
    attn_dict = torch.load(file_path)
    
    # Extract components
    query = attn_dict['query'].float()  # Convert to float32
    key = attn_dict['key'].float()      # Convert to float32
    value = attn_dict['value'].float()  # Convert to float32
    heads = attn_dict['heads']
    head_dim = attn_dict['head_dim']
    attention_name = attn_dict['attention_name']
    
    # Print shapes for debugging
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Number of heads: {heads}")
    print(f"Head dimension: {head_dim}")
    
    # Get number of heads from the tensor shape
    num_heads = query.shape[1]
    
    # Compute attention scores for the first batch item
    scale_factor = 1 / math.sqrt(head_dim)
    
    # Compute attention weights
    print("Computing attention weights...")
    attn_weights = torch.matmul(query[0], key[0].transpose(-2, -1)) * scale_factor
    
    # Apply softmax to get attention probabilities
    print("Applying softmax...")
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Determine the feature size from the key shape instead of query
    # Key shape is [batch, heads, seq_len, head_dim]
    key_seq_len = key.shape[2]
    feature_size = int(math.sqrt(key_seq_len // 2))  # Divide by 2 because we have A and B parts
    print(f"Detected feature size: {feature_size}x{feature_size}")
    
    # If no pixel indices provided, use just the center pixel
    if pixel_indices is None:
        # Find the center pixel index in the grid
        center_y, center_x = feature_size // 2, feature_size // 2
        pixel_indices = [(center_y, center_x)]
    
    # Load and resize the source image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img_gray = ImageOps.grayscale(img)
    
    # Create separate overlays for A and B parts
    img_A = img_gray.resize((feature_size, feature_size), Image.LANCZOS)
    img_B = img_gray.resize((feature_size, feature_size), Image.LANCZOS)
    
    # Convert to numpy arrays
    img_A_np = np.array(img_A) / 255.0  # Normalize to [0, 1]
    img_B_np = np.array(img_B) / 255.0
    
    # Expand grayscale to 3 channels for blending
    img_A_np = np.stack([img_A_np, img_A_np, img_A_np], axis=2)
    img_B_np = np.stack([img_B_np, img_B_np, img_B_np], axis=2)
    
    # Dim the grayscale image to make attention more visible
    img_A_dimmed = img_A_np * 0.4  # Reduce brightness
    img_B_dimmed = img_B_np * 0.4
    
    # Process each pixel
    for y, x in pixel_indices:
        pixel_idx = y * feature_size + x
        
        # Create a subdirectory for this pixel
        # pixel_dir = os.path.join(save_dir, f"pixel_y{y}_x{x}")
        pixel_dir = save_dir
        os.makedirs(pixel_dir, exist_ok=True)
        
        # Extract attention map for this pixel
        pixel_attn = attn_probs[:, pixel_idx, :].cpu().numpy()
        
        # Split the attention into A and B parts
        half_seq_len = feature_size * feature_size
        pixel_attn_A = pixel_attn[:, :half_seq_len].reshape(num_heads, feature_size, feature_size)
        pixel_attn_B = pixel_attn[:, half_seq_len:half_seq_len*2].reshape(num_heads, feature_size, feature_size)
        
        print(f'Processing pixel at y={y}, x={x}')
        
        # Also create an average across all heads
        avg_attn_A = pixel_attn_A.mean(0)
        avg_attn_B = pixel_attn_B.mean(0)
        
        # Create a figure with 2 subplots for the average
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Find the global max across both A and B for the average
        global_max_avg = max(np.max(avg_attn_A), np.max(avg_attn_B))
        avg_attn_A_norm = avg_attn_A / global_max_avg if global_max_avg > 0 else avg_attn_A
        avg_attn_B_norm = avg_attn_B / global_max_avg if global_max_avg > 0 else avg_attn_B
        
        # Create colored heatmaps
        cmap = cm.get_cmap('hot')  # 'hot' colormap is more visible on grayscale
        avg_attn_A_colored = cmap(avg_attn_A_norm)[:, :, :3]
        avg_attn_B_colored = cmap(avg_attn_B_norm)[:, :, :3]
        
        # Blend with grayscale images
        overlay_A_avg = img_A_dimmed * 0.5 + avg_attn_A_colored * 0.5
        overlay_B_avg = img_B_dimmed * 0.5 + avg_attn_B_colored * 0.5
        
        axes[0].imshow(overlay_A_avg)
        axes[0].set_title(f'Average Across Heads, Self-Attention (A)\nSum: {avg_attn_A.sum():.3f}')
        axes[0].axis('off')
        axes[0].plot(x, y, 'gx', markersize=10)
        
        axes[1].imshow(overlay_B_avg)
        axes[1].set_title(f'Average Across Heads, Cross-Attention (B)\nSum: {avg_attn_B.sum():.3f}')
        axes[1].axis('off')
        axes[1].plot(x, y, 'gx', markersize=10)
        
        # Add text about joint normalization
        plt.figtext(0.5, 0.01, f"Attention maps normalized jointly (max={global_max_avg:.6f})", 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
        
        plt.tight_layout()
        plt.savefig(f"{pixel_dir}/y{y}_x{x}_avg.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved attention overlay visualizations to {save_dir}")


def determine_stride_from_filename(file_path, default_stride=8):
    """
    Determine the appropriate stride based on the file name.
    
    Args:
        file_path: Path to the attention file
        default_stride: Default stride to use if no pattern matches
        
    Returns:
        int: The stride to use
    """
    file_name = os.path.basename(file_path)
    
    if 'downblock_0' in file_name or 'upblock_3' in file_name:
        return 8
    elif 'downblock_1' in file_name or 'upblock_2' in file_name:
        return 4
    elif 'downblock_2' in file_name or 'upblock_1' in file_name:
        return 2
    elif 'mid_block' in file_name:
        return 1
    else:
        return default_stride


def visualize_all_pixels(file_path, image_path, save_dir='attention_all_pixels', stride=None):
    """
    Visualize attention maps for all pixels in the feature grid with a given stride.
    
    Args:
        file_path: Path to the saved attention dictionary
        image_path: Path to the source image
        save_dir: Directory to save visualizations
        stride: Pixel stride to use (to avoid creating too many visualizations).
               If None, stride will be determined based on the file name.
    """
    # If stride is None, determine it from the file name
    if stride is None:
        stride = determine_stride_from_filename(file_path)
    
    # Load the dictionary to determine feature size
    attn_dict = torch.load(file_path)
    key = attn_dict['key']
    key_seq_len = key.shape[2]
    feature_size = int(math.sqrt(key_seq_len // 2))  # Divide by 2 because we have A and B parts
    
    # Generate pixel indices with the given stride
    pixel_indices = []
    for y in range(0, feature_size, stride):
        for x in range(0, feature_size, stride):
            pixel_indices.append((y, x))
    
    print(f"Feature size: {feature_size}x{feature_size}")
    print(f"Using stride: {stride} (determined from file name)")
    print(f"Visualizing attention for {len(pixel_indices)} pixels")
    
    overlay_attention_on_image(file_path, image_path, save_dir, pixel_indices)


def process_folder(folder_path, image_path, output_base_dir='attention_visualizations', pattern='*spatial_0_attn1.pt', default_stride=8):
    """
    Process all attention files in a folder that match the given pattern.
    
    Args:
        folder_path: Path to the folder containing attention files
        image_path: Path to the source image
        output_base_dir: Base directory for saving visualizations
        pattern: Pattern to match files (default: '*spatial_0_attn1.pt')
        default_stride: Default stride for pixel sampling (will be overridden based on file name)
    """
    # Find all matching files
    file_pattern = os.path.join(folder_path, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern {file_pattern}")
        return
    
    print(f"Found {len(files)} files matching pattern {pattern} in {folder_path}")
    
    # Process each file
    for file_path in tqdm(files, desc="Processing attention files"):
        # Extract file name without extension for the output directory
        file_name = os.path.basename(file_path).split('.')[0]
        
        # Create output directory for this file
        save_dir = os.path.join(output_base_dir, file_name)
        
        # Determine stride based on file name
        stride = determine_stride_from_filename(file_path, default_stride)
        
        print(f"\nProcessing {file_path} with stride {stride}...")
        visualize_all_pixels(file_path, image_path, save_dir, stride)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize spatial attention maps')
    parser.add_argument('--file', help='Path to a single attention data file')
    parser.add_argument('--dir', help='Directory containing attention data files')
    parser.add_argument('--image', required=True, help='Path to the source image')
    parser.add_argument('--pattern', default='*spatial_0_attn1.pt', help='Pattern to match files in directory')
    parser.add_argument('--output_dir', default='attention_visualizations', help='Base directory to save visualizations')
    parser.add_argument('--stride', type=int, help='Override stride for pixel sampling (if not specified, determined from file name)')
    
    args = parser.parse_args()
    
    if args.file:
        # Process a single file
        file_name = os.path.basename(args.file).split('.')[0]
        save_dir = os.path.join(args.output_dir, file_name)
        visualize_all_pixels(args.file, args.image, save_dir, args.stride)
    elif args.dir:
        # Process all matching files in the directory
        process_folder(args.dir, args.image, args.output_dir, args.pattern, args.stride)
    else:
        parser.error("Either --file or --dir must be specified")