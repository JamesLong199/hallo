import torch
import numpy as np
import os
import math
import argparse
from tqdm import tqdm
import glob

def compute_attention_statistics(file_path, output_dir='attention_statistics'):
    """
    Compute statistics about attention distributions across all pixels in image feature A.
    
    Args:
        file_path: Path to the saved attention dictionary
        output_dir: Directory to save statistics
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the saved dictionary
    print(f"Loading attention data from {file_path}")
    attn_dict = torch.load(file_path)
    
    # Extract components
    query = attn_dict['query'].float()  # Convert to float32
    key = attn_dict['key'].float()      # Convert to float32
    heads = attn_dict['heads']
    head_dim = attn_dict['head_dim']
    attention_name = attn_dict['attention_name']
    
    # Print shapes for debugging
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Number of heads: {heads}")
    print(f"Head dimension: {head_dim}")
    
    # Get number of heads from the tensor shape
    num_heads = query.shape[1]
    
    # Determine the feature size from the key shape
    key_seq_len = key.shape[2]
    feature_size = int(math.sqrt(key_seq_len // 2))  # Divide by 2 because we have A and B parts
    print(f"Detected feature size: {feature_size}x{feature_size}")
    
    # Compute attention scores for the first batch item
    scale_factor = 1 / math.sqrt(head_dim)
    
    # Compute attention weights
    print("Computing attention weights...")
    attn_weights = torch.matmul(query[0], key[0].transpose(-2, -1)) * scale_factor
    
    # Apply softmax to get attention probabilities
    print("Applying softmax...")
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Initialize arrays to store statistics
    half_seq_len = feature_size * feature_size
    
    # Arrays to store per-head statistics
    attn_A_sum_per_head = np.zeros(num_heads)
    attn_B_sum_per_head = np.zeros(num_heads)
    
    # Process each pixel in image feature A
    print(f"Processing attention for all {feature_size}x{feature_size} pixels...")
    for y in tqdm(range(feature_size)):
        for x in range(feature_size):
            pixel_idx = y * feature_size + x
            
            # Extract attention map for this pixel
            pixel_attn = attn_probs[:, pixel_idx, :].cpu().numpy()
            
            # Split the attention into A and B parts
            pixel_attn_A = pixel_attn[:, :half_seq_len].reshape(num_heads, feature_size, feature_size)
            pixel_attn_B = pixel_attn[:, half_seq_len:half_seq_len*2].reshape(num_heads, feature_size, feature_size)
            
            # Add to the per-head sums
            attn_A_sum_per_head += pixel_attn_A.sum(axis=(1, 2))
            attn_B_sum_per_head += pixel_attn_B.sum(axis=(1, 2))
    
    # Calculate averages
    total_pixels = feature_size * feature_size
    attn_A_avg_per_head = attn_A_sum_per_head / total_pixels
    attn_B_avg_per_head = attn_B_sum_per_head / total_pixels
    
    # Calculate overall averages
    attn_A_overall_avg = attn_A_avg_per_head.mean()
    attn_B_overall_avg = attn_B_avg_per_head.mean()
    
    # Print statistics
    print("\nAttention Statistics:")
    print(f"Feature size: {feature_size}x{feature_size}")
    print(f"Number of heads: {num_heads}")
    print("\nAverage attention sum per head:")
    for h in range(num_heads):
        print(f"  Head {h}: A={attn_A_avg_per_head[h]:.6f}, B={attn_B_avg_per_head[h]:.6f}, Ratio A/B={attn_A_avg_per_head[h]/attn_B_avg_per_head[h] if attn_B_avg_per_head[h] > 0 else float('inf'):.6f}")
    
    print(f"\nOverall average attention sum: A={attn_A_overall_avg:.6f}, B={attn_B_overall_avg:.6f}")
    print(f"Overall ratio A/B: {attn_A_overall_avg/attn_B_overall_avg if attn_B_overall_avg > 0 else float('inf'):.6f}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, f"{attention_name}_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Attention Statistics for {attention_name}\n")
        f.write(f"Feature size: {feature_size}x{feature_size}\n")
        f.write(f"Number of heads: {num_heads}\n\n")
        f.write("Average attention sum per head:\n")
        for h in range(num_heads):
            f.write(f"  Head {h}: A={attn_A_avg_per_head[h]:.6f}, B={attn_B_avg_per_head[h]:.6f}, Ratio A/B={attn_A_avg_per_head[h]/attn_B_avg_per_head[h] if attn_B_avg_per_head[h] > 0 else float('inf'):.6f}\n")
        
        f.write(f"\nOverall average attention sum: A={attn_A_overall_avg:.6f}, B={attn_B_overall_avg:.6f}\n")
        f.write(f"Overall ratio A/B: {attn_A_overall_avg/attn_B_overall_avg if attn_B_overall_avg > 0 else float('inf'):.6f}\n")
    
    print(f"Saved statistics to {stats_file}")
    
    return {
        'feature_size': feature_size,
        'num_heads': num_heads,
        'attn_A_avg_per_head': attn_A_avg_per_head,
        'attn_B_avg_per_head': attn_B_avg_per_head,
        'attn_A_overall_avg': attn_A_overall_avg,
        'attn_B_overall_avg': attn_B_overall_avg
    }


def process_multiple_files(file_paths, output_dir='attention_statistics_comparison'):
    """
    Process multiple attention files and create a summary of statistics.
    
    Args:
        file_paths: List of paths to attention files
        output_dir: Directory to save statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = {}
    summary_file = os.path.join(output_dir, "attention_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Summary of Attention Statistics Across Files\n")
        f.write("==========================================\n\n")
        
        for file_path in file_paths:
            attention_name = os.path.basename(file_path).split('.')[0]
            print(f"\nProcessing {attention_name}...")
            
            # Compute statistics for this file
            stats = compute_attention_statistics(file_path, output_dir)
            all_stats[attention_name] = stats
            
            # Write summary for this file
            f.write(f"File: {attention_name}\n")
            f.write(f"Feature size: {stats['feature_size']}x{stats['feature_size']}\n")
            f.write(f"Number of heads: {stats['num_heads']}\n")
            f.write(f"Overall average attention sum: A={stats['attn_A_overall_avg']:.6f}, B={stats['attn_B_overall_avg']:.6f}\n")
            f.write(f"Overall ratio A/B: {stats['attn_A_overall_avg']/stats['attn_B_overall_avg'] if stats['attn_B_overall_avg'] > 0 else float('inf'):.6f}\n\n")
    
    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute attention statistics')
    parser.add_argument('--files', nargs='+', help='Path(s) to attention data file(s)')
    parser.add_argument('--dir', help='Directory containing attention data files')
    parser.add_argument('--pattern', default='*spatial_0_attn1.pt', help='Pattern to match files in directory')
    parser.add_argument('--output_dir', default='attention_statistics', help='Directory to save statistics')
    
    args = parser.parse_args()
    
    # Get files from directory and pattern if specified
    if args.dir:
        file_pattern = os.path.join(args.dir, args.pattern)
        files = glob.glob(file_pattern)
        if not files:
            print(f"No files found matching pattern {file_pattern}")
            exit(1)
        print(f"Found {len(files)} files matching pattern {args.pattern} in {args.dir}")
    elif args.files:
        files = args.files
    else:
        parser.error("Either --files or --dir must be specified")
    
    if len(files) == 1:
        compute_attention_statistics(files[0], args.output_dir)
    else:
        process_multiple_files(files, args.output_dir)