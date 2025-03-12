import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import torch
from PIL import Image
import matplotlib.cm as cm

def visualize_attention_and_mask(attention_file, mask_path, image_path=None, output_dir='attention_mask_visualization'):
    """
    Visualize the attention heatmap and the 64x64 segmentation mask.
    
    Args:
        attention_file: Path to the saved attention dictionary (.pt file)
        mask_path: Path to the .npy mask file
        image_path: Optional path to the source image for reference
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the mask
    print(f"Loading mask from {mask_path}")
    fg_mask = np.load(mask_path).astype(np.uint8)
    
    # Create 64x64 version of the mask (as used in attention)
    fg_mask_64 = cv2.resize(fg_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # Load the attention data if provided
    if attention_file:
        print(f"Loading attention data from {attention_file}")
        try:
            attn_dict = torch.load(attention_file)
            
            # Extract attention components
            query = attn_dict.get('query', None)
            key = attn_dict.get('key', None)
            attention_name = attn_dict.get('attention_name', 'Unknown')
            
            if query is not None and key is not None:
                # Compute attention scores for the first batch item
                scale_factor = 1 / np.sqrt(key.shape[-1])
                attn_weights = torch.matmul(query[0], key[0].transpose(-2, -1)) * scale_factor
                
                # Apply softmax to get attention probabilities
                attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
                
                # Average across heads
                avg_attn = attn_probs.mean(0).cpu().numpy()
                
                # Determine the feature size
                key_seq_len = key.shape[2]
                feature_size = int(np.sqrt(key_seq_len // 2))  # Divide by 2 because we have A and B parts
                
                # Split into A and B parts
                half_seq_len = feature_size * feature_size
                avg_attn_A = avg_attn[:half_seq_len].reshape(feature_size, feature_size)
                avg_attn_B = avg_attn[half_seq_len:half_seq_len*2].reshape(feature_size, feature_size)
                
                has_attention_data = True
            else:
                print("Warning: Query or key data missing in attention file")
                has_attention_data = False
        except Exception as e:
            print(f"Error loading attention data: {e}")
            has_attention_data = False
    else:
        has_attention_data = False
    
    # Create a figure for the mask visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(fg_mask_64, cmap='gray')
    plt.title('64x64 Segmentation Mask')
    plt.colorbar(label='Mask Value')
    
    # Add grid for better visualization
    plt.grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Save the mask visualization
    plt.savefig(f"{output_dir}/segmentation_mask_64x64.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a binary version of the mask for clearer visualization
    binary_mask = (fg_mask_64 > 0).astype(np.uint8)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, cmap='gray')
    plt.title('64x64 Binary Segmentation Mask')
    
    # Add grid for better visualization
    plt.grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Save the binary mask visualization
    plt.savefig(f"{output_dir}/binary_segmentation_mask_64x64.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # If we have attention data, visualize it
    if has_attention_data:
        # Create a figure with 2 subplots for attention A and B
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Visualize attention A
        im_A = axes[0].imshow(avg_attn_A, cmap='hot')
        axes[0].set_title(f'Self-Attention (A) - {attention_name}')
        axes[0].grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
        fig.colorbar(im_A, ax=axes[0], label='Attention Weight')
        
        # Visualize attention B
        im_B = axes[1].imshow(avg_attn_B, cmap='hot')
        axes[1].set_title(f'Cross-Attention (B) - {attention_name}')
        axes[1].grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
        fig.colorbar(im_B, ax=axes[1], label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/attention_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create a figure showing the mask and attention side by side
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Show binary mask
        axes[0].imshow(binary_mask, cmap='gray')
        axes[0].set_title('Binary Segmentation Mask (64x64)')
        axes[0].grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Show attention A
        axes[1].imshow(avg_attn_A, cmap='hot')
        axes[1].set_title(f'Self-Attention (A)')
        axes[1].grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Show attention B
        axes[2].imshow(avg_attn_B, cmap='hot')
        axes[2].set_title(f'Cross-Attention (B)')
        axes[2].grid(True, color='blue', alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mask_and_attention_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # If image is provided, show mask overlay on image
    if image_path:
        try:
            img = Image.open(image_path)
            img_np = np.array(img)
            
            # Resize mask to match image dimensions
            img_height, img_width = img_np.shape[:2]
            fg_mask_full = cv2.resize(fg_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # Create a figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot original image
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Create colored mask overlay
            mask_overlay = np.zeros_like(img_np)
            mask_overlay[fg_mask_full > 0] = [255, 0, 0]  # Red for foreground
            
            # Blend the mask with the original image
            alpha = 0.5
            blended = cv2.addWeighted(img_np, 1, mask_overlay, alpha, 0)
            axes[1].imshow(blended)
            axes[1].set_title('Image with Mask Overlay')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/image_with_mask_overlay.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize attention heatmap and segmentation mask')
    parser.add_argument('--attention', help='Path to the attention data file (.pt)')
    parser.add_argument('--mask', required=True, help='Path to the .npy mask file')
    parser.add_argument('--image', help='Path to the source image (optional)')
    parser.add_argument('--output_dir', default='attention_mask_visualization', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_attention_and_mask(args.attention, args.mask, args.image, args.output_dir)