from typing import Any, Dict, List, Optional

import torch
from diffusers.models.attention import (AdaLayerNorm, AdaLayerNormZero,
                                        Attention, FeedForward)
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange
from torch import nn
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class AttentionScoreSavingProcessor(AttnProcessor):
    """Custom attention processor that saves intermediate attention scores."""
    
    def __init__(self):
        super().__init__()
        self.attention_scores = None
        self.save_attention = False
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Get attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
        
        # Save attention scores if requested
        if self.save_attention:
            # Store the raw attention scores before softmax
            self.attention_scores = attention_scores.detach().clone()
        
        # Continue with normal processing
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(value.dtype)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttentionVisualizationProcessor(AttnProcessor):
    """Processor that visualizes attention scores as heatmaps without displaying them."""
    
    def __init__(self, save_dir="attention_heatmaps"):
        super().__init__()
        self.save_dir = save_dir
        self.visualize_attention = False
        self.block_name = "unknown_block"
        self.downsample_factor = 32  # Downsample factor for visualization
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Visualize attention if requested
        if self.visualize_attention:
            # Get dimensions
            head_size = attn.heads
            batch_dim = query.shape[0] // head_size
            query_len = query.shape[1]
            key_len = key.shape[1]
            
            # Calculate downsampled dimensions
            ds_query_len = max(1, query_len // self.downsample_factor)
            ds_key_len = max(1, key_len // self.downsample_factor)
            
            # Create timestamp for unique filenames
            timestamp = int(time.time())
            
            # Process in chunks to avoid OOM
            chunk_size = 128  # Adjust based on your GPU memory
            num_chunks = (query.shape[0] + chunk_size - 1) // chunk_size
            
            # Initialize accumulator for the average attention map
            avg_attention = torch.zeros(ds_query_len, ds_key_len, dtype=torch.float32, device="cpu")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, query.shape[0])
                
                # Process a chunk of queries
                q_chunk = query[start_idx:end_idx]
                k_chunk = key[start_idx:end_idx] if key.shape[0] == query.shape[0] else key
                
                # Downsample query and key for memory efficiency
                if query_len > ds_query_len:
                    q_indices = torch.linspace(0, query_len-1, ds_query_len, dtype=torch.long, device=query.device)
                    q_chunk = q_chunk[:, q_indices]
                
                if key_len > ds_key_len:
                    k_indices = torch.linspace(0, key_len-1, ds_key_len, dtype=torch.long, device=key.device)
                    k_chunk = k_chunk[:, k_indices] if key.shape[0] == query.shape[0] else key[:, k_indices]
                
                # Compute attention scores for the chunk
                chunk_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * attn.scale
                
                # Apply softmax to get proper attention weights
                chunk_attn = chunk_scores.softmax(dim=-1)
                
                # Average across batch and heads within this chunk
                chunk_avg = chunk_attn.mean(dim=0).cpu()
                
                # Accumulate to the average
                avg_attention += chunk_avg
            
            # Normalize by number of chunks
            avg_attention /= num_chunks
            
            # Convert to numpy for matplotlib
            avg_attention_np = avg_attention.numpy()
            
            # Create and save standard heatmap
            fig = plt.figure(figsize=(10, 8))
            plt.matshow(avg_attention_np, cmap='viridis', fignum=fig.number)
            plt.colorbar(label='Attention Weight')
            plt.title(f'Average Attention Map - {self.block_name}')
            plt.xlabel('Key Position (downsampled)')
            plt.ylabel('Query Position (downsampled)')
            
            # Save the figure directly without displaying
            filename = f"{self.save_dir}/{self.block_name}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create and save high-contrast version
            fig = plt.figure(figsize=(10, 8))
            plt.matshow(avg_attention_np, cmap='hot', fignum=fig.number, 
                       norm=matplotlib.colors.Normalize(0, avg_attention_np.max() * 0.2))
            plt.colorbar(label='Attention Weight (High Contrast)')
            plt.title(f'High Contrast Attention Map - {self.block_name}')
            plt.xlabel('Key Position (downsampled)')
            plt.ylabel('Query Position (downsampled)')
            
            # Save the high-contrast figure directly
            filename = f"{self.save_dir}/{self.block_name}_high_contrast_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved attention heatmaps for {self.block_name}")
            
            # Also save the raw numpy array for further analysis
            np.save(f"{self.save_dir}/{self.block_name}_{timestamp}.npy", avg_attention_np)
        
        # Continue with normal processing
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states