# /kaggle/working/ARAA-Net/spatial_attn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundarySpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(BoundarySpatialAttention, self).__init__()
        # Use a kernel size that can capture local spatial gradients/features
        padding = kernel_size // 2
        
        # This layer will learn to generate an attention map that highlights spatial features.
        # We use 2 input channels because we'll combine average and max pooling.
        # Output is 1 channel for the final attention map.
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # Capture average pooling and max pooling along the channel dimension
        # This emphasizes spatial locations with strong activation
        avg_out = torch.mean(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)
        
        # Concatenate pooled features along the channel dimension
        x_pooled = torch.cat([avg_out, max_out], dim=1) # Shape: (B, 2, H, W)
        
        # Apply convolution to learn spatial attention weights
        attention_map = self.conv(x_pooled) # Shape: (B, 1, H, W)
        attention_map = self.sigmoid(attention_map) # Normalize to [0, 1]
        
        # Apply attention to the original feature map
        # This scales the features based on their spatial importance
        return x * attention_map