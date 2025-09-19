# /kaggle/working/ARAA-Net/lasa.py (RE-IMPLEMENTED for Lightweight Attention)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper module to explicitly squeeze dimensions within nn.Sequential
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)

class LASA(nn.Module):
    """
    Lightweight Adaptive Scale-Aware Attention Module (LASA).
    This module performs grouped channel and spatial attention, using L_list
    to define diverse receptive fields for channel excitation.
    It's designed to be efficient and effective for 2D feature enhancement.
    """
    def __init__(self, in_channels, M=4, L_list=[3, 5, 7, 9]): # Adjusted default L_list for common kernel sizes
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"
        self.M = M
        self.L_list = L_list # Used to influence kernel size for channel attention
        self.group_channels = in_channels // M

        self.channel_excitation_layers = nn.ModuleList()
        self.spatial_attention_layers = nn.ModuleList()

        for i in range(M):
            L_k = L_list[i] # Use L_list for kernel size for 1D conv
            if L_k % 2 == 0: L_k += 1 # Ensure odd kernel size for symmetric padding

            # Grouped Channel Attention (ECA-like)
            self.channel_excitation_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), # (B, C_g, 1, 1)
                    Squeeze(dim=-1),         # (B, C_g, 1) - ready for Conv1d
                    nn.Conv1d(self.group_channels, self.group_channels, kernel_size=L_k, 
                              padding=(L_k - 1) // 2, bias=False),
                    nn.BatchNorm1d(self.group_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.group_channels, self.group_channels, kernel_size=1, bias=False) # Final projection
                )
            )

            # Grouped Spatial Attention (Simplified, 1x1 conv + large kernel conv)
            self.spatial_attention_layers.append(
                nn.Sequential(
                    nn.Conv2d(self.group_channels, self.group_channels // 8, kernel_size=1, bias=False), # Squeeze channels
                    nn.BatchNorm2d(self.group_channels // 8),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.group_channels // 8, 1, kernel_size=7, padding=3, bias=False), # Generate single spatial map
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True) # Sigmoid applied later for weights
                )
            )
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.split(x, self.group_channels, dim=1)
        
        output_features_per_group = []

        for i in range(self.M):
            x_g = x_groups[i] # (B, C_g, H, W)

            # --- Channel Attention ---
            channel_vector_1d = self.channel_excitation_layers[i](x_g) # (B, C_g, 1)
            channel_weights = self.sigmoid(channel_vector_1d.unsqueeze(-1)) # (B, C_g, 1, 1)
            
            # --- Spatial Attention ---
            spatial_map = self.spatial_attention_layers[i](x_g) # (B, 1, H, W)
            spatial_weights = self.sigmoid(spatial_map) # (B, 1, H, W)

            # --- Combine Attention ---
            # Element-wise product of feature map with channel and spatial weights
            # This is a common way to fuse both attention types in CBAM-like modules
            # We scale the group features by both attention maps
            attn_out_g = x_g * channel_weights * spatial_weights
            
            output_features_per_group.append(attn_out_g)
        
        # Concatenate features from all groups
        concatenated_features = torch.cat(output_features_per_group, dim=1)
        
        # Residual connection: Add the enhanced features back to the original input
        return concatenated_features + x