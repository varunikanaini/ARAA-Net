# /kaggle/working/ARAA-Net/lasa.py (RE-IMPLEMENTED as Grouped CBAM-like Attention)
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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size_1d=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP for both avg and max pooled features
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze the 2D spatial dimensions before MLP
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1).unsqueeze(-1)) # (B, C, 1) after MLP
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1).unsqueeze(-1)) # (B, C, 1) after MLP
        
        # Add and apply sigmoid
        out = avg_out + max_out # (B, C, 1)
        return self.sigmoid(out.unsqueeze(-1)) # Reshape to (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7): # L_list can influence this kernel_size
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        x_out = self.conv1(x_cat)
        x_out = self.bn(x_out)
        return self.sigmoid(x_out) # (B, 1, H, W)


class LASA(nn.Module):
    """
    Lightweight Adaptive Scale-Aware Attention Module (LASA) - CBAM-like Grouped.
    This module applies sequential channel and spatial attention to groups of features.
    L_list is used to potentially vary attributes of the attention sub-modules per group.
    """
    def __init__(self, in_channels, M=4, L_list=[3, 5, 7, 9]):
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"
        self.M = M
        self.L_list = L_list # Can be used to influence reduction_ratio or kernel_size
        self.group_channels = in_channels // M

        self.channel_attention_modules = nn.ModuleList()
        self.spatial_attention_modules = nn.ModuleList()

        for i in range(M):
            # For each group, we can set different parameters based on L_list[i]
            # e.g., reduction_ratio could vary, or spatial kernel size.
            # Here, we keep reduction_ratio constant but could make kernel_size_1d for ChannelAttention
            # or kernel_size for SpatialAttention dynamic based on L_list[i] if desired.
            
            # Channel Attention for each group
            self.channel_attention_modules.append(
                ChannelAttention(self.group_channels, reduction_ratio=16) # Keeping reduction ratio constant for simplicity
            )
            # Spatial Attention for each group
            self.spatial_attention_modules.append(
                SpatialAttention(kernel_size=7) # Keeping kernel size constant, but L_list could influence this
            )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.split(x, self.group_channels, dim=1) # Split features into M groups
        
        output_features_per_group = []

        for i in range(self.M):
            x_g = x_groups[i] # (B, C_g, H, W)

            # Apply Channel Attention
            channel_attn_map = self.channel_attention_modules[i](x_g) # (B, C_g, 1, 1)
            x_g_channel_attn = x_g * channel_attn_map # Modulate features

            # Apply Spatial Attention to channel-modulated features
            spatial_attn_map = self.spatial_attention_modules[i](x_g_channel_attn) # (B, 1, H, W)
            attn_out_g = x_g_channel_attn * spatial_attn_map # Final modulated features for this group
            
            output_features_per_group.append(attn_out_g)
        
        # Concatenate features from all groups
        concatenated_features = torch.cat(output_features_per_group, dim=1)
        
        # Residual connection: Add the original input features back
        return concatenated_features + x