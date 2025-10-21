# /kaggle/working/ARAA-Net/simple_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SimplifiedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y # Scale input features by channel attention weights

class SimplifiedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SimplifiedSpatialAttention, self).__init__()
        # Using a simple 3x3 convolution for spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate them and apply convolution
        spatial_map = torch.cat([avg_out, max_out], dim=1)
        spatial_map = self.conv(spatial_map)
        return x * self.sigmoid(spatial_map) # Scale input features by spatial attention weights

class SimpleAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super(SimpleAttentionModule, self).__init__()
        self.channel_attention = SimplifiedChannelAttention(in_channels, reduction)
        self.spatial_attention = SimplifiedSpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention first
        x_out = self.channel_attention(x)
        # Apply spatial attention to the output of channel attention
        x_out = self.spatial_attention(x_out)
        return x_out