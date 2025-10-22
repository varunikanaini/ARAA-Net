# hybrid_decoder_block.py

import torch
import torch.nn as nn
from spatial_attn import BoundarySpatialAttention

class HybridDecoderBlock(nn.Module):
    """
    A hybrid decoder block that combines the strengths of previous experiments.
    1. It first uses a context module with dilated convolutions to capture multi-scale information.
    2. It then applies the proven BoundarySpatialAttention to refine the spatial details of the context-aware features.
    """
    def __init__(self, in_channels, out_channels):
        super(HybridDecoderBlock, self).__init__()

        # Initial convolution to process concatenated features from the skip connection
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Part 1: Context Module (inspired by LRD branch of AFFBlock)
        # This captures wider contextual information.
        self.context_module = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Part 2: Spatial Refinement (the successful BoundarySpatialAttention)
        # This acts as a magnifying glass on the context-aware features.
        self.spatial_attention = BoundarySpatialAttention(kernel_size=7)
        
        # Final convolution to stabilize the features after attention
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x_context = self.context_module(x)
        x_refined = self.spatial_attention(x_context)
        out = self.conv2(x_refined)
        return out