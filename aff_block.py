# aff_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AFFBlock(nn.Module):
    """
    An Adaptive Feature Fusion (AFF) inspired block that replaces a standard decoder block.
    It processes features through three parallel branches to capture long-range, multi-scale,
    and channel-wise semantic information.
    """
    def __init__(self, in_channels, out_channels):
        super(AFFBlock, self).__init__()

        # --- Branch 1: Long-Range Dependencies (LRD) ---
        # Uses dilated convolutions to capture a wider receptive field.
        self.lrd_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Branch 2: Multi-Scale Feature Fusion (MFF) ---
        # A simple stem to process the input, followed by multi-scale convolutions.
        self.mff_stem = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.mff_b1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.mff_b2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.mff_bn_relu = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

        # --- Branch 3: Adaptive Semantic Center (ASC) ---
        # This branch acts like a channel-attention and spatial-attention mechanism.
        self.asc_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to fuse the outputs of the three branches.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False), # We fuse LRD + MFF
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # --- LRD Branch ---
        lrd_out = self.lrd_branch(x)

        # --- MFF Branch ---
        mff_stem_out = self.mff_stem(x)
        mff_b1_out = self.mff_b1(mff_stem_out)
        mff_b2_out = self.mff_b2(mff_b1_out)
        mff_out = self.mff_bn_relu(torch.cat([mff_b1_out, mff_b2_out], dim=1))
        
        # --- ASC Branch ---
        asc_out = self.asc_branch(x)
        
        # --- Fusion ---
        # 1. Fuse LRD and MFF streams
        fused_lrd_mff = self.fusion_conv(torch.cat([lrd_out, mff_out], dim=1))
        
        # 2. Apply ASC as a channel-wise attention mechanism
        final_out = fused_lrd_mff * asc_out.expand_as(fused_lrd_mff)
        
        return final_out