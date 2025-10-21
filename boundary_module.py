# /kaggle/working/ARAA-Net/boundary_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilation_rates=[1, 2, 4]):
        super(BoundaryModule, self).__init__()
        
        # Ensure we have a match for kernel_sizes and dilation_rates length
        if len(kernel_sizes) != len(dilation_rates):
            raise ValueError("kernel_sizes and dilation_rates must have the same length")

        self.conv_blocks = nn.ModuleList()
        
        # 1x1 convolution to process input channels to match the number of output channels
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different rates
        for i, (k, rate) in enumerate(zip(kernel_sizes, dilation_rates)):
            # Use the same out_channels for all dilated convolutions and the 1x1 conv
            # This simplifies concatenation later.
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=rate * (k // 2), dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution to fuse features.
        # The input channels will be out_channels (from 1x1) + out_channels * num_blocks
        # For simplicity and to avoid increasing channels, let's make the intermediate convs output out_channels/N
        # and then the final conv has in_channels = (out_channels/N) * (N+1) and outputs out_channels.
        # Or, a simpler approach: keep all intermediate convs outputting out_channels, and then use a 1x1 conv to reduce channels.

        # Let's try this simpler approach:
        # 1x1 conv -> out_channels
        # Dilated convs -> out_channels
        # Concatenate these -> in_channels = out_channels * (1 + num_blocks)
        # Final conv -> out_channels
        
        # Re-initializing based on common practice for ASPP-like modules
        # All branches should ideally produce the same spatial resolution and similar channel depth.
        # Let's make each branch output out_channels // (len(kernel_sizes) + 1) channels,
        # and then the final conv sums them up.

        # Revised structure for simplicity and common practice:
        # All branches will output out_channels // num_branches
        branches = len(kernel_sizes) + 1 # 1x1 conv + dilated convs
        channels_per_branch = out_channels // branches
        
        self.conv1x1 = self._make_dilated_block(in_channels, channels_per_branch, kernel_size=1, dilation=1)

        self.conv_blocks = nn.ModuleList()
        for i, (k, rate) in enumerate(zip(kernel_sizes, dilation_rates)):
            self.conv_blocks.append(
                self._make_dilated_block(in_channels, channels_per_branch, kernel_size=k, dilation=rate)
            )
        
        # Image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels_per_branch, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_per_branch),
            nn.ReLU(inplace=True)
        )

        # Final convolution to combine features
        self.conv_final = nn.Sequential(
            nn.Conv2d(channels_per_branch * (len(kernel_sizes) + 2), out_channels, kernel_size=1, bias=False), # +1 for 1x1, +1 for image_pool
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_dilated_block(self, in_channels, out_channels, kernel_size, dilation):
        padding = dilation * (kernel_size // 2)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: (B, C_in, H, W)
        size = x.shape[2:]

        # Process with 1x1 conv
        conv1x1_out = self.conv1x1(x)

        # Process with dilated convolutions
        dilated_outputs = [conv1x1_out]
        for block in self.conv_blocks:
            dilated_outputs.append(block(x))
        
        # Process with image pooling
        image_pool_out = self.image_pool(x)
        # Resize image_pool_out to match spatial dimensions of other outputs
        image_pool_out = F.interpolate(image_pool_out, size=size, mode='bilinear', align_corners=True)
        dilated_outputs.append(image_pool_out)

        # Concatenate all features
        combined_features = torch.cat(dilated_outputs, dim=1)
        
        # Final convolution to fuse features
        output = self.conv_final(combined_features)
        
        return output