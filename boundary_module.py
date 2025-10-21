# /kaggle/working/ARAA-Net/boundary_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilation_rates=[1, 2, 4]):
        super(BoundaryModule, self).__init__()
        
        # Use dilated convolutions to capture context at multiple scales
        # This helps in distinguishing edges even with blur.
        self.conv_blocks = nn.ModuleList()
        
        # Convolution with 1x1 kernel for initial processing
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // len(kernel_sizes), kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // len(kernel_sizes))
        self.relu1 = nn.ReLU(inplace=True)

        for i, (k, rate) in enumerate(zip(kernel_sizes, dilation_rates)):
            # Dilated convolutions capture context without increasing parameters
            # We'll use out_channels // len(kernel_sizes) for each block's output channels
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // len(kernel_sizes), kernel_size=k, stride=1, padding=rate*(k//2), dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels // len(kernel_sizes)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution to fuse features
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels + (out_channels // len(kernel_sizes)) * len(kernel_sizes), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        size = x.shape[2:]

        # Process with 1x1 conv first
        conv1x1_out = self.conv1x1(x)

        # Process with dilated convolutions
        dilated_outputs = [conv1x1_out]
        for block in self.conv_blocks:
            dilated_outputs.append(block(x))
        
        # Concatenate all features
        combined_features = torch.cat(dilated_outputs, dim=1)
        
        # Final convolution to fuse features
        output = self.conv_final(combined_features)
        
        return output