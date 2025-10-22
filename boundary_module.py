import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryModule(nn.Module):
    def __init__(self, in_channels, out_channels=1, dilations=[1, 3]):
        super(BoundaryModule, self).__init__()
        
        # Multi-scale convolutions for boundary detection
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for dilation in dilations
        ])
        
        # Fuse multi-scale features
        self.fuse_conv = nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_channels)
        self.relu_fuse = nn.ReLU(inplace=True)
        
        # Refinement layer to sharpen boundaries
        self.refine_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply multi-scale convolutions
        boundary_features = [conv(x) for conv in self.conv_layers]
        # Concatenate along channel dimension
        boundary_map = torch.cat(boundary_features, dim=1)
        # Fuse features
        boundary_map = self.fuse_conv(boundary_map)
        boundary_map = self.bn_fuse(boundary_map)
        boundary_map = self.relu_fuse(boundary_map)
        # Refine
        boundary_map = self.refine_conv(boundary_map)
        boundary_map = self.sigmoid(boundary_map)
        
        return boundary_map