# /kaggle/working/ARAA-Net/boundary_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryModule(nn.Module):
    def __init__(self, in_channels, out_channels=1): # Output a single channel for boundary prediction
        super(BoundaryModule, self).__init__()
        
        # Use a simple sequence of convolutions to detect boundaries
        # This is designed to be lightweight.
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # A second convolution to refine the boundary prediction
        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=1) # 1x1 for final output
        self.sigmoid = nn.Sigmoid() # Output probabilities for boundary presence

    def forward(self, x):
        # Input x is a feature map
        x = self.conv_layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        boundary_map = self.conv_layer2(x)
        boundary_map = self.sigmoid(boundary_map)
        
        return boundary_map