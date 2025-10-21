# /kaggle/working/ARAA-Net/spatial_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SimpleSpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_pooled = torch.cat([avg_out, max_out], dim=1) # Shape: (B, 2, H, W)
        
        attention_map = self.conv(x_pooled) # Shape: (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map # Apply attention