# /kaggle/working/ARAA-Net/lasa.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    This module enhances feature discrimination by re-weighting pixels
    based on multi-scale local features, inspired by radiologists' workflow.
    """
    def __init__(self, in_channels, M=4, L_list=[5, 7, 9, 11]):
        """
        Args:
            in_channels (int): Number of input channels.
            M (int): Number of groups/scales to split the channels into.
            L_list (list of int): List of odd integers representing the side lengths of the local squares for each scale.
        """
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"

        self.M = M
        self.L_list = L_list
        group_channels = in_channels // M

        self.q_convs = nn.ModuleList([nn.Conv2d(group_channels, group_channels, 1, bias=False) for _ in range(M)])
        self.k_convs = nn.ModuleList([nn.Conv2d(group_channels, group_channels, 1, bias=False) for _ in range(M)])

        # Learnable positional encodings
        self.r_q = nn.Parameter(torch.randn(1, in_channels, 1, 1), requires_grad=True)
        self.r_ks = nn.ParameterList([nn.Parameter(torch.randn(1, group_channels, 1, 2 * L - 1), requires_grad=True) for L in L_list])

        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split features into M groups
        x_groups = torch.split(x, C // self.M, dim=1)
        
        # Add positional encoding to query
        q_groups = torch.split(self.r_q + x, C // self.M, dim=1)

        attention_groups = []

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i]) # (B, C/M, H, W)
            k_unpadded = self.k_convs[i](x_groups[i])
            
            L = self.L_list[i]
            pad = (L - 1) // 2
            
            # Pad k to create local context windows
            k_padded = F.pad(k_unpadded, [pad, pad, pad, pad])
            
            # Extract axial features from k for each position
            k_axial = []
            for r in range(H):
                for c in range(W):
                    # Height-axis and Width-axis pixels, excluding the center pixel twice
                    k_h = k_padded[:, :, r:r+L, c+pad]
                    k_w = k_padded[:, :, r+pad, c:c+L]
                    k_axial_pixel = torch.cat([k_h, k_w[:, :, k_w.shape[2] // 2 + 1:], k_w[:, :, :k_w.shape[2] // 2]], dim=2)
                    k_axial.append(k_axial_pixel.permute(0, 1, 3, 2))

            k_axial = torch.stack(k_axial, dim=-1).reshape(B, C//self.M, 2*L-1, H, W)
            k_axial = k_axial.squeeze(-2).permute(0, 1, 3, 4, 2) # (B, C/M, H, W, 2L-1)
            
            # Add positional encoding to key
            k_axial_pos = k_axial + self.r_ks[i].unsqueeze(0).unsqueeze(2).unsqueeze(2)

            # Calculate attention
            q_expanded = q.unsqueeze(-1) # (B, C/M, H, W, 1)
            
            # Element-wise product and sum
            energy = torch.matmul(q_expanded.transpose(-1, -2), k_axial_pos.transpose(-1, -2).transpose(-1,-2)).squeeze(-2) # (B, C/M, H, W)
            
            attention_groups.append(energy)
            
        # Concatenate and fuse attention maps
        attention_map = torch.cat(attention_groups, dim=1)
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention
        return x * attention_map