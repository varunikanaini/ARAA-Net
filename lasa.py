# /kaggle/working/araa/ARAA-Net/lasa.py (FINAL CORRECTED AND OPTIMIZED VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    This version uses a vectorized approach (torch.unfold) for massive speed improvements
    and corrects the tensor dimension errors from the previous version.
    """
    def __init__(self, in_channels, M=4, L_list=[5, 7, 9, 11]):
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"
        self.M = M
        self.L_list = L_list
        self.group_channels = in_channels // M

        self.q_convs = nn.ModuleList([nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False) for _ in range(M)])
        self.k_convs = nn.ModuleList([nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False) for _ in range(M)])

        # Learnable positional encodings
        self.r_q = nn.Parameter(torch.randn(1, in_channels, 1, 1), requires_grad=True)
        # Positional encoding for the combined axial keys
        self.r_ks = nn.ParameterList([nn.Parameter(torch.randn(1, self.group_channels, 2 * L - 1, 1), requires_grad=True) for L in L_list])

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
        x_groups = torch.split(x, self.group_channels, dim=1)
        
        # Add positional encoding to query
        q_groups = torch.split(self.r_q + x, self.group_channels, dim=1)

        attention_groups = []

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i])         # (B, C/M, H, W)
            k_unpadded = self.k_convs[i](x_groups[i])   # (B, C/M, H, W)
            
            L = self.L_list[i]
            pad = (L - 1) // 2
            
            # --- VECTORIZED AXIAL KEY EXTRACTION (THE FIX) ---
            # 1. Create patches for all pixels at once
            k_unfolded = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad) # (B, C/M * L*L, H*W)
            k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W)   # (B, C/M, L, L, H, W)
            
            # 2. Extract the vertical and horizontal axes from the patches
            k_h = k_unfolded[:, :, :, pad, :, :] # (B, C/M, L, H, W) - Vertical axis
            k_w = k_unfolded[:, :, pad, :, :, :] # (B, C/M, L, H, W) - Horizontal axis

            # 3. Concatenate axes, removing the duplicate center pixel
            # Center pixel is at index `pad`
            k_w_pre, k_w_post = k_w.split([pad, L - pad - 1], dim=2)
            k_axial = torch.cat((k_h, k_w_pre, k_w_post), dim=2) # (B, C/M, 2L-1, H, W)

            # 4. Add positional encoding
            k_axial = k_axial + self.r_ks[i]
            
            # 5. Reshape for dot product
            # q: (B, C/M, H, W) -> (B, C/M, 1, H*W)
            # k: (B, C/M, 2L-1, H, W) -> (B, C/M, 2L-1, H*W)
            q_flat = q.reshape(B, self.group_channels, 1, H * W)
            k_flat = k_axial.reshape(B, self.group_channels, 2 * L - 1, H * W)

            # 6. Calculate attention with batch matrix multiplication
            energy = torch.matmul(q_flat.transpose(2, 3), k_flat) # (B, C/M, H*W, 2L-1)
            energy = torch.sum(energy, dim=-1) # Sum over the axial dimension (B, C/M, H*W)
            energy = energy.view(B, self.group_channels, H, W) # Reshape back to image format
            
            attention_groups.append(energy)
            
        # Concatenate and fuse attention maps
        attention_map = torch.cat(attention_groups, dim=1)
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention
        return x * attention_map