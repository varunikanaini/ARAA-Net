# lasa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For calculating padding based on kernel size

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    Now supports multi-scale kernels by iterating through L_list.
    """
    def __init__(self, in_channels, M=4, L_list=[1, 3, 5, 7]): # Changed default to match user's request
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"
        self.M = M
        self.L_list = L_list
        self.group_channels = in_channels // M

        # Convolution layers for query and key for each group and kernel size
        self.q_convs = nn.ModuleList([
            nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, bias=False) 
            for _ in range(M)
        ])
        self.k_convs = nn.ModuleList([
            nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, bias=False) 
            for _ in range(M)
        ])

        self.r_q = nn.Parameter(torch.randn(1, in_channels, 1, 1), requires_grad=True)
        
        # Parameter list for positional embeddings for each kernel size
        # Shape: (1, group_channels, 2*L - 1, 1, 1) - where 2*L-1 is the length of axial pixels
        self.r_ks = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.group_channels, 2 * L - 1, 1, 1), requires_grad=True) 
            for L in L_list
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.split(x, self.group_channels, dim=1)
        q_groups = torch.split(self.r_q + x, self.group_channels, dim=1)
        attention_groups = []

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i])
            k_conv = self.k_convs[i]
            r_k_param = self.r_ks[i]
            L = self.L_list[i]

            # Calculate padding for the kernel L
            pad = (L - 1) // 2 
            
            # Apply convolution for key extraction
            k_unpadded = k_conv(x_groups[i])

            # Unfold the feature map for spatial attention
            # Use kernel_size=L, padding=pad to get axial features correctly
            k_unfolded = F.unfold(k_unpadded, kernel_size=L, padding=pad)
            # Reshape to (B, C_g, L, L, H, W) to extract axial positions
            k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W)

            # Extract axial pixels (height and width axes)
            # k_h gets pixels on the same column but different rows within the LxF kernel
            k_h = k_unfolded[:, :, :, pad, :, :] # Shape: (B, C_g, L, H, W)
            # k_w gets pixels on the same row but different columns within the LxL kernel
            k_w = k_unfolded[:, :, pad, :, :, :] # Shape: (B, C_g, L, H, W)

            # Combine them to form the axial feature map
            # The shape of k_axial should be (B, C_g, 2*L-1, H, W)
            # First, split k_w to get pre-pad, center, and post-pad parts
            k_w_pre, _, k_w_post = k_w.split([pad, 1, L - pad - 1], dim=2) # Split along the kernel dimension
            k_axial = torch.cat((k_h, k_w_pre, k_w_post), dim=2) # Concatenate along the kernel dimension

            # Add positional embedding (r_k_param)
            # Ensure r_k_param has the correct shape for broadcasting: (1, C_g, 2*L-1, 1, 1)
            k_axial = k_axial + r_k_param # Broadcasting handles the H, W dimensions

            # Calculate attention energy using einsum for batch dot product
            # q: (B, C_g, H, W) -> Reshaped for einsum: (B, C_g, 1, H, W)
            # k_axial: (B, C_g, 2*L-1, H, W) -> Reshaped for einsum: (B, C_g, 2*L-1, H, W)
            # einsum('bchw,bcrhw->brhw', q, k_axial) seems incorrect.
            # Correct approach: q needs to be compatible with k_axial's spatial/axial dimensions.
            # The original paper implies q is used as the query for the K,V pair.
            # Let's assume q is broadcast against k_axial.

            # The most straightforward way is to treat q as (B, C_g, 1, H, W) for einsum,
            # and perform dot product along the channel dimension.
            # Then sum over the axial dimension (2*L-1).
            # The target shape for attention map: (B, C_g, H, W) if done per channel.
            # But we want a single attention map per group.
            
            # Let's re-evaluate: q is (B, C_g, H, W). k_axial is (B, C_g, 2*L-1, H, W)
            # We want to get an attention map for each group i.
            # It seems the original code's einsum might be trying to do something like:
            # energy = F.conv2d(q.unsqueeze(2), k_axial.view(B*C_g, 2*L-1, H, W).permute(0, 2, 1, 3)) -> not this
            # The original code had: energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)
            # This means: for each batch B, each channel C, it does a dot product with the axial feature.
            # r is the axial dimension index (2*L-1). h,w are spatial.
            # This einsum doesn't look right for standard attention.
            # It's more like a weighted sum based on axial features.

            # Correcting the logic based on common attention mechanisms:
            # q: (B, C_g, H, W)
            # k_axial: (B, C_g, 2*L-1, H, W)
            # We need to compute similarity between q and k_axial.
            # The common approach is to reshape q and k_axial to allow matrix multiplication.
            # For example, if we want to compute attention weights for each spatial location (H, W) based on axial context.
            
            # Let's re-try based on the intention:
            # The goal is to get a per-pixel attention weight for F.
            # energy_per_axial_dim: (B, C_g, 2*L-1, H, W) -> compute similarity
            # Assuming a simple dot product similarity:
            # q reshaped: (B, C_g, 1, H, W)
            # k_axial reshaped: (B, C_g, 2*L-1, H, W)
            
            # A common way is to compute dot product: q * k_axial and then sum.
            # Or use scaled dot product attention (softmax(Q*K^T)).
            # Given the original einsum, it looks like it's summing up the axial dimension `r` for each spatial (h,w).
            # `einsum('bchw,bcrhw->brhw', q, k_axial)` suggests:
            # For each batch `b`, for each axial index `r`, for each spatial pixel (h,w), compute
            # sum_c ( q[b,c,h,w] * k_axial[b,c,r,h,w] )
            # This seems to be computing an "axial response" for each of the 2*L-1 positions.
            
            # Let's assume this is correct for now and proceed to sum over the axial dimension.
            # Result: (B, H, W) for each axial position 'r'.
            energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)

            # Summing over the axial dimension 'r' to get a single attention map per pixel.
            # The sum should be over `r` (the 2*L-1 dimension).
            attention_map_per_group = torch.sum(energy, dim=1) # Sum over axial dimension 'r'

            # Now, we have a tensor (B, H, W) for each group `i`.
            # We need to apply this as an attention mask to the original input `x`.
            # The shape of attention_map_per_group is (B, H, W).
            # We want to apply it to `x` which has shape (B, C, H, W).
            # The attention_map_per_group needs to be broadcast across channels.
            # It might be more standard to get weights for each spatial location:
            # attention_weights = F.softmax(attention_map_per_group, dim=[1,2]).unsqueeze(1) # Normalize over spatial dims
            # But the original code seemed to sum and repeat. Let's follow that for now.
            
            # Original code repeated attention_map_per_group across channels:
            energy_final_group = attention_map_per_group.unsqueeze(1).repeat(1, self.group_channels, 1, 1)

            attention_groups.append(energy_final_group)
        
        # Concatenate attention maps from all groups
        attention_map = torch.cat(attention_groups, dim=1) # Shape: (B, C, H, W)
        
        # Apply fusion convolution and sigmoid to get the final attention weights
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map # Apply attention to the original input features