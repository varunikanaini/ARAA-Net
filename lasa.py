# /kaggle/working/ARAA-Net/lasa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    This version uses a corrected axial feature extraction mechanism.
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

        self.r_q = nn.Parameter(torch.randn(1, in_channels, 1, 1), requires_grad=True)
        
        # r_ks should have shape (1, C_g, 2L-1, 1, 1) for adding to axial features.
        # The 2L-1 comes from L vertical + L horizontal pixels, with the center pixel counted once.
        self.r_ks = nn.ParameterList([nn.Parameter(torch.randn(1, self.group_channels, 2 * L - 1, 1, 1), requires_grad=True) for L in L_list])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.split(x, self.group_channels, dim=1)
        q_groups = torch.split(self.r_q + x, self.group_channels, dim=1) # Add r_q to input x before splitting for q
        
        attention_groups = []

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i]) # Query for this group (B, C_g, H, W)
            k_unpadded = self.k_convs[i](x_groups[i]) # Key for this group (B, C_g, H, W)

            L = self.L_list[i]
            pad = (L - 1) // 2 # Padding for kernel_size L

            # --- CORRECTED AXIAL FEATURE EXTRACTION ---
            # Extract vertical axial features: L pixels along height, centered at 'pad'
            # kernel_size=(L, 1) means L rows, 1 column. padding=(pad, 0) for height, width.
            k_axial_h_unfolded = F.unfold(k_unpadded, kernel_size=(L, 1), padding=(pad, 0)) # Shape: (B, C_g * L, H, W)
            k_axial_h_unfolded = k_axial_h_unfolded.view(B, self.group_channels, L, H, W) # Reshape: (B, C_g, L, H, W)
            # Select the central row (pad) across all L column features. This gives L features for each pixel.
            k_axial_h_features = k_axial_h_unfolded[:, :, pad, :, :] # Shape: (B, C_g, H, W) -- Wait, this is L features for each pixel, not 1 feature.
            
            # Let's rethink the axial features: The paper implies for each pixel, we extract L vertical neighbors and L horizontal neighbors.
            # The total number of these features is 2L-1 (center pixel counted once).
            # The resulting `k_axial` should likely be `(B, C_g, 2L-1, H, W)` for the einsum to work with `q` of `(B, C_g, H, W)`.
            
            # Re-generating k_axial as a feature vector of size 2L-1 for each pixel:
            
            # Vertical axial pixels: Extract L pixels along the height axis.
            # Use unfold with kernel (L, 1) and padding (pad, 0)
            k_vertical_unfolded = k_unpadded.unfold(2, L, 1).unfold(3, 1, 1) # (B, C_g, H-L+1, W, 1, L) -- this is not right
            
            # Let's use the paper's formulation: V_o(i,j) and U_o(i,j) which are SETS of pixels.
            # N^u = F(V_o(i,j), U_o(i,j)) is flattening them into a vector of length 2L-1.
            
            # Correct way to extract axial features:
            # For vertical features: For each pixel (h, w), take k_unpadded[:, :, h-pad:h+pad+1, w]
            # For horizontal features: For each pixel (h, w), take k_unpadded[:, :, h, w-pad:w+pad+1]
            
            # This is best done by using F.unfold on height and width SEPARATELY.
            
            # Vertical axial features (L pixels): kernel size (L, 1), padding (pad, 0)
            k_axial_v = F.unfold(k_unpadded, kernel_size=(L, 1), padding=(pad, 0)) # shape: (B, C_g * L, H, W)
            k_axial_v = k_axial_v.view(B, self.group_channels, L, H, W) # Reshape to (B, C_g, L, H, W)
            
            # Horizontal axial features (L pixels): kernel size (1, L), padding (0, pad)
            k_axial_h = F.unfold(k_unpadded, kernel_size=(1, L), padding=(0, pad)) # shape: (B, C_g * L, H, W)
            k_axial_h = k_axial_h.view(B, self.group_channels, L, H, W) # Reshape to (B, C_g, L, H, W)

            # Combine these to get 2L-1 features. We need to flatten the L axial pixels and concatenate.
            # For each pixel (h,w), we have L vertical features and L horizontal features.
            # The simplest way is to flatten them and concatenate.
            
            k_axial_v_flat = k_axial_v.permute(0, 1, 3, 4, 2).reshape(B, self.group_channels, H, W, L) # (B, C_g, H, W, L)
            k_axial_h_flat = k_axial_h.permute(0, 1, 3, 4, 2).reshape(B, self.group_channels, H, W, L) # (B, C_g, H, W, L)
            
            # Concatenate along the last dimension (L) and then flatten to get 2L features.
            # This is still not quite 2L-1. The paper implies selection, not just concatenation.
            # A common way to get 2L-1 features is to take L vertical and L horizontal, then remove duplicates.
            # Or, to select specific indices.
            
            # Let's use the paper's description more closely:
            # V_o(i,j) = pixels on height axis, U_o(i,j) = pixels on width axis.
            # Flattening these sets gives N^u.
            # The critical point is that the input to `einsum` must match the `r_ks` shape's axial dimension (2L-1).
            
            # Let's construct `k_axial` of shape (B, C_g, 2L-1, H, W) directly.
            # This requires carefully selecting pixels from the kernel grid.
            # `F.unfold` with kernel (L,L) produces (B, C_g, L*L, H*W)
            # We need to extract axial indices from the L*L dimension.
            
            # Let's re-implement the axial extraction using the `k_unpadded.unfold` approach correctly.
            # `k_unpadded` is (B, C_g, H, W)
            # Unfold with kernel (L, L), padding (pad, pad)
            k_unfolded_kernel = k_unpadded.unfold(2, L, 1).unfold(3, L, 1) # shape: (B, C_g, H-L+1, W-L+1, L, L)
            # This `H-L+1` and `W-L+1` means the output spatial dims are reduced.
            # We need to work with the original H, W spatial dims.
            # The correct way is to apply padding before unfold or use different unfold parameters.
            
            # Let's try padding `k_unpadded` first to ensure output spatial size remains H,W.
            k_padded = F.pad(k_unpadded, (pad, pad, pad, pad), "constant", 0) # Pad width and height
            # Now unfold: kernel (L, L), stride (1, 1). output spatial size H, W.
            # `k_unpadded.unfold(dim, size, step)`
            # For H dimension: dim=2, size=L, step=1
            # For W dimension: dim=3, size=L, step=1
            k_unfolded_correct = k_padded.unfold(2, L, 1).unfold(3, L, 1) # shape: (B, C_g, H, W, L, L)
            
            # Now, from this (B, C_g, H, W, L, L) tensor, extract axial pixels.
            # For each (h, w) position:
            # Vertical axial pixels are at k_unfolded_correct[:, :, h, w, :, pad]
            # Horizontal axial pixels are at k_unfolded_correct[:, :, h, w, pad, :]
            
            # Flattening these directly:
            k_axial_v = k_unfolded_correct[:, :, :, :, :, pad] # (B, C_g, H, W, L)
            k_axial_h = k_unfolded_correct[:, :, :, :, pad, :] # (B, C_g, H, W, L)

            # Concatenate these L vertical and L horizontal features.
            # To get 2L-1, we must be careful about the center pixel.
            # The paper's diagram suggests selecting indices along the kernel's axes.
            # A simpler but likely correct interpretation for "2L-1 features" might be:
            # Take L vertical pixels, and then L-1 horizontal pixels (excluding the center already taken by vertical).
            
            # Let's combine these:
            k_axial_v_flat = k_axial_v.reshape(B, self.group_channels, H, W, L) # (B, C_g, H, W, L)
            # Take horizontal pixels excluding the center row (pad)
            k_axial_h_excl_center = k_axial_h[:, :, :, :, torch.arange(L) != pad] # (B, C_g, H, W, L-1)
            
            # Concatenate vertical and horizontal (excl. center)
            k_axial = torch.cat((k_axial_v_flat, k_axial_h_excl_center), dim=-1) # (B, C_g, H, W, L + L-1) = (B, C_g, H, W, 2L-1)
            
            # Now k_axial has the correct axial feature dimension (2L-1).
            # Add r_ks (shape (1, C_g, 2L-1, 1, 1)) to k_axial (shape (B, C_g, H, W, 2L-1))
            # This requires reordering k_axial's dimensions for broadcasting.
            k_axial = k_axial.permute(0, 1, 4, 2, 3) # -> (B, C_g, 2L-1, H, W)
            
            # Add r_ks (which is (1, C_g, 2L-1, 1, 1)) to k_axial (B, C_g, 2L-1, H, W)
            # Broadcasting works as intended now.
            k_axial = k_axial + self.r_ks[i] # Add positional encoding

            # Energy calculation using einsum:
            # q is (B, C_g, H, W). k_axial is (B, C_g, 2L-1, H, W).
            # einsum signature: 'bchw,bcrhw->brhw'
            # b=batch, c=channel, h=height, w=width, r=axial_feature_index
            # This implies `q` is treated as `(B, C_g, 1, H, W)` and `k_axial` as `(B, C_g, 2L-1, H, W)`.
            # The dot product is over 'c' (channel) and 'r' (axial feature index).
            # The result shape is `(B, R, H, W)`.
            energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)

            # Sum over the axial feature dimension (R, which is 2L-1) to get a single attention map per group.
            energy_summed = torch.sum(energy, dim=1) # Shape: (B, H, W)

            # Repeat this summed energy map `self.group_channels` times to match channel dim for fusion.
            energy_final_group = energy_summed.unsqueeze(1).repeat(1, self.group_channels, 1, 1)

            attention_groups.append(energy_final_group)
        
        attention_map = torch.cat(attention_groups, dim=1) # Shape: (B, C, H, W)
        
        attention_map = self.fusion_conv(attention_map) # Apply fusion conv
        attention_map = self.sigmoid(attention_map) # Apply sigmoid to get attention weights
        
        # Apply attention map to original input feature map
        return x * attention_map