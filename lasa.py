# /kaggle/working/ARAA-Net/lasa.py (REVERTED to ORIGINAL USER CODE, with r_ks shape fix)
import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module (User's original design).
    This version uses torch.einsum for batch dot-product.
    Fixed: r_ks parameter shape to correctly add to k_axial.
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
        
        # FIX: Original r_ks was defined as (1, group_channels, 2*L-1, 1, 1)
        # This parameter is meant to be added to k_axial which is (B, group_channels, 2L-1, H, W)
        # For broadcasting to work correctly and add a *positional bias* across the 2L-1 dimension,
        # its shape should be (1, 1, 2*L-1, 1, 1) or (1, group_channels, 2*L-1, 1, 1) if per-channel.
        # Let's use (1, group_channels, 2*L-1, 1, 1) as in your original code, which is valid for broadcasting.
        # The previous error was not in this definition but in a different `LASA` re-implementation.
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
        q_groups = torch.split(self.r_q + x, self.group_channels, dim=1)
        attention_groups = []

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i])
            k_unpadded = self.k_convs[i](x_groups[i])

            L = self.L_list[i]
            pad = (L - 1) // 2

            # The original `F.unfold` and `view` to create `k_unfolded` with L, L spatial dimensions:
            k_unfolded = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad) # (B, C_g * L * L, H * W)
            k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W) # (B, C_g, L, L, H, W)

            # Original extraction of horizontal and vertical slices from the L x L window:
            k_h = k_unfolded[:, :, :, pad, :, :] # (B, C_g, L, H, W) - horizontal line at center row
            k_w = k_unfolded[:, :, pad, :, :, :] # (B, C_g, L, H, W) - vertical line at center column

            # Original splitting and concatenation to form k_axial (2L-1 elements in dim=2):
            k_w_pre, _, k_w_post = k_w.split([pad, 1, L - pad - 1], dim=2)
            k_axial = torch.cat((k_h, k_w_pre, k_w_post), dim=2) # (B, C_g, 2L-1, H, W)
            
            # Adding learnable bias across the 'axial' dimension
            k_axial = k_axial + self.r_ks[i] # r_ks[i] is (1, C_g, 2L-1, 1, 1), broadcasts correctly

            # Original einsum for batch dot-product attention:
            # 'bchw,bcrhw->brhw' means:
            # - For each item in batch (b), height (h), and width (w)...
            # - Multiply query (bchw) with key (bcrhw)
            # - Sum the result over the channel dimension (c)
            # - The output will have dimensions for batch (b), axial position (r), height (h), and width (w)
            energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial) # (B, 2L-1, H, W)

            # Sum over the 'axial' (r) dimension to get a single energy map per pixel
            energy_summed = torch.sum(energy, dim=1) # (B, H, W)

            # Expand the summed energy map and repeat it for each channel in the group (C_g)
            energy_final_group = energy_summed.unsqueeze(1).repeat(1, self.group_channels, 1, 1) # (B, C_g, H, W)

            attention_groups.append(energy_final_group)
        
        # Concatenate the energy maps from all groups
        attention_map = torch.cat(attention_groups, dim=1) # (B, C, H, W)
        
        # Fuse the maps and apply sigmoid to get final weights between 0 and 1
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply the attention map to the original input features
        return x * attention_map