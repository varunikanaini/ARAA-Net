# /kaggle/working/araa/ARAA-Net/lasa.py (FINAL DEFINITIVE VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    This version uses torch.einsum for a correct, fast, and readable
    implementation of the batch dot-product for attention.
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

            k_unfolded = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad)
            k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W)

            k_h = k_unfolded[:, :, :, pad, :, :]
            k_w = k_unfolded[:, :, pad, :, :, :]

            k_w_pre, _, k_w_post = k_w.split([pad, 1, L - pad - 1], dim=2)
            k_axial = torch.cat((k_h, k_w_pre, k_w_post), dim=2)
            k_axial = k_axial + self.r_ks[i]

            # --- THIS IS THE FIX ---
            # Use Einstein summation for a direct and error-free batch dot-product.
            # 'bchw,bcrhw->brhw' means:
            # - For each item in batch (b), height (h), and width (w)...
            # - Multiply query (bchw) with key (bcrhw)
            # - Sum the result over the channel dimension (c)
            # - The output will have dimensions for batch (b), axial position (r), height (h), and width (w)
            energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)

            # The resulting `energy` has shape (B, 2L-1, H, W).
            # This represents the compatibility score for each of the 2L-1 axial positions.
            # We sum these scores to get a single energy value per pixel for this group.
            energy_summed = torch.sum(energy, dim=1)  # Sum over the 'r' dimension.

            # We need to produce C/M output channels for this group's attention map.
            # We expand the summed energy map and repeat it for each channel in the group.
            energy_final_group = energy_summed.unsqueeze(1).repeat(1, self.group_channels, 1, 1)

            attention_groups.append(energy_final_group)
            # --- END OF FIX ---

        # Concatenate the energy maps from all groups
        attention_map = torch.cat(attention_groups, dim=1)
        
        # Fuse the maps and apply sigmoid to get final weights between 0 and 1
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply the attention map to the original input
        return x * attention_map