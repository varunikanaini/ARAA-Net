# /kaggle/working/ARAA-Net/lasa.py (User's ORIGINAL code, with structural fix for r_ks)
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
        
        # FIX: Ensure r_ks shape for broadcast compatibility with k_axial (B, C_g, 2L-1, H, W)
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

            energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)

            energy_summed = torch.sum(energy, dim=1)

            energy_final_group = energy_summed.unsqueeze(1).repeat(1, self.group_channels, 1, 1)

            attention_groups.append(energy_final_group)
        
        attention_map = torch.cat(attention_groups, dim=1)
        
        attention_map = self.fusion_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map