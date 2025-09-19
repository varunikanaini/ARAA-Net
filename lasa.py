# /kaggle/working/ARAA-Net/lasa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper module to explicitly squeeze dimensions within nn.Sequential
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)


class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module (Re-interpreted for 2D robustness).
    This version performs grouped channel attention where each group's
    attention weights are derived from its local spatial context,
    with 'L_list' specifying diverse local receptive fields (using 1D conv kernel size).
    """
    def __init__(self, in_channels, M=4, L_list=[5, 7, 9, 11]):
        super(LASA, self).__init__()
        assert in_channels % M == 0, "in_channels must be divisible by the number of groups M"
        assert len(L_list) == M, "Length of L_list must be equal to M"
        self.M = M
        self.L_list = L_list
        self.group_channels = in_channels // M

        # Learnable global bias for the input features
        self.global_bias = nn.Parameter(torch.randn(1, in_channels, 1, 1), requires_grad=True)

        self.local_context_extractors = nn.ModuleList()
        self.channel_attention_fcs = nn.ModuleList()

        for i in range(M):
            L = L_list[i]
            
            # 1. Local context extractor for each group
            # Using a simple 3x3 convolution with padding to capture local features.
            self.local_context_extractors.append(
                nn.Sequential(
                    nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.group_channels),
                    nn.ReLU(inplace=True)
                )
            )

            # 2. Channel attention generation for each group
            # This follows an ECA-like approach: Global Average Pool -> Squeeze -> 1D Conv -> Sigmoid
            # The kernel size of the 1D conv is determined by L from L_list.
            kernel_size_1d = L 
            if kernel_size_1d % 2 == 0: kernel_size_1d += 1 # Ensure odd kernel size for symmetric padding

            self.channel_attention_fcs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), # Output: (B, C_g, 1, 1)
                    Squeeze(dim=-1),         # Output: (B, C_g, 1) - Correct 3D for Conv1d
                    nn.Conv1d(self.group_channels, self.group_channels, kernel_size=kernel_size_1d, 
                              padding=(kernel_size_1d - 1) // 2, bias=False),
                    nn.BatchNorm1d(self.group_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.group_channels, self.group_channels, kernel_size=1, bias=False) # Final projection
                )
            )
            
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Add global learnable bias to the input features first
        x_biased = x + self.global_bias
        x_groups = torch.split(x_biased, self.group_channels, dim=1) # Split into M groups

        modulated_features_per_group = []

        for i in range(self.M):
            x_group = x_groups[i] # (B, C_g, H, W)
            
            # 1. Extract local context for this group
            local_context_features = self.local_context_extractors[i](x_group) # (B, C_g, H, W)
            
            # 2. Generate channel attention weights from the local context features
            # channel_attention_fcs[i] now correctly produces (B, C_g, 1) from (B, C_g, H, W)
            channel_weights_1d = self.channel_attention_fcs[i](local_context_features) # (B, C_g, 1)
            
            # Reshape back to (B, C_g, 1, 1) for 2D element-wise multiplication
            channel_weights = channel_weights_1d.unsqueeze(-1) # (B, C_g, 1, 1)
            
            # Apply sigmoid to get attention map (0-1)
            group_attention_map = self.sigmoid(channel_weights)
            
            # 3. Modulate the original (biased) group features with the attention map
            modulated_features_per_group.append(x_group * group_attention_map)
                
        # Concatenate modulated features from all groups
        concatenated_modulated_features = torch.cat(modulated_features_per_group, dim=1) # (B, C, H, W)
        
        # Apply final fusion conv for potential inter-group interaction and refinement
        final_output = self.fusion_conv(concatenated_modulated_features)
        
        return final_output