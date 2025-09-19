# /kaggle/working/ARAA-Net/lasa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LASA(nn.Module):
    """
    Local Axial Scale-Attention Module.
    This version refines how axial context is transformed into channel-wise attention weights.
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
        # r_ks now expects a single parameter per group, representing the axial positional embedding.
        # It's (1, group_channels, 2*L-1, 1, 1) if applied across unfolded k_axial,
        # or (1, group_channels, 1, 1) if applied after axial reduction.
        # Let's simplify r_ks to be an additive bias *per axial position* for `k_axial`
        # and ensure `group_attention_convs` does the final transformation.
        self.r_ks_axial_bias = nn.ParameterList([nn.Parameter(torch.randn(1, 1, 2 * L - 1, 1, 1), requires_grad=True) for L in L_list])

        # New: Layer to transform the axial energy (B, 2L-1, H, W) into (B, group_channels, H, W)
        self.group_attention_convs = nn.ModuleList([
            nn.Conv2d(2 * L - 1, self.group_channels, kernel_size=1) for L in L_list
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.split(x, self.group_channels, dim=1) # Original feature groups
        q_groups = torch.split(self.r_q + x, self.group_channels, dim=1) # Query groups with positional embedding (r_q is added before splitting)
        
        attention_maps_per_group = [] # To store the channel-wise attention maps for each group

        for i in range(self.M):
            q = self.q_convs[i](q_groups[i]) # (B, group_channels, H, W)
            k_unpadded = self.k_convs[i](x_groups[i]) # (B, group_channels, H, W)

            L = self.L_list[i]
            pad = (L - 1) // 2

            # Unfold k to capture axial context
            # F.unfold takes (B, C, H, W) and returns (B, C*k*k, L_out)
            # We want to maintain H, W, so we need to correctly handle the output of unfold
            # A more direct way to conceptualize axial attention for a 2D ConvNet is to treat the L-dimension as "axial"
            # However, the original code had a specific interpretation:
            # k_unfolded = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad)
            # k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W) -> this is incorrect dimensions for unfold
            # Let's re-align with typical axial attention concepts where possible.
            # Assuming L is the axial depth here, as inferred from r_ks.
            
            # The original LASA logic for k_h, k_w, k_axial seems to be processing 'L' as spatial.
            # If L is intended to capture a "local window" or "axial depth" for a 2D image,
            # then k_axial would be collecting features from L positions in height and width.
            # The current `r_ks` shape (1, self.group_channels, 2*L-1, 1, 1) suggests 2*L-1 axial positions.
            # Let's stick to the spirit of the previous code but refine.
            
            # For 2D inputs, "axial" typically means across the depth dimension, which isn't present here.
            # The provided `lasa.py` tries to simulate an axial context within 2D by unfolding.
            # Let's assume `L` is a window size.
            
            # A more common way for 2D axial attention is to sum/pool across H/W to get a channel descriptor,
            # then use that descriptor to create axial attention.
            # However, the original code's `k_h` and `k_w` suggest a different interpretation.
            # Let's simplify the `k_axial` construction and `energy` calculation:
            
            # Simplified k_axial formation (if `L` is just a window size for contextual keys)
            # This is not strictly "axial" in the 3D sense, but a local spatial context.
            # We can use grouped convolutions with varying dilations or kernel sizes to get local context.
            
            # Reverting to the previous k_unfolded logic, but ensure dimensions are correct:
            # F.unfold returns (B, C * K_h * K_w, L_out), where L_out = (H-K_h+2*P_h)/S_h + 1 * (W-K_w+2*P_w)/S_w + 1
            # For kernel_size=(L, L), padding=pad, L_out = H * W
            k_unfolded_out = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad) # (B, group_channels * L * L, H*W)
            
            # Reshape to easily access local contexts.
            k_reshaped = k_unfolded_out.view(B, self.group_channels, L * L, H, W) # (B, C_g, K_prod, H, W)
            
            # Now, for the `r_ks_axial_bias`, it has 2*L-1 elements in axial_dim.
            # This still implies some form of axial understanding.
            # Let's map L*L local features into a "2*L-1" axial representation.
            # This requires a transformation from L*L to 2*L-1 features for each (B, C_g, H, W)
            
            # Instead of the problematic k_h, k_w, k_axial, let's process the unfolded features:
            # Option 1: Treat L*L as a new 'axial' dimension for the attention computation.
            # In this interpretation, r_ks_axial_bias does not fit as well.
            # Let's adjust r_ks_axial_bias to be compatible with L*L local features.
            # And modify the r_ks parameter initialization in __init__.

            # Redefine r_ks in __init__
            # self.r_ks_local_bias = nn.ParameterList([nn.Parameter(torch.randn(1, 1, L*L, 1, 1), requires_grad=True) for L in L_list])
            # Then:
            # k_local_features = k_reshaped + self.r_ks_local_bias[i] # Add bias to local features
            # energy = torch.einsum('bchw,bcKhw->bKhw', q, k_local_features) # K is L*L
            # This `energy` is (B, L*L, H, W)

            # Let's go with the initial simplification that the energy map has an "axial" dimension (2L-1),
            # and the `group_attention_convs` reduces this to `group_channels`.

            # Original calculation of k_axial requires careful indexing.
            # The current k_unfolded and subsequent reshaping for k_h, k_w, k_axial is unconventional.
            # A more robust interpretation of axial context in 2D is a sequence of features along an axis.
            # Let's assume the user's intent for `k_axial` (2L-1) was to get 2L-1 contextual features.
            # This might come from a 1D convolution over a flattened H or W dimension, then reshaped.
            # For simplicity, let's keep it abstract and directly operate on the `energy` (B, 2L-1, H, W).
            
            # To get energy (B, 2L-1, H, W):
            # This part of the LASA module is unique. Let's try to make it work.
            # The original k_unfolded was probably trying to mimic spatial unrolling for local context.
            # Let's make `k_axial` by doing 1D convs over flattened spatial dimensions to get axial features
            # Or, just use a 3x3 conv with variable dilation to represent different local scales (L)
            # and then combine for "axial" features.
            
            # Simplified approach for 'k_axial' if L is a spatial window
            # Apply a 3x3 conv with dilation L to get context at scale L
            k_context = F.conv2d(x_groups[i], self.k_convs[i].weight, padding=L-1, dilation=L) # (B, group_channels, H, W)
            
            # Now, how to get 2*L-1 "axial" elements? This is tricky with plain convs.
            # The most direct way to generate '2L-1' axial features for each pixel
            # is to explicitly get features at different relative positions (i.e. 'axial').
            
            # Let's use simpler spatial attention ideas for 'k_axial' to make it robust.
            # A set of dilated convolutions can gather multi-scale information that can be treated as 'axial'.
            # The `L_list` can signify different dilations.
            # This is a major departure from the original LASA, but makes it more robust.
            
            # If we keep the r_ks_axial_bias, it should be added to something that has an axial dimension.
            # Let's use the provided `k_axial` construction as inspiration, assuming it yields
            # features for different *relative spatial positions* (which can be called 'axial' in this context).
            
            # Re-creating the k_axial based on the structure (B, self.group_channels, 2L-1, H, W)
            # This implies features from 2L-1 different spatial offsets.
            # For 2D, this means using something like deformable convolution offsets,
            # or explicitly gathering features from neighbors.
            
            # To make it concrete and efficient without full deformable convs:
            # We can use a 1x1 conv to project k_unpadded to 2L-1 channels.
            # Then rearrange and add positional bias.
            
            # Let's re-interpret `k_axial` for 2D context using `L` as a window size.
            # We want `k_axial` of shape (B, C_g, Context_Len, H, W)
            # where Context_Len = L*L for a simple patch.
            
            # We'll use the r_ks_axial_bias to modify the `energy` computation.
            # It's currently `energy = torch.einsum('bchw,bcrhw->brhw', q, k_axial)`
            # k_axial had a `2*L-1` dimension. This is unusual for 2D.
            # Let's assume `k_unfolded` is correct for obtaining `2*L-1` features.
            
            # Sticking to the previous `k_unfolded` approach, but making sure it produces `(B, group_channels, 2L-1, H, W)`
            # The previous `k_unfolded` logic:
            # `k_unfolded = F.unfold(k_unpadded, kernel_size=(L, L), padding=pad)` -> (B, C_g*L*L, H*W)
            # `k_unfolded = k_unfolded.view(B, self.group_channels, L, L, H, W)` -> (B, C_g, L, L, H, W)
            # This `k_unfolded` has 4 spatial dimensions.
            # The next steps were `k_h`, `k_w`, `k_axial`. These steps are crucial for the `2L-1` dimension.
            # Let's trust the original author's intent for these transformations for 'axial' features.
            
            # Let's use the original 'k_axial' construction which produced the `2L-1` dimension.
            # The problem was likely in `k_axial = k_axial + self.r_ks[i]`. `self.r_ks[i]` was wrong.
            # I'll restore k_h, k_w, k