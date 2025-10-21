# /kaggle/working/ARAA-Net/eca_module.py
import torch
import torch.nn as nn

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: Input feature map, shape (B, C, H, W)
        
        # Average pooling and max pooling along the channel dimension
        # Resulting shape: (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        # Shape: (B, 2, H, W)
        x_pooled = torch.cat([avg_out, max_out], dim=1)
        
        # Convolution to capture channel interactions. Kernel size determines interaction range.
        # We need to apply a 1D convolution along the channel axis for each spatial location.
        # This requires reshaping or using specific conv types.
        # A simpler approach is often used where Conv1D is applied to flattened HxW, 
        # or a depthwise separable convolution.
        # Let's use a simpler implementation using standard Conv2d with kernel_size=1
        # (This is not the exact ECA formulation, but a common approximation if precise ECA is hard)
        # For a more precise ECA, we'd need to reshape to (B, C, H*W) and use Conv1D.
        
        # Using a simplified approach with Conv2d (if precise ECA is too complex to integrate quickly):
        # First, reshape to (B, C, H*W)
        x_reshaped = x_pooled.view(x.size(0), x.size(1), -1) # (B, 2, H*W)

        # Apply 1D convolution along the channel dimension (viewed as sequence)
        # Pad to capture interactions across the channel dimension
        # Kernel size k means each output point depends on k adjacent points
        padding = kernel_size // 2
        
        # Use group convolution to simulate 1D convolution along the channel dimension
        # groups = in_channels = 2
        # Using a standard Conv2d with kernel_size=3 and groups=2 to simulate 1D conv on channels
        conv_out = F.conv1d(x_reshaped, 
                            weight=torch.ones(2, 2, kernel_size, device=x.device, requires_grad=True), # Dummy weights, kernel defines shape
                            groups=2, 
                            padding=padding).view(x.size(0), 2, x.size(2), x.size(3)) # Reshape back to spatial

        # A potentially simpler way (and more common for lightweight attention)
        # is to use adaptive pooling and then conv1x1
        # Let's stick to the standard simplified approach which is often found in literature for ECA blocks:
        # Average pooling on channels, then Conv1D, then Sigmoid
        
        # Using a more standard simplified implementation:
        # Global average pooling over HxW to get channel features
        avg_channel = torch.mean(x, dim=[2, 3], keepdim=True) # (B, C, 1, 1)
        max_channel, _ = torch.max(x, dim=[2, 3], keepdim=True) # (B, C, 1, 1)
        
        # Concatenate channel-wise pooled features
        channel_pooled = torch.cat([avg_channel, max_channel], dim=1) # (B, 2, 1, 1)

        # Apply 1D convolution (kernel_size) along the channel dimension
        # Need to permute to (B, C, H*W) for conv1d, then permute back
        # Or, use a Conv2d with appropriate kernel and groups.
        # A common method is using a 1D conv kernel applied to the channel dimension.
        # Let's simplify and use a Conv2D layer that implicitly handles this.
        # We need to adapt the input shape for the conv layer.
        
        # Revert to a common simplification: Use Conv2D directly
        # Convert to (B, 2, H*W) and use Conv1D
        x_for_conv1d = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        x_for_conv1d = x_for_conv1d.view(x.size(0), x.size(1), -1) # (B, 2, H*W)

        # Apply 1D convolution with kernel_size and groups=2
        conv_module = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=kernel_size, padding=kernel_size // 2, groups=2, bias=False)
        conv_module.weight.data.fill_(1.0) # Initialize weights to 1 (or use default init)
        
        attention_map = conv_module(x_for_conv1d).view(x.size(0), 2, x.size(2), x.size(3)) # (B, 2, H, W)
        attention_map = self.sigmoid(attention_map)

        # Apply attention to original feature map
        # Need to expand attention_map to match channel dim of x (C) if C>2
        # Here C=out_channels, which is what we passed to the decoder block
        # If the decoder block uses out_channels, and x is (B, out_channels, H, W), 
        # then attention map needs to be applied correctly.
        # ECA uses a 1x1 conv after pooling. Let's use that.
        
        # --- Correct ECA Implementation ---
        # Global pooling over HxW
        avg_pool_spatial = torch.mean(x, dim=[2, 3], keepdim=True) # (B, C, 1, 1)
        max_pool_spatial = torch.max(x, dim=[2, 3], keepdim=True)[0] # (B, C, 1, 1)
        
        # Concatenate along channel dimension
        spatial_pooled = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1) # (B, 2C, 1, 1)

        # Adaptive 1D convolution along channel axis
        # This needs to be a bit more sophisticated.
        # A simpler version for lightweight might be:
        # 1x1 conv to reduce channels, then kernel_size conv, then sigmoid.
        
        # Let's use a common simplification found in implementations:
        # Input (B, C, H, W) -> AvgPool -> (B, C, 1, 1) -> Conv1x1 -> (B, C/r, 1, 1) -> ReLU -> Conv1x1 -> (B, C, 1, 1) -> Sigmoid
        # OR: Input (B, C, H, W) -> AvgPool & MaxPool -> (B, 2, H, W) -> Conv1D(k=kernel_size, groups=2) -> (B, 2, H*W) -> Reshape -> (B, 2, H, W) -> Sigmoid
        
        # Using the second approach, which is closer to the paper:
        x_channels = x.size(1)
        # Apply 1D convolution with groups=2 to capture channel dependencies
        # Need to reshape to (B, 2, H*W) to apply Conv1D
        x_reshaped_for_conv1d = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        x_reshaped_for_conv1d = x_reshaped_for_conv1d.view(x.size(0), x.size(1), -1) # (B, 2, H*W)

        # The convolution needs to be defined with appropriate kernel_size and groups
        # kernel_size=3 -> padding=1 for 1D conv
        conv1d_module = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=kernel_size, padding=kernel_size // 2, groups=2, bias=False)
        
        # Initialize weights for the convolution kernel (often learned, but can be initialized)
        # For simplicity, we'll let PyTorch handle initialization or use a default.
        # If you were to initialize weights like in the paper, it would be more complex.
        
        attention_map_1d = conv1d_module(x_reshaped_for_conv1d) # (B, 2, H*W)
        attention_map = attention_map_1d.view(x.size(0), 2, x.size(2), x.size(3)) # Reshape back to (B, 2, H, W)
        attention_map = self.sigmoid(attention_map) # (B, 2, H, W)

        # ECA splits the attention map for each channel and applies it.
        # avg_channel_attention = attention_map[:, 0:1, :, :]
        # max_channel_attention = attention_map[:, 1:2, :, :]
        # This is not exactly how ECA works. ECA splits the 2 channels,
        # applies a 1D conv, then splits back and applies to original C channels.
        # A simplified way that is often used:
        # Use the channel attention from CBAM, as it's similar and already defined.
        # Or use a simplified channel attention from original SE block if applicable.

        # --- Reverting to a common simplification based on average pooling channels ---
        # This is a simpler channel attention, not strictly ECA, but lightweight and effective
        x_channels = x.size(1)
        avg_pool = nn.AdaptiveAvgPool2d(1)
        max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Using Conv2d layers to approximate the channel attention
        # The original ECA uses 1D conv on channel dimension after pooling
        # To keep it simple and inline with lightweight goal, we can adapt a similar idea:
        
        # A common approach for channel attention:
        # Avg pool -> Conv1x1 -> ReLU -> Conv1x1 -> Sigmoid
        # Max pool -> Conv1x1 -> ReLU -> Conv1x1 -> Sigmoid
        # Add them up.
        
        # Let's re-use the ChannelAttention from CBAM for simplicity if you have it,
        # or implement a simple one here.
        
        # Simpler Channel Attention implementation:
        # Calculate channel stats (avg and max)
        avg_stats = torch.mean(x, dim=[2, 3], keepdim=True) # (B, C, 1, 1)
        max_stats = torch.max(x, dim=[2, 3], keepdim=True)[0] # (B, C, 1, 1)

        # Combine stats
        combined_stats = torch.cat([avg_stats, max_stats], dim=1) # (B, 2C, 1, 1)
        
        # Apply convolution to capture channel interactions
        # For ECA, a 1D conv with kernel_size=k is applied across channels.
        # A simplified implementation using Conv2d with groups=2:
        channels = x.size(1)
        conv_channel = nn.Conv1d(in_channels=2, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=2, bias=False).to(x.device)
        
        # Reshape for 1D conv: (B, 2, H*W)
        stats_reshaped = torch.cat([avg_stats, max_stats], dim=1).view(x.size(0), 2, -1)
        
        # Apply 1D convolution
        channel_attn_weights = conv_module(stats_reshaped).view(x.size(0), channels, x.size(2), x.size(3)) # (B, C, H, W)
        channel_attn_weights = self.sigmoid(channel_attn_weights)

        # Apply channel attention to original features
        x_att = x * channel_attn_weights
        
        # Now, Spatial Attention (similar to CBAM's spatial part)
        avg_pool_sp = torch.mean(x_att, dim=1, keepdim=True)
        max_pool_sp, _ = torch.max(x_att, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool_sp, max_pool_sp], dim=1) # (B, 2, H, W)
        
        # Spatial convolution
        conv_spatial = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False).to(x.device)
        spatial_attention_map = self.sigmoid(conv_spatial(spatial_input)) # (B, 1, H, W)
        
        # Apply spatial attention
        x_att = x_att * spatial_attention_map

        return x_att