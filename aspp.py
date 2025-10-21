# /kaggle/working/ARAA-Net/aspp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different rates
        self.aspp_blocks = nn.ModuleList()
        for rate in atrous_rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Image pooling (e.g., Global Average Pooling)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final convolution to combine features
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False), # +2 for 1x1 conv and image_pool
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: B x C x H x W
        size = x.shape[2:]

        # 1x1 convolution output
        conv1x1_out = self.conv1x1(x)

        # ASPP blocks output
        aspp_outs = [conv1x1_out]
        for block in self.aspp_blocks:
            aspp_outs.append(block(x))

        # Image pooling output
        image_pool_out = self.image_pool(x)
        # Resize image_pool_out to match spatial dimensions of other outputs
        image_pool_out = F.interpolate(image_pool_out, size=size, mode='bilinear', align_corners=True)
        aspp_outs.append(image_pool_out)

        # Concatenate all outputs
        x = torch.cat(aspp_outs, dim=1)

        # Final convolution
        return self.conv_final(x)