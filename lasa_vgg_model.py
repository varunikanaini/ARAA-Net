# /kaggle/working/ARAA-Net/lasa_vgg_model.py
import torch
import torch.nn as nn
import torchvision.models as models
from lasa import LASA # We will reuse your existing lasa.py file

class LASA_VGG_Unet(nn.Module):
    """
    A standalone segmentation model using a VGG16 backbone, a LASA module for
    feature enhancement, and a U-Net style decoder.
    """
    def __init__(self, num_classes=2):
        super(LASA_VGG_Unet, self).__init__()

        # --- 1. Load Pre-trained VGG16 Backbone ---
        # We use the 'features' part of VGG16 with Batch Normalization
        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
        
        # --- 2. Define Encoder Blocks from VGG ---
        # We slice the VGG network to create encoder blocks and save their outputs
        # for skip connections in the U-Net decoder.
        self.encoder1 = vgg_features[:6]   # Output channels: 64
        self.encoder2 = vgg_features[6:13]  # Output channels: 128
        self.encoder3 = vgg_features[13:23] # Output channels: 256
        self.encoder4 = vgg_features[23:33] # Output channels: 512
        
        # --- 3. Insert the LASA Module ---
        # LASA will enhance the features from the 4th encoder block.
        self.lasa = LASA(in_channels=512)

        # --- 4. Define the Bottleneck ---
        self.bottleneck = vgg_features[33:43] # Output channels: 512

        # --- 5. Define Decoder Blocks ---
        # These blocks will upsample the features and merge them with skip connections.
        self.decoder4 = self._decoder_block(512 + 512, 512) # Input: bottleneck + lasa-enhanced e4
        self.decoder3 = self._decoder_block(512 + 256, 256) # Input: upsampled d4 + e3
        self.decoder2 = self._decoder_block(256 + 128, 128) # Input: upsampled d3 + e2
        self.decoder1 = self._decoder_block(128 + 64, 64)   # Input: upsampled d2 + e1

        # --- 6. Final Output Convolution ---
        # This converts the final 64-channel feature map to a 2-channel segmentation map.
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Apply LASA enhancement
        e4_lasa = self.lasa(e4)
        
        bottleneck = self.bottleneck(e4_lasa)

        # --- Decoder Path with Skip Connections ---
        d4 = nn.functional.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_lasa], dim=1) # Skip connection from LASA-enhanced features
        d4 = self.decoder4(d4)
        
        d3 = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # The model only has one final output, unlike ARAA-Net's deep supervision
        final_output = self.final_conv(d1)

        # To maintain compatibility with the validation/testing logic that expects 5 outputs,
        # we return the final output 5 times. This is a simple trick to reuse the existing code.
        return final_output, final_output, final_output, final_output, final_output