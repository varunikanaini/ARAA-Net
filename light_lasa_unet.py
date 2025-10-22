# /kaggle/working/ARAA-Net/light_lasa_unet.py
# --- FINAL VERSION: MobileNetV2 + SE Blocks + Deep Supervision + BoundaryModule ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA
from se_block import SEBlock # SE Blocks in encoder
from boundary_module import BoundaryModule # <-- IMPORT THIS NEW MODULE
from spatial_attn import BoundarySpatialAttention 

class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # --- 1. ENHANCED MobileNetV2 ENCODER (with SE Blocks) ---
        self.encoder1 = nn.Sequential(*mobilenet.features[0:2], SEBlock(16))
        self.encoder2 = nn.Sequential(*mobilenet.features[2:4], SEBlock(24))
        self.encoder3 = nn.Sequential(*mobilenet.features[4:7], SEBlock(32))
        self.encoder4 = nn.Sequential(*mobilenet.features[7:14], SEBlock(96))
        self.bottleneck_layer = nn.Sequential(*mobilenet.features[14:])

        e1_ch, e2_ch, e3_ch, e4_ch, bottle_ch = 16, 24, 32, 96, 1280

        self.lasa_module = LASA(in_channels=e4_ch, L_list=lasa_kernels)

        # --- 2. DECODER with DEEP SUPERVISION (and Boundary Module) ---
        # Decoder 4: Input bottle_ch + e4_ch, Output 256 channels
        self.decoder4 = self._decoder_block(bottle_ch + e4_ch, 256)
        self.aux_conv_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Decoder 3: Input 256 (from decoder4) + e3_ch, Output 128 channels
        self.decoder3 = self._decoder_block(256 + e3_ch, 128)
        self.aux_conv_d3 = nn.Conv2d(128, num_classes, kernel_size=1)

        # Decoder 2: Input 128 (from decoder3) + e2_ch, Output 64 channels
        self.decoder2 = self._decoder_block(128 + e2_ch, 64)
        self.aux_conv_d2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # Decoder 1: Input 64 (from decoder2) + e1_ch, Output 64 channels
        self.decoder1 = self._decoder_block(64 + e1_ch, 64)
        self.aux_conv_d1 = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # First, capture multi-scale context for boundaries
            BoundaryModule(in_channels=out_channels, out_channels=out_channels),
            
            # Second, apply spatial attention to focus on important regions
            BoundarySpatialAttention(kernel_size=7), # <-- INTEGRATE THE ATTENTION MODULE HERE
            
            # Final refinement convolution in the block
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_h, input_w = x.shape[2:]

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # LASA module applied to the last encoder output
        e4_enhanced = self.lasa_module(e4)
        
        # Bottleneck layer
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- Decoder Path with Deep Supervision ---
        aux_outputs = []

        # Decoder 4
        d4 = torch.cat([F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1)
        d4_out = self.decoder4(d4) # This now includes BoundaryModule
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        d3 = torch.cat([F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1)
        d3_out = self.decoder3(d3) # This now includes BoundaryModule
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 2
        d2 = torch.cat([F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1)
        d2_out = self.decoder2(d2) # This now includes BoundaryModule
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 1
        d1 = torch.cat([F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1)
        d1_out = self.decoder1(d1) # This now includes BoundaryModule
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Final output
        final_output = self.final_conv(d1_out) # Attention is already applied in the last decoder block
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])