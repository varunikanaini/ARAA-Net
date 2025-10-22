# light_lasa_unet.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA
from se_block import SEBlock
# --- REMOVE OLD IMPORTS and ADD THE NEW ONE ---
# from boundary_module import BoundaryModule
# from spatial_attn import BoundarySpatialAttention
from aff_block import AFFBlock  # <-- IMPORT THE NEW AGGRESSIVE BLOCK

class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        self.encoder1 = nn.Sequential(*mobilenet.features[0:2], SEBlock(16))
        self.encoder2 = nn.Sequential(*mobilenet.features[2:4], SEBlock(24))
        self.encoder3 = nn.Sequential(*mobilenet.features[4:7], SEBlock(32))
        self.encoder4 = nn.Sequential(*mobilenet.features[7:14], SEBlock(96))
        self.bottleneck_layer = nn.Sequential(*mobilenet.features[14:])

        e1_ch, e2_ch, e3_ch, e4_ch, bottle_ch = 16, 24, 32, 96, 1280

        self.lasa_module = LASA(in_channels=e4_ch, L_list=lasa_kernels)

        # --- REPLACE THE DECODER BLOCKS WITH THE NEW AFFBlock ---
        # The AFFBlock handles all the complexity internally.
        self.decoder4 = AFFBlock(bottle_ch + e4_ch, 256)
        self.aux_conv_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.decoder3 = AFFBlock(256 + e3_ch, 128)
        self.aux_conv_d3 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.decoder2 = AFFBlock(128 + e2_ch, 64)
        self.aux_conv_d2 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.decoder1 = AFFBlock(64 + e1_ch, 64)
        self.aux_conv_d1 = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    # NO NEED TO CHANGE _decoder_block as it is no longer used.
    # You can safely delete the old _decoder_block method.

    def forward(self, x):
        # The forward pass logic remains IDENTICAL.
        # This is the beauty of a modular, plug-and-play design.
        input_h, input_w = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        e4_enhanced = self.lasa_module(e4)
        
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        aux_outputs = []

        d4_input = torch.cat([F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1)
        d4_out = self.decoder4(d4_input) # Pass the concatenated tensor here
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d3_input = torch.cat([F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1)
        d3_out = self.decoder3(d3_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d2_input = torch.cat([F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1)
        d2_out = self.decoder2(d2_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d1_input = torch.cat([F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1)
        d1_out = self.decoder1(d1_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])