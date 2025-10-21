# /kaggle/working/ARAA-Net/light_lasa_unet.py
# --- FINAL VERSION: MobileNetV2 + SE Blocks + Deep Supervision + CBAM ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA
from se_block import SEBlock # SE Blocks in encoder
from cbam import CBAM # <-- ADD THIS IMPORT
# In light_lasa_unet.py

# ... imports ...
from aspp import ASPP # <-- ADD THIS IMPORT
# ... other imports ...

class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # --- ENCODER (with SE Blocks) ---
        self.encoder1 = nn.Sequential(*mobilenet.features[0:2], SEBlock(16))
        self.encoder2 = nn.Sequential(*mobilenet.features[2:4], SEBlock(24))
        self.encoder3 = nn.Sequential(*mobilenet.features[4:7], SEBlock(32))
        self.encoder4 = nn.Sequential(*mobilenet.features[7:14], SEBlock(96))
        
        # --- BOTTLENECK LAYER REPLACED WITH ASPP ---
        # ASPP takes input from encoder4 (e4_ch = 96) and outputs out_channels (e.g., 256)
        self.aspp = ASPP(in_channels=96, out_channels=256, atrous_rates=[6, 12, 18]) 
        
        e1_ch, e2_ch, e3_ch, e4_ch = 16, 24, 32, 96
        aspp_out_ch = 256 # Output channels from ASPP

        self.lasa_module = LASA(in_channels=e4_ch, L_list=lasa_kernels)

        # --- DECODER ---
        # Decoder 4: Input is ASPP output (256) + encoder4 (96) -> 256 + 96 = 352
        self.decoder4 = self._decoder_block(aspp_out_ch + e4_ch, 256)
        self.aux_conv_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Decoder 3: Input is decoder4 output (256) + encoder3 (32) -> 256 + 32 = 288
        self.decoder3 = self._decoder_block(256 + e3_ch, 128)
        self.aux_conv_d3 = nn.Conv2d(128, num_classes, kernel_size=1)

        # Decoder 2: Input is decoder3 output (128) + encoder2 (24) -> 128 + 24 = 152
        self.decoder2 = self._decoder_block(128 + e2_ch, 64)
        self.aux_conv_d2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # Decoder 1: Input is decoder2 output (64) + encoder1 (16) -> 64 + 16 = 80
        self.decoder1 = self._decoder_block(64 + e1_ch, 64)
        self.aux_conv_d1 = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        # You can choose where to put CBAM here if you still want it,
        # or just use standard conv-bn-relu.
        # Let's assume for now we are adding ASPP and keeping the decoder blocks standard for simplicity.
        # If you want CBAM AND ASPP, you'd integrate CBAM into this block.
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            # --- If you want CBAM, add it here: ---
            # CBAM(out_channels),
            # ---
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_h, input_w = x.shape[2:]

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # LASA module is applied to the last encoder output
        e4_enhanced = self.lasa_module(e4)
        
        # --- Bottleneck replaced with ASPP ---
        bottleneck = self.aspp(e4_enhanced) # Use ASPP output

        # --- Decoder Path with Deep Supervision ---
        aux_outputs = []

        # Decoder 4
        # Input: ASPP output (256) + enhanced e4 (96)
        d4 = torch.cat([F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        d3 = torch.cat([F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 2
        d2 = torch.cat([F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 1
        d1 = torch.cat([F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Final output
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])
