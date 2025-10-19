# /kaggle/working/ARAA-Net/light_lasa_unet.py
# --- FINAL, FACT-BASED LIGHTWEIGHT MODEL ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA
import config

class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()

        # --- 1. Load MobileNetV2 Encoder Stages (Based on Architecture Printout) ---
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.encoder1 = nn.Sequential(*mobilenet.features[0:2])   # out: 16 ch
        self.encoder2 = nn.Sequential(*mobilenet.features[2:4])   # out: 24 ch
        self.encoder3 = nn.Sequential(*mobilenet.features[4:7])   # out: 32 ch
        self.encoder4 = nn.Sequential(*mobilenet.features[7:14])  # out: 96 ch
        self.bottleneck_layer = nn.Sequential(*mobilenet.features[14:]) # out: 1280 ch

        # Channel counts are hard-coded based on the known architecture
        e1_ch, e2_ch, e3_ch, e4_ch, bottle_ch = 16, 24, 32, 96, 1280

        # --- 2. LASA Module ---
        self.lasa_module = LASA(in_channels=e4_ch, L_list=lasa_kernels)

        # --- 3. Decoder ---
        self.decoder4 = self._decoder_block(bottle_ch + e4_ch, 256)
        self.decoder3 = self._decoder_block(256 + e3_ch, 128)
        self.decoder2 = self._decoder_block(128 + e2_ch, 64)
        self.decoder1 = self._decoder_block(64 + e1_ch, 64)
        
        # --- 4. Final Output ---
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        e4_enhanced = self.lasa_module(e4)
        bottleneck = self.bottleneck_layer(e4_enhanced)

        d4 = F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4_out = self.decoder4(d4)

        d3 = F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3_out = self.decoder3(d3)

        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)

        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        
        out = self.final_conv(d1_out)
        
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)