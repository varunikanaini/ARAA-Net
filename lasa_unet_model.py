# /kaggle/working/ARAA-Net/lasa_unet_model.py
# --- FINAL VERSION: VGG19 Baseline + Feature Refinement Module (FRM) ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from lasa import LASA
from frm_module import FeatureRefinementModule # <-- The new high-impact module
import config

BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS

class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg19', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Original VGG19 Encoder ---
        vgg_features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
        self.encoder1 = nn.Sequential(*vgg_features[:6])   # 64 channels
        self.encoder2 = nn.Sequential(*vgg_features[6:13])  # 128 channels
        self.encoder3 = nn.Sequential(*vgg_features[13:26]) # 256 channels
        self.encoder4 = nn.Sequential(*vgg_features[26:39]) # 512 channels
        
        # --- 2. Instantiate the NEW Feature Refinement Module ---
        # This module will act as our intelligent bottleneck
        encoder_channels = [64, 128, 256, 512]
        self.frm = FeatureRefinementModule(in_channels_list=encoder_channels, out_channels=512)

        # --- 3. LASA Module (applied to the final encoder stage before FRM) ---
        self.lasa_module = LASA(in_channels=512, L_list=lasa_kernels)

        # --- 4. Simplified Decoder ---
        # The decoder now starts from the powerful FRM output
        self.decoder4 = self._decoder_block(512 + 512, 256) # FRM_out + e4 -> 256
        self.decoder3 = self._decoder_block(256 + 256, 128) # Dec4_out + e3 -> 128
        self.decoder2 = self._decoder_block(128 + 128, 64)  # Dec3_out + e2 -> 64
        self.decoder1 = self._decoder_block(64 + 64, 64)    # Dec2_out + e1 -> 64
        
        # --- 5. Final Output Layer ---
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # NOTE: Deep supervision is removed for this version to simplify and
        # focus on the impact of the FRM. It can be added back later if needed.

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
        input_h, input_w = x.shape[2:]

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # --- FRM as the new Bottleneck ---
        # 1. Pass all encoder features to the FRM
        frm_out = self.frm([e1, e2, e3, e4])
        
        # --- Decoder Path ---
        # Start decoding from the powerful FRM output
        d4 = F.interpolate(frm_out, size=e4.shape[2:], mode='bilinear', align_corners=True)
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
        
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # Return a single output tensor
        return (final_output_upsampled,) # Return as a tuple to match trainer logic