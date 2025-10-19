# /kaggle/working/ARAA-Net/lasa_unet_model.py
# --- FINAL VERSION: VGG19 + SE Blocks + MSFA Module ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from lasa import LASA
from se_block import SEBlock             # Enhancement from Paper 2
from msfa_module import MSFA_Module     # Enhancement from Paper 1
import config

BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS

class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg19', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name != 'vgg19':
            raise NotImplementedError("This model is specifically enhanced for the 'vgg19' backbone.")

        # --- 1. ENHANCED VGG19 ENCODER (with SE Blocks) ---
        vgg_features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
        self.encoder1 = nn.Sequential(*vgg_features[:6], SEBlock(64))
        self.encoder2 = nn.Sequential(*vgg_features[6:13], SEBlock(128))
        self.encoder3 = nn.Sequential(*vgg_features[13:26], SEBlock(256))
        self.encoder4 = nn.Sequential(*vgg_features[26:39], SEBlock(512))
        self.bottleneck_layer = nn.Sequential(*vgg_features[39:52])

        # Channel info from config
        channel_info = BACKBONE_CHANNELS_INFO[backbone_name]
        self.encoder1_channels, self.encoder2_channels, self.encoder3_channels, self.encoder4_channels, self.bottleneck_channels = \
            channel_info['e1'], channel_info['e2'], channel_info['e3'], channel_info['e4'], channel_info['bottleneck']

        # --- 2. LASA Module ---
        self.lasa_module = LASA(in_channels=self.encoder4_channels, L_list=lasa_kernels)

        # --- 3. NEW: Multi-Scale Feature Aggregation Module ---
        encoder_channels_list = [self.encoder1_channels, self.encoder2_channels, self.encoder3_channels, self.encoder4_channels]
        self.msfa = MSFA_Module(in_channels_list=encoder_channels_list, out_channels=self.bottleneck_channels)

        # --- 4. Original Decoder Path ---
        # The first decoder block now takes the standard bottleneck + the enhanced bottleneck from MSFA
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.encoder4_channels, self.encoder4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.encoder4_channels, num_classes, kernel_size=1)

        self.decoder3 = self._decoder_block(self.encoder4_channels + self.encoder3_channels, self.encoder3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.encoder3_channels, num_classes, kernel_size=1)

        self.decoder2 = self._decoder_block(self.encoder3_channels + self.encoder2_channels, self.encoder2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.encoder2_channels, num_classes, kernel_size=1)

        self.decoder1 = self._decoder_block(self.encoder2_channels + self.encoder1_channels, self.encoder1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)
        
        self.final_conv = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_h, input_w = x.shape[2:]

        # --- Enhanced Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Apply LASA enhancement
        e4_enhanced = self.lasa_module(e4)
        
        # Standard Bottleneck path
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- NEW: Inject Multi-Scale Context from MSFA ---
        msfa_out = self.msfa([e1, e2, e3, e4])
        # Upsample MSFA output and add it to the main bottleneck path
        upsampled_msfa = F.interpolate(msfa_out, size=bottleneck.shape[2:], mode='bilinear', align_corners=True)
        bottleneck = bottleneck + upsampled_msfa # Fuse the two contexts

        # --- Original Decoder Path (now benefits from enhanced bottleneck) ---
        aux_outputs = [] 

        d4_interp_size = e4.shape[2:] 
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4_out = self.decoder4(d4) 
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d3_interp_size = e3.shape[2:]
        d3 = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1) 
        d3_out = self.decoder3(d3) 
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2_interp_size = e2.shape[2:]
        d2 = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1) 
        d2_out = self.decoder2(d2) 
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d1_interp_size = e1.shape[2:]
        d1 = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1) 
        d1_out = self.decoder1(d1) 
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) 
        
        final_output = self.final_conv(d1_out) 
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])