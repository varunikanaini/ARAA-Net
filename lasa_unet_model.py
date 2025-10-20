# /kaggle/working/ARAA-Net/lasa_unet_model.py
# --- FINAL REVERTED VERSION ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from lasa import LASA
import config

BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS

class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg19', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels

        channel_info = BACKBONE_CHANNELS_INFO.get(backbone_name)
        if channel_info is None:
            raise ValueError(f"Channel info for '{backbone_name}' not in config.py")
        
        self._get_backbone_features(backbone_name)
        
        self.encoder1_channels, self.encoder2_channels, self.encoder3_channels, self.encoder4_channels, self.bottleneck_channels = \
            channel_info['e1'], channel_info['e2'], channel_info['e3'], channel_info['e4'], channel_info['bottleneck']

        self.lasa_module = LASA(in_channels=self.encoder4_channels, L_list=self.lasa_kernels)

        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.encoder4_channels, self.encoder4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.encoder4_channels, num_classes, kernel_size=1)
        self.decoder3 = self._decoder_block(self.encoder4_channels + self.encoder3_channels, self.encoder3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.encoder3_channels, num_classes, kernel_size=1)
        self.decoder2 = self._decoder_block(self.encoder3_channels + self.encoder2_channels, self.encoder2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.encoder2_channels, num_classes, kernel_size=1)
        self.decoder1 = self._decoder_block(self.encoder2_channels + self.encoder1_channels, self.encoder1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)
        self.final_conv = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)

    def _get_backbone_features(self, backbone_name):
        if backbone_name == 'vgg19':
            features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
        elif backbone_name == 'vgg16':
            features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' not supported.")
        
        self.encoder1 = nn.Sequential(*features[:6])
        self.encoder2 = nn.Sequential(*features[6:13])
        self.encoder3 = nn.Sequential(*features[13:26])
        self.encoder4 = nn.Sequential(*features[26:39])
        self.bottleneck_layer = nn.Sequential(*features[39:52])

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    # In /kaggle/working/ARAA-Net/lasa_unet_model.py

    # --- REPLACE THE ENTIRE forward method with this ---
    def forward(self, x):
        input_h, input_w = x.shape[2:]

        # --- Encoder Path (Corrected sequential execution) ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Apply LASA enhancement to the final encoder output
        e4_enhanced = self.lasa_module(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- Decoder Path ---
        aux_outputs = [] 

        # Decoder 4
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