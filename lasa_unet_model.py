# /kaggle/working/ARAA-Net/lasa_unet_model.py
# --- FINAL VERSION: VGG19 Baseline + SE Blocks ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Your custom modules
from lasa import LASA
from se_block import SEBlock # The new enhancement
import config

BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS

class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg19', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels

        # --- 1. Load Backbone and Get Channel Info ---
        channel_info = BACKBONE_CHANNELS_INFO.get(backbone_name)
        if channel_info is None:
            raise ValueError(f"Channel info for '{backbone_name}' not found in config.py")
        
        self._get_backbone_features(backbone_name)
        
        self.encoder1_channels = channel_info['e1']
        self.encoder2_channels = channel_info['e2']
        self.encoder3_channels = channel_info['e3']
        self.encoder4_channels = channel_info['e4']
        self.bottleneck_channels = channel_info['bottleneck']

        # --- 2. LASA Module (Applied to the final encoder output) ---
        self.lasa_module = LASA(in_channels=self.encoder4_channels, L_list=self.lasa_kernels)

        # --- 3. Decoder Blocks and Auxiliary Heads (Reverted to original) ---
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
            vgg_features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
            
            # --- ENHANCEMENT: Rebuild VGG blocks with SE layers injected ---
            self.encoder1 = nn.Sequential(*vgg_features[:6], SEBlock(64))
            self.encoder2 = nn.Sequential(*vgg_features[6:13], SEBlock(128))
            self.encoder3 = nn.Sequential(*vgg_features[13:26], SEBlock(256))
            self.encoder4 = nn.Sequential(*vgg_features[26:39], SEBlock(512))
            self.bottleneck_layer = nn.Sequential(*vgg_features[39:52])
        else:
            raise NotImplementedError(f"This model version is optimized for 'vgg19'. Other backbones are not supported.")

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

        # Apply LASA enhancement
        e4_enhanced = self.lasa_module(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- Original Decoder Path with Skip Connections ---
        aux_outputs = [] 

        # Decoder 4
        d4_interp_size = e4.shape[2:] 
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1) # Use original e4 for skip connection
        d4_out = self.decoder4(d4) 
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        d3_interp_size = e3.shape[2:]
        d3 = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1) 
        d3_out = self.decoder3(d3) 
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        d2_interp_size = e2.shape[2:]
        d2 = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1) 
        d2_out = self.decoder2(d2) 
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1
        d1_interp_size = e1.shape[2:]
        d1 = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1) 
        d1_out = self.decoder1(d1) 
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) 
        
        # Final output
        final_output = self.final_conv(d1_out) 
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])