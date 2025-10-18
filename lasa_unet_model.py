# /kaggle/working/ARAA-Net/lasa_unet_model.py
# --- FINAL CORRECTED VERSION ---

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Your custom modules remain essential
from lasa import LASA
from attention_gate import AttentionGate
import config

# Load channel info from config
BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS

class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg19', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Backbone Layers Explicitly ---
        self._load_backbone_layers(backbone_name)

        # --- 2. Get Channel Info from Config (Must Match Slicing) ---
        channel_info = BACKBONE_CHANNELS_INFO.get(backbone_name)
        if channel_info is None:
            raise ValueError(f"Channel info for '{backbone_name}' not found in config.py")

        self.encoder1_channels = channel_info['e1']
        self.encoder2_channels = channel_info['e2']
        self.encoder3_channels = channel_info['e3']
        self.encoder4_channels = channel_info['e4']
        self.bottleneck_channels = channel_info['bottleneck']

        # --- 3. Instantiate Custom Modules (LASA and Attention Gates) ---
        self.lasa_module = LASA(in_channels=self.encoder4_channels, L_list=lasa_kernels)
        
        # Attention Gates for skip connections
        self.Att4 = AttentionGate(F_g=self.bottleneck_channels, F_l=self.encoder4_channels, F_int=self.encoder4_channels // 2)
        self.Att3 = AttentionGate(F_g=self.encoder4_channels, F_l=self.encoder3_channels, F_int=self.encoder3_channels // 2)
        self.Att2 = AttentionGate(F_g=self.encoder3_channels, F_l=self.encoder2_channels, F_int=self.encoder2_channels // 2)
        self.Att1 = AttentionGate(F_g=self.encoder2_channels, F_l=self.encoder1_channels, F_int=self.encoder1_channels // 2)
        
        # --- 4. Decoder Blocks and Auxiliary Heads ---
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.encoder4_channels, self.encoder4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.encoder4_channels, num_classes, kernel_size=1)

        self.decoder3 = self._decoder_block(self.encoder4_channels + self.encoder3_channels, self.encoder3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.encoder3_channels, num_classes, kernel_size=1)

        self.decoder2 = self._decoder_block(self.encoder3_channels + self.encoder2_channels, self.encoder2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.encoder2_channels, num_classes, kernel_size=1)

        self.decoder1 = self._decoder_block(self.encoder2_channels + self.encoder1_channels, self.encoder1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)
        
        self.final_conv = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1)

    def _load_backbone_layers(self, backbone_name):
        """
        Loads backbone and assigns its layers explicitly to prevent slicing errors.
        """
        if backbone_name == 'efficientnet_b4':
            effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            features = effnet.features
            # Define each encoder stage as a distinct layer
            self.encoder_stem = features[0]
            self.encoder_e1 = features[1]
            self.encoder_e2 = features[2]
            self.encoder_e3 = features[3]
            self.encoder_e4 = features[4]
            self.bottleneck_layer = nn.Sequential(*features[5:])
        elif backbone_name == 'vgg19':
            vgg_features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
            self.encoder_e1 = nn.Sequential(*list(vgg_features.children())[:13])
            self.encoder_e2 = nn.Sequential(*list(vgg_features.children())[13:26])
            self.encoder_e3 = nn.Sequential(*list(vgg_features.children())[26:39])
            self.encoder_e4 = nn.Sequential(*list(vgg_features.children())[39:52])
            self.bottleneck_layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2) # VGG needs an extra MaxPool
            ) 
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

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

        # --- Explicit Encoder Path ---
        if self.backbone_name == 'efficientnet_b4':
            s0 = self.encoder_stem(x)
            e1 = self.encoder_e1(s0)
            e2 = self.encoder_e2(e1)
            e3 = self.encoder_e3(e2)
            e4 = self.encoder_e4(e3) # This will have 160 channels
        else: # Fallback for VGG-like structures
            e1 = self.encoder_e1(x)
            e2 = self.encoder_e2(e1)
            e3 = self.encoder_e3(e2)
            e4 = self.encoder_e4(e3)

        # Apply LASA enhancement to e4 (Input: 160 channels)
        e4_enhanced = self.lasa_module(e4)

        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced)

        # --- Decoder Path with Attention Gates ---
        aux_outputs = []

        # Decoder 4
        d4_interp_size = e4.shape[2:]
        d4_gating = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        e4_att = self.Att4(g=d4_gating, x=e4)
        d4 = torch.cat([d4_gating, e4_att], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        d3_interp_size = e3.shape[2:]
        d3_gating = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        e3_att = self.Att3(g=d3_gating, x=e3)
        d3 = torch.cat([d3_gating, e3_att], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        d2_interp_size = e2.shape[2:]
        d2_gating = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        e2_att = self.Att2(g=d2_gating, x=e2)
        d2 = torch.cat([d2_gating, e2_att], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1
        d1_interp_size = e1.shape[2:]
        d1_gating = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        e1_att = self.Att1(g=d1_gating, x=e1)
        d1 = torch.cat([d1_gating, e1_att], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])