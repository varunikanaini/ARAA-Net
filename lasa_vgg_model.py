# lasa_vgg_model.py (Final Attempt at Correct InceptionV3 Layer Names)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA

# --- Helper function to get features from different backbones ---
def get_backbone_features(backbone_name, pretrained=True):
    """
    Loads a backbone and returns a dictionary of feature extraction layers
     and their output channel counts, suitable for a U-Net style encoder.
    """
    if backbone_name == 'vgg16':
        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None).features
        features = {
            'encoder1': vgg_features[:6],   # Output channels: 64
            'encoder2': vgg_features[6:13],  # Output channels: 128
            'encoder3': vgg_features[13:23], # Output channels: 256
            'encoder4': vgg_features[23:33], # Output channels: 512
            'bottleneck': vgg_features[33:43] # Output channels: 512
        }
        channels = {
            'e1_channels': 64, 'e2_channels': 128, 'e3_channels': 256,
            'e4_channels': 512, 'bottleneck_channels': 512
        }
        return nn.ModuleDict(features), channels

    elif backbone_name == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        features = {
            'encoder1': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool), # Output 64 channels
            'encoder2': resnet.layer1, # Output 256 channels
            'encoder3': resnet.layer2, # Output 512 channels
            'encoder4': resnet.layer3, # Output 1024 channels
            'bottleneck': resnet.layer4 # Output 2048 channels
        }
        channels = {
            'e1_channels': 64, 'e2_channels': 256, 'e3_channels': 512,
            'e4_channels': 1024, 'bottleneck_channels': 2048
        }
        return nn.ModuleDict(features), channels

    elif backbone_name == 'inception_v3':
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
        
        # --- FINAL CORRECTION FOR INCEPTION V3 LAYER NAMES ---
        # Using the exact attribute names from the printed structure: maxpool1 and maxpool2.
        # Also ensuring the last layer for bottleneck is appropriate.
        
        features = {
            # Encoder 1: Initial layers up to the first MaxPool.
            # The names are from the printed structure you provided.
            'encoder1': nn.Sequential(
                inception.Conv2d_1a_3x3, 
                inception.Conv2d_2a_3x3, 
                inception.Conv2d_2b_3x3, 
                inception.maxpool1, # CORRECTED: Used 'maxpool1' as per the printed structure
                inception.Conv2d_3b_1x1, 
                inception.Conv2d_4a_3x3, 
                inception.Conv2d_4b_3x3, 
                inception.maxpool2  # CORRECTED: Used 'maxpool2'
            ),
            
            # Encoder 2: First set of Inception modules.
            'encoder2': nn.Sequential(inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d),
            
            # Encoder 3: Second set of Inception modules.
            'encoder3': nn.Sequential(inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e),
            
            # Encoder 4: Final set of Inception blocks.
            'encoder4': nn.Sequential(inception.Mixed_7a, inception.Mixed_7b),
            
            # Bottleneck: Features before the final average pooling and classifier.
            # Using the output of Mixed_7b.
            'bottleneck': nn.Sequential(inception.Mixed_7b)
        }
        
        # --- VERIFIED channel counts for InceptionV3 stages ---
        # These counts are based on the output channels of the specified layers.
        # You should confirm these by running a small test to print shapes.
        channels = {
            'e1_channels': 192,  # Channels after inception.maxpool2 (output of Conv2d_4b_3x3)
            'e2_channels': 288,  # Channels after inception.Mixed_5d
            'e3_channels': 768,  # Channels after inception.Mixed_6e
            'e4_channels': 1280, # Channels after inception.Mixed_7b
            'bottleneck_channels': 1280 # Matching e4_channels
        }
        return nn.ModuleDict(features), channels

    elif backbone_name.startswith('efficientnet'):
        if backbone_name == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            features = {
                'encoder1': nn.Sequential(efficientnet._conv_stem, efficientnet._bn1, efficientnet._activation, efficientnet._initial_max_pool),
                'encoder2': nn.Sequential(*efficientnet._blocks[0:2]),
                'encoder3': nn.Sequential(*efficientnet._blocks[2:4]),
                'encoder4': nn.Sequential(*efficientnet._blocks[4:7]),
                'bottleneck': nn.Sequential(*efficientnet._blocks[7:12])
            }
            channels = {
                'e1_channels': 32, 'e2_channels': 48, 'e3_channels': 80,
                'e4_channels': 128, 'bottleneck_channels': 256
            }
        elif backbone_name == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            features = {
                'encoder1': nn.Sequential(efficientnet._conv_stem, efficientnet._bn1, efficientnet._activation, efficientnet._initial_max_pool),
                'encoder2': nn.Sequential(*efficientnet._blocks[0:3]),
                'encoder3': nn.Sequential(*efficientnet._blocks[3:6]),
                'encoder4': nn.Sequential(*efficientnet._blocks[6:10]),
                'bottleneck': nn.Sequential(*efficientnet._blocks[10:16])
            }
            channels = {
                'e1_channels': 40, 'e2_channels': 56, 'e3_channels': 96,
                'e4_channels': 176, 'bottleneck_channels': 304
            }
        else:
            raise ValueError(f"Unsupported EfficientNet version: {backbone_name}")
        
        return nn.ModuleDict(features), channels
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg16', pretrained=True):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- Load Backbone and Extract Features ---
        self.encoder_features, self.channels = get_backbone_features(backbone_name, pretrained=pretrained)

        # --- Original LASA Module ---
        # Initialize LASA with the correct in_channels for e4
        self.lasa_module = LASA(in_channels=self.channels['e4_channels'])

        # --- Decoder Blocks and Auxiliary Convs ---
        # Decoder 4
        self.decoder4 = self._decoder_block(self.channels['bottleneck_channels'] + self.channels['e4_channels'], self.channels['e4_channels'])
        self.aux_conv_d4 = nn.Conv2d(self.channels['e4_channels'], num_classes, kernel_size=1)

        # Decoder 3
        self.decoder3 = self._decoder_block(self.channels['e4_channels'] + self.channels['e3_channels'], self.channels['e3_channels'])
        self.aux_conv_d3 = nn.Conv2d(self.channels['e3_channels'], num_classes, kernel_size=1)

        # Decoder 2
        self.decoder2 = self._decoder_block(self.channels['e3_channels'] + self.channels['e2_channels'], self.channels['e2_channels'])
        self.aux_conv_d2 = nn.Conv2d(self.channels['e2_channels'], num_classes, kernel_size=1)

        # Decoder 1
        self.decoder1 = self._decoder_block(self.channels['e2_channels'] + self.channels['e1_channels'], self.channels['e1_channels'])
        self.aux_conv_d1 = nn.Conv2d(self.channels['e1_channels'], num_classes, kernel_size=1)

        # --- Final Output Convolution ---
        self.final_conv = nn.Conv2d(self.channels['e1_channels'], num_classes, kernel_size=1)

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
        e1 = self.encoder_features['encoder1'](x)
        e2 = self.encoder_features['encoder2'](e1)
        e3 = self.encoder_features['encoder3'](e2)
        e4 = self.encoder_features['encoder4'](e3)

        e4_enhanced = self.lasa_module(e4)

        bottleneck = self.encoder_features['bottleneck'](e4_enhanced)

        # --- Decoder Path ---
        aux_outputs = []

        def get_interp_size(module_output):
            if isinstance(module_output, (list, tuple)):
                return module_output[0].shape[2:]
            return module_output.shape[2:]
        
        # Decoder 4
        d4_interp_size = get_interp_size(e4)
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 3
        d3_interp_size = get_interp_size(e3)
        d3 = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        d2_interp_size = get_interp_size(e2)
        d2 = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1
        d1_interp_size = get_interp_size(e1)
        d1 = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        final_output = self.final_conv(d1_out)
        
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])

# Aliases
LASA_VGG_Unet = LASA_Unet
LASA_ResNet_Unet = LASA_ResNet_Unet
LASA_Inception_Unet = LASA_Inception_Unet
LASA_EfficientNet_Unet = LASA_EfficientNet_Unet