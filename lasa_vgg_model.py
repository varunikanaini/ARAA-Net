# /kaggle/working/ARAA-Net/lasa_vgg_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # We will reuse your existing lasa.py file

class LASA_Unet(nn.Module):
    """
    A standalone segmentation model using a VGG16 or ResNet50 backbone,
    a LASA module for feature enhancement, and a U-Net style decoder with deep supervision.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            # VGG16 with Batch Normalization
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Encoder blocks and their output channels
            self.encoder1 = vgg_features[:6]   # Output channels: 64 (conv1_2)
            self.encoder2 = vgg_features[6:13]  # Output channels: 128 (conv2_2)
            self.encoder3 = vgg_features[13:23] # Output channels: 256 (conv3_3)
            self.encoder4 = vgg_features[23:33] # Output channels: 512 (conv4_3)
            self.bottleneck_layer = vgg_features[33:43] # Output channels: 512 (conv5_3)

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512
            self.bottleneck_channels = 512

        elif backbone_name == 'resnet50':
            # ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # Encoder blocks for U-Net style skip connections
            # encoder1: conv1 + maxpool (output 64 channels, 1/4 spatial resolution)
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
            self.encoder2 = resnet.layer1 # Output 256 channels (1/4 spatial resolution)
            self.encoder3 = resnet.layer2 # Output 512 channels (1/8 spatial resolution)
            self.encoder4 = resnet.layer3 # Output 1024 channels (1/16 spatial resolution)
            self.bottleneck_layer = resnet.layer4 # Output 2048 channels (1/32 spatial resolution)

            self.e1_channels = 64
            self.e2_channels = 256
            self.e3_channels = 512
            self.e4_channels = 1024
            self.bottleneck_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Insert the LASA Module ---
        # LASA will enhance the features from the 4th encoder block (e4)
        self.lasa = LASA(in_channels=self.e4_channels)

        # --- 3. Define Decoder Blocks ---
        # These blocks will upsample the features and merge them with skip connections.
        # Auxiliary segmentation heads are also added for deep supervision.
        
        # Decoder 4: upsample bottleneck + LASA-enhanced e4
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.e4_channels, self.e4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1)

        # Decoder 3: upsample d4_out + e3
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1)

        # Decoder 2: upsample d3_out + e2
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1)

        # Decoder 1: upsample d2_out + e1
        self.decoder1 = self._decoder_block(self.e2_channels + self.e1_channels, self.e1_channels)
        
        # --- 4. Final Output Convolution ---
        self.final_conv = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1)

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
        input_size = x.shape[2:] # Store original input dimensions for upsampling

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3) # Features for LASA module

        # Apply LASA enhancement
        e4_lasa = self.lasa(e4) # LASA output has same dims as e4

        bottleneck = self.bottleneck_layer(e4_lasa)

        # --- Decoder Path with Skip Connections and Deep Supervision ---
        aux_outputs = [] # To collect intermediate predictions

        # Decoder 4
        # Upsample bottleneck and concatenate with e4_lasa
        d4 = F.interpolate(bottleneck, size=e4_lasa.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_lasa], dim=1) # Skip connection from LASA-enhanced e4
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=input_size, mode='bilinear', align_corners=True))
        
        # Decoder 3
        # Upsample d4_out and concatenate with e3
        d3 = F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=input_size, mode='bilinear', align_corners=True))

        # Decoder 2
        # Upsample d3_out and concatenate with e2
        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=input_size, mode='bilinear', align_corners=True))

        # Decoder 1 (Final segmentation head)
        # Upsample d2_out and concatenate with e1
        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        final_output = self.final_conv(d1_out)
        
        # Append final output to aux_outputs and return as a tuple
        return tuple(aux_outputs + [final_output])

# Alias for backward compatibility if the original model name is referenced elsewhere
LASA_VGG_Unet = LASA_Unet 