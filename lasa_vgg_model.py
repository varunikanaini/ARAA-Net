# /kaggle/working/ARAA-Net/lasa_vgg_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # <<< Imports the user's original LASA module

class LASA_Unet(nn.Module): # Renamed for general backbone compatibility (can be LASA_VGG_Unet if preferred)
    """
    A standalone segmentation model using a VGG16 or ResNet50 backbone,
    the original LASA module for feature enhancement, and a U-Net style decoder with true deep supervision.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            # Use VGG16_BN weights
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Encoder blocks and their output channels for U-Net style skip connections
            # VGG16 structure: 
            # block1: conv1_1, relu, conv1_2, bn1_2, relu, pool1 (output channels 64)
            # block2: conv2_1, relu, conv2_2, bn2_2, relu, pool2 (output channels 128)
            # block3: conv3_1, relu, conv3_2, bn3_2, relu, conv3_3, bn3_3, relu, pool3 (output channels 256)
            # block4: conv4_1, relu, conv4_2, bn4_2, relu, conv4_3, bn4_3, relu, pool4 (output channels 512)
            # block5: conv5_1, relu, conv5_2, bn5_2, relu, conv5_3, bn5_3, relu (output channels 512)
            
            self.encoder1 = vgg_features[:6]   # Up to bn1_2. Output spatial size is 1/2, channels 64.
            self.encoder2 = vgg_features[6:13]  # Up to bn2_2. Output spatial size is 1/4, channels 128.
            self.encoder3 = vgg_features[13:23] # Up to bn3_3. Output spatial size is 1/8, channels 256.
            self.encoder4 = vgg_features[23:33] # Up to bn4_3. Output spatial size is 1/16, channels 512.
            # Bottleneck uses the last convolutional block of VGG
            self.bottleneck_layer = vgg_features[33:43] # Up to bn5_3. Output spatial size is 1/32, channels 512.

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512
            self.bottleneck_channels = 512

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # ResNet encoder stages for U-Net style skip connections
            # ResNet output channel configurations:
            # layer1: 256 channels, spatial 1/4
            # layer2: 512 channels, spatial 1/8
            # layer3: 1024 channels, spatial 1/16
            # layer4: 2048 channels, spatial 1/32
            
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # Output 64 channels, spatial 1/4
            self.encoder2 = resnet.layer1 # Output 256 channels, spatial 1/4 (This is incorrect, layer1 is 1/4 spatial)
            self.encoder3 = resnet.layer2 # Output 512 channels, spatial 1/8
            self.encoder4 = resnet.layer3 # Output 1024 channels, spatial 1/16
            self.bottleneck_layer = resnet.layer4 # Output 2048 channels, spatial 1/32

            self.e1_channels = 64 # conv1 output channels
            self.e2_channels = 256 # layer1 output channels
            self.e3_channels = 512 # layer2 output channels
            self.e4_channels = 1024 # layer3 output channels
            self.bottleneck_channels = 2048 # layer4 output channels
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module ---
        # LASA will enhance features from the 4th encoder block (e4)
        # For ResNet, e4_channels = 1024. For VGG, e4_channels = 512.
        self.lasa_module = LASA(in_channels=self.e4_channels)

        # --- 3. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # The decoder block takes input from upsampled previous decoder stage + skip connection.
        # The number of output channels for the decoder block is usually reduced.
        
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        # Bottleneck channels = 2048 (ResNet) or 512 (VGG)
        # e4 channels = 1024 (ResNet) or 512 (VGG)
        # The LASA module operates on e4, so its output has the same channels as e4.
        # Thus, input to decoder4 is bottleneck_channels + e4_channels.
        # Output channels are set to e4_channels to match skip connection.
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.e4_channels, self.e4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1) # Aux head for d4_out

        # Decoder 3: Input from upsampled d4_out + e3
        # Input channels: e4_channels (from d4_out) + e3_channels. Output channels e3_channels.
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1) # Aux head for d3_out

        # Decoder 2: Input from upsampled d3_out + e2
        # Input channels: e3_channels (from d3_out) + e2_channels. Output channels e2_channels.
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1) # Aux head for d2_out

        # Decoder 1: Input from upsampled d2_out + e1
        # Input channels: e2_channels (from d2_out) + e1_channels. Output channels e1_channels.
        self.decoder1 = self._decoder_block(self.e2_channels + self.e1_channels, self.e1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1) # Aux head for d1_out (before final conv)
        
        # --- 4. Final Output Convolution ---
        # Takes output from decoder 1 and maps to num_classes.
        self.final_conv = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1) # This is the main output

    def _decoder_block(self, in_channels, out_channels):
        """ Defines a standard U-Net decoder block (Conv-BN-ReLU-Conv-BN-ReLU). """
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
        # Extract features from encoder stages
        if self.backbone_name == 'vgg16':
            e1 = self.encoder1(x) 
            e2 = self.encoder2(e1) 
            e3 = self.encoder3(e2) 
            e4 = self.encoder4(e3) 
            bottleneck = self.bottleneck_layer(e4) 
        elif self.backbone_name == 'resnet50':
            # ResNet encoder path without the final fc layer and avgpool
            e1 = self.encoder1(x) # Output channels 64, spatial 1/4
            e2 = self.encoder2(e1) # Output channels 256, spatial 1/4
            e3 = self.encoder3(e2) # Output channels 512, spatial 1/8
            e4 = self.encoder4(e3) # Output channels 1024, spatial 1/16
            bottleneck = self.bottleneck_layer(e4) # Output channels 2048, spatial 1/32

        # Apply LASA enhancement to the 4th encoder stage output (e4)
        e4_enhanced = self.lasa_module(e4)
        
        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] # List to store outputs from auxiliary heads

        # Decoder 4: Takes bottleneck features and LASA-enhanced e4, outputs features of e4_channels size
        # Interpolate bottleneck features to match spatial size of e4_enhanced
        d4_interp_size = e4_enhanced.shape[2:] 
        d4_upsampled_bottleneck = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        # Concatenate with skip connection (LASA-enhanced e4)
        d4_input = torch.cat([d4_upsampled_bottleneck, e4_enhanced], dim=1) 
        d4_out = self.decoder4(d4_input) 
        # Generate auxiliary prediction for this decoder stage and upsample to original image size
        aux_pred_d4 = self.aux_conv_d4(d4_out)
        aux_outputs.append(F.interpolate(aux_pred_d4, size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3: Takes upsampled d4_out + e3 skip connection
        d3_interp_size = e3.shape[2:]
        d3_upsampled_prev = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3_input = torch.cat([d3_upsampled_prev, e3], dim=1) 
        d3_out = self.decoder3(d3_input) 
        aux_pred_d3 = self.aux_conv_d3(d3_out)
        aux_outputs.append(F.interpolate(aux_pred_d3, size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2: Takes upsampled d3_out + e2 skip connection
        d2_interp_size = e2.shape[2:]
        d2_upsampled_prev = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2_input = torch.cat([d2_upsampled_prev, e2], dim=1) 
        d2_out = self.decoder2(d2_input) 
        aux_pred_d2 = self.aux_conv_d2(d2_out)
        aux_outputs.append(F.interpolate(aux_pred_d2, size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1: Takes upsampled d2_out + e1 skip connection
        d1_interp_size = e1.shape[2:]
        d1_upsampled_prev = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1_input = torch.cat([d1_upsampled_prev, e1], dim=1) 
        d1_out = self.decoder1(d1_input) 
        aux_pred_d1 = self.aux_conv_d1(d1_out)
        aux_outputs.append(F.interpolate(aux_pred_d1, size=(input_h, input_w), mode='bilinear', align_corners=True)) 
        
        # Final output of the network: applies a 1x1 convolution to d1_out
        final_output = self.final_conv(d1_out) 
        
        # Upsample final output to original image size
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # Return all auxiliary predictions plus the final prediction
        return tuple(aux_outputs + [final_output_upsampled])

# Alias for convenience
LASA_VGG_Unet = LASA_Unet 
