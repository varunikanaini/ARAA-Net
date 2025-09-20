# /kaggle/working/ARAA-Net/lasa_vgg_model.py (Modified for True Deep Supervision and Backbone Flexibility)
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
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Encoder blocks and their output channels for U-Net style
            # These correspond to stages before pooling layers (or specific conv layers)
            self.encoder1 = vgg_features[:6]   # Output channels: 64 (after conv1_2_bn)
            self.encoder2 = vgg_features[6:13]  # Output channels: 128 (after conv2_2_bn)
            self.encoder3 = vgg_features[13:23] # Output channels: 256 (after conv3_3_bn)
            self.encoder4 = vgg_features[23:33] # Output channels: 512 (after conv4_3_bn)
            self.bottleneck_layer = vgg_features[33:43] # Output channels: 512 (after conv5_3_bn)

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512
            self.bottleneck_channels = 512

            # Define spatial scale factors for VGG encoder stages relative to input:
            # Input -> e1 (1/2) -> e2 (1/4) -> e3 (1/8) -> e4 (1/16) -> bottleneck (1/32)
            # This is important for correct upsampling to matching sizes.
            self.e_spatial_scales = {'e1': 2, 'e2': 4, 'e3': 8, 'e4': 16, 'bottleneck': 32}

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # ResNet encoder stages for U-Net style skip connections
            # encoder1: conv1 + bn1 + relu + maxpool (output 64 channels, spatial 1/4)
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
            self.encoder2 = resnet.layer1 # Output 256 channels, spatial 1/4
            self.encoder3 = resnet.layer2 # Output 512 channels, spatial 1/8
            self.encoder4 = resnet.layer3 # Output 1024 channels, spatial 1/16
            self.bottleneck_layer = resnet.layer4 # Output 2048 channels, spatial 1/32

            self.e1_channels = 64
            self.e2_channels = 256
            self.e3_channels = 512
            self.e4_channels = 1024
            self.bottleneck_channels = 2048
            
            # Define spatial scale factors for ResNet encoder stages relative to input:
            # Input -> e1 (1/4) -> e2 (1/4) -> e3 (1/8) -> e4 (1/16) -> bottleneck (1/32)
            self.e_spatial_scales = {'e1': 4, 'e2': 4, 'e3': 8, 'e4': 16, 'bottleneck': 32}

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module ---
        # LASA will enhance features from the 4th encoder block (e4)
        self.lasa_module = LASA(in_channels=self.e4_channels)

        # --- 3. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.e4_channels, self.e4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1) # Aux head for d4_out

        # Decoder 3: Input from upsampled d4_out + e3
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1) # Aux head for d3_out

        # Decoder 2: Input from upsampled d3_out + e2
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1) # Aux head for d2_out

        # Decoder 1: Input from upsampled d2_out + e1
        self.decoder1 = self._decoder_block(self.e2_channels + self.e1_channels, self.e1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1) # Aux head for d1_out (before final conv)
        
        # --- 4. Final Output Convolution ---
        self.final_conv = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1) # This is the main output

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
        e1 = self.encoder1(x) # (B, e1_C, H/scale_e1, W/scale_e1)
        e2 = self.encoder2(e1) # (B, e2_C, H/scale_e2, W/scale_e2)
        e3 = self.encoder3(e2) # (B, e3_C, H/scale_e3, W/scale_e3)
        e4 = self.encoder4(e3) # (B, e4_C, H/scale_e4, W/scale_e4)

        # Apply LASA enhancement to e4
        e4_enhanced = self.lasa_module(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced) # (B, B_C, H/scale_bottleneck, W/scale_bottleneck)

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4 (highest stride, lowest resolution decoder stage)
        # Upsample bottleneck to match e4_enhanced spatial size
        d4 = F.interpolate(bottleneck, size=e4_enhanced.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1) # Skip connection from LASA-enhanced e4
        d4_out = self.decoder4(d4) # Output of d4 block
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        # Upsample d4_out to match e3 spatial size
        d3 = F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1) # Skip connection from e3
        d3_out = self.decoder3(d3) # Output of d3 block
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        # Upsample d3_out to match e2 spatial size
        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection from e2
        d2_out = self.decoder2(d2) # Output of d2 block
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1 (Lowest stride, highest resolution decoder stage)
        # Upsample d2_out to match e1 spatial size
        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection from e1
        d1_out = self.decoder1(d1) # Output of d1 block
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) # Aux head for d1_out
        
        # Final output of the network
        final_output = self.final_conv(d1_out) # (B, num_classes, H/scale_e1, W/scale_e1)
        
        # Finally, upsample the final_output to the original input size.
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # The tuple of outputs will now contain 4 auxiliary outputs and 1 final main output.
        return tuple(aux_outputs + [final_output_upsampled])

# Alias for backward compatibility if the original model name is referenced elsewhere
# (But you should ideally update references to LASA_Unet)
LASA_VGG_Unet = LASA_Unet 