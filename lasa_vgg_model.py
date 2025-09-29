# /kaggle/working/ARAA-Net/lasa_vgg_model.py (Updated with Multiple LASA Modules)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # <<< Imports the user's original LASA module

class LASA_Unet(nn.Module):
    """
    A standalone segmentation model using a VGG16 or ResNet50 backbone,
    the original LASA module for feature enhancement, and a U-Net style decoder with true deep supervision.
    Incorporates multiple LASA modules at different encoder stages (e3 and e4)
    as suggested by the paper's ablation studies for increased accuracy.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Encoder blocks and their output channels for U-Net style
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

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # ResNet encoder stages for U-Net style skip connections
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
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Multiple LASA Modules for Feature Enhancement ---
        # Now, LASA will enhance features from the 3rd (e3) and 4th (e4) encoder blocks.
        # This aligns with the ablation studies in the paper suggesting benefits from earlier blocks.
        self.lasa_module_e3 = LASA(in_channels=self.e3_channels)
        self.lasa_module_e4 = LASA(in_channels=self.e4_channels)


        # --- 3. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.e4_channels, self.e4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1) # Aux head for d4_out

        # Decoder 3: Input from upsampled d4_out + LASA-enhanced e3
        # e3_enhanced will be used here.
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1) # Aux head for d3_out

        # Decoder 2: Input from upsampled d3_out + e2 (e2 remains unenhanced by LASA)
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1) # Aux head for d2_out

        # Decoder 1: Input from upsampled d2_out + e1 (e1 remains unenhanced by LASA)
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
        e1 = self.encoder1(x) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 

        # Apply LASA enhancement to e3 and e4
        e3_enhanced = self.lasa_module_e3(e3)
        e4_enhanced = self.lasa_module_e4(e4)
        
        # Bottleneck uses the enhanced e4 features
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4 (highest stride, lowest resolution decoder stage)
        # Uses the enhanced e4 features for concatenation
        d4_interp_size = e4_enhanced.shape[2:] 
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1) 
        d4_out = self.decoder4(d4) 
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3
        # Uses the enhanced e3 features for concatenation
        d3_interp_size = e3_enhanced.shape[2:] 
        d3 = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3_enhanced], dim=1) 
        d3_out = self.decoder3(d3) 
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        # Uses original e2 features
        d2_interp_size = e2.shape[2:]
        d2 = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1) 
        d2_out = self.decoder2(d2) 
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1 (Lowest stride, highest resolution decoder stage)
        # Uses original e1 features
        d1_interp_size = e1.shape[2:]
        d1 = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1) 
        d1_out = self.decoder1(d1) 
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) 
        
        # Final output of the network
        final_output = self.final_conv(d1_out) 
        
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])

LASA_VGG_Unet = LASA_Unet