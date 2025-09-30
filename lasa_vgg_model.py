# /kaggle/working/ARAA-Net/lasa_vgg_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # <<< Imports the user's original LASA module

# --- ASPP Module Definition ---
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    Incorporates parallel atrous convolutions with different dilation rates
    to capture multi-scale contextual information.
    """
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates

        # 1x1 convolution for parallel branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), # BatchNorm here
            nn.ReLU(inplace=True)
        )

        # Parallel atrous convolutions
        self.conv_aspp = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels), # BatchNorm here
                    nn.ReLU(inplace=True)
                )
            )

        # Image Pooling (Global Average Pooling)
        # AdaptiveAvgPool2d handles any spatial size, including 1x1
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), # BatchNorm here
            nn.ReLU(inplace=True)
        )

        # Final convolution to fuse features
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), # BatchNorm here
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Get spatial dimensions before any pooling
        spatial_dims = x.shape[2:]

        # Branch 1: 1x1 convolution
        branch1 = self.conv1x1(x)

        # Branches 2 to N: Atrous convolutions
        branches_aspp = [conv(x) for conv in self.conv_aspp]

        # Branch N+1: Global Average Pooling
        branch_pool = self.global_avg_pool(x)
        # Upsample pooled features to match spatial dimensions AFTER other branches
        branch_pool = F.interpolate(branch_pool, size=spatial_dims, mode='bilinear', align_corners=True)

        # Concatenate all branches
        features = [branch1] + branches_aspp + [branch_pool]
        fused_features = torch.cat(features, dim=1)

        # Fuse features with the final convolution
        output = self.conv_fuse(fused_features)
        return output

# --- Modified LASA_Unet Class ---
class LASA_Unet(nn.Module):
    """
    A standalone segmentation model using a VGG16 or ResNet50 backbone,
    the original LASA module for feature enhancement, and a U-Net style decoder
    with true deep supervision and an integrated ASPP module at the bottleneck.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            self.encoder1 = vgg_features[:6]   # Output channels: 64
            self.encoder2 = vgg_features[6:13]  # Output channels: 128
            self.encoder3 = vgg_features[13:23] # Output channels: 256
            self.encoder4 = vgg_features[23:33] # Output channels: 512
            # This CNN part for VGG's bottleneck is no longer directly used in the same way
            # as before, as ASPP takes over the primary bottleneck processing.
            # We'll remove direct use to simplify. The encoder4 output is the critical one.
            # self.bottleneck_cnn_part = vgg_features[33:43] 

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512
            # self.bottleneck_cnn_channels = 512 # Not directly used anymore

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
            self.encoder2 = resnet.layer1 # Output 256 channels, spatial 1/4
            self.encoder3 = resnet.layer2 # Output 512 channels, spatial 1/8
            self.encoder4 = resnet.layer3 # Output 1024 channels, spatial 1/16
            # resnet.layer4 is the bottleneck CNN part, we'll use encoder4 output for LASA/ASPP.
            # self.bottleneck_cnn_part = resnet.layer4 

            self.e1_channels = 64
            self.e2_channels = 256
            self.e3_channels = 512
            self.e4_channels = 1024
            # self.bottleneck_cnn_channels = 2048

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module ---
        # LASA operates on the output of the last encoder block (e4)
        self.lasa_module = LASA(in_channels=self.e4_channels)

        # --- 3. ASPP Module Integration ---
        # ASPP operates on the LASA-enhanced features from encoder4.
        # Let ASPP output channels match e4_channels for consistent concatenation with e4_enhanced.
        aspp_output_channels = self.e4_channels 
        self.aspp = ASPP(in_channels=self.e4_channels, 
                         out_channels=aspp_output_channels, 
                         atrous_rates=[6, 12, 18, 24]) # Common rates for ASPP

        # The input to decoder4 will be the concatenation of the ASPP output and e4_enhanced.
        # So, input to decoder4 is aspp_output_channels + self.e4_channels
        # Which simplifies to 2 * self.e4_channels if aspp_output_channels == self.e4_channels
        
        # --- 4. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        self.decoder4 = self._decoder_block(2 * self.e4_channels, self.e4_channels) # Input channels = ASPP_out + e4_enhanced_out
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1)

        # Decoder 3: Input from upsampled d4_out + e3
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1)

        # Decoder 2: Input from upsampled d3_out + e2
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1)

        # Decoder 1: Input from upsampled d2_out + e1
        self.decoder1 = self._decoder_block(self.e2_channels + self.e1_channels, self.e1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.e1_channels, num_classes, kernel_size=1)
        
        # --- 5. Final Output Convolution ---
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
        input_h, input_w = x.shape[2:] 

        # --- Encoder Path ---
        e1 = self.encoder1(x) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 

        # Apply LASA enhancement to e4
        e4_enhanced = self.lasa_module(e4)
        
        # --- Bottleneck with ASPP ---
        # Pass LASA-enhanced features through ASPP
        aspp_output = self.aspp(e4_enhanced)

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4 (highest stride, lowest resolution decoder stage)
        d4_interp_size = e4_enhanced.shape[2:] # Spatial size of e4_enhanced
        
        # Upsample ASPP output to match the spatial size of e4_enhanced for concatenation
        aspp_upsampled = F.interpolate(aspp_output, size=d4_interp_size, mode='bilinear', align_corners=True)
        
        # Concatenate ASPP output and e4_enhanced (skip connection)
        d4 = torch.cat([aspp_upsampled, e4_enhanced], dim=1) 
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

        # Decoder 1 (Lowest stride, highest resolution decoder stage)
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