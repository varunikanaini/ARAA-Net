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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Parallel atrous convolutions
        self.conv_aspp = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Image Pooling (Global Average Pooling)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final convolution to fuse features
        # The output channels from all branches will be out_channels
        # Total input channels to the final conv will be out_channels * (1 + len(atrous_rates) + 1)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Branch 1: 1x1 convolution
        branch1 = self.conv1x1(x)

        # Branches 2 to N: Atrous convolutions
        branches_aspp = [conv(x) for conv in self.conv_aspp]

        # Branch N+1: Global Average Pooling
        branch_pool = self.global_avg_pool(x)
        # Upsample pooled features to match spatial dimensions
        branch_pool = F.interpolate(branch_pool, size=x.shape[2:], mode='bilinear', align_corners=True)

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
            self.bottleneck_cnn_part = vgg_features[33:43] # CNN part of bottleneck for VGG

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512
            self.bottleneck_cnn_channels = 512 # Channels after the CNN part of bottleneck

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
            self.encoder2 = resnet.layer1 # Output 256 channels, spatial 1/4
            self.encoder3 = resnet.layer2 # Output 512 channels, spatial 1/8
            self.encoder4 = resnet.layer3 # Output 1024 channels, spatial 1/16
            self.bottleneck_cnn_part = resnet.layer4 # Output 2048 channels, spatial 1/32

            self.e1_channels = 64
            self.e2_channels = 256
            self.e3_channels = 512
            self.e4_channels = 1024
            self.bottleneck_cnn_channels = 2048
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module ---
        # LASA will enhance features from the 4th encoder block (e4)
        # It should operate on the number of channels output by encoder4
        self.lasa_module = LASA(in_channels=self.e4_channels)

        # --- 3. ASPP Module Integration ---
        # ASPP will operate on the LASA-enhanced features from encoder4
        # We set the output channels of ASPP to match the input channels of the decoder blocks
        # Let's assume decoder blocks will have channels related to the encoder4_channels
        aspp_output_channels = self.e4_channels # or a different value if desired
        self.aspp = ASPP(in_channels=self.e4_channels, 
                         out_channels=aspp_output_channels, 
                         atrous_rates=[6, 12, 18, 24]) # Common rates for ASPP

        # The bottleneck now will be a combination of the CNN part and ASPP output
        # For VGG, this will be out_channels from ASPP + out_channels from bottleneck_cnn_part
        # For ResNet, this will be out_channels from ASPP + out_channels from bottleneck_cnn_part
        self.bottleneck_channels = aspp_output_channels + self.bottleneck_cnn_channels 

        # --- 4. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Adjust input channels for decoder blocks based on concatenated features
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        # Bottleneck output is now from ASPP, which is then concatenated with the CNN bottleneck part (for VGG)
        # OR directly fed into the decoder if we decide not to concatenate CNN bottleneck part for ResNet
        # Let's refine the bottleneck processing for clarity.

        # Re-evaluating bottleneck processing:
        # Option A: ASPP takes LASA_enhanced_e4, its output is then concatenated with the CNN bottleneck part.
        # Option B: ASPP takes LASA_enhanced_e4, and its output directly feeds into decoder. CNN bottleneck part is skipped or concatenated earlier.
        # Let's stick to Option A for now, for VGG compatibility where bottleneck_cnn_part is defined.
        # For ResNet, layer4 is the bottleneck CNN part.

        # The input to decoder4 should be the concatenation of the ASPP output and the CNN bottleneck output.
        # We need to ensure consistent channel counts.
        # Let's make ASPP output channels = self.e4_channels (same as encoder4).
        # Then the combined bottleneck will be self.e4_channels (from ASPP) + self.bottleneck_cnn_channels (from CNN bottleneck part).
        
        # Adjusting decoder input channels:
        # Decoder 4 now takes input from the combined bottleneck.
        # The original logic was bottleneck_layer (self.bottleneck_layer) + e4_enhanced.
        # Let's make the bottleneck output the combined features.
        # The input to decoder4 will be the *output of ASPP* + *e4_enhanced*.
        # The previous `bottleneck_layer` was VGG's conv5_3_bn. We'll replace it.

        # Let's redefine the bottleneck to be the ASPP output followed by a 1x1 conv for channel adjustment,
        # and then combine it with the CNN bottleneck part if needed.
        # For simplicity and cleaner integration, let's have ASPP output directly feed into decoder path.
        # We'll use ASPP output + e4_enhanced as input to decoder4.
        # The CNN bottleneck part (vgg_features[33:43] or resnet.layer4) will be treated as part of the encoder.

        # Redefining bottleneck processing:
        # Encoder4 output -> LASA module -> ASPP module -> Decoder 4 input

        # Revised bottleneck_channels: Output channels from ASPP module
        self.bottleneck_channels = aspp_output_channels 

        # Decoder 4: Input from ASPP output + LASA-enhanced e4
        # ASPP output channels should be consistent, let's set it to self.e4_channels
        # Then the concatenation will be self.e4_channels (ASPP) + self.e4_channels (e4_enhanced)
        # So, input to decoder4 is 2 * self.e4_channels
        self.decoder4 = self._decoder_block(2 * self.e4_channels, self.e4_channels) 
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
        # Ensure ASPP output channels match the expected input for concatenation
        aspp_output = self.aspp(e4_enhanced)

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4 (highest stride, lowest resolution decoder stage)
        # Concatenate ASPP output with LASA-enhanced e4 (skip connection)
        d4_interp_size = e4_enhanced.shape[2:] # Spatial size of e4_enhanced
        
        # Upsample ASPP output to match the spatial size of e4_enhanced for concatenation
        aspp_upsampled = F.interpolate(aspp_output, size=d4_interp_size, mode='bilinear', align_corners=True)
        
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