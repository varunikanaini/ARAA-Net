# lasa_unet_model.py (Renamed from lasa_vgg_model.py)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # Imports the user's original LASA module

# --- New imports for backbones ---
from torchvision.models import Inception3, EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B0_Weights, Inception_V3_Weights

class LASA_Unet(nn.Module):
    """
    A segmentation model using a flexible backbone (VGG16, ResNet50, InceptionV3, EfficientNet),
    the LASA module for feature enhancement with multi-scale attention, and a U-Net style decoder.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16', lasa_kernels=[1, 3, 5, 7]): # Added lasa_kernels
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels # Store kernel sizes

        self.encoder1_channels, self.encoder2_channels, self.encoder3_channels, self.encoder4_channels, self.bottleneck_channels = 0, 0, 0, 0, 0
        self.encoder_outs = {} # To store output of encoder blocks

        # --- 1. Load Pre-trained Backbone and Extract Features ---
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            # Extract features at specific points for U-Net style skip connections
            self.encoder1 = nn.Sequential(*list(vgg_features.children())[:6])   # Output channels: 64
            self.encoder2 = nn.Sequential(*list(vgg_features.children())[6:13])  # Output channels: 128
            self.encoder3 = nn.Sequential(*list(vgg_features.children())[13:23]) # Output channels: 256
            self.encoder4 = nn.Sequential(*list(vgg_features.children())[23:33]) # Output channels: 512
            self.bottleneck_layer = nn.Sequential(*list(vgg_features.children())[33:43]) # Output channels: 512

            self.encoder1_channels = 64
            self.encoder2_channels = 128
            self.encoder3_channels = 256
            self.encoder4_channels = 512
            self.bottleneck_channels = 512

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # Output: 64 channels, spatial 1/4
            self.encoder2 = resnet.layer1 # Output: 256 channels, spatial 1/4
            self.encoder3 = resnet.layer2 # Output: 512 channels, spatial 1/8
            self.encoder4 = resnet.layer3 # Output: 1024 channels, spatial 1/16
            self.bottleneck_layer = resnet.layer4 # Output: 2048 channels, spatial 1/32

            self.encoder1_channels = 64
            self.encoder2_channels = 256
            self.encoder3_channels = 512
            self.encoder4_channels = 1024
            self.bottleneck_channels = 2048

        # --- NEW: InceptionV3 ---
        elif backbone_name == 'inception_v3':
            inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
            # Extract features from InceptionV3's blocks
            self.encoder1 = nn.Sequential(*list(inception.children())[:3]) # Output: 64 channels (conv1, bn1, relu, maxpool)
            self.encoder2 = nn.Sequential(*list(inception.children())[3:4]) # Mixed_3 (e.g., output channels ~256)
            self.encoder3 = nn.Sequential(*list(inception.children())[4:5]) # Mixed_4 (e.g., output channels ~768)
            self.encoder4 = nn.Sequential(*list(inception.children())[5:7]) # Mixed_5, Inception_ResNet_Block (e.g., output ~1280)
            self.bottleneck_layer = nn.Sequential(*list(inception.children())[7:8]) # AdaptiveAvgPool2d, Dropout, FC - need to adapt this

            self.encoder1_channels = 64
            self.encoder2_channels = inception.Mixed_3.b1.conv1.out_channels + inception.Mixed_3.b2.conv1.out_channels # Example calculation
            self.encoder3_channels = inception.Mixed_4.b1.conv1.out_channels + inception.Mixed_4.b2.conv1.out_channels # Example
            self.encoder4_channels = inception.Mixed_5.b1.conv1.out_channels + inception.Mixed_5.b2.conv1.out_channels # Example
            self.bottleneck_channels = 2048 # InceptionV3's final layer before FC

            # Adjust encoder blocks for Inception structure if needed
            # This part needs careful inspection of InceptionV3's internal structure to correctly extract features at different spatial resolutions.
            # The current extraction is illustrative and might need refinement.
            # The last part (FC layer) will be replaced by our decoder/bottleneck logic.

        # --- NEW: EfficientNetB0 and B3 ---
        elif backbone_name in ['efficientnet_b0', 'efficientnet_b3']:
            if backbone_name == 'efficientnet_b0':
                weights = EfficientNet_B0_Weights.DEFAULT
                effnet = models.efficientnet_b0(weights=weights)
            elif backbone_name == 'efficientnet_b3':
                weights = EfficientNet_B3_Weights.DEFAULT
                effnet = models.efficientnet_b3(weights=weights)
            
            self.encoder1 = nn.Sequential(effnet.features[0], effnet.features[1]) # stem, block1
            self.encoder2 = nn.Sequential(effnet.features[2]) # block2
            self.encoder3 = nn.Sequential(*list(effnet.features.children())[3:5]) # block3, block4
            self.encoder4 = nn.Sequential(*list(effnet.features.children())[5:7]) # block5, block6
            self.bottleneck_layer = nn.Sequential(*list(effnet.features.children())[7:8]) # block7 + classifier (we'll use block7)

            # Channel counts for EfficientNet stages (approximate based on common structures)
            # These might need precise verification from torchvision's model definition
            self.encoder1_channels = 32 if backbone_name == 'efficientnet_b0' else 48 # Initial channel count after stem/block1
            self.encoder2_channels = 48 if backbone_name == 'efficientnet_b0' else 80 # Output of block2
            self.encoder3_channels = 128 if backbone_name == 'efficientnet_b0' else 160 # Output of block4
            self.encoder4_channels = 256 if backbone_name == 'efficientnet_b0' else 272 # Output of block6
            self.bottleneck_channels = 1280 if backbone_name == 'efficientnet_b0' else 1536 # Output of block7 (before projection head)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module with Multi-Scale Kernels ---
        # LASA will enhance features from the 4th encoder block (e4)
        # Pass kernel sizes to LASA module
        self.lasa_module = LASA(in_channels=self.encoder4_channels, L_list=self.lasa_kernels) 

        # --- 3. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        self.decoder4 = self._decoder_block(self.bottleneck_channels + self.encoder4_channels, self.encoder4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.encoder4_channels, num_classes, kernel_size=1)

        # Decoder 3: Input from upsampled d4_out + e3
        self.decoder3 = self._decoder_block(self.encoder4_channels + self.encoder3_channels, self.encoder3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.encoder3_channels, num_classes, kernel_size=1)

        # Decoder 2: Input from upsampled d3_out + e2
        self.decoder2 = self._decoder_block(self.encoder3_channels + self.encoder2_channels, self.encoder2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.encoder2_channels, num_classes, kernel_size=1)

        # Decoder 1: Input from upsampled d2_out + e1
        self.decoder1 = self._decoder_block(self.encoder2_channels + self.encoder1_channels, self.encoder1_channels)
        self.aux_conv_d1 = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1) 
        
        # --- 4. Final Output Convolution ---
        self.final_conv = nn.Conv2d(self.encoder1_channels, num_classes, kernel_size=1) 

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
        # Pass through encoder blocks, storing outputs for skip connections
        # NOTE: Need to ensure that the output channels match the expected channels for the decoder.
        # If not, an intermediate 1x1 conv might be needed to adjust channels.

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Apply LASA enhancement to e4
        e4_enhanced = self.lasa_module(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # --- Decoder Path with Skip Connections and Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4
        d4_interp_size = e4_enhanced.shape[2:] 
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1) 
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

# Helper to create aliases for convenience
LASA_VGG_Unet = LASA_Unet