# lasa_unet_model.py (with dynamic channel determination and updated backbone extraction)
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
            # Extract features from InceptionV3's blocks.
            # These are based on common U-Net like feature extraction points for Inception.
            # You may need to adjust these based on the exact spatial resolutions and channel counts from your torchvision version.
            self.encoder1 = nn.Sequential(*list(inception.children())[:3]) # After maxpool (spatial 1/4), 64 channels
            self.encoder2 = nn.Sequential(*list(inception.children())[3:4]) # Mixed_3 (spatial 1/8), ~256 channels
            self.encoder3 = nn.Sequential(*list(inception.children())[4:5]) # Mixed_4 (spatial 1/16), ~768 channels
            self.encoder4 = nn.Sequential(*list(inception.children())[5:7]) # Mixed_5, Inception_ResNet_Block (spatial 1/32), ~1280 channels
            self.bottleneck_layer = nn.Sequential(*list(inception.children())[7:8]) # AdaptiveAvgPool2d + FC (FC is replaced by bottleneck logic)

            self.encoder1_channels = 64
            self.encoder2_channels = 256 # This is a simplified estimation for Mixed_3's output channels
            self.encoder3_channels = 768 # Simplified for Mixed_4
            self.encoder4_channels = 1280 # Simplified for Mixed_5
            self.bottleneck_channels = 2048 # InceptionV3's final state before classification, though we might need to adapt this if using its raw feature output.

            # To get exact channels, you'd need to pass a dummy tensor:
            dummy_input_for_channels = torch.randn(1, 3, 299, 299) # typical input size for InceptionV3
            e1_channels = self.encoder1(dummy_input_for_channels).shape[1]
            e2_channels = self.encoder2(self.encoder1(dummy_input_for_channels)).shape[1]
            e3_channels = self.encoder3(self.encoder2(self.encoder1(dummy_input_for_channels))).shape[1]
            e4_channels = self.encoder4(self.encoder3(self.encoder2(self.encoder1(dummy_input_for_channels)))).shape[1]
            bottleneck_channels = self.bottleneck_layer(self.encoder4(self.encoder3(self.encoder2(self.encoder1(dummy_input_for_channels))))).shape[1]
            
            self.encoder1_channels = e1_channels
            self.encoder2_channels = e2_channels
            self.encoder3_channels = e3_channels
            self.encoder4_channels = e4_channels
            self.bottleneck_channels = bottleneck_channels


        # --- NEW: EfficientNetB0 and B3 ---
        elif backbone_name in ['efficientnet_b0', 'efficientnet_b3']:
            if backbone_name == 'efficientnet_b0':
                weights = EfficientNet_B0_Weights.DEFAULT
                effnet = models.efficientnet_b0(weights=weights)
            elif backbone_name == 'efficientnet_b3':
                weights = EfficientNet_B3_Weights.DEFAULT
                effnet = models.efficientnet_b3(weights=weights)
            
            # Extracting features at common stages for U-Net style skip connections
            # These are based on common segmentation implementations with EfficientNets.
            # The `features` attribute is a Sequential module that needs careful slicing.
            self.encoder1 = nn.Sequential(effnet.features[0], effnet.features[1]) # Stem + Block1
            self.encoder2 = nn.Sequential(effnet.features[2]) # Block2
            self.encoder3 = nn.Sequential(*list(effnet.features.children())[3:5]) # Block3, Block4
            self.encoder4 = nn.Sequential(*list(effnet.features.children())[5:7]) # Block5, Block6
            self.bottleneck_layer = nn.Sequential(*list(effnet.features.children())[7:8]) # Block7 + Classifier (we use block7 part)

            # Dynamically determine channel counts by passing a dummy tensor
            dummy_input_for_channels = torch.randn(1, 3, 224, 224) # Typical input size for EfficientNets (can adjust if needed)
            
            e1_out = self.encoder1(dummy_input_for_channels)
            e2_out = self.encoder2(e1_out)
            e3_out = self.encoder3(e2_out)
            e4_out = self.encoder4(e3_out)
            bottleneck_out = self.bottleneck_layer(e4_out)
            
            self.encoder1_channels = e1_out.shape[1]
            self.encoder2_channels = e2_out.shape[1]
            self.encoder3_channels = e3_out.shape[1]
            self.encoder4_channels = e4_out.shape[1]
            self.bottleneck_channels = bottleneck_out.shape[1]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module with Multi-Scale Kernels ---
        # LASA will enhance features from the 4th encoder block (e4)
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