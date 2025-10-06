# lasa_vgg_model.py (Updated)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # <<< Imports the user's original LASA module

# --- Helper function to get features from different backbones ---
def get_backbone_features(backbone_name, pretrained=True):
    """
    Loads a backbone and returns a dictionary of feature extraction layers
    and their output channel counts, suitable for a U-Net style encoder.
    """
    if backbone_name == 'vgg16':
        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None).features
        # Extracting features at different stages
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
            # Layer 1 output: 256 channels, spatial 1/4
            'encoder1': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            # Layer 2 output: 512 channels, spatial 1/8
            'encoder2': resnet.layer1,
            # Layer 3 output: 1024 channels, spatial 1/16
            'encoder3': resnet.layer2,
            # Layer 4 output: 2048 channels, spatial 1/32
            'encoder4': resnet.layer3,
            'bottleneck': resnet.layer4 # Output 2048 channels, spatial 1/32
        }
        channels = {
            'e1_channels': 64, 'e2_channels': 256, 'e3_channels': 512,
            'e4_channels': 1024, 'bottleneck_channels': 2048
        }
        return nn.ModuleDict(features), channels

    elif backbone_name == 'inception_v3':
        # InceptionV3 has a more complex structure. We'll try to extract features from
        # mid-level and later blocks. Adjustments might be needed for optimal U-Net integration.
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
        
        # Example: Taking features before the final classifier and auxiliary classifier
        # These are just examples; the exact layers to use for U-Net can be tricky.
        # Block A3 (features after AuxLogits) might be a good starting point for encoder3
        # Block B4 (features before final fc) might be good for encoder4/bottleneck
        
        # IMPORTANT: InceptionV3 feature extraction for U-Net is non-trivial.
        # The following is a best-effort attempt. You might need to inspect
        # the model's structure more deeply and choose specific `nn.ModuleList` sections.
        
        # For simplicity and demonstration, let's try to use some sequential blocks.
        # This might require more fine-tuning or a different feature extraction approach.
        
        # Example: Using the main path structure before final avgpool
        # This part needs careful inspection of InceptionV3's output.
        # Let's define some simple sequential blocks as placeholders.
        # You'd typically need to find specific sequential parts that reduce spatial resolution.
        
        # A more robust approach would be to use hooks to capture intermediate outputs.
        # For now, we'll define placeholder modules.
        
        # Placeholder for encoder1 (e.g., first few conv layers) - needs detailed inspection
        # Likely output channels after first few layers: 32 or 64
        encoder1 = nn.Sequential(*list(inception.children())[:2]) # Crude example, might need adjustment
        
        # Placeholder for encoder2 (intermediate blocks) - needs detailed inspection
        # Reduction in spatial resolution occurs in inception blocks
        encoder2 = nn.Sequential(*list(inception.children())[2:3]) # Crude example
        
        # Placeholder for encoder3 (later inception blocks) - needs detailed inspection
        encoder3 = nn.Sequential(*list(inception.children())[3:4]) # Crude example

        # Placeholder for encoder4 and bottleneck (before final avgpool) - needs detailed inspection
        encoder4 = nn.Sequential(*list(inception.children())[4:5]) # Crude example

        bottleneck = nn.Sequential(*list(inception.children())[5:6]) # Crude example before final avgpool

        features = {
            'encoder1': encoder1,
            'encoder2': encoder2,
            'encoder3': encoder3,
            'encoder4': encoder4,
            'bottleneck': bottleneck
        }
        channels = {
            # These channel counts need verification by inspecting the output of each module
            # InceptionV3 channels can vary. This is an ESTIMATE.
            'e1_channels': 128, # Likely some value between 64-256
            'e2_channels': 256, # Likely some value between 256-512
            'e3_channels': 512, # Likely some value between 512-768
            'e4_channels': 768, # Likely some value between 768-1024
            'bottleneck_channels': 1024 # Likely some value between 1024-2048
        }
        # WARNING: InceptionV3's complex block structure makes direct U-Net feature extraction challenging.
        # The channel counts and module selections above are ESTIMATES and may need significant adjustment.
        # Consider using hooks to get intermediate outputs for better integration.
        return nn.ModuleDict(features), channels


    elif backbone_name.startswith('efficientnet'):
        # Load EfficientNet and extract features. We need to map to specific blocks.
        if backbone_name == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        elif backbone_name == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported EfficientNet version: {backbone_name}")
            
        # EfficientNet features are often accessed via internal blocks.
        # We'll use the standard blocks commonly used for segmentation backbones.
        # These blocks typically represent downsampling stages.
        
        # Structure for EfficientNet feature extraction:
        # Stage 0: Stem (input conv, bn, activation, pool)
        # Stage 1: Blocks 1 (MBConv1)
        # Stage 2: Blocks 2 (MBConv2)
        # Stage 3: Blocks 3 (MBConv3)
        # Stage 4: Blocks 4 (MBConv4)
        # Stage 5: Blocks 5 (MBConv5) - This is usually the final main feature block
        
        # We need to access these blocks. The exact naming might vary slightly with torchvision versions.
        # The following is a common pattern:
        
        features = {
            # Encoder 1: Stem output (after first pool)
            'encoder1': nn.Sequential(
                efficientnet._conv_stem, efficientnet._bn1, efficientnet._activation, efficientnet._initial_max_pool
            ),
            # Encoder 2: First block group (typically reducing spatial resolution and increasing channels)
            'encoder2': efficientnet._blocks[0:3], # Example range, adjust based on model structure
            # Encoder 3: Second block group
            'encoder3': efficientnet._blocks[3:7], # Example range, adjust based on model structure
            # Encoder 4: Third block group
            'encoder4': efficientnet._blocks[7:13], # Example range, adjust based on model structure
            # Bottleneck: Final block group
            'bottleneck': efficientnet._blocks[13:19] # Example range, adjust based on model structure
        }

        # CHANNEL COUNTS NEED VERIFICATION BY INSPECTING OUTPUTS
        # For EfficientNets, channel counts grow significantly.
        # These are GENERALIZED estimates and may need fine-tuning.
        channels = {
            'e1_channels': 32, # Output from stem after first pool
            'e2_channels': 48, # Output from first block group (highly variable)
            'e3_channels': 80, # Output from second block group
            'e4_channels': 128, # Output from third block group
            'bottleneck_channels': 256 # Output from final block group
        }
        # WARNING: The block slicing and channel counts for EfficientNet are ESTIMATES.
        # You MUST verify these by printing the output shapes of each module during a forward pass.
        return nn.ModuleDict(features), channels
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


class LASA_Unet(nn.Module): # Renamed for general backbone compatibility
    """
    A segmentation model using a specified backbone,
    the original LASA module for feature enhancement, and a U-Net style decoder with true deep supervision.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16', pretrained=True):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone and Get Features/Channels ---
        self.encoder_features, self.channels = get_backbone_features(backbone_name, pretrained=pretrained)

        # --- 2. Original LASA Module ---
        # LASA will enhance features from the 4th encoder block (e4)
        # Dynamically set in_channels for LASA based on backbone's e4 output channels
        self.lasa_module = LASA(in_channels=self.channels['e4_channels'])

        # --- 3. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Decoder 4: Input from bottleneck + LASA-enhanced e4
        self.decoder4 = self._decoder_block(self.channels['bottleneck_channels'] + self.channels['e4_channels'], self.channels['e4_channels'])
        self.aux_conv_d4 = nn.Conv2d(self.channels['e4_channels'], num_classes, kernel_size=1)

        # Decoder 3: Input from upsampled d4_out + e3
        self.decoder3 = self._decoder_block(self.channels['e4_channels'] + self.channels['e3_channels'], self.channels['e3_channels'])
        self.aux_conv_d3 = nn.Conv2d(self.channels['e3_channels'], num_classes, kernel_size=1)

        # Decoder 2: Input from upsampled d3_out + e2
        self.decoder2 = self._decoder_block(self.channels['e3_channels'] + self.channels['e2_channels'], self.channels['e2_channels'])
        self.aux_conv_d2 = nn.Conv2d(self.channels['e2_channels'], num_classes, kernel_size=1)

        # Decoder 1: Input from upsampled d2_out + e1
        self.decoder1 = self._decoder_block(self.channels['e2_channels'] + self.channels['e1_channels'], self.channels['e1_channels'])
        self.aux_conv_d1 = nn.Conv2d(self.channels['e1_channels'], num_classes, kernel_size=1)

        # --- 4. Final Output Convolution ---
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
        # Pass through each encoder stage
        e1 = self.encoder_features['encoder1'](x)
        e2 = self.encoder_features['encoder2'](e1)
        e3 = self.encoder_features['encoder3'](e2)
        e4 = self.encoder_features['encoder4'](e3)

        # Apply LASA enhancement to e4
        e4_enhanced = self.lasa_module(e4)

        # Bottleneck
        bottleneck = self.encoder_features['bottleneck'](e4_enhanced)

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = []

        # Helper function to get the correct interpolation size
        def get_interp_size(module_output):
            if isinstance(module_output, tuple): # Handle cases where a module returns multiple outputs
                return module_output[0].shape[2:] # Assuming the first element is the feature map
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

        # Final output of the network
        final_output = self.final_conv(d1_out)

        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)

        return tuple(aux_outputs + [final_output_upsampled])

# You might want to create aliases for specific backbones if needed
LASA_VGG_Unet = LASA_Unet
LASA_ResNet_Unet = LASA_Unet
LASA_Inception_Unet = LASA_Unet
LASA_EfficientNet_Unet = LASA_Unet