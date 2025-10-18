# lasa_unet_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # Imports the user's original LASA module
from cbam import CBAM
from attention_gate import AttentionGate
# --- New imports for backbones ---
# Import specific weights for newer torchvision versions if needed,
# but generally models.MODEL_NAME_Weights.DEFAULT works.
# from torchvision.models import Inception3, EfficientNet_B0_Weights, EfficientNet_B3_Weights
# etc.

# --- Import BACKBONE_CHANNELS from config ---
# Assuming config.py is accessible and contains BACKBONE_CHANNELS
# You might need to adjust the import path if config.py is in a different directory
try:
    import config
    BACKBONE_CHANNELS_INFO = config.BACKBONE_CHANNELS
except ImportError:
    print("Error: Could not import BACKBONE_CHANNELS from config.py. Ensure config.py is in the Python path.")
    # Define a fallback or raise an error
    BACKBONE_CHANNELS_INFO = {
        'vgg16': {'e1': 64, 'e2': 128, 'e3': 256, 'e4': 512, 'bottleneck': 512},
        'resnet50': {'e1': 64, 'e2': 256, 'e3': 512, 'e4': 1024, 'bottleneck': 2048},
        # Add other backbones here if config is not available
    }


class LASA_Unet(nn.Module):
    """
    A segmentation model using a flexible backbone (VGG16, ResNet50, InceptionV3, EfficientNet),
    the LASA module for feature enhancement with multi-scale attention, and a U-Net style decoder.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16', lasa_kernels=[1, 3, 5, 7]):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels

        # --- 1. Load Pre-trained Backbone and Extract Features ---
        self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.bottleneck_layer = self._get_backbone_features(backbone_name)
        
        # Get channel info from config or estimate if not found
        channel_info = BACKBONE_CHANNELS_INFO.get(backbone_name)
        if channel_info is None:
            print(f"Warning: Channel information for backbone '{backbone_name}' not found in config. Attempting dynamic estimation (may be slow or inaccurate).")
            channel_info = self._estimate_backbone_channels(backbone_name)
            if channel_info is None:
                 raise ValueError(f"Could not determine channel info for backbone '{backbone_name}'.")
        
        self.encoder1_channels = channel_info['e1']
        self.encoder2_channels = channel_info['e2']
        self.encoder3_channels = channel_info['e3']
        self.encoder4_channels = channel_info['e4']
        self.bottleneck_channels = channel_info['bottleneck']

        # --- 2. Original LASA Module ---
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
        self.Att4 = AttentionGate(F_g=self.bottleneck_channels, F_l=self.encoder4_channels, F_int=self.encoder4_channels // 2)
        self.Att3 = AttentionGate(F_g=self.encoder4_channels, F_l=self.encoder3_channels, F_int=self.encoder3_channels // 2)
        self.Att2 = AttentionGate(F_g=self.encoder3_channels, F_l=self.encoder2_channels, F_int=self.encoder2_channels // 2)
        self.Att1 = AttentionGate(F_g=self.encoder2_channels, F_l=self.encoder1_channels, F_int=self.encoder1_channels // 2)

    def _get_backbone_features(self, backbone_name):
        """ Helper to extract encoder features and bottleneck from various backbones. """
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            encoder1 = nn.Sequential(*list(vgg_features.children())[:6])
            encoder2 = nn.Sequential(*list(vgg_features.children())[6:13])
            encoder3 = nn.Sequential(*list(vgg_features.children())[13:23])
            encoder4 = nn.Sequential(*list(vgg_features.children())[23:33])
            bottleneck_layer = nn.Sequential(*list(vgg_features.children())[33:43])
            return encoder1, encoder2, encoder3, encoder4, bottleneck_layer
        
        elif backbone_name == 'vgg19': # Add VGG19 extraction
            vgg_features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
            encoder1 = nn.Sequential(*list(vgg_features.children())[:6])
            encoder2 = nn.Sequential(*list(vgg_features.children())[6:13])
            encoder3 = nn.Sequential(*list(vgg_features.children())[13:23])
            encoder4 = nn.Sequential(*list(vgg_features.children())[23:33])
            bottleneck_layer = nn.Sequential(*list(vgg_features.children())[33:43]) # Adjust if VGG19 has different final blocks
            return encoder1, encoder2, encoder3, encoder4, bottleneck_layer

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            encoder2 = resnet.layer1
            encoder3 = resnet.layer2
            encoder4 = resnet.layer3
            bottleneck_layer = resnet.layer4
            return encoder1, encoder2, encoder3, encoder4, bottleneck_layer

        elif backbone_name == 'inception_v3':
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=False) # aux_logits=False for cleaner feature extraction
            encoder1 = nn.Sequential(*list(inception.children())[:3]) # Up to conv3/3x3 + BN + ReLU
            encoder2 = nn.Sequential(*list(inception.children())[3:4]) # Mixed_3a
            encoder3 = nn.Sequential(*list(inception.children())[4:5]) # Mixed_4d
            encoder4 = nn.Sequential(*list(inception.children())[5:7]) # Mixed_5c
            bottleneck_layer = nn.Sequential(*list(inception.children())[7:]) # From Mixed_6 onwards, adapt to remove final classifiers
            return encoder1, encoder2, encoder3, encoder4, bottleneck_layer # Placeholder

        elif backbone_name in ['efficientnet_b0', 'efficientnet_b3']:
            if backbone_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT
                effnet = models.efficientnet_b0(weights=weights)
            elif backbone_name == 'efficientnet_b3':
                weights = models.EfficientNet_B3_Weights.DEFAULT
                effnet = models.efficientnet_b3(weights=weights)
            
            # Extracting features at common stages for U-Net style skip connections
            # The `features` attribute is a Sequential module that needs careful slicing.
            encoder1 = nn.Sequential(effnet.features[0], effnet.features[1]) # Stem + Block1
            encoder2 = nn.Sequential(effnet.features[2]) # Block2
            encoder3 = nn.Sequential(*list(effnet.features.children())[3:5]) # Block3, Block4
            encoder4 = nn.Sequential(*list(effnet.features.children())[5:7]) # Block5, Block6
            bottleneck_layer = nn.Sequential(*list(effnet.features.children())[7:]) # Block7 + Classifier (we use block7 part)
            return encoder1, encoder2, encoder3, encoder4, bottleneck_layer
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def _estimate_backbone_channels(self, backbone_name):
        """ Dynamically estimate channel counts by passing a dummy tensor. """
        print(f"Dynamically estimating channels for backbone: {backbone_name}")
        try:
            if backbone_name == 'vgg16':
                dummy_input_for_channels = torch.randn(1, 3, 224, 224)
                e1 = self.encoder1(dummy_input_for_channels).shape[1]
                e2 = self.encoder2(e1).shape[1]
                e3 = self.encoder3(e2).shape[1]
                e4 = self.encoder4(e3).shape[1]
                bottleneck = self.bottleneck_layer(e4).shape[1]
                return {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'bottleneck': bottleneck}

            elif backbone_name == 'resnet50':
                dummy_input_for_channels = torch.randn(1, 3, 224, 224)
                e1 = self.encoder1(dummy_input_for_channels).shape[1]
                e2 = self.encoder2(e1).shape[1]
                e3 = self.encoder3(e2).shape[1]
                e4 = self.encoder4(e3).shape[1]
                bottleneck = self.bottleneck_layer(e4).shape[1]
                return {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'bottleneck': bottleneck}

            elif backbone_name == 'inception_v3':
                dummy_input_for_channels = torch.randn(1, 3, 299, 299) # Typical input size for InceptionV3
                e1_out = self.encoder1(dummy_input_for_channels)
                e2_out = self.encoder2(e1_out)
                e3_out = self.encoder3(e2_out)
                e4_out = self.encoder4(e3_out)
                # Bottleneck might need a dedicated conv if inception's final part is removed
                # For now, we pass e4_out to a dummy conv to estimate.
                # This needs careful tuning based on exact inception feature extraction.
                dummy_bottleneck_in = e4_out
                bottleneck_out = nn.Conv2d(dummy_bottleneck_in.shape[1], 2048, 1)(dummy_bottleneck_in) # Example bottleneck conv
                
                return {'e1': e1_out.shape[1], 'e2': e2_out.shape[1], 'e3': e3_out.shape[1], 'e4': e4_out.shape[1], 'bottleneck': bottleneck_out.shape[1]}
            
            elif backbone_name in ['efficientnet_b0', 'efficientnet_b3']:
                dummy_input_for_channels = torch.randn(1, 3, 224, 224) # Typical input size for EfficientNets
                e1_out = self.encoder1(dummy_input_for_channels)
                e2_out = self.encoder2(e1_out)
                e3_out = self.encoder3(e2_out)
                e4_out = self.encoder4(e3_out)
                bottleneck_out = self.bottleneck_layer(e4_out)
                return {'e1': e1_out.shape[1], 'e2': e2_out.shape[1], 'e3': e3_out.shape[1], 'e4': e4_out.shape[1], 'bottleneck': bottleneck_out.shape[1]}

            else:
                return None
        except Exception as e:
            print(f"Error during dynamic channel estimation for {backbone_name}: {e}")
            return None


    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                CBAM(out_channels)  # <-- ADD THIS LINE
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
        
        # Bottleneck
        bottleneck = self.bottleneck_layer(e4_enhanced) 

        # In lasa_unet_model.py, replace the entire "Decoder Path" section in the forward method

        aux_outputs = [] 

        d4_interp_size = e4_enhanced.shape[2:] 
        d4_gating = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        e4_att = self.Att4(g=d4_gating, x=e4_enhanced) # Apply Attention Gate
        d4 = torch.cat([d4_gating, e4_att], dim=1)      # Concatenate attended features
        d4_out = self.decoder4(d4) 
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d3_interp_size = e3.shape[2:]
        d3_gating = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        e3_att = self.Att3(g=d3_gating, x=e3)           # Apply Attention Gate
        d3 = torch.cat([d3_gating, e3_att], dim=1)      # Concatenate attended features
        d3_out = self.decoder3(d3) 
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2_interp_size = e2.shape[2:]
        d2_gating = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        e2_att = self.Att2(g=d2_gating, x=e2)           # Apply Attention Gate
        d2 = torch.cat([d2_gating, e2_att], dim=1)      # Concatenate attended features
        d2_out = self.decoder2(d2) 
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d1_interp_size = e1.shape[2:]
        d1_gating = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        e1_att = self.Att1(g=d1_gating, x=e1)           # Apply Attention Gate
        d1 = torch.cat([d1_gating, e1_att], dim=1)      # Concatenate attended features
        d1_out = self.decoder1(d1) 
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) 

        final_output = self.final_conv(d1_out) 
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)

        return tuple(aux_outputs + [final_output_upsampled])
# Helper to create aliases for convenience
# LASA_VGG_Unet = LASA_Unet # No longer needed as it's generalized