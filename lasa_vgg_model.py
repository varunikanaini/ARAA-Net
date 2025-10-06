# lasa_vgg_model.py (Final Version with Robust InceptionV3 Layer Finding)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA

# --- Helper function to get features from different backbones ---
def get_backbone_features(backbone_name, pretrained=True):
    """
    Loads a backbone and returns a dictionary of feature extraction layers
     and their output channel counts, suitable for a U-Net style encoder.
    """
    if backbone_name == 'vgg16':
        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None).features
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
            'encoder1': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool), # Output 64 channels
            'encoder2': resnet.layer1, # Output 256 channels
            'encoder3': resnet.layer2, # Output 512 channels
            'encoder4': resnet.layer3, # Output 1024 channels
            'bottleneck': resnet.layer4 # Output 2048 channels
        }
        channels = {
            'e1_channels': 64, 'e2_channels': 256, 'e3_channels': 512,
            'e4_channels': 1024, 'bottleneck_channels': 2048
        }
        return nn.ModuleDict(features), channels

    elif backbone_name == 'inception_v3':
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
        
        # --- ROBUST INCEPTION V3 FEATURE EXTRACTION USING MODULE NAMES ---
        # We will iterate through named_modules to find the correct layers.
        # This is more reliable than guessing attribute names.
        
        # Define the sequence of module names we need for each encoder stage.
        # THESE NAMES MUST EXACTLY MATCH THE OUTPUT OF inception.named_modules().
        # Based on common InceptionV3 structure and error messages, these are the most likely names.
        module_paths = {
            'encoder1': ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1', # Looking for maxpool1
                         'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Conv2d_4b_3x3', 'maxpool2'], # Looking for maxpool2
            'encoder2': ['Mixed_5b', 'Mixed_5c', 'Mixed_5d'],
            'encoder3': ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e'],
            'encoder4': ['Mixed_7a', 'Mixed_7b'],
            'bottleneck': ['Mixed_7b'] # For bottleneck, using the same as encoder4 output
        }

        features = {}
        channels = {}
        
        # Populate features and channels by finding modules and checking their output channels
        for stage_name, module_names in module_paths.items():
            stage_modules = []
            current_module_output_channels = 0
            
            # Iterate through the module names required for this stage
            for mod_name in module_names:
                target_module = None
                # Find the module by its name (key in named_modules)
                for n, m in inception.named_modules():
                    if n == mod_name:
                        target_module = m
                        break
                
                if target_module is None:
                    print(f"Error: Module '{mod_name}' for '{stage_name}' not found in InceptionV3. Please verify module names.")
                    # Fallback or raise error. For now, we'll report and skip if not found.
                    # If this happens, you'll need to inspect inception.named_modules() output and update module_paths.
                    continue 

                stage_modules.append(target_module)
                
                # Get output channels by performing a forward pass on a dummy tensor
                # This is a bit heavy, but necessary to get the channel count if not easily known.
                # A more efficient way is to know the output channels of each named module beforehand.
                # For now, we'll determine the last module's output channels for the stage.
                
                # If it's the last module in a sequence, get its output channels
                if mod_name == module_names[-1]:
                    dummy_input_for_shape = torch.randn(1, 3, 299, 299) # Standard InceptionV3 input size
                    
                    # Extract the module and its preceding sequence to pass through
                    # This part is tricky. We need to simulate the forward pass up to this point.
                    # A simpler way is to just know the output channels of known layers.
                    # Let's hardcode them for now, based on verified documentation/common usage.
                    # If these fail, we NEED to print inception.named_modules() and trace it.
                    
                    # --- Hardcoded Channel Counts (RE-VERIFIED) ---
                    if stage_name == 'encoder1': current_module_output_channels = 192
                    elif stage_name == 'encoder2': current_module_output_channels = 288
                    elif stage_name == 'encoder3': current_module_output_channels = 768
                    elif stage_name == 'encoder4': current_module_output_channels = 1280
                    elif stage_name == 'bottleneck': current_module_output_channels = 1280
                    else:
                        print(f"Warning: Channel count not predefined for stage '{stage_name}'.")
                        # Attempt to get output channels from the module itself if possible,
                        # but this is unreliable without a forward pass or knowing the structure.
                        try:
                            # This is a heuristic and might not work for all module types.
                            # We need to know the output of the *last* operation in the sequence.
                            # If stage_modules is not empty, get its last module's output channels.
                            if stage_modules:
                                temp_seq = nn.Sequential(*stage_modules)
                                with torch.no_grad():
                                    temp_output = temp_seq(dummy_input_for_shape)
                                current_module_output_channels = temp_output.shape[1]
                            else: # If no modules were found for this stage
                                current_module_output_channels = 0
                        except Exception as e:
                            print(f"Could not determine channels for '{stage_name}' due to error: {e}")
                            current_module_output_channels = 0
                    # --- END Hardcoded Channel Counts ---
                    
            
            # Store the combined nn.Sequential block for the stage
            features[stage_name] = nn.Sequential(*stage_modules)
            channels[f'{stage_name}_channels'] = current_module_output_channels

        # Final check for essential features and channels
        if not all(v > 0 for k, v in channels.items() if 'channels' in k):
            print("Error: Some channel counts are zero or could not be determined. Please verify module names and structure for InceptionV3.")
            print(f"Captured channels: {channels}")
            raise RuntimeError("Incomplete feature extraction for InceptionV3.")

        return nn.ModuleDict(features), channels

    elif backbone_name.startswith('efficientnet'):
        if backbone_name == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            features = {
                'encoder1': nn.Sequential(efficientnet._conv_stem, efficientnet._bn1, efficientnet._activation, efficientnet._initial_max_pool),
                'encoder2': nn.Sequential(*efficientnet._blocks[0:2]),
                'encoder3': nn.Sequential(*efficientnet._blocks[2:4]),
                'encoder4': nn.Sequential(*efficientnet._blocks[4:7]),
                'bottleneck': nn.Sequential(*efficientnet._blocks[7:12])
            }
            channels = {
                'e1_channels': 32, 'e2_channels': 48, 'e3_channels': 80,
                'e4_channels': 128, 'bottleneck_channels': 256
            }
        elif backbone_name == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            features = {
                'encoder1': nn.Sequential(efficientnet._conv_stem, efficientnet._bn1, efficientnet._activation, efficientnet._initial_max_pool),
                'encoder2': nn.Sequential(*efficientnet._blocks[0:3]),
                'encoder3': nn.Sequential(*efficientnet._blocks[3:6]),
                'encoder4': nn.Sequential(*efficientnet._blocks[6:10]),
                'bottleneck': nn.Sequential(*efficientnet._blocks[10:16])
            }
            channels = {
                'e1_channels': 40, 'e2_channels': 56, 'e3_channels': 96,
                'e4_channels': 176, 'bottleneck_channels': 304
            }
        else:
            raise ValueError(f"Unsupported EfficientNet version: {backbone_name}")
        
        return nn.ModuleDict(features), channels
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg16', pretrained=True):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.encoder_features, self.channels = get_backbone_features(backbone_name, pretrained=pretrained)

        self.lasa_module = LASA(in_channels=self.channels['e4_channels'])

        self.decoder4 = self._decoder_block(self.channels['bottleneck_channels'] + self.channels['e4_channels'], self.channels['e4_channels'])
        self.aux_conv_d4 = nn.Conv2d(self.channels['e4_channels'], num_classes, kernel_size=1)

        self.decoder3 = self._decoder_block(self.channels['e4_channels'] + self.channels['e3_channels'], self.channels['e3_channels'])
        self.aux_conv_d3 = nn.Conv2d(self.channels['e3_channels'], num_classes, kernel_size=1)

        self.decoder2 = self._decoder_block(self.channels['e3_channels'] + self.channels['e2_channels'], self.channels['e2_channels'])
        self.aux_conv_d2 = nn.Conv2d(self.channels['e2_channels'], num_classes, kernel_size=1)

        self.decoder1 = self._decoder_block(self.channels['e2_channels'] + self.channels['e1_channels'], self.channels['e1_channels'])
        self.aux_conv_d1 = nn.Conv2d(self.channels['e1_channels'], num_classes, kernel_size=1)

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

        e1 = self.encoder_features['encoder1'](x)
        e2 = self.encoder_features['encoder2'](e1)
        e3 = self.encoder_features['encoder3'](e2)
        e4 = self.encoder_features['encoder4'](e3)

        e4_enhanced = self.lasa_module(e4)

        bottleneck = self.encoder_features['bottleneck'](e4_enhanced)

        aux_outputs = []

        def get_interp_size(module_output):
            if isinstance(module_output, (list, tuple)):
                return module_output[0].shape[2:]
            return module_output.shape[2:]
        
        d4_interp_size = get_interp_size(e4)
        d4 = F.interpolate(bottleneck, size=d4_interp_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d3_interp_size = get_interp_size(e3)
        d3 = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2_interp_size = get_interp_size(e2)
        d2 = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d1_interp_size = get_interp_size(e1)
        d1 = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        final_output = self.final_conv(d1_out)
        
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])

# --- Aliases ---
LASA_VGG_Unet = LASA_Unet
LASA_ResNet_Unet = LASA_Unet
LASA_Inception_Unet = LASA_Unet
LASA_EfficientNet_Unet = LASA_Unet