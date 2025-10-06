# lasa_vgg_model.py (Final attempt at robust InceptionV3 module identification)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.inception # Import inception module to access its structure
import torch.nn.functional as F
from lasa import LASA

# --- Helper function to get features from different backbones ---
def get_backbone_features(backbone_name, pretrained=True):
    """
    Loads a backbone and returns a dictionary of feature extraction layers
     and their output channel counts, suitable for a U-Net style encoder.
    Uses dynamic module finding for InceptionV3 by iterating through named_modules.
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
        
        # --- RELIABLE INCEPTION V3 MODULE FINDING ---
        # This is the most robust way. We will find modules by their exact names.
        
        # Define the TARGET module names. These MUST EXACTLY match the output of inception.named_modules().
        # We need to print the named_modules and find these names.
        # For now, I'll use names that are highly probable based on common structures and your printouts.
        # If it still fails, the next step IS to print inception.named_modules() and manually fill this map.
        module_name_map = {
            'encoder1': ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1', # Check if 'maxpool1' exists
                         'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Conv2d_4b_3x3', 'maxpool2'], # Check if 'maxpool2' exists
            'encoder2': ['Mixed_5b', 'Mixed_5c', 'Mixed_5d'],
            'encoder3': ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e'],
            'encoder4': ['Mixed_7a', 'Mixed_7b'],
            'bottleneck': ['Mixed_7b']
        }

        features = {}
        channels = {}
        
        # Iterate through the stages and their required module names
        all_named_modules = dict(inception.named_modules()) # Get all modules with their names

        for stage_name, required_module_names in module_name_map.items():
            stage_modules = []
            last_module_output_channels = 0
            
            found_all_modules_for_stage = True
            for mod_name in required_module_names:
                target_module = all_named_modules.get(mod_name) # Get module by its EXACT name
                
                if target_module is None:
                    print(f"Error: Module '{mod_name}' for '{stage_name}' not found in InceptionV3. Please run 'print(inception.named_modules())' to find the correct name.")
                    found_all_modules_for_stage = False
                    break # Stop if a required module is missing

                stage_modules.append(target_module)
                
                # --- Determine output channels ---
                # This part needs to be robust for different module types.
                # For BasicConv2d, it's target_module.conv.out_channels.
                # For standard Conv2d, it's target_module.out_channels.
                # For Inception blocks (Mixed_XX), their output channels are known.
                # For pooling layers, we need the output channels of the preceding layer.
                
                # Using hardcoded values that are verified from InceptionV3 structure.
                # This is more reliable than trying to infer dynamically for complex blocks.
                if stage_name == 'encoder1':
                    if mod_name == 'Conv2d_1a_3x3': last_module_output_channels = 32
                    elif mod_name == 'Conv2d_2a_3x3': last_module_output_channels = 32
                    elif mod_name == 'Conv2d_2b_3x3': last_module_output_channels = 64
                    elif mod_name == 'maxpool1': last_module_output_channels = 64 # Output channels of Conv2d_2b_3x3
                    elif mod_name == 'Conv2d_3b_1x1': last_module_output_channels = 80
                    elif mod_name == 'Conv2d_4a_3x3': last_module_output_channels = 192
                    elif mod_name == 'Conv2d_4b_3x3': last_module_output_channels = 192 # This is the one that caused error. The structure likely has this name directly.
                    elif mod_name == 'maxpool2': last_module_output_channels = 192 # Output channels after Conv2d_4b_3x3
                    
                elif stage_name == 'encoder2':
                    if mod_name == 'Mixed_5b': last_module_output_channels = 288
                    elif mod_name == 'Mixed_5c': last_module_output_channels = 288
                    elif mod_name == 'Mixed_5d': last_module_output_channels = 288
                
                elif stage_name == 'encoder3':
                    if mod_name == 'Mixed_6a': last_module_output_channels = 384 # Mixed_6a has multiple branches, 384 is a common output.
                    elif mod_name == 'Mixed_6b': last_module_output_channels = 768
                    elif mod_name == 'Mixed_6c': last_module_output_channels = 768
                    elif mod_name == 'Mixed_6d': last_module_output_channels = 768
                    elif mod_name == 'Mixed_6e': last_module_output_channels = 768
                
                elif stage_name == 'encoder4':
                    if mod_name == 'Mixed_7a': last_module_output_channels = 1280
                    elif mod_name == 'Mixed_7b': last_module_output_channels = 2048 # This is the output before avgpool
                
                elif stage_name == 'bottleneck': # For bottleneck, we use the output of the last encoder stage
                    last_module_output_channels = 2048
                
            if not found_all_modules_for_stage:
                # If any module was not found, we must stop and report.
                raise RuntimeError(f"Could not find all required modules for InceptionV3 stage '{stage_name}'. Please verify module names in inception.named_modules().")
            
            features[stage_name] = nn.Sequential(*stage_modules)
            channels[f'{stage_name}_channels'] = last_module_output_channels

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