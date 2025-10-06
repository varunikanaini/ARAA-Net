# lasa_vgg_model.py (Further Corrected with InceptionV3 Hooks)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA

# --- Global dictionary to store hooked features ---
# This is a simple way to pass features from hooks back to the main function.
# In a more complex scenario, you might use a class attribute or a callback.
hooked_features_cache = {}

def get_features_hook(name):
    """Returns a hook function that stores the output tensor and its shape."""
    def hook(model, input, output):
        if isinstance(output, torch.Tensor):
            hooked_features_cache[name] = output
        elif isinstance(output, (list, tuple)): # Handle models that return multiple outputs
            hooked_features_cache[name] = output[0] # Assume the first is the primary feature map
        else:
            print(f"Warning: Unexpected output type from hook '{name}': {type(output)}")
    return hook

def get_backbone_features(backbone_name, pretrained=True):
    """
    Loads a backbone and returns a dictionary of feature extraction layers
     and their output channel counts, suitable for a U-Net style encoder.
    Uses hooks for InceptionV3 to capture intermediate outputs.
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
        
        # --- Using Hooks for InceptionV3 Feature Extraction ---
        # We need to capture intermediate outputs. The exact layers to hook into are crucial.
        # Common choices for U-Net encoders are often after major block groups.
        # Let's define some target module names and their expected channel counts.
        # This part requires careful inspection of the InceptionV3 architecture.
        
        # Names of modules to hook into (from inspecting inception_v3 structure):
        # These are representative names and might need slight adjustment.
        hook_modules = {
            'encoder1': 'Conv2d_4a_3x3',  # After initial conv blocks and first pooling
            'encoder2': 'Mixed_5d',       # After the first group of Inception modules
            'encoder3': 'Mixed_6e',       # After the second group of Inception modules
            'encoder4': 'Mixed_7b',       # After the last group of Inception modules (before final classifier)
            'bottleneck': 'Mixed_7b'      # Using the same as encoder4 for bottleneck
        }

        # Register forward hooks to capture the output of these modules
        handles = []
        for name, module_name in hook_modules.items():
            try:
                module = dict(inception.named_children())[module_name]
                # Some modules might be nested, so we might need to traverse deeper
                # Example: if 'Mixed_5d' was nested, you'd need something like:
                # module = inception.layer3.Mixed_5d
                # For now, assume direct access works or needs specific path.
                # If 'module_name' is not directly found, you'll need to find its parent or full path.
                
                # A more reliable way for nested modules is to iterate through named_modules()
                target_module = None
                for n, m in inception.named_modules():
                    if n == module_name: # Direct match
                        target_module = m
                        break
                    # If module is nested, e.g., 'layer3.Mixed_5d', you might need to split and traverse.
                    # For simplicity, we rely on direct name match or immediate children.
                
                if target_module is None:
                    print(f"Warning: Module '{module_name}' for '{name}' not found directly. Trying to find within 'children'.")
                    # Fallback: Try to find in direct children if not found by name
                    if hasattr(inception, module_name): # Check if it's a direct attribute
                        target_module = getattr(inception, module_name)
                    else:
                        print(f"Error: Module '{module_name}' not found for '{name}'. Skipping.")
                        continue # Skip if module isn't found

                handle = target_module.register_forward_hook(get_features_hook(name))
                handles.append((target_module, handle)) # Store module and handle to remove later
                
            except AttributeError as e:
                print(f"AttributeError when trying to hook '{module_name}' for '{name}': {e}")
                # If direct attribute access fails, it means the name is wrong or it's deeply nested.
                # For a robust solution, you'd need to print(inception.named_modules()) and find the correct path.
                print("Please inspect inception_v3 structure and update module names in hook_modules.")
                # Attempting a more programmatic way to find the module if name is not direct:
                try:
                    potential_module = None
                    # Iterate through module names and try to find a match
                    for n, m in inception.named_modules():
                        if n == module_name: # Check if the full path matches
                            potential_module = m
                            break
                    if potential_module:
                        handle = potential_module.register_forward_hook(get_features_hook(name))
                        handles.append((potential_module, handle))
                        print(f"Successfully hooked '{module_name}' for '{name}'.")
                    else:
                        print(f"Failed to find module '{module_name}' for '{name}' even with named_modules().")
                except Exception as hook_err:
                    print(f"An error occurred during hook registration for '{module_name}': {hook_err}")

        # --- Forward pass with dummy input to trigger hooks ---
        # This is necessary to populate hooked_features_cache
        dummy_input = torch.randn(1, 3, 299, 299) # InceptionV3 typically uses 299x299
        if pretrained:
            # Remove classifier and aux logits for feature extraction
            original_fc = inception.fc
            original_aux_logits = inception.AuxLogits
            inception.fc = nn.Identity()
            inception.AuxLogits = nn.Identity() # Effectively remove classifier and aux
            
            try:
                inception(dummy_input) # Run forward pass to capture features
            except Exception as forward_err:
                print(f"Error during dummy forward pass for InceptionV3 feature capture: {forward_err}")
                print("This might be due to incorrect module names or shapes in the InceptionV3 structure.")
            finally:
                # Restore original layers if they were modified (though here we just bypassed them)
                inception.fc = original_fc
                inception.AuxLogits = original_aux_logits

        # --- Remove hooks after capture ---
        for module, handle in handles:
            handle.remove()
            
        # --- Extract features and channels from the captured outputs ---
        if not hooked_features_cache:
            raise RuntimeError("Failed to capture any features using hooks for InceptionV3. Cannot proceed.")
        
        features = {}
        channels = {}
        
        # Assign captured tensors to features and determine channels
        # The names used here ('encoder1', 'encoder2', etc.) must match those in hook_modules
        features['encoder1'] = hooked_features_cache.get('encoder1')
        channels['e1_channels'] = features['encoder1'].shape[1] if features.get('encoder1') is not None else 0
        
        features['encoder2'] = hooked_features_cache.get('encoder2')
        channels['e2_channels'] = features['encoder2'].shape[1] if features.get('encoder2') is not None else 0
        
        features['encoder3'] = hooked_features_cache.get('encoder3')
        channels['e3_channels'] = features['encoder3'].shape[1] if features.get('encoder3') is not None else 0
        
        features['encoder4'] = hooked_features_cache.get('encoder4')
        channels['e4_channels'] = features['encoder4'].shape[1] if features.get('encoder4') is not None else 0
        
        features['bottleneck'] = hooked_features_cache.get('bottleneck')
        channels['bottleneck_channels'] = features['bottleneck'].shape[1] if features.get('bottleneck') is not None else 0
        
        # Clear cache for next iteration if needed
        hooked_features_cache.clear()

        # --- Final check for essential features ---
        if not all(v > 0 for v in channels.values()):
            print("Error: Some channel counts are zero. Please verify module names and feature extraction for InceptionV3.")
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

        # --- Load Backbone and Extract Features ---
        self.encoder_features, self.channels = get_backbone_features(backbone_name, pretrained=pretrained)

        # --- Original LASA Module ---
        # Initialize LASA with the correct in_channels for e4
        self.lasa_module = LASA(in_channels=self.channels['e4_channels'])

        # --- Decoder Blocks and Auxiliary Convs ---
        # Decoder 4
        self.decoder4 = self._decoder_block(self.channels['bottleneck_channels'] + self.channels['e4_channels'], self.channels['e4_channels'])
        self.aux_conv_d4 = nn.Conv2d(self.channels['e4_channels'], num_classes, kernel_size=1)

        # Decoder 3
        self.decoder3 = self._decoder_block(self.channels['e4_channels'] + self.channels['e3_channels'], self.channels['e3_channels'])
        self.aux_conv_d3 = nn.Conv2d(self.channels['e3_channels'], num_classes, kernel_size=1)

        # Decoder 2
        self.decoder2 = self._decoder_block(self.channels['e3_channels'] + self.channels['e2_channels'], self.channels['e2_channels'])
        self.aux_conv_d2 = nn.Conv2d(self.channels['e2_channels'], num_classes, kernel_size=1)

        # Decoder 1
        self.decoder1 = self._decoder_block(self.channels['e2_channels'] + self.channels['e1_channels'], self.channels['e1_channels'])
        self.aux_conv_d1 = nn.Conv2d(self.channels['e1_channels'], num_classes, kernel_size=1)

        # --- Final Output Convolution ---
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
        # Forward pass through each encoder stage to capture features via hooks
        e1 = self.encoder_features['encoder1'](x)
        e2 = self.encoder_features['encoder2'](e1)
        e3 = self.encoder_features['encoder3'](e2)
        e4 = self.encoder_features['encoder4'](e3)

        # Apply LASA enhancement to e4
        e4_enhanced = self.lasa_module(e4)

        # Bottleneck
        bottleneck = self.encoder_features['bottleneck'](e4_enhanced)

        # --- Decoder Path ---
        aux_outputs = []

        # Helper to get interpolation size, handling potential tuple outputs from modules
        def get_interp_size(module_output):
            if isinstance(module_output, (list, tuple)):
                return module_output[0].shape[2:]
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

        # Final output
        final_output = self.final_conv(d1_out)
        
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])

# Aliases
LASA_VGG_Unet = LASA_Unet
LASA_ResNet_Unet = LASA_Unet
LASA_Inception_Unet = LASA_Unet
LASA_EfficientNet_Unet = LASA_Unet