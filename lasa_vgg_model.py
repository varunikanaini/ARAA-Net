# /kaggle/working/ARAA-Net/lasa_vgg_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# Ensure lasa.py is correctly imported
try:
    from lasa import LASA
except ImportError:
    print("ERROR: Could not import LASA module. Make sure 'lasa.py' is in the same directory or in sys.path.")
    # You might want to exit or raise an error here if LASA is critical
    # For demonstration, we'll define a dummy LASA if import fails, but it won't work
    class LASA(nn.Module):
        def __init__(self, in_channels, M=4, L_list=[5, 7, 9, 11]):
            super(LASA, self).__init__()
            print("WARNING: Using dummy LASA module due to import error. Performance will be affected.")
            # Simple placeholder to allow the model to run, but without functionality
            self.conv = nn.Conv2d(in_channels, in_channels, 1) 
        def forward(self, x):
            return self.conv(x)


# --- Attention Gate Module ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_intermediate):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_intermediate, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_intermediate)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_intermediate, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_intermediate)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_intermediate, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, x_gating):
        g1 = self.W_g(x_gating)
        x1 = self.W_x(x_skip)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x_skip * psi

# --- ASPP Module Definition ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        self.gn = lambda num_channels: nn.GroupNorm(num_groups=min(num_channels, 32), num_channels=num_channels)

        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), self.gn(out_channels), nn.ReLU(inplace=True))
        self.conv_aspp = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_aspp.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False), self.gn(out_channels), nn.ReLU(inplace=True)))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), self.gn(out_channels), nn.ReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False), self.gn(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        spatial_dims = x.shape[2:]
        branch1 = self.conv1x1(x)
        branches_aspp = [conv(x) for conv in self.conv_aspp]
        branch_pool = self.global_avg_pool(x)
        branch_pool = F.interpolate(branch_pool, size=spatial_dims, mode='bilinear', align_corners=True)
        features = [branch1] + branches_aspp + [branch_pool]
        fused_features = torch.cat(features, dim=1)
        output = self.conv_fuse(fused_features)
        return output

# --- LASA_Unet Class with LASA in Encoder + ASPP + Attention Gates ---
class LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Define encoder stages and their channel counts
            # We'll need to capture features BEFORE the max pooling layers to insert LASA
            self.encoder1 = vgg_features[:5]   # conv1_1, conv1_2. Output: 64 channels
            self.encoder2 = vgg_features[5:12] # conv2_1, conv2_2. Output: 128 channels
            self.encoder3 = vgg_features[12:22] # conv3_1, conv3_2, conv3_3. Output: 256 channels
            self.encoder4 = vgg_features[22:32] # conv4_1, conv4_2, conv4_3. Output: 512 channels
            # The LASA paper suggests LASA in dense blocks 1 & 2. In VGG, this is roughly before conv3_3 and conv4_3.
            # So, we'll apply LASA after encoder3 and encoder4 outputs.

            self.e1_channels = 64
            self.e2_channels = 128
            self.e3_channels = 256
            self.e4_channels = 512

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # Output 64 channels
            self.encoder2 = resnet.layer1 # Output 256 channels
            self.encoder3 = resnet.layer2 # Output 512 channels
            self.encoder4 = resnet.layer3 # Output 1024 channels
            # ResNet structure is different. The original LASA paper used DenseNet.
            # For ResNet, "dense block 1" usually means after layer1, and "dense block 2" after layer2.
            # Let's adapt the strategy to apply LASA after encoder3 and encoder4.

            self.e1_channels = 64
            self.e2_channels = 256
            self.e3_channels = 512
            self.e4_channels = 1024

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. LASA Modules at Different Encoder Stages ---
        # Apply LASA after encoder3 and encoder4 outputs, as suggested by the paper's findings.
        # LASA output channels should match the input channels of the next stage or the bottleneck.
        self.lasa_module_3 = LASA(in_channels=self.e3_channels)
        self.lasa_module_4 = LASA(in_channels=self.e4_channels)

        # --- 3. ASPP Module Integration ---
        # ASPP operates on the LASA-enhanced features from encoder4.
        # ASPP output channels match e4_channels for consistency.
        aspp_output_channels = self.e4_channels 
        self.aspp = ASPP(in_channels=self.e4_channels, 
                         out_channels=aspp_output_channels, 
                         atrous_rates=[6, 12, 18, 24])

        # --- 4. Define Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Attention Gates will be added to each skip connection.
        # Intermediate channels for Attention Gate are typically half of the input channels.

        # Decoder 4: Input from ASPP output + Attention(LASA_enhanced_e4, ASPP_output)
        ag_intermediate_channels_d4 = self.e4_channels // 2 
        self.attention_gate_d4 = AttentionGate(F_g=aspp_output_channels, 
                                               F_x=self.e4_channels, # LASA output has self.e4_channels
                                               F_intermediate=ag_intermediate_channels_d4)
        # Input to decoder4: upsampled ASPP output + attended LASA_enhanced_e4
        self.decoder4 = self._decoder_block(aspp_output_channels + self.e4_channels, self.e4_channels)
        self.aux_conv_d4 = nn.Conv2d(self.e4_channels, num_classes, kernel_size=1)

        # Decoder 3: Input from upsampled d4_out + Attention(LASA_enhanced_e3, d4_out)
        ag_intermediate_channels_d3 = self.e3_channels // 2
        self.attention_gate_d3 = AttentionGate(F_g=self.e4_channels, # F_g is from d4_out
                                               F_x=self.e3_channels, # F_x is from LASA_enhanced_e3
                                               F_intermediate=ag_intermediate_channels_d3)
        # Input to decoder3: upsampled d4_out + attended LASA_enhanced_e3
        self.decoder3 = self._decoder_block(self.e4_channels + self.e3_channels, self.e3_channels)
        self.aux_conv_d3 = nn.Conv2d(self.e3_channels, num_classes, kernel_size=1)

        # Decoder 2: Input from upsampled d3_out + Attention(e2, d3_out)
        # Note: We are NOT applying LASA to e2, so F_x is e2_channels
        ag_intermediate_channels_d2 = self.e2_channels // 2
        self.attention_gate_d2 = AttentionGate(F_g=self.e3_channels, # F_g is from d3_out
                                               F_x=self.e2_channels, # F_x is from e2
                                               F_intermediate=ag_intermediate_channels_d2)
        # Input to decoder2: upsampled d3_out + attended e2
        self.decoder2 = self._decoder_block(self.e3_channels + self.e2_channels, self.e2_channels)
        self.aux_conv_d2 = nn.Conv2d(self.e2_channels, num_classes, kernel_size=1)

        # Decoder 1: Input from upsampled d2_out + Attention(e1, d2_out)
        # Note: We are NOT applying LASA to e1
        ag_intermediate_channels_d1 = self.e1_channels // 2
        self.attention_gate_d1 = AttentionGate(F_g=self.e2_channels, # F_g is from d2_out
                                               F_x=self.e1_channels, # F_x is from e1
                                               F_intermediate=ag_intermediate_channels_d1)
        # Input to decoder1: upsampled d2_out + attended e1
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

        # Apply LASA enhancement to e3 and e4 outputs as suggested by the paper
        e3_enhanced = self.lasa_module_3(e3)
        e4_enhanced = self.lasa_module_4(e4)
        
        # --- Bottleneck with ASPP ---
        # ASPP operates on the LASA-enhanced e4 features
        aspp_output = self.aspp(e4_enhanced)

        # --- Decoder Path with Skip Connections and True Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4: Uses ASPP output and attended e4_enhanced
        d4_interp_size = e4_enhanced.shape[2:] 
        aspp_upsampled = F.interpolate(aspp_output, size=d4_interp_size, mode='bilinear', align_corners=True)
        
        attended_e4 = self.attention_gate_d4(e4_enhanced, aspp_upsampled)
        
        d4 = torch.cat([aspp_upsampled, attended_e4], dim=1) 
        d4_out = self.decoder4(d4) 
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 3: Uses upsampled d4_out and attended e3
        d3_interp_size = e3.shape[2:]
        d3_upsampled = F.interpolate(d4_out, size=d3_interp_size, mode='bilinear', align_corners=True)
        attended_e3 = self.attention_gate_d3(e3_enhanced, d3_upsampled) # Use LASA-enhanced e3
        d3 = torch.cat([d3_upsampled, attended_e3], dim=1) 
        d3_out = self.decoder3(d3) 
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2: Uses upsampled d3_out and attended e2 (e2 is not LASA enhanced)
        d2_interp_size = e2.shape[2:]
        d2_upsampled = F.interpolate(d3_out, size=d2_interp_size, mode='bilinear', align_corners=True)
        attended_e2 = self.attention_gate_d2(e2, d2_upsampled) # Use raw e2
        d2 = torch.cat([d2_upsampled, attended_e2], dim=1) 
        d2_out = self.decoder2(d2) 
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 1: Uses upsampled d2_out and attended e1 (e1 is not LASA enhanced)
        d1_interp_size = e1.shape[2:]
        d1_upsampled = F.interpolate(d2_out, size=d1_interp_size, mode='bilinear', align_corners=True)
        attended_e1 = self.attention_gate_d1(e1, d1_upsampled) # Use raw e1
        d1 = torch.cat([d1_upsampled, attended_e1], dim=1) 
        d1_out = self.decoder1(d1) 
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True)) 
        
        # Final output
        final_output = self.final_conv(d1_out) 
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        return tuple(aux_outputs + [final_output_upsampled])

LASA_VGG_Unet = LASA_Unet