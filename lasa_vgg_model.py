# /kaggle/working/ARAA-Net/lasa_vgg_model.py (Modified for FPN and Deep Supervision)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # <<< Imports the user's original LASA module

class LASA_Unet(nn.Module):
    """
    A standalone segmentation model with VGG16/ResNet50 backbone,
    User's original LASA module, an FPN for multi-scale feature fusion,
    and a U-Net style decoder with deep supervision.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            
            # Encoder outputs at different stages
            self.encoder1 = vgg_features[:6]   # 64 channels (stride 1/2 from input)
            self.encoder2 = vgg_features[6:13]  # 128 channels (stride 1/4)
            self.encoder3 = vgg_features[13:23] # 256 channels (stride 1/8)
            self.encoder4 = vgg_features[23:33] # 512 channels (stride 1/16)
            self.bottleneck_layer = vgg_features[33:43] # 512 channels (stride 1/32)

            self.c1_channels = 64  # Corresponds to e1
            self.c2_channels = 128 # Corresponds to e2
            self.c3_channels = 256 # Corresponds to e3
            self.c4_channels = 512 # Corresponds to e4
            self.c5_channels = 512 # Corresponds to bottleneck
            
            fpn_out_channels = 256 # Common FPN feature dimension

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # Encoder outputs at different stages
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # 64 channels (stride 1/4)
            self.encoder2 = resnet.layer1 # 256 channels (stride 1/4)
            self.encoder3 = resnet.layer2 # 512 channels (stride 1/8)
            self.encoder4 = resnet.layer3 # 1024 channels (stride 1/16)
            self.bottleneck_layer = resnet.layer4 # 2048 channels (stride 1/32)

            self.c1_channels = 64  # Corresponds to e1
            self.c2_channels = 256 # Corresponds to e2
            self.c3_channels = 512 # Corresponds to e3
            self.c4_channels = 1024 # Corresponds to e4
            self.c5_channels = 2048 # Corresponds to bottleneck

            fpn_out_channels = 256 # Common FPN feature dimension
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Original LASA Module ---
        # LASA will enhance features from the 4th encoder block (e4)
        self.lasa_module = LASA(in_channels=self.c4_channels)

        # --- 3. FPN Layers ---
        # Lateral connections (1x1 convs to match fpn_out_channels)
        self.fpn_lateral5 = nn.Conv2d(self.c5_channels, fpn_out_channels, kernel_size=1)
        self.fpn_lateral4 = nn.Conv2d(self.c4_channels, fpn_out_channels, kernel_size=1)
        self.fpn_lateral3 = nn.Conv2d(self.c3_channels, fpn_out_channels, kernel_size=1)
        self.fpn_lateral2 = nn.Conv2d(self.c2_channels, fpn_out_channels, kernel_size=1)
        self.fpn_lateral1 = nn.Conv2d(self.c1_channels, fpn_out_channels, kernel_size=1) # For e1

        # Smooth connections (3x3 convs to refine combined features)
        self.fpn_smooth5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.fpn_smooth4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.fpn_smooth3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.fpn_smooth2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.fpn_smooth1 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1) # For e1


        # --- 4. Decoder Blocks and Auxiliary Convs for Deep Supervision ---
        # Decoder structure changes to use FPN features.
        # Input channels for decoders will be (fpn_out_channels + fpn_out_channels) due to concat
        
        # d4: upsample p5 + p4_smooth. Uses p4_smooth as skip.
        # We'll use FPN outputs (s5, s4, s3, s2, s1) for skip connections and main path.
        self.decoder4 = self._decoder_block(fpn_out_channels * 2, fpn_out_channels)
        self.aux_conv_d4 = nn.Conv2d(fpn_out_channels, num_classes, kernel_size=1)

        self.decoder3 = self._decoder_block(fpn_out_channels * 2, fpn_out_channels)
        self.aux_conv_d3 = nn.Conv2d(fpn_out_channels, num_classes, kernel_size=1)

        self.decoder2 = self._decoder_block(fpn_out_channels * 2, fpn_out_channels)
        self.aux_conv_d2 = nn.Conv2d(fpn_out_channels, num_classes, kernel_size=1)

        self.decoder1 = self._decoder_block(fpn_out_channels * 2, fpn_out_channels)
        
        # --- 5. Final Output Convolution ---
        self.final_conv = nn.Conv2d(fpn_out_channels, num_classes, kernel_size=1)


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
        input_size = x.shape[2:] 

        # --- Encoder Path (C1, C2, C3, C4, C5) ---
        c1 = self.encoder1(x) # e1
        c2 = self.encoder2(c1) # e2
        c3 = self.encoder3(c2) # e3
        c4 = self.encoder4(c3) # e4

        # Apply LASA enhancement to c4 (e4)
        c4_enhanced = self.lasa_module(c4)
        
        # Bottleneck (C5) from enhanced c4
        c5 = self.bottleneck_layer(c4_enhanced)

        # --- FPN Top-down Pathway with Lateral Connections ---
        # p5 is the highest level feature
        p5 = self.fpn_lateral5(c5)
        
        # p4 combines lateral from c4_enhanced and upsampled p5
        p4 = self._upsample_add(p5, self.fpn_lateral4(c4_enhanced)) 
        p4 = self.fpn_smooth4(p4)

        # p3 combines lateral from c3 and upsampled p4
        p3 = self._upsample_add(p4, self.fpn_lateral3(c3))
        p3 = self.fpn_smooth3(p3)

        # p2 combines lateral from c2 and upsampled p3
        p2 = self._upsample_add(p3, self.fpn_lateral2(c2))
        p2 = self.fpn_smooth2(p2)

        # p1 combines lateral from c1 and upsampled p2
        p1 = self._upsample_add(p2, self.fpn_lateral1(c1))
        p1 = self.fpn_smooth1(p1)


        # --- Decoder Path with FPN Features and Deep Supervision ---
        aux_outputs = [] 

        # Decoder 4 (highest resolution skip connection for deep supervision)
        # Main decoder path starts from p5 (highest level FPN feature)
        # It's common to connect p5 to d4, and subsequent layers get input from lower FPN levels
        # Here, we will use FPN outputs (p1, p2, p3, p4, p5) as the primary skip connections in the U-Net decoder.
        
        # Let's adjust the decoder to use FPN features as direct input to corresponding levels
        # The FPN features p1, p2, p3, p4, p5 have all fpn_out_channels.
        
        # Decoder's highest level input from p5 (or p4 for d4 if we follow a different convention)
        # Using a standard U-Net decoder, the highest level of detail (input_size) is fed by p1.
        # The lowest resolution in the decoder is the highest resolution in FPN (p5).
        # Reversing the FPN path for the decoder:
        
        # d_bottleneck_out from p5, or directly integrate p5 into the decoder chain
        # For deep supervision, we need 5 outputs.
        # p5 is lowest resolution. d4 output will be from p5-like features.

        # Upsample p5 (or the last FPN feature) to the next level's size
        # Our current decoder architecture works from bottleneck up. Let's make FPN feed into it.
        # e4_enhanced has same spatial as c4 (stride 1/16 or 1/8 for vgg).
        # bottleneck has spatial as c5 (stride 1/32 or 1/16 for vgg).
        
        # Let's consider the FPN outputs as `fpn_features = [p1, p2, p3, p4, p5]` in ascending order of stride (descending spatial resolution)
        # So p5 is highest stride, p1 is lowest stride (highest spatial res).
        
        # Highest level of decoder (d4) takes processed bottleneck and p4 (from FPN)
        # Here `bottleneck` is `c5` from ResNet structure, which is `1/32` or `1/16` resolution.
        # `p5` is `c5` after `fpn_lateral5` (channels reduced to `fpn_out_channels`).
        # `p4` is `1/16` resolution.
        
        # We will feed `p5` as the initial feature for the decoder's highest stage.
        # Then concatenate with `p4`, `p3`, `p2`, `p1` for skip connections.
        
        # Decoder from p5 (lowest resolution FPN feature)
        current_decoder_features = p5 # (B, fpn_out_channels, H/32, W/32) or (H/16, W/16)

        # d4 output, from p5 and p4
        d4 = F.interpolate(current_decoder_features, size=p4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, p4], dim=1) # (B, fpn_out_channels * 2, H/16, W/16)
        d4_out = self.decoder4(d4) # (B, fpn_out_channels, H/16, W/16)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=input_size, mode='bilinear', align_corners=True))
        
        # d3 output, from d4_out and p3
        d3 = F.interpolate(d4_out, size=p3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, p3], dim=1) # (B, fpn_out_channels * 2, H/8, W/8)
        d3_out = self.decoder3(d3) # (B, fpn_out_channels, H/8, W/8)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=input_size, mode='bilinear', align_corners=True))

        # d2 output, from d3_out and p2
        d2 = F.interpolate(d3_out, size=p2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, p2], dim=1) # (B, fpn_out_channels * 2, H/4, W/4)
        d2_out = self.decoder2(d2) # (B, fpn_out_channels, H/4, W/4)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=input_size, mode='bilinear', align_corners=True))

        # d1 output, from d2_out and p1 (final level, highest resolution)
        d1 = F.interpolate(d2_out, size=p1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, p1], dim=1) # (B, fpn_out_channels * 2, H, W)
        d1_out = self.decoder1(d1) # (B, fpn_out_channels, H, W)
        final_output = self.final_conv(d1_out)
        
        return tuple(aux_outputs + [final_output])

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps, aligning their spatial sizes."""
        # x is higher level FPN feature (smaller spatial size), y is lower level encoder feature (larger spatial size)
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

# Alias for backward compatibility if the original model name is referenced elsewhere
LASA_VGG_Unet = LASA_Unet 