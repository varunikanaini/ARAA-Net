# /kaggle/working/ARAA-Net/lasa_unet_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA 

# --- New Lightweight Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LASA_UNet(nn.Module):
    """
    An enhanced segmentation model using a flexible backbone (VGG16 or ResNet50),
    the original LASA module for feature enhancement at a strategic layer,
    and a U-Net style decoder with SE blocks and true deep supervision.
    """
    def __init__(self, num_classes=2, backbone_name='resnet50'):
        super(LASA_UNet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # --- 1. Load Pre-trained Backbone ---
        if backbone_name == 'vgg16':
            features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            self.encoder1 = features[:6]      # 64 channels
            self.encoder2 = features[6:13]    # 128 channels
            self.encoder3 = features[13:23]   # 256 channels
            self.encoder4 = features[23:33]   # 512 channels
            self.bottleneck = features[33:43] # 512 channels
            
            # Channel dimensions for VGG
            (e1_c, e2_c, e3_c, e4_c, bn_c) = (64, 128, 256, 512, 512)

        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.encoder2 = resnet.layer1
            self.encoder3 = resnet.layer2
            self.encoder4 = resnet.layer3
            self.bottleneck = resnet.layer4
            
            # Channel dimensions for ResNet50
            (e1_c, e2_c, e3_c, e4_c, bn_c) = (64, 256, 512, 1024, 2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- 2. Strategically Placed LASA Module ---
        # Applied to encoder3 output based on paper's findings for better performance.
        self.lasa_module = LASA(in_channels=e3_c)

        # --- 3. U-Net Decoder with SE Blocks and Deep Supervision Heads ---
        self.decoder4 = self._decoder_block(bn_c + e4_c, e4_c)
        self.decoder3 = self._decoder_block(e4_c + e3_c, e3_c)
        self.decoder2 = self._decoder_block(e3_c + e2_c, e2_c)
        self.decoder1 = self._decoder_block(e2_c + e1_c, e1_c)
        
        # Auxiliary heads for deep supervision
        self.ds_out4 = nn.Conv2d(e4_c, num_classes, kernel_size=1)
        self.ds_out3 = nn.Conv2d(e3_c, num_classes, kernel_size=1)
        self.ds_out2 = nn.Conv2d(e2_c, num_classes, kernel_size=1)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(e1_c, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        # Decoder block now includes an SEBlock for channel-wise feature refinement
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(channel=out_channels) # <-- Added SE block
        )

    def forward(self, x):
        input_h, input_w = x.shape[2:] 

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Apply LASA enhancement to e3 (mid-level features)
        e3_enhanced = self.lasa_module(e3)
        
        e4 = self.encoder4(e3_enhanced) 
        bottleneck = self.bottleneck(e4)

        # --- Decoder Path with Skip Connections & Deep Supervision ---
        aux_outputs = []

        # Decoder 4
        d4 = F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.ds_out4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 3
        d3 = F.interpolate(d4_out, size=e3_enhanced.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3_enhanced], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.ds_out3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        # Decoder 2
        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.ds_out2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Decoder 1
        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        
        # Final Output
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # Return all decoder outputs for deep supervision loss calculation
        return tuple(aux_outputs + [final_output_upsampled])