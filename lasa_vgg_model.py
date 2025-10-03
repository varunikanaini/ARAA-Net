# /kaggle/working/ARAA-Net/lasa_vgg_model.py
# ### THIS IS THE CORRECTED AND ENHANCED FILE ###
# It keeps the original class name "LASA_Unet" and enhances the bottleneck with ASPP.

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # Your original LASA module remains unchanged

# --- Atrous Spatial Pyramid Pooling (ASPP) Module ---
class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(in_channels, out_channels, 1, 0, 1)
        self.aspp2 = _ASPPModule(in_channels, out_channels, 3, dilations[1], dilations[1])
        self.aspp3 = _ASPPModule(in_channels, out_channels, 3, dilations[2], dilations[2])
        self.aspp4 = _ASPPModule(in_channels, out_channels, 3, dilations[3], dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d((len(dilations) + 1) * out_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1, x2, x3, x4 = self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv1(x_cat)
        out = self.bn1(out)
        out = self.relu(out)
        return self.dropout(out)

# --- The Main Model, corrected and enhanced but with the ORIGINAL NAME ---
class LASA_Unet(nn.Module):
    """
    This is your original LASA_Unet, enhanced with an ASPP bottleneck.
    The filename and class name are unchanged to ensure compatibility.
    """
    def __init__(self, num_classes=2, backbone_name='vgg16'):
        super(LASA_Unet, self).__init__()
        if backbone_name != 'vgg16':
            raise NotImplementedError("This model has been fixed and optimized for VGG16 only.")

        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
        
        # Encoder blocks from your original model
        self.encoder1 = vgg_features[:6]
        self.encoder2 = vgg_features[6:13]
        self.encoder3 = vgg_features[13:23]
        self.encoder4 = vgg_features[23:33]
        
        # LASA module from your original model
        self.lasa_module = LASA(in_channels=512)

        # ### ENHANCEMENT: Replace the bottleneck ###
        # Original bottleneck is removed: vgg_features[33:43]
        self.aspp = ASPP(in_channels=512, out_channels=256)

        # Decoder blocks (channels adjusted for ASPP's output)
        self.decoder4 = self._decoder_block(256 + 512, 512) # in: aspp_out (256) + e4 (512)
        self.decoder3 = self._decoder_block(512 + 256, 256) # in: up(d4) + e3
        self.decoder2 = self._decoder_block(256 + 128, 128) # in: up(d3) + e2
        self.decoder1 = self._decoder_block(128 + 64, 64)   # in: up(d2) + e1
        
        # Deep Supervision heads from your original model
        self.aux_conv_d4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.aux_conv_d3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux_conv_d2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux_conv_d1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

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

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3) 
        e4_enhanced = self.lasa_module(e4)
        
        # Use ASPP as the bottleneck
        bottleneck = self.aspp(e4_enhanced)

        aux_outputs = []

        d4 = F.interpolate(bottleneck, size=e4_enhanced.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4_enhanced], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d3 = F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # The model returns 5 outputs for deep supervision
        return tuple(aux_outputs + [final_output_upsampled])