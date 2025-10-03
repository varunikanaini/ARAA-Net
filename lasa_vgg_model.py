# /kaggle/working/ARAA-Net/enhanced_lasa_vgg_unet.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA # Your original LASA module

# --- New Lightweight Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    """ A very lightweight channel attention block. """
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

# --- The Main Enhanced Model ---
class Enhanced_LASA_VGG_UNet(nn.Module):
    """
    The enhanced U-Net using a VGG16 backbone with three key improvements:
    1.  LASA module is strategically moved to an earlier layer (encoder3).
    2.  Lightweight SE blocks are added to each decoder stage for better feature fusion.
    3.  A robust 5-output deep supervision structure is maintained.
    """
    def __init__(self, num_classes=2):
        super(Enhanced_LASA_VGG_UNet, self).__init__()
        
        # --- 1. Load Pre-trained VGG16 Backbone ---
        vgg_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
        
        self.encoder1 = vgg_features[:6]      # Output channels: 64
        self.encoder2 = vgg_features[6:13]    # Output channels: 128
        self.encoder3 = vgg_features[13:23]   # Output channels: 256
        self.encoder4 = vgg_features[23:33]   # Output channels: 512
        self.bottleneck = vgg_features[33:43] # Output channels: 512

        # --- 2. Strategically Placed LASA Module ---
        # Moved to encoder3 output (256 channels) for better focus on mid-level features.
        self.lasa_module = LASA(in_channels=256)

        # --- 3. U-Net Decoder with SE Blocks ---
        self.decoder4 = self._decoder_block(512 + 512, 512) # in: bottleneck + e4
        self.decoder3 = self._decoder_block(512 + 256, 256) # in: up(d4) + e3_enhanced
        self.decoder2 = self._decoder_block(256 + 128, 128) # in: up(d3) + e2
        self.decoder1 = self._decoder_block(128 + 64, 64)   # in: up(d2) + e1
        
        # --- 4. Deep Supervision Heads ---
        # 4 auxiliary heads + 1 final output head
        self.ds_out4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.ds_out3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.ds_out2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.ds_out1 = nn.Conv2d(64, num_classes, kernel_size=1) # Aux head from final decoder stage
        
        # The main final output convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        """ Decoder block now includes an SEBlock for smarter feature fusion. """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(channel=out_channels) # <<< Lightweight enhancement
        )

    def forward(self, x):
        input_h, input_w = x.shape[2:] 

        # --- Encoder Path ---
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Apply LASA enhancement to mid-level features
        e3_enhanced = self.lasa_module(e3)
        
        e4 = self.encoder4(e3_enhanced) 
        bottleneck = self.bottleneck(e4)

        # --- Decoder Path with Deep Supervision ---
        aux_outputs = []

        d4 = F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4_out = self.decoder4(d4)
        aux_outputs.append(F.interpolate(self.ds_out4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d3 = F.interpolate(d4_out, size=e3_enhanced.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3_enhanced], dim=1)
        d3_out = self.decoder3(d3)
        aux_outputs.append(F.interpolate(self.ds_out3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2 = F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2_out = self.decoder2(d2)
        aux_outputs.append(F.interpolate(self.ds_out2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        d1 = F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1_out = self.decoder1(d1)
        aux_outputs.append(F.interpolate(self.ds_out1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))
        
        # Final, primary output
        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)
        
        # Return all 5 outputs for deep supervision loss calculation
        return tuple(aux_outputs + [final_output_upsampled])