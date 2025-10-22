import torch
import torch.nn as nn
import torchvision.models as models

class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels
        
        self.encoder = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').features
        self.encoder_channels = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        
        self.decoder = nn.ModuleList()
        self.boundary_heads = nn.ModuleList()
        self.center_heads = nn.ModuleList()
        
        for i in range(4):
            in_channels = self.encoder_channels[-i-2] if i < 3 else self.encoder_channels[-1]
            out_channels = self.encoder_channels[-i-3] if i < 3 else 32
            self.decoder.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.boundary_heads.append(nn.Conv2d(out_channels, num_classes, kernel_size=1))
            self.center_heads.append(nn.Conv2d(out_channels, 1, kernel_size=1))

        self.final_seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final_center_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 6, 13, 17]:
                features.append(x)
        
        outputs = []
        for i in range(4):
            x = self.decoder[i](features[-i-2] if i < 3 else features[-1])
            seg_out = self.boundary_heads[i](x)
            center_out = self.center_heads[i](x)
            outputs.extend([seg_out, seg_out, center_out])
        
        x = self.decoder[-1](features[0])
        final_seg = self.final_seg_head(x)
        final_center = self.final_center_head(x)
        outputs.extend([final_seg, final_center])
        
        return outputs