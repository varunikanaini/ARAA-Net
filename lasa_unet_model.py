import torch
import torch.nn as nn
import torchvision.models as models

class LASAUNet(nn.Module):
    def __init__(self, backbone='vgg16', num_classes=2, lasa_kernels=[1, 3, 5, 7]):
        super(LASAUNet, self).__init__()
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels
        
        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=True)
        self.backbone = vgg.features
        
        # Encoder feature channels
        self.encoder_channels = [64, 128, 256, 512, 512]
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.encoder_channels[-i-1] + self.encoder_channels[-i-2], self.encoder_channels[-i-2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2),  # Added dropout
                nn.Conv2d(self.encoder_channels[-i-2], self.encoder_channels[-i-2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2)  # Added dropout
            ) for i in range(1, len(self.encoder_channels))
        ])
        
        # Upsampling
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(self.encoder_channels[-i-1], self.encoder_channels[-i-2], 2, stride=2)
            for i in range(1, len(self.encoder_channels))
        ])
        
        # LASA block (simplified)
        self.lasa = nn.ModuleList([
            nn.Conv2d(self.encoder_channels[i], num_classes, kernel_size=k, padding=k//2)
            for i in range(len(self.encoder_channels))
            for k in lasa_kernels
        ])
        
        # Final conv
        self.final_conv = nn.Conv2d(self.encoder_channels[0], num_classes, 1)
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:  # VGG16 feature map indices
                encoder_features.append(x)
        
        # Decoder
        outputs = []
        for i in range(len(self.decoder)):
            x = self.upsample[i](encoder_features[-i-1])
            x = torch.cat([x, encoder_features[-i-2]], dim=1)
            x = self.decoder[i](x)
            # Apply LASA convolutions
            for j in range(len(self.lasa_kernels)):
                lasa_out = self.lasa[i * len(self.lasa_kernels) + j](x)
                outputs.append(lasa_out)
        
        x = self.final_conv(x)
        outputs.append(x)
        
        return outputs

def get_model(backbone='vgg16', num_classes=2, lasa_kernels=[1, 3, 5, 7]):
    return LASAUNet(backbone=backbone, num_classes=num_classes, lasa_kernels=lasa_kernels)