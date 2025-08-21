#!/usr/bin/env python3
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- Helper Modules ---

class CA_Block(nn.Module):
    """Channel Attention Block"""
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)

        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class SA_Block(nn.Module):
    """Spatial Attention Block"""
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.channels_single = input_channels // 4
        
        reductions = []
        for _ in range(4):
            reductions.append(nn.Sequential(
                nn.Conv2d(input_channels, self.channels_single, 1),
                nn.BatchNorm2d(self.channels_single), nn.ReLU(inplace=True)
            ))
        self.channel_reductions = nn.ModuleList(reductions)

        dilations = [1, 2, 4, 8]
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels_single, self.channels_single, 3, padding=d, dilation=d),
                nn.BatchNorm2d(self.channels_single), nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outs = []
        for i in range(4):
            feat = self.channel_reductions[i](x)
            outs.append(self.dilated_convs[i](feat))
        
        ce = self.fusion(torch.cat(outs, 1))
        return ce

class Positioning(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cab = CA_Block(channel)
        self.sab = SA_Block(channel)
        self.map = nn.Conv2d(channel, 1, 3, 1, 1)

    def forward(self, x):
        x = self.cab(x)
        x = self.sab(x)
        map_out = self.map(x)
        return x, map_out

class Focus(nn.Module):
    def __init__(self, channel1, channel2, is_last=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channel2, channel1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel1), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        
        output_channels = 2 if is_last else 1
        self.output_map = nn.Conv2d(channel1, output_channels, 3, 1, 1)

        self.fp = Context_Exploration_Block(channel1)
        self.fn = Context_Exploration_Block(channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, y, in_map):
        up_y = self.up(y)
        input_map_scaled = self.input_map(in_map)
        
        fp = self.fp(x * input_map_scaled)
        fn = self.fn(x * (1 - input_map_scaled))

        refine = up_y - (self.alpha * fp) + (self.beta * fn)
        output_map = self.output_map(refine)

        return refine, output_map

# --- Main Model ---

class ARAA_Net(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        if backbone_name not in ['resnet50', 'resnet101', 'vgg16', 'inception_v3']:
            raise ValueError(f"Backbone '{backbone_name}' is not supported.")

        self.backbone_name = backbone_name
        self.decoder_channels = [2048, 1024, 512, 256, 64]
        
        self._load_backbone(pretrained)

        self.positioning = Positioning(self.decoder_channels[0])
        self.focus3 = Focus(self.decoder_channels[1], self.decoder_channels[0])
        self.focus2 = Focus(self.decoder_channels[2], self.decoder_channels[1])
        self.focus1 = Focus(self.decoder_channels[3], self.decoder_channels[2])
        self.focus0 = Focus(self.decoder_channels[4], self.decoder_channels[3], is_last=True)

    def _load_backbone(self, pretrained):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None

        if self.backbone_name in ['resnet50', 'resnet101']:
            if self.backbone_name == 'resnet101':
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                resnet = models.resnet101(weights=weights)
            else:
                resnet = models.resnet50(weights=weights)
            
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(5)])

        elif self.backbone_name == 'vgg16':
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            vgg = models.vgg16(weights=weights).features
            self.layer0 = vgg[:4]    # 64 channels
            self.layer1 = vgg[4:9]   # 128 channels
            self.layer2 = vgg[9:16]  # 256 channels
            self.layer3 = vgg[16:23] # 512 channels
            self.layer4 = vgg[23:]   # 512 channels
            
            self.adapters = nn.ModuleList([
                nn.Conv2d(512, 2048, 1), nn.Conv2d(512, 1024, 1),
                nn.Conv2d(256, 512, 1), nn.Conv2d(128, 256, 1), nn.Identity()
            ])

        elif self.backbone_name == 'inception_v3':
            # Note: InceptionV3 expects input size of (3, 299, 299)
            weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
            inception = models.inception_v3(weights=weights, aux_logits=False)
            self.layer0 = nn.Sequential(inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(3, 2))
            self.layer1 = nn.Sequential(inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(3, 2))
            self.layer2 = nn.Sequential(inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d)
            self.layer3 = nn.Sequential(inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e)
            self.layer4 = nn.Sequential(inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c)

            self.adapters = nn.ModuleList([
                nn.Identity(), nn.Conv2d(768, 1024, 1),
                nn.Conv2d(288, 512, 1), nn.Conv2d(192, 256, 1), nn.Identity()
            ])

    def forward(self, x):
        if self.backbone_name == 'inception_v3' and self.training:
             x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        cr4 = self.adapters[0](l4) if not isinstance(self.adapters[0], nn.Identity) else l4
        cr3 = self.adapters[1](l3) if not isinstance(self.adapters[1], nn.Identity) else l3
        cr2 = self.adapters[2](l2) if not isinstance(self.adapters[2], nn.Identity) else l2
        cr1 = self.adapters[3](l1) if not isinstance(self.adapters[3], nn.Identity) else l1
        cr0 = self.adapters[4](l0) if not isinstance(self.adapters[4], nn.Identity) else l0
        
        pos_feat, pred4 = self.positioning(cr4)
        f3_feat, pred3 = self.focus3(cr3, pos_feat, pred4)
        f2_feat, pred2 = self.focus2(cr2, f3_feat, pred3)
        f1_feat, pred1 = self.focus1(cr1, f2_feat, pred2)
        _, pred0 = self.focus0(cr0, f1_feat, pred1)

        pred4 = F.interpolate(pred4, size=x.size()[2:], mode='bilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=x.size()[2:], mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, size=x.size()[2:], mode='bilinear', align_corners=True)
        pred0 = F.interpolate(pred0, size=x.size()[2:], mode='bilinear', align_corners=True)

        return pred4, pred3, pred2, pred1, pred0