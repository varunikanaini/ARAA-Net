# daseg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dar import DARConv2d # Assumes dar.py is in the same directory

# --- HELPER TO GET BACKBONE LAYERS (With InceptionV3) ---
def get_backbone(backbone_name):
    if backbone_name == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        return {
            'layer0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
            'layer1': nn.Sequential(resnet.maxpool, resnet.layer1),
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4,
            'channels': {'c0': 64, 'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}
        }
    elif backbone_name == 'resnet101':
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        return {
            'layer0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
            'layer1': nn.Sequential(resnet.maxpool, resnet.layer1),
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4,
            'channels': {'c0': 64, 'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}
        }
    elif backbone_name == 'vgg16':
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
        return {
            'layer0': vgg[:6],
            'layer1': vgg[6:13],
            'layer2': vgg[13:23],
            'layer3': vgg[23:33],
            'layer4': vgg[33:43],
            'channels': {'c0': 64, 'c1': 128, 'c2': 256, 'c3': 512, 'c4': 512}
        }
    elif backbone_name == 'inception_v3':
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.aux_logits = False # We don't need the auxiliary output
        return {
            'layer0': nn.Sequential(
                inception.Conv2d_1a_32,
                inception.Conv2d_2a_32,
                inception.Conv2d_2b_64,
            ),
            'layer1': nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_192,
            ),
            'layer2': nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
            ),
            'layer3': nn.Sequential(
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ),
            'layer4': nn.Sequential(
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ),
            'channels': {'c0': 64, 'c1': 192, 'c2': 288, 'c3': 768, 'c4': 2048}
        }
    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' not supported.")

# --- ORIGINAL HELPER MODULES (Unchanged) ---
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
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
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.p1_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8), nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0), nn.BatchNorm2d(self.input_channels), nn.ReLU())
    def forward(self, x):
        p1_input = self.p1_channel_reduction(x); p1_dc = self.p1_dc(p1_input)
        p2_input = self.p2_channel_reduction(x); p2_dc = self.p2_dc(p2_input)
        p3_input = self.p3_channel_reduction(x); p3_dc = self.p3_dc(p3_input)
        p4_input = self.p4_channel_reduction(x); p4_dc = self.p4_dc(p4_input)
        return self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.cab = CA_Block(channel); self.sab = SA_Block(channel); self.map = nn.Conv2d(channel, 1, 3, 1, 1)
    def forward(self, x):
        cab = self.cab(x); sab = self.sab(cab); map_out = self.map(sab); return sab, map_out

class Focus(nn.Module):
    def __init__(self, channel1, channel2, is_last=False):
        super(Focus, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(channel2, channel1, 3, 1, 1, bias=False), nn.BatchNorm2d(channel1), nn.ReLU(inplace=True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(channel1, 2 if is_last else 1, 3, 1, 1)
        self.fp = Context_Exploration_Block(channel1); self.fn = Context_Exploration_Block(channel1)
        self.alpha = nn.Parameter(torch.ones(1)); self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x, y, in_map):
        up_y = self.up(y); input_map_scaled = self.input_map(in_map)
        fp = self.fp(x * input_map_scaled); fn = self.fn(x * (1 - input_map_scaled))
        refine = up_y - (self.alpha * fp) + (self.beta * fn)
        output_map = self.output_map(refine)
        return refine, output_map

# --- MAIN DASEG CLASS ---
class daseg(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(daseg, self).__init__()
        
        backbone_data = get_backbone(backbone_name)
        
        self.layer0 = backbone_data['layer0']
        self.layer1 = backbone_data['layer1']
        self.layer2 = backbone_data['layer2']
        self.layer3 = backbone_data['layer3']
        self.layer4 = backbone_data['layer4']
        
        ch = backbone_data['channels']
        
        # Decoder layers
        self.positioning = Positioning(ch['c4'])
        self.focus3 = Focus(ch['c3'], ch['c4'])
        self.focus2 = Focus(ch['c2'], ch['c3'])
        self.focus1 = Focus(ch['c1'], ch['c2'])
        self.focus0 = Focus(ch['c0'], ch['c1'], is_last=True)

    def forward(self, x):
        # Backbone forward pass
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # Decoder forward pass
        pos_feat, pred4 = self.positioning(l4)
        f3_feat, pred3 = self.focus3(l3, pos_feat, pred4)
        f2_feat, pred2 = self.focus2(l2, f3_feat, pred3)
        f1_feat, pred1 = self.focus1(l1, f2_feat, pred2)
        _, pred0 = self.focus0(l0, f1_feat, pred1)

        # Upsample all predictions to original size
        original_size = x.size()[2:]
        pred4 = F.interpolate(pred4, size=original_size, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=original_size, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=original_size, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, size=original_size, mode='bilinear', align_corners=True)
        pred0 = F.interpolate(pred0, size=original_size, mode='bilinear', align_corners=True)

        return pred4, pred3, pred2, pred1, pred0