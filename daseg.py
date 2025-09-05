# /kaggle/working/ARAA-Net/daseg.py (FINAL INTEGRATED VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Import our custom modules
from lasa import LASA
from mfr import MultiscaleFeatureRefinement
from dar import DARConv2d # Assuming dar.py is in the same directory

# The get_backbone function remains unchanged, supporting multiple architectures
def get_backbone(backbone_name):
    if backbone_name == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        return {'layer0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu), 'layer1': nn.Sequential(resnet.maxpool, resnet.layer1), 'layer2': resnet.layer2, 'layer3': resnet.layer3, 'layer4': resnet.layer4, 'channels': {'c0': 64, 'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}}
    elif backbone_name == 'resnet101':
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        return {'layer0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu), 'layer1': nn.Sequential(resnet.maxpool, resnet.layer1), 'layer2': resnet.layer2, 'layer3': resnet.layer3, 'layer4': resnet.layer4, 'channels': {'c0': 64, 'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}}
    # Add other backbones like vgg16, inception_v3 as needed
    else: raise NotImplementedError(f"Backbone '{backbone_name}' not supported.")

# Original ARAA-Net components (CA_Block, SA_Block, etc.) are preserved
class CA_Block(nn.Module): # Compacted for brevity
    def __init__(self, in_dim): super(CA_Block, self).__init__(); self.gamma = nn.Parameter(torch.zeros(1)); self.softmax = nn.Softmax(dim=-1); self.query_conv = nn.Conv2d(in_dim, in_dim, 1); self.key_conv = nn.Conv2d(in_dim, in_dim, 1); self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
    def forward(self, x): m_batchsize, C, H, W = x.size(); proj_query = self.query_conv(x).view(m_batchsize, C, -1); proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1); energy = torch.bmm(proj_query, proj_key); attention = self.softmax(energy); proj_value = self.value_conv(x).view(m_batchsize, C, -1); out = torch.bmm(attention, proj_value).view(m_batchsize, C, H, W); return self.gamma * out + x

class SA_Block(nn.Module): # Compacted for brevity
    def __init__(self, in_dim): super(SA_Block, self).__init__(); self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1); self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1); self.value_conv = nn.Conv2d(in_dim, in_dim, 1); self.gamma = nn.Parameter(torch.zeros(1)); self.softmax = nn.Softmax(dim=-1)
    def forward(self, x): m_batchsize, C, H, W = x.size(); proj_query = self.query_conv(x).view(m_batchsize, -1, W * H).permute(0, 2, 1); proj_key = self.key_conv(x).view(m_batchsize, -1, W * H); energy = torch.bmm(proj_query, proj_key); attention = self.softmax(energy); proj_value = self.value_conv(x).view(m_batchsize, -1, W * H); out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(m_batchsize, C, H, W); return self.gamma * out + x

class Context_Exploration_Block(nn.Module): # Compacted for brevity
    def __init__(self, c): super(Context_Exploration_Block, self).__init__(); cs = c // 4; self.p1_cr = nn.Sequential(nn.Conv2d(c, cs, 1), nn.BatchNorm2d(cs), nn.ReLU()); self.p2_cr = nn.Sequential(nn.Conv2d(c, cs, 1), nn.BatchNorm2d(cs), nn.ReLU()); self.p3_cr = nn.Sequential(nn.Conv2d(c, cs, 1), nn.BatchNorm2d(cs), nn.ReLU()); self.p4_cr = nn.Sequential(nn.Conv2d(c, cs, 1), nn.BatchNorm2d(cs), nn.ReLU()); self.p1_dc = nn.Sequential(nn.Conv2d(cs, cs, 3, padding=1, dilation=1), nn.BatchNorm2d(cs), nn.ReLU()); self.p2_dc = nn.Sequential(nn.Conv2d(cs, cs, 3, padding=2, dilation=2), nn.BatchNorm2d(cs), nn.ReLU()); self.p3_dc = nn.Sequential(nn.Conv2d(cs, cs, 3, padding=4, dilation=4), nn.BatchNorm2d(cs), nn.ReLU()); self.p4_dc = nn.Sequential(nn.Conv2d(cs, cs, 3, padding=8, dilation=8), nn.BatchNorm2d(cs), nn.ReLU()); self.fusion = nn.Sequential(nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU())
    def forward(self, x): p1_in = self.p1_cr(x); p1 = self.p1_dc(p1_in); p2_in = self.p2_cr(x) + p1; p2 = self.p2_dc(p2_in); p3_in = self.p3_cr(x) + p2; p3 = self.p3_dc(p3_in); p4_in = self.p4_cr(x) + p3; p4 = self.p4_dc(p4_in); return self.fusion(torch.cat((p1, p2, p3, p4), 1))

class Positioning(nn.Module): # Compacted for brevity
    def __init__(self, c): super(Positioning, self).__init__(); self.cab = CA_Block(c); self.sab = SA_Block(c); self.map = nn.Conv2d(c, 1, 3, 1, 1)
    def forward(self, x): sab_out = self.sab(self.cab(x)); map_out = self.map(sab_out); return sab_out, map_out

class Focus(nn.Module): # Compacted for brevity
    def __init__(self, c1, c2, is_last=False): super(Focus, self).__init__(); self.up = nn.Sequential(nn.Conv2d(c2, c1, 3, 1, 1), nn.BatchNorm2d(c1), nn.ReLU()); self.map_proc = nn.Sigmoid(); self.out_map = nn.Conv2d(c1, 2 if is_last else 1, 3, 1, 1); self.fp = Context_Exploration_Block(c1); self.fn = Context_Exploration_Block(c1); self.alpha = nn.Parameter(torch.ones(1)); self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x, y, in_map): up_y = self.up(F.interpolate(y, x.size()[2:], mode='bilinear')); map_s = self.map_proc(F.interpolate(in_map, x.size()[2:], mode='bilinear')); fp = self.fp(x * map_s); fn = self.fn(x * (1 - map_s)); refine = up_y - self.alpha * fp + self.beta * fn; return refine, self.out_map(refine)


class daseg(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(daseg, self).__init__()
        backbone_data = get_backbone(backbone_name)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = \
            backbone_data['layer0'], backbone_data['layer1'], backbone_data['layer2'], \
            backbone_data['layer3'], backbone_data['layer4']
        ch = backbone_data['channels']

        # --- INTEGRATION 1: Instantiate LASA modules for early feature enhancement ---
        self.lasa1 = LASA(in_channels=ch['c1'])
        self.lasa2 = LASA(in_channels=ch['c2'])

        # --- INTEGRATION 2: Instantiate the new MFR module ---
        self.mfr_module = MultiscaleFeatureRefinement(c2_in=ch['c2'], c3_in=ch['c3'], c4_in=ch['c4'])

        # Original ARAA-Net decoder components
        self.positioning = Positioning(ch['c4'])
        self.focus3 = Focus(ch['c3'], ch['c4'])
        self.focus2 = Focus(ch['c2'], ch['c3'])
        self.focus1 = Focus(ch['c1'], ch['c2'])
        self.focus0 = Focus(ch['c0'], ch['c1'], is_last=True)

    def forward(self, x):
        original_size = x.size()[2:]

        # --- Encoder Path ---
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        
        # --- Stage 1 Refinement: Apply LASA to early layers ---
        l1_lasa = self.lasa1(l1)
        l2 = self.layer2(l1_lasa)
        l2_lasa = self.lasa2(l2)
        
        l3 = self.layer3(l2_lasa)
        l4 = self.layer4(l3)
        
        # Get deepest feature context
        pos_feat, pred4 = self.positioning(l4)

        # --- Stage 2 Refinement: Apply MFR in a top-down manner ---
        # It refines the LASA-enhanced features using context from the deepest layer.
        l2_refined, l3_refined = self.mfr_module(l2_lasa, l3, pos_feat)

        # --- Decoder Path: Use the refined features ---
        f3_feat, pred3 = self.focus3(l3_refined, pos_feat, pred4)
        f2_feat, pred2 = self.focus2(l2_refined, f3_feat, pred3)
        f1_feat, pred1 = self.focus1(l1_lasa, f2_feat, pred2)
        _, pred0 = self.focus0(l0, f1_feat, pred1)
        
        # Upsample all predictions to the original size
        preds = [pred4, pred3, pred2, pred1, pred0]
        for i in range(len(preds)):
            preds[i] = F.interpolate(preds[i], size=original_size, mode='bilinear', align_corners=True)
            
        return preds[0], preds[1], preds[2], preds[3], preds[4]