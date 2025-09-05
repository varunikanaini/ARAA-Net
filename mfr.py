# /kaggle/working/ARAA-Net/mfr.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleFeatureRefinement(nn.Module):
    """
    Implements a top-down feature refinement path with spatial gating.
    Refines deeper encoder features (l2, l3) using context from the deepest layer (l4).
    """
    def __init__(self, c2_in, c3_in, c4_in, dropout_rate=0.2):
        super(MultiscaleFeatureRefinement, self).__init__()

        # Path from l4's features (pos_feat) to l3
        self.upsample_c4_to_c3 = nn.Sequential(
            nn.Conv2d(c4_in, c3_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(c3_in),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention_l3 = nn.Sequential(
            nn.Conv2d(c3_in * 2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.l3_refinement = nn.Sequential(
            nn.Conv2d(c3_in * 2, c3_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3_in), nn.ReLU(inplace=True),
            nn.Conv2d(c3_in, c3_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3_in), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Path from the newly refined l3 to l2
        self.upsample_c3_to_c2 = nn.Sequential(
            nn.Conv2d(c3_in, c2_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2_in),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention_l2 = nn.Sequential(
            nn.Conv2d(c2_in * 2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.l2_refinement = nn.Sequential(
            nn.Conv2d(c2_in * 2, c2_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2_in), nn.ReLU(inplace=True),
            nn.Conv2d(c2_in, c2_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2_in), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, l2_in, l3_in, pos_feat_in):
        # Refine l3 using pos_feat (from l4)
        pos_feat_upsampled_to_l3 = F.interpolate(pos_feat_in, size=l3_in.size()[2:], mode='bilinear', align_corners=True)
        pos_feat_upsampled_to_l3 = self.upsample_c4_to_c3(pos_feat_upsampled_to_l3)
        
        l3_fused_for_attention = torch.cat([l3_in, pos_feat_upsampled_to_l3], dim=1)
        attention_map_l3 = self.spatial_attention_l3(l3_fused_for_attention)
        
        l3_in_attended = l3_in * attention_map_l3
        l3_fused = torch.cat([l3_in_attended, pos_feat_upsampled_to_l3], dim=1)
        l3_refined = self.l3_refinement(l3_fused)

        # Refine l2 using the already refined l3
        l3_refined_upsampled_to_l2 = F.interpolate(l3_refined, size=l2_in.size()[2:], mode='bilinear', align_corners=True)
        l3_refined_upsampled_to_l2 = self.upsample_c3_to_c2(l3_refined_upsampled_to_l2)
        
        l2_fused_for_attention = torch.cat([l2_in, l3_refined_upsampled_to_l2], dim=1)
        attention_map_l2 = self.spatial_attention_l2(l2_fused_for_attention)
        
        l2_in_attended = l2_in * attention_map_l2
        l2_fused = torch.cat([l2_in_attended, l3_refined_upsampled_to_l2], dim=1)
        l2_refined = self.l2_refinement(l2_fused)

        return l2_refined, l3_refined