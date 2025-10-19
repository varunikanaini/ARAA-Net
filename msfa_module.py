# /kaggle/working/ARAA-Net/msfa_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSFA_Module(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MSFA_Module, self).__init__()
        total_in_channels = sum(in_channels_list)
        
        # A refinement block to process the aggregated features from all scales
        self.refinement_block = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_features):
        e1, e2, e3, e4 = encoder_features
        target_size = e4.shape[2:] # Target size is the size of the last feature map

        # Pool all features to the common target size
        pooled_e1 = F.adaptive_avg_pool2d(e1, target_size)
        pooled_e2 = F.adaptive_avg_pool2d(e2, target_size)
        pooled_e3 = F.adaptive_avg_pool2d(e3, target_size)
        
        # Concatenate along the channel dimension
        concatenated_features = torch.cat([pooled_e1, pooled_e2, pooled_e3, e4], dim=1)
        
        # Refine the aggregated features
        refined_features = self.refinement_block(concatenated_features)
        
        return refined_features