# /kaggle/working/ARAA-Net/frm_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureRefinementModule(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size=(14, 14)):
        """
        A module inspired by the FRM paper to aggregate multi-stage features.

        Args:
            in_channels_list (list of int): A list of the number of channels for each encoder stage (e.g., [64, 128, 256, 512]).
            out_channels (int): The number of output channels for the refined features.
            target_size (tuple): The uniform size to which all feature maps will be pooled.
        """
        super(FeatureRefinementModule, self).__init__()
        self.target_size = target_size
        
        # Total number of channels after concatenating all pooled features
        total_in_channels = sum(in_channels_list)
        
        # A refinement block to process the aggregated features
        self.refinement_block = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_features):
        # encoder_features should be a list or tuple of tensors [e1, e2, e3, e4]
        
        # Pool each feature map to the target size
        pooled_features = [
            F.adaptive_avg_pool2d(feat, self.target_size) for feat in encoder_features
        ]
        
        # Concatenate along the channel dimension
        concatenated_features = torch.cat(pooled_features, dim=1)
        
        # Refine the aggregated features
        refined_features = self.refinement_block(concatenated_features)
        
        return refined_features