#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # For compatibility with older PyTorch versions or specific scenarios,
        # ensure 'reduce' is not used if 'none' is intended for reduction parameter.
        # F.binary_cross_entropy_with_logits already supports 'reduction'.
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6) # Added epsilon for stability

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter + 1e-6) # Added epsilon for stability
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


###################################################################
# ########################## Focal Loss ###########################
###################################################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are raw logits (N, C, H, W), targets are class indices (N, H, W)
        # First, compute the standard Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # For targets with ignore_index, ce_loss will be 0 if reduction='none' is used
        # and ignore_index is handled by F.cross_entropy.
        # However, to explicitly mask them for pt calculation:
        pt = torch.exp(-ce_loss) # Calculate probability of the correct class
        
        # If ignore_index is present, we must ensure it doesn't affect focal_loss.
        # F.cross_entropy with reduction='none' sets loss to 0 for ignore_index.
        # So, the (1-pt)**gamma factor will apply to these 0 losses, which is fine.
        
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            # Only average over non-ignored pixels
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index).float()
                return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss