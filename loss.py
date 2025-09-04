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
        # Add a small epsilon to prevent division by zero
        iou = 1 - (inter + 1e-6) / (union + 1e-6)

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
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # The 'reduce' argument is deprecated. 'reduction' is the correct one.
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        # Add a small epsilon to prevent division by zero
        wiou = 1 - (inter + 1e-6) / (union - inter + 1e-6)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


# ===================================================================
#      ✅ NEWLY ADDED FOCAL LOSS CLASS ✅
# ===================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are raw logits (N, C, H, W)
        # targets are class indices (N, H, W)
        
        # Calculate the standard cross entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            # Only average over non-ignored pixels for a correct mean
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss
# ===================================================================