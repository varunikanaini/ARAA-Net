# /kaggle/working/ARAA-Net/custom_losses.py (NEW FILE)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
#      FOCAL LOSS CLASS
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        inputs = inputs.to(targets.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            # Ensure mask is on the same device as focal_loss
            mask = mask.to(focal_loss.device)
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===================================================================
#      DICE LOSS CLASS
# ===================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        inputs = inputs.to(targets.device)
        targets = targets.to(inputs.device) # Ensure targets are on the same device as inputs

        num_classes = inputs.shape[1]
        
        if num_classes > 1:
            pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1) 
            true_oh = (targets == 1).float().unsqueeze(1) 
        else: 
            pred_probs = F.sigmoid(inputs)
            true_oh = targets.float().unsqueeze(1) 

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            mask = mask.to(pred_probs.device)
            pred_probs = pred_probs * mask.unsqueeze(1)
            true_oh = true_oh * mask.unsqueeze(1)
        
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] 
        else:
            return loss 