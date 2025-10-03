# /kaggle/working/ARAA-Net/seg_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""

import numpy as np
import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.dice = [] # Store dice scores for each batch

    def update(self, a, b):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
            
            # Ensure inputs are on the same device and have correct types
            a = a.long().to(self.mat.device)
            b = b.long().to(self.mat.device)

            # Compute confusion matrix update
            k = (a >= 0) & (a < n)
            inds = n * a[k] + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
            
            # Calculate Dice score for this batch
            # Assuming binary segmentation (class 1 vs background class 0) for Dice calculation
            # If multi-class dice is needed, this part requires adjustment.
            # For now, we assume we are interested in the foreground Dice.
            if n > 1:
                # Extract predicted and target for class 1 (foreground)
                # If targets can be multi-class, this needs more careful handling.
                # Assuming binary target for simplicity here as well.
                target_binary = (a == 1).float()
                pred_binary = (b == 1).float()
            else: # Binary case already
                target_binary = a.float()
                pred_binary = b.float()

            overlap = (pred_binary * target_binary).sum()
            dice = torch.clamp(((2. * overlap) / (target_binary.sum() + pred_binary.sum() + 1e-6)), 1e-4, 0.9999)
            self.dice.append(dice.item()) # Append scalar item to list

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()
        self.dice = [] # Clear dice scores

    def compute(self):
        with torch.no_grad():
            h = self.mat.float()
            
            # Global Accuracy
            acc_global = torch.diag(h).sum() / (h.sum() + 1e-6)

            # Class-wise Accuracy
            h_sum1 = h.sum(1)
            acc = torch.diag(h) / (h_sum1 + 1e-6)
            
            # Class-wise IoU (Intersection over Union)
            h_sum0 = h.sum(0)
            iu = torch.diag(h) / (h_sum1 + h_sum0 - torch.diag(h) + 1e-6)

            # Frequency weighted IoU
            freq = h_sum1 / (h.sum() + 1e-6)
            FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
            
            # Mean Dice (average of all batch dice scores)
            mDice = np.mean(self.dice) if self.dice else 0.0

            # Mean IoU (average of all class IoUs)
            mIoU = iu.mean().item() if not iu.nelement() == 0 else 0.0
            
        return acc_global, acc, iu, FWIoU, mDice, mIoU

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)
        # Reduce dice scores if needed (requires distributed averaging)
        # For now, assuming single GPU or manual aggregation of dice

    def __str__(self):
        acc_global, acc, iu, FWIoU, mDice, mIoU = self.compute()
        return (
            'Global Correct: {:.1f}%\n'
            'Class Accuracy: {}\n'
            'Class IoU: {}\n'
            'Mean IoU: {:.1f}%\n'
            'FWIoU: {:.1f}%\n'
            'Mean Dice: {:.1f}%').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                mIoU * 100,
                FWIoU.item() * 100,
                mDice * 100)


def blend_seg(img, seg, color_map=None, alpha=0.5, ignore_index=0):
    """ Blend images with their corresponding segmentation prediction.

    Args:
        img (torch.Tensor): A batch of image tensors of shape (B, 3, H, W) where B is the batch size,
            H is the images height and W is the images width
        seg (torch.Tensor): A batch of segmentation predictions of shape (B, C, H, W) where B is the batch size,
            C is the number of segmentation classes, H is the images height and W is the images width
        alpha: alpha (float): Opacity value for the segmentation in the range [0, 1] where 0 is completely transparent
            and 1 is completely opaque

    Returns:
        torch.Tensor: The blended image.
    """
    color_map_tensor = torch.from_numpy(np.array(color_map, dtype='float32')).to(img.device).div_(128.).sub_(1.)
    seg_classes = seg.argmax(1) if seg.dim() == 4 else seg.clone()
    seg_classes[seg_classes >= color_map_tensor.shape[0]] = ignore_index
    seg_rgb = color_map_tensor[seg_classes].permute(0, 3, 1, 2)
    alpha_mask = (1. - (seg_classes != ignore_index).float() * alpha).unsqueeze(1).repeat(1, 3, 1, 1)

    return img * alpha_mask + seg_rgb * (1. - alpha_mask)
