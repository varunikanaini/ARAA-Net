#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2025-10-22
@author: XuWang, modified by Grok for boundary IoU
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.dice = []
        self.boundary_mat = None  # For boundary IoU

    def update(self, a, b, a_boundary=None, b_boundary=None):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
                self.boundary_mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)

            # Update standard confusion matrix
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

            # Update boundary confusion matrix
            if a_boundary is not None and b_boundary is not None:
                k_boundary = (a_boundary >= 0) & (a_boundary < n)
                inds_boundary = n * a_boundary[k_boundary].to(torch.int64) + b_boundary[k_boundary]
                self.boundary_mat += torch.bincount(inds_boundary, minlength=n**2).reshape(n, n)

            # Compute Dice
            input_flatten = b.cpu().numpy().flatten()
            target_flatten = a.cpu().numpy().flatten()
            overlap = np.sum(input_flatten * target_flatten)
            dice = np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + 1e-6)), 1e-4, 0.9999)
            self.dice.append(dice)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()
        if self.boundary_mat is not None:
            self.boundary_mat.zero_()
        self.dice = []

    def compute(self):
        with torch.no_grad():
            h = self.mat.float()
            acc_global = torch.diag(h).sum() / (h.sum() + 1e-6)
            h_sum1 = h.sum(1)
            acc = torch.diag(h) / (h_sum1 + 1e-6)
            iu = torch.diag(h) / (h_sum1 + h.sum(0) - torch.diag(h) + 1e-6)
            freq = h_sum1 / (h.sum() + 1e-6)
            FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
            mDice = np.mean(self.dice)

            # Boundary IoU
            boundary_iu = 0.0
            if self.boundary_mat is not None:
                h_boundary = self.boundary_mat.float()
                h_boundary_sum1 = h_boundary.sum(1)
                boundary_iu = torch.diag(h_boundary) / (h_boundary_sum1 + h_boundary.sum(0) - torch.diag(h_boundary) + 1e-6)
                boundary_iu = boundary_iu.mean().item()

        return acc_global, acc, iu, FWIoU, mDice, boundary_iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)
        if self.boundary_mat is not None:
            torch.distributed.all_reduce(self.boundary_mat)

    def __str__(self):
        acc_global, acc, iu, FWIoU, mDice, boundary_iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}\n'
            'FWIoU: {:.1f}\n'
            'mean Dice: {:.1f}\n'
            'boundary IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100,
                FWIoU.item() * 100,
                mDice * 100,
                boundary_iu * 100)

class IOUBenchmark(object):
    def __init__(self, num_classes=None):
        self.confmat = None if num_classes is None else ConfusionMatrix(num_classes)

    def reset(self):
        if self.confmat is not None:
            self.confmat.reset()

    def to(self, device):
        return self

    def __call__(self, pred, target, pred_boundary=None, target_boundary=None):
        if self.confmat is None:
            assert pred.dim() == 4, 'prediction must be of 4 dimensions if num_classes was not specified'
            self.confmat = ConfusionMatrix(pred.shape[1])
        self.confmat.update(target.flatten(), pred.argmax(1).flatten() if pred.dim() == 4 else pred.flatten(),
                           pred_boundary.flatten() if pred_boundary is not None else None,
                           target_boundary.flatten() if target_boundary is not None else None)
        acc_global, acc, iou, FWIoU, mDice, boundary_iu = self.confmat.compute()
        return {'iou': iou.mean().item(), 'boundary_iou': boundary_iu}

def get_boundary_mask(label, distance_threshold=3):
    """Compute binary boundary mask using distance transform."""
    label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
    boundary = np.zeros_like(label_np, dtype=np.uint8)
    for b in range(label_np.shape[0]):
        dist = distance_transform_edt(label_np[b] > 0)
        boundary[b] = (dist <= distance_threshold).astype(np.uint8)
    return torch.tensor(boundary, device=label.device) if isinstance(label, torch.Tensor) else boundary

def blend_seg(img, seg, color_map=None, alpha=0.5, ignore_index=0):
    color_map_tensor = torch.from_numpy(np.array(color_map, dtype='float32')).to(img.device).div_(128.).sub_(1.)
    seg_classes = seg.argmax(1) if seg.dim() == 4 else seg.clone()
    seg_classes[seg_classes >= color_map_tensor.shape[0]] = ignore_index
    seg_rgb = color_map_tensor[seg_classes].permute(0, 3, 1, 2)
    alpha_mask = (1. - (seg_classes != ignore_index).float() * alpha).unsqueeze(1).repeat(1, 3, 1, 1)
    return img * alpha_mask + seg_rgb * (1. - alpha_mask)