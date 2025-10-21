# /kaggle/working/ARAA-Net/boundary_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoundaryLoss(nn.Module):
    def __init__(self, device, alpha=2.0, gamma=0.5):
        super(BoundaryLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma

    def get_gradient(self, pred_mask): # Expects pred_mask to be Bx1xHxW
        """Computes the gradient of a binary mask using Sobel filters."""
        # Ensure pred_mask is float and on the correct device
        pred_tensor = pred_mask.float().to(self.device)
        
        # Get original spatial dimensions BEFORE any potential padding issues
        batch_size, channels, height, width = pred_tensor.shape

        # Define kernels as float32 tensors on the correct device
        # Kernels are 3x3, padding=1 ensures spatial dimensions are preserved for conv2d
        dx_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Calculate horizontal gradient (dx)
        grad_x = torch.abs(F.conv2d(pred_tensor, dx_kernel, padding=1))
        
        # Calculate vertical gradient (dy)
        grad_y = torch.abs(F.conv2d(pred_tensor, dy_kernel, padding=1))

        # Stack gradients: results in shape (B, 2, H, W)
        gradients = torch.cat([grad_x, grad_y], dim=1)
        return gradients

    def forward(self, pred, target):
        """
        Computes the Boundary Loss.
        Args:
            pred (torch.Tensor): Predicted segmentation mask (probabilities, shape BxCxHxW).
            target (torch.Tensor): Ground truth mask (class indices, shape BxHxW).
        Returns:
            torch.Tensor: The computed Boundary Loss.
        """
        # --- Prepare Prediction for Gradient Calculation ---
        # Ensure pred is Bx1xHxW. If it's BxCxHxW (e.g., num_classes=2), take the foreground channel.
        if pred.shape[1] > 1: # If it's multi-class output
            pred_prob = F.softmax(pred, dim=1)
            # Ensure we correctly select the foreground channel (assuming class 1 is foreground)
            # Use sum across channels if num_classes > 2 and you want any foreground gradient
            # For binary (num_classes=2), pred_prob[:, 1, :, :] is correct.
            if pred_prob.shape[1] >= 2:
                 pred_for_grad = pred_prob[:, 1, :, :].unsqueeze(1) # Shape Bx1xHxW
            else: # Handle case where only background channel might be present
                 pred_for_grad = torch.zeros(pred.shape[0], 1, pred.shape[2], pred.shape[3], device=self.device)
        else: # If pred is already single-channel probabilities (Bx1xHxW)
            pred_for_grad = pred.float().to(self.device)

        # --- Prepare Target for Gradient Calculation ---
        # Ensure target is Bx1xHxW with binary values (0.0 or 1.0)
        if target.ndim == 3 and target.shape[1] == 1: # Already Bx1xHxW
            target_for_grad = target.float().to(self.device)
        else: # Assuming target is BxHxW with class indices
            # Create binary mask where target > 0 is foreground (1.0)
            target_binary = (target > 0).float().to(self.device) # Ensure it's float
            target_for_grad = target_binary.unsqueeze(1) # Shape Bx1xHxW

        # --- Compute Gradients ---
        # Ensure both inputs to get_gradient have the same shape Bx1xHxW
        target_gradients = self.get_gradient(target_for_grad) 
        pred_gradients = self.get_gradient(pred_for_grad)      

        # --- Calculate Loss ---
        # Ensure spatial dimensions match before subtraction
        if target_gradients.shape != pred_gradients.shape:
            # This is a critical debugging point. If shapes mismatch,
            # it's likely due to how conv2d affects dimensions or mismatches in input.
            # With padding=1 and kernel 3x3, dimensions should be preserved.
            # If they don't match, it could be an issue with input size or kernel application.
            # For now, let's try to resize the larger one to match the smaller one if there's a mismatch.
            # (This is a workaround; ideally, dimensions should match naturally)
            
            h_pred, w_pred = pred_gradients.shape[2], pred_gradients.shape[3]
            h_target, w_target = target_gradients.shape[2], target_gradients.shape[3]

            if h_pred != h_target or w_pred != w_target:
                print(f"Warning: Spatial dimension mismatch in gradients. Pred: {pred_gradients.shape}, Target: {target_gradients.shape}")
                # Resize target gradients to match prediction gradients
                target_gradients = F.interpolate(target_gradients, size=(h_pred, w_pred), mode='bilinear', align_corners=False)
                
        # Now, pred_gradients and target_gradients should have compatible shapes
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss