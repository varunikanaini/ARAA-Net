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

        # Define kernels directly as torch tensors with float32 dtype
        # These kernels are used for Sobel operators (gradient approximation)
        dx_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Calculate horizontal gradient (dx)
        # Padding=1 ensures output spatial dimensions match input
        grad_x = torch.abs(F.conv2d(pred_tensor, dx_kernel, padding=1))
        
        # Calculate vertical gradient (dy)
        grad_y = torch.abs(F.conv2d(pred_tensor, dy_kernel, padding=1))

        # Stack gradients: results in shape (B, 2, H, W) where 2 is for dx and dy
        gradients = torch.cat([grad_x, grad_y], dim=1)
        return gradients

     def forward(self, pred, target):
        """
        Computes the Boundary Loss.
        Args:
            pred (torch.Tensor): Predicted segmentation mask (logits or probabilities, shape BxCxHxW).
            target (torch.Tensor): Ground truth mask (class indices, shape BxHxW).
        Returns:
            torch.Tensor: The computed Boundary Loss.
        """
        # --- Prepare Prediction for Gradient Calculation ---
        # Ensure pred is Bx1xHxW (single channel probability mask)
        if pred.shape[1] > 1:
            # If it's multi-class logits/probabilities (BxCxHxW), convert to probabilities and select foreground
            pred_prob = F.softmax(pred, dim=1)
            # Ensure we correctly select the foreground channel. Assuming class 1 is foreground.
            pred_for_grad = pred_prob[:, 1:, :, :].sum(dim=1, keepdim=True) # Sum all foreground channels if multi-class
            # If you only have binary, pred_prob[:, 1, :, :].unsqueeze(1) is fine,
            # but summing is safer if num_classes could be > 2 and you want *any* foreground.
            # For binary (num_classes=2), pred_prob[:, 1, :, :] is correct.
            # Let's use the binary assumption from num_classes=2 and pick channel 1.
            if pred_prob.shape[1] >= 2:
                 pred_for_grad = pred_prob[:, 1, :, :].unsqueeze(1) # Shape Bx1xHxW
            else: # If only one channel (e.g., background), handle appropriately or raise error
                 pred_for_grad = torch.zeros_like(pred_prob) # Or handle as background, if that's all that's predicted

        else:
            # If pred is already single-channel probabilities (Bx1xHxW), use it directly
            pred_for_grad = pred.float().to(self.device)

        # --- Prepare Target for Gradient Calculation ---
        # Ensure target is Bx1xHxW with binary values (0.0 or 1.0)
        if target.ndim == 3 and target.shape[1] == 1: # Already Bx1xHxW
            target_for_grad = target.float().to(self.device)
        else: # Assuming target is BxHxW
            # Create binary mask where target > 0 is foreground (1.0)
            target_binary = (target > 0).float().to(self.device)
            target_for_grad = target_binary.unsqueeze(1) # Shape Bx1xHxW

        # --- Compute Gradients ---
        # Make sure get_gradient handles the Bx1xHxW input correctly
        target_gradients = self.get_gradient(target_for_grad) 
        pred_gradients = self.get_gradient(pred_for_grad)      

        # --- Calculate Loss ---
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss