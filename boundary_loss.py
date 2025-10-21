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
            pred (torch.Tensor): Predicted segmentation mask (probabilities, shape BxCxHxW).
            target (torch.Tensor): Ground truth mask (class indices, shape BxHxW).
        Returns:
            torch.Tensor: The computed Boundary Loss.
        """
        # --- Prepare Prediction for Gradient Calculation ---
        # pred is BxCxHxW (e.g., Bx2xHxW from segmentation output)
        # We need the probability of the foreground class (assuming class 1 is foreground)
        if pred.shape[1] > 1:
            pred_prob = F.softmax(pred, dim=1)
            pred_for_grad = pred_prob[:, 1, :, :].unsqueeze(1) # Take foreground channel, shape Bx1xHxW
        else:
            # If pred is already probabilities (Bx1xHxW), use it directly
            pred_for_grad = pred

        # --- Prepare Target for Gradient Calculation ---
        # target is BxHxW with class indices
        # Convert to one-hot encoding BxCxHxW, then select foreground channel (index 1)
        # Ensure target is float for calculations
        
        # Find the maximum class index in the target tensor to determine the number of channels
        num_classes_in_target = int(target.max()) + 1 if target.numel() > 0 else 1
        
        # Ensure target_one_hot has at least 2 channels for binary segmentation (background + foreground)
        # If num_classes is 2, F.one_hot will create BxHxWx2
        # We need to permute it to Bx2xHxW
        if num_classes_in_target < 2: # This might happen if a batch only contains background
             # Create a dummy target with two channels if only background is present
             target_one_hot = torch.zeros(target.shape[0], 2, target.shape[1], target.shape[2], device=self.device, dtype=torch.float32)
             # If target is BxHxW, we need to unsqueeze to add channel dim
             target_one_hot[:, 0, :, :] = (target == 0).float() # Background channel
             if target.shape[1] == 1: # If target is already Bx1xHxW, we might not need this
                 pass # Handle cases where target might already be in Bx1xHxW format
             else:
                 # Assuming target is BxHxW, convert to Bx1xHxW for gradient calculation
                 target_for_grad = target_one_hot[:, 1, :, :].unsqueeze(1) # Foreground channel, Bx1xHxW

        else:
            target_one_hot = F.one_hot(target.long(), num_classes=num_classes_in_target).permute(0, 3, 1, 2).float()
            target_for_grad = target_one_hot[:, 1, :, :].unsqueeze(1) # Foreground channel, Bx1xHxW

        # Compute gradients for target and prediction
        # Shape of gradients: (B, 2, H, W)
        target_gradients = self.get_gradient(target_for_grad) 
        pred_gradients = self.get_gradient(pred_for_grad)      

        # Calculate loss between gradients
        # L1 loss: sum(|pred_grad - target_grad|)
        # Average over the 2 gradient channels (dx, dy) and spatial dimensions, then batch dimension
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss