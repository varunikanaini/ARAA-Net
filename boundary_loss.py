# /kaggle/working/ARAA-Net/boundary_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoundaryLoss(nn.Module):
    def __init__(self, device, alpha=2.0, gamma=0.5):
        """
        Boundary Loss implementation.
        Args:
            device (torch.device): The device (e.g., 'cuda') to run the loss on.
            alpha (float): Weight for the loss calculation.
            gamma (float): Weight for the gradient loss.
        """
        super(BoundaryLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma

    def get_gradient(self, pred):
        """Computes the gradient of a tensor using Sobel filters."""
        pred_np = pred.detach().cpu().numpy()
        
        # Use Sobel filters for gradient calculation
        # Pad to avoid boundary artifacts during gradient computation
        padded_pred = np.pad(pred_np, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
        
        # Horizontal gradient (dx)
        dx_kernel_right = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        dx_kernel_left = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        
        # Vertical gradient (dy)
        dy_kernel_down = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        dy_kernel_up = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradients_x = []
        gradients_y = []

        for batch_item in padded_pred:
            img_tensor = torch.from_numpy(batch_item).float().unsqueeze(0).to(self.device) # Add batch dim, move to device

            # Apply conv2d with kernels
            # Horizontal gradient
            dx_right = F.conv2d(img_tensor, torch.from_numpy(dx_kernel_right).unsqueeze(0).unsqueeze(0).to(self.device), padding=1)
            dx_left = F.conv2d(img_tensor, torch.from_numpy(dx_kernel_left).unsqueeze(0).unsqueeze(0).to(self.device), padding=1)
            dx = torch.abs(dx_right - dx_left)

            # Vertical gradient
            dy_down = F.conv2d(img_tensor, torch.from_numpy(dy_kernel_down).unsqueeze(0).unsqueeze(0).to(self.device), padding=1)
            dy_up = F.conv2d(img_tensor, torch.from_numpy(dy_kernel_down).unsqueeze(0).unsqueeze(0).to(self.device), padding=1)
            dy = torch.abs(dy_down - dy_up)
            
            gradients_x.append(dx.squeeze(0))
            gradients_y.append(dy.squeeze(0))

        # Stack gradients along a new dimension (e.g., channel dimension)
        # Result shape: (B, 2, H, W) where 2 is for dx and dy
        gradients_x_stacked = torch.stack(gradients_x, dim=0)
        gradients_y_stacked = torch.stack(gradients_y, dim=0)

        # Concatenate for output shape (B, 2, H, W)
        return torch.cat([gradients_x_stacked, gradients_y_stacked], dim=1)


    def forward(self, pred, target):
        """
        Computes the Boundary Loss.
        Args:
            pred (torch.Tensor): Predicted segmentation mask (logits or probabilities, shape BxCxHxW).
            target (torch.Tensor): Ground truth mask (class indices, shape BxHxW).
        Returns:
            torch.Tensor: The computed Boundary Loss.
        """
        # Ensure target is float and one-hot encoded
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply softmax to predictions to get probabilities
        pred_prob = F.softmax(pred, dim=1)

        # Compute gradients for target and prediction
        # Shape of gradients: (B, 2, H, W) where 2 is for dx, dy
        target_gradients = self.get_gradient(target_one_hot[:, 1, :, :].unsqueeze(1)) # Focus on class 1 (lesion)
        pred_gradients = self.get_gradient(pred_prob[:, 1, :, :].unsqueeze(1))      # Focus on class 1 (lesion)

        # Calculate loss between gradients
        # L1 loss: sum(|pred_grad - target_grad|)
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss