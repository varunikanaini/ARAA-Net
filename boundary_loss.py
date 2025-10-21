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

    # In boundary_loss.py

# ... (imports and class definition) ...

    def get_gradient(self, pred): # pred is expected to be Bx1xHxW (binary mask)
        """Computes the gradient of a tensor using Sobel filters."""
        # Ensure pred is float and on the correct device
        pred_tensor = pred.float().to(self.device)

        # Define kernels as float32 tensors on the correct device
        dx_kernel_right = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dx_kernel_left = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel_down = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel_up = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        gradients_x = []
        gradients_y = []

        for i in range(pred_tensor.shape[0]): # Iterate through batch
            img_tensor = pred_tensor[i].unsqueeze(0) # Shape becomes 1x1xHxW for conv2d

            # Horizontal gradient
            dx_right = F.conv2d(img_tensor, dx_kernel_right, padding=1)
            dx_left = F.conv2d(img_tensor, dx_kernel_left, padding=1)
            dx = torch.abs(dx_right - dx_left)

            # Vertical gradient
            dy_down = F.conv2d(img_tensor, dy_kernel_down, padding=1)
            dy_up = F.conv2d(img_tensor, dy_kernel_up, padding=1)
            dy = torch.abs(dy_down - dy_up)
            
            gradients_x.append(dx.squeeze(0)) # Remove batch dim
            gradients_y.append(dy.squeeze(0))

        gradients_x_stacked = torch.stack(gradients_x, dim=0) # Shape (B, H, W)
        gradients_y_stacked = torch.stack(gradients_y, dim=0) # Shape (B, H, W)

        # Concatenate for output shape (B, 2, H, W) where 2 is for dx and dy
        return torch.cat([gradients_x_stacked.unsqueeze(1), gradients_y_stacked.unsqueeze(1)], dim=1)


    def forward(self, pred, target):
        """
        Computes the Boundary Loss.
        Args:
            pred (torch.Tensor): Predicted segmentation mask (probabilities, shape BxCxHxW, where C=1 for binary prediction).
            target (torch.Tensor): Ground truth mask (class indices, shape BxHxW).
        Returns:
            torch.Tensor: The computed Boundary Loss.
        """
        # Ensure pred is Bx1xHxW for get_gradient
        if pred.shape[1] > 1:
            # If pred is multi-class probabilities (BxCxHxW), select the foreground class (index 1)
            pred_for_grad = pred[:, 1, :, :].unsqueeze(1) # Shape Bx1xHxW
        else:
            # If pred is already binary (Bx1xHxW), use it directly
            pred_for_grad = pred
        
        # Ensure target is float and one-hot encoded if pred was multi-class before softmax,
        # but since we are focusing on class 1 boundary, let's assume binary target from input.
        # If target is BxHxW with class indices, and num_classes=2, then F.one_hot will give BxHxWx2.
        # We need Bx1xHxW for get_gradient.
        
        # Correct one-hot encoding for binary target (class 0 and 1)
        # If your target is already binary (0 or 1), it's fine. If it's like 0/255, it needs thresholding.
        target_np = target.cpu().numpy() # Convert to numpy to check max value
        if target_np.max() > 1: # Assuming mask values are 0 or 255
            target_binary = (target == 1).float() # Convert to 0s and 1s, float
        else:
            target_binary = target.float() # Already 0s or 1s, ensure float

        # Ensure target_binary has shape Bx1xHxW for get_gradient
        target_for_grad = target_binary.unsqueeze(1).to(self.device) # Shape Bx1xHxW

        # Compute gradients for target and prediction
        target_gradients = self.get_gradient(target_for_grad) # Use Bx1xHxW target
        pred_gradients = self.get_gradient(pred_for_grad)      # Use Bx1xHxW pred

        # Calculate loss between gradients
        # Average over the 2 gradient channels (dx, dy) and spatial dimensions, then batch dimension
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss