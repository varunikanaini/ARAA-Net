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

    def get_gradient(self, pred):
        """Computes the gradient of a tensor using Sobel filters."""
        # pred is already a torch tensor, no need to convert to numpy first if moving directly
        # pred = pred.detach().cpu().numpy() # REMOVE THIS LINE

        # Define kernels directly as torch tensors or ensure correct dtype after numpy
        # Option 1: Define directly as torch tensors (preferred)
        dx_kernel_right = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dx_kernel_left = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel_down = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel_up = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Option 2: If you prefer numpy, ensure dtype is float32
        # dx_kernel_right_np = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        # dx_kernel_left_np = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        # dy_kernel_down_np = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        # dy_kernel_up_np = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # dx_kernel_right = torch.from_numpy(dx_kernel_right_np).unsqueeze(0).unsqueeze(0).to(self.device)
        # dx_kernel_left = torch.from_numpy(dx_kernel_left_np).unsqueeze(0).unsqueeze(0).to(self.device)
        # dy_kernel_down = torch.from_numpy(dy_kernel_down_np).unsqueeze(0).unsqueeze(0).to(self.device)
        # dy_kernel_up = torch.from_numpy(dy_kernel_up_np).unsqueeze(0).unsqueeze(0).to(self.device)

        gradients_x = []
        gradients_y = []

        # --- IMPORTANT: pred is expected to be BxCxHxW ---
        # Ensure pred has the correct shape for conv2d (e.g., Bx1xHxW if it's a single-channel mask)
        # Your current code seems to pass target_one_hot[:, 1, :, :].unsqueeze(1) which is Bx1xHxW, so that's good.
        
        for i in range(pred.shape[0]): # Iterate through batch
            img_tensor = pred[i].unsqueeze(0) # Shape becomes 1xCxHxW for conv2d

            # Apply conv2d with kernels
            # Horizontal gradient
            dx_right = F.conv2d(img_tensor, dx_kernel_right, padding=1)
            dx_left = F.conv2d(img_tensor, dx_kernel_left, padding=1)
            dx = torch.abs(dx_right - dx_left)

            # Vertical gradient
            dy_down = F.conv2d(img_tensor, dy_kernel_down, padding=1)
            dy_up = F.conv2d(img_tensor, dy_kernel_up, padding=1)
            dy = torch.abs(dy_down - dy_up)
            
            # Concatenate dx and dy for this sample
            gradients_x.append(dx.squeeze(0)) # Remove batch dim
            gradients_y.append(dy.squeeze(0))

        # Stack gradients from all batch items
        # Result shape: (B, 1, H, W) for each gradient type
        gradients_x_stacked = torch.stack(gradients_x, dim=0)
        gradients_y_stacked = torch.stack(gradients_y, dim=0)

        # Concatenate for output shape (B, 2, H, W) where 2 is for dx and dy
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
        # target_one_hot shape should be BxCxHxW (e.g., Bx2xHxW)
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply softmax to predictions to get probabilities
        pred_prob = F.softmax(pred, dim=1)

        # Compute gradients for target and prediction
        # We need gradients for class 1 (lesion). Extract the channel for class 1.
        # target_gradients shape: (B, 2, H, W)
        target_gradients = self.get_gradient(target_one_hot[:, 1, :, :].unsqueeze(1)) # Focus on class 1 (lesion)
        # pred_gradients shape: (B, 2, H, W)
        pred_gradients = self.get_gradient(pred_prob[:, 1, :, :].unsqueeze(1))      # Focus on class 1 (lesion)

        # Calculate loss between gradients
        # L1 loss: sum(|pred_grad - target_grad|)
        # Average over the 2 gradient channels (dx, dy) and spatial dimensions, then batch dimension
        boundary_loss = torch.mean(torch.abs(pred_gradients - target_gradients))

        return boundary_loss