import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoundaryLoss(nn.Module):
    def __init__(self, device, alpha=2.0, gamma=0.5, hausdorff_weight=0.3):
        super(BoundaryLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.hausdorff_weight = hausdorff_weight

    def get_gradient(self, pred_mask):
        """Computes the gradient of a binary mask using Sobel filters."""
        pred_tensor = pred_mask.float().to(self.device)
        batch_size, channels, height, width = pred_tensor.shape

        dx_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dy_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        grad_x = torch.abs(F.conv2d(pred_tensor, dx_kernel, padding=1))
        grad_y = torch.abs(F.conv2d(pred_tensor, dy_kernel, padding=1))
        gradients = torch.cat([grad_x, grad_y], dim=1)
        return gradients

    def hausdorff_distance(self, pred, target):
        """Approximate Hausdorff distance for boundary points."""
        pred_binary = (pred > 0.5).float()
        target_binary = target.float()

        pred_grad = self.get_gradient(pred_binary)
        target_grad = self.get_gradient(target_binary)

        pred_points = torch.nonzero(pred_grad.sum(dim=1) > 0, as_tuple=False)
        target_points = torch.nonzero(target_grad.sum(dim=1) > 0, as_tuple=False)

        if pred_points.size(0) == 0 or target_points.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        pred_points = pred_points[:, 2:].float()
        target_points = target_points[:, 2:].float()

        dists_to_target = torch.cdist(pred_points, target_points, p=2)
        dists_to_pred = torch.cdist(target_points, pred_points, p=2)

        max_dist_to_target = dists_to_target.min(dim=1)[0].max() if dists_to_target.size(0) > 0 else 0.0
        max_dist_to_pred = dists_to_pred.min(dim=0)[0].max() if dists_to_pred.size(0) > 0 else 0.0

        return (max_dist_to_target + max_dist_to_pred) / 2.0

    def forward(self, pred, target):
        """
        Computes the Boundary Loss with gradient and Hausdorff components.
        Args:
            pred (torch.Tensor): Predicted boundary mask (probabilities, Bx1xHxW).
            target (torch.Tensor): Ground truth mask (class indices, BxHxW).
        Returns:
            torch.Tensor: Combined boundary loss.
        """
        if pred.shape[1] > 1:
            pred_prob = F.softmax(pred, dim=1)
            pred_for_grad = pred_prob[:, 1, :, :].unsqueeze(1)
        else:
            pred_for_grad = pred.float().to(self.device)

        target_binary = (target > 0).float().to(self.device)
        target_for_grad = target_binary.unsqueeze(1)

        target_gradients = self.get_gradient(target_for_grad)
        pred_gradients = self.get_gradient(pred_for_grad)

        if target_gradients.shape != pred_gradients.shape:
            h_pred, w_pred = pred_gradients.shape[2], pred_gradients.shape[3]
            target_gradients = F.interpolate(target_gradients, size=(h_pred, w_pred), mode='bilinear', align_corners=False)

        gradient_loss = torch.mean(torch.abs(pred_gradients - target_gradients))
        hausdorff_loss = self.hausdorff_distance(pred_for_grad, target_for_grad)

        return self.alpha * gradient_loss + self.hausdorff_weight * hausdorff_loss