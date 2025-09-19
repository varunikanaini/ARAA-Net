# /kaggle/working/ARAA-Net/train_lasa_vgg.py (FINAL, CORRECTED STANDALONE SCRIPT)
import os
import time
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F # Needed for Focal Loss

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet # Renamed the model class
# Assuming config.py exists and defines DATA_ROOT and CKPT_ROOT
from config import DATA_ROOT, CKPT_ROOT 
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# ===================================================================
#      ✅ FOCAL LOSS CLASS - INCLUDED DIRECTLY IN THIS SCRIPT ✅
# ===================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification, included directly in the script.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are raw logits (N, C, H, W)
        # targets are class indices (N, H, W)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            # Only sum over valid pixels (not ignore_index)
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ===================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Train Standalone LASA-Unet Model with Deep Supervision and Amplification')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=448, help='Resize height for input images')
    parser.add_argument('--scale-w', type=int, default=448, help='Resize width for input images')
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    
    # Arguments for CenterAmplification
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Minimum lesion area in pixels to trigger CenterAmplification (STS-Net default 576)')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Factor by which to expand the bounding box during CenterAmplification')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Minimum height of the expanded bounding box in pixels for CenterAmplification')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Minimum width of the expanded bounding box in pixels for CenterAmplification')


    try:
        args = parser.parse_args()
    except SystemExit:
        # This block allows running in environments like Jupyter where sys.argv might be empty
        # or where a SystemExit from argparse is undesirable.
        # It sets default arguments for interactive use.
        args = parser.parse_args([]) 
        # You might want to explicitly set some defaults here for quick testing in notebooks
        # For example, if you want to test resnet50 by default in notebook:
        # args = parser.parse_args(['--dataset-name', 'TSRS_RSNA-Epiphysis', '--backbone', 'resnet50', '--epochs', '10', '--batch-size', '2'])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    # Remove existing handlers to avoid duplicate log entries in some environments
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def validate(net, test_loader, device, focal_loss_fn, deep_supervision_weights):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            # Model now returns 5 outputs due to deep supervision
            outputs = net(inputs) 
            final_pred = outputs[-1] # The last output is always the final one for evaluation metrics
            
            # Calculate total loss for validation reporting (using all deep supervision losses)
            total_loss = 0
            for i, pred in enumerate(outputs):
                total_loss += deep_supervision_weights[i] * focal_loss_fn(pred, labels.long())
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            # For mIoU calculation, use only the final output's prediction
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- Validation mIoU: {mIoU:.4f} | Validation Loss: {loss_recorder.avg:.4f} ---")
    net.train()
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    # Create experiment directory for checkpoints and logs
    exp_name = f"{args.backbone}_LASA_Unet_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training for '{exp_name}' with arguments: {args}")

    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    train_set = ImageFolder(train_path, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(val_path, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # Instantiate the model with chosen backbone
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2).to(device)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    
    # Load model and optimizer state if checkpoint exists for resuming
    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict']) # Corrected: Pass the state dict to load
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data in train_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Model returns multiple outputs for deep supervision
            outputs = net(inputs)
            
            # Calculate total loss, summing weighted losses from all deep supervision heads
            total_loss = 0
            for i, pred in enumerate(outputs):
                total_loss += args.deep_supervision_weights[i] * focal_loss_fn(pred, labels.long())
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        current_mIoU = validate(net, test_loader, device, focal_loss_fn, args.deep_supervision_weights)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save latest checkpoint including patience_counter for robust resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # FIXED THIS LINE!
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered due to no improvement.")
            break

if __name__ == '__main__':
    main()