# /kaggle/working/ARAA-Net/train.py
# --- FINAL VERSION: Reverted to VGG19 Baseline + Seg-CutMix Enhancement ---

import sys
import os
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import time
import datetime
from sklearn.model_selection import KFold
import copy

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Config and Other Modules ---
import config
from lasa_unet_model import LASA_Unet
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# --- Loss Functions (Unchanged from original) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        if num_classes > 1:
            pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1)
            true_oh = (targets == 1).float().unsqueeze(1)
        else:
            pred_probs = F.sigmoid(inputs).unsqueeze(1)
            true_oh = targets.float().unsqueeze(1)
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            if mask.ndim == 3: mask = mask.unsqueeze(1)
            pred_probs = pred_probs * mask
            true_oh = true_oh * mask
        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        loss = 1. - dice
        return loss

# --- NEW: Seg-CutMix Augmentation ---
def seg_cutmix(data, targets, alpha=1.0):
    """Applies CutMix to a batch of images and segmentation masks."""
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    targets[:, :, bbx1:bbx2, bby1:bby2] = shuffled_targets[:, :, bbx1:bbx2, bby1:bby2]
    return data, targets

def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# --- Backbone Freezing/Unfreezing Helper Functions (Unchanged) ---
def freeze_backbone(model, backbone_name):
    # This part can be simplified if only VGG19 is used, but kept for compatibility
    frozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4']
    for layer_name in frozen_layers:
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = False
    logging.info(f"Backbone '{backbone_name}' frozen.")

def unfreeze_backbone(model, backbone_name):
    unfrozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4']
    for layer_name in unfrozen_layers:
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True
    logging.info(f"Backbone '{backbone_name}' unfrozen.")

# --- Argument Parsing (Unchanged from original) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--backbone', type=str, default='vgg19', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lasa-kernels', type=int, nargs='+', default=[1, 3, 5, 7])
    parser.add_argument('--deep-supervision-weights', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--focal-loss-weight', type=float, default=1.0)
    parser.add_argument('--dice-loss-weight', type=float, default=1.0)
    parser.add_argument('--scheduler-type', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--scheduler-T0', type=int, default=10, help='T_0 for CosineAnnealingWarmRestarts.')
    parser.add_argument('--scheduler-patience', type=int, default=5)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    
    args = parser.parse_args()
    
    # Post-processing to add dataset-specific configs
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path = dataset_info['path']
    args.num_classes = dataset_info['num_classes']
    backbone_res = config.get_backbone_resolution(args.backbone)
    args.scale_h, args.scale_w = backbone_res[0], backbone_res[1]
    
    # Add other default args from your config if needed
    for key, value in config.DEFAULT_ARGS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            
    return args

# --- Logging Setup (Unchanged) ---
def setup_logging(log_dir, filename='training.log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Evaluation Function (Reverted to original simple version) ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes)
    loss_recorder = AvgMeter()

    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None:
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs)
            final_pred = outputs[-1]
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    global_acc, _, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA: {global_acc.item():.4f}, mIoU: {mIoU:.4f}, FWIoU: {fwiou.item():.4f}, Dice: {mDice:.4f}")
    
    if mode == "Validating":
        net.train()
    return mIoU

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Main Function ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = f"{args.backbone}_LASA{'_'.join(map(str, args.lasa_kernels))}_DSW{'_'.join(map(str, args.deep_supervision_weights))}_FLW{args.focal_loss_weight}_DLW{args.dice_loss_weight}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    check_mkdir(base_exp_path)
    setup_logging(base_exp_path)

    logging.info(f"Starting experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")

    train_dataset = ImageFolder(os.path.join(args.dataset_path, 'train'), args.dataset_name, args, split='train')
    val_dataset = ImageFolder(os.path.join(args.dataset_path, 'val'), args.dataset_name, args, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2, eta_min=1e-6)
    else: # Default to ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.scheduler_patience)

    if args.test_only:
        test_dataset = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, split='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)
        checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found at {checkpoint_path}")
            return
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Model loaded from {checkpoint_path}")
        evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing")
        return
        
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(base_exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
    
    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for i, data in enumerate(train_iterator):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            # --- ENHANCEMENT: Seg-CutMix Augmentation ---
            if np.random.rand() < 0.5: # Apply to 50% of batches
                inputs, labels_float = seg_cutmix(inputs, labels.unsqueeze(1).float())
                labels = labels_float.squeeze(1).long() # Convert back to Long type for loss
            # ------------------------------------

            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)
            
            total_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                f_loss = focal_loss_fn(pred_output, labels)
                d_loss = dice_loss_fn(pred_output, labels)
                combined_loss = (args.focal_loss_weight * f_loss) + (args.dice_loss_weight * d_loss)
                total_loss += args.deep_supervision_weights[head_idx] * combined_loss
            
            total_loss.backward()
            optimizer.step()
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_mIoU)
        else:
            scheduler.step()
        
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), best_checkpoint_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Model saved.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ mIoU did not improve for {patience_counter} epoch(s). Best: {best_mIoU:.4f}")

        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered after {patience_counter} epochs.")
            break
            
    logging.info("Training finished.")

if __name__ == '__main__':
    main()