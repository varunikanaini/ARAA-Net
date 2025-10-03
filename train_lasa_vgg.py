# /kaggle/working/ARAA-Net/train_lasa_unet.py
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
import torch.nn.functional as F

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_unet_model import Enhanced_LASA_VGG_UNet
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# ===================================================================
#      LOSS FUNCTION CLASSES (Focal Loss, Dice Loss)
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean', ignore_index=255):
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
            valid_sum = mask.sum()
            return (focal_loss * mask).sum() / (valid_sum + 1e-6) if valid_sum > 0 else (focal_loss * mask).sum()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred_probs = F.softmax(inputs, dim=1)[:, 1] # Get probability of class 1
        true_flat = (targets == 1).float().view(-1)
        pred_flat = pred_probs.view(-1)

        mask = (targets.view(-1) != self.ignore_index).float()
        pred_flat = pred_flat * mask
        true_flat = true_flat * mask
        
        intersection = (pred_flat * true_flat).sum()
        union = pred_flat.sum() + true_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1. - dice_score
# ===================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-UNet Model with Enhancements')
    # --- Key Arguments ---
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'], help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4) # Smaller batch size for ResNet50 if memory is an issue
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # --- Loss Function Arguments ---
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.4, 0.6, 0.8, 1.0], help='Weights for deep supervision losses (4 outputs)')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component')
    parser.add_argument('--dice-loss-weight', type=float, default=1.2, help='Weight for Dice Loss component (increased emphasis)')

    # --- Scheduler and Optimization Arguments ---
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--scheduler-patience', type=int, default=5)
    parser.add_argument('--scheduler-factor', type=float, default=0.5)

    # --- Dataloader and Augmentation (Dummy args for ImageFolder) ---
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576)
    parser.add_argument('--expansion-factor', type=float, default=1.5)
    parser.add_argument('--min-bbox-h', type=int, default=32)
    parser.add_argument('--min-bbox-w', type=int, default=32)
    parser.add_argument('--wavelet-type', type=str, default='haar')
    parser.add_argument('--wavelet-level', type=int, default=1)
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5)

    try: args = parser.parse_args()
    except SystemExit: args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 4:
        parser.error(f"deep-supervision-weights must have 4 values. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def evaluate(net, data_loader, device, loss_fns, weights, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    focal_loss_fn, dice_loss_fn = loss_fns
    focal_w, dice_w, ds_weights = weights

    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs)
            final_pred = outputs[-1]
            
            total_loss = sum(ds_weights[i] * (focal_w * focal_loss_fn(out, labels.long()) + dice_w * dice_loss_fn(out, labels.long())) for i, out in enumerate(outputs))
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- {mode} mIoU: {mIoU:.4f} | Loss: {loss_recorder.avg:.4f} ---")
    if mode == "Validating": net.train()
    return mIoU

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = f"ENHANCED_{args.backbone}_LASA_UNet_{args.dataset_name}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)
    logging.info(f"Starting experiment '{exp_name}' with args: {args}")

    base_path = DATASET_PATHS[args.dataset_name]
    train_path = os.path.join(base_path, 'train') if 'TSRS_RSNA' in args.dataset_name else base_path
    val_path = os.path.join(base_path, 'val') if 'TSRS_RSNA' in args.dataset_name else base_path

    train_set = ImageFolder(train_path, args.dataset_name, args, split='train')
    val_set = ImageFolder(val_path, args.dataset_name, args, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)

    net = LASA_UNet(num_classes=2, backbone_name=args.backbone).to(device)
    focal_loss_fn = FocalLoss(alpha=0.5, gamma=2).to(device)
    dice_loss_fn = DiceLoss().to(device)
    loss_fns = (focal_loss_fn, dice_loss_fn)
    loss_weights = (args.focal_loss_weight, args.dice_loss_weight, args.deep_supervision_weights)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)

    best_mIoU, patience_counter = 0.0, 0
    for epoch in range(args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data in train_iterator:
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)
            
            total_loss = sum(loss_weights[2][i] * (loss_weights[0] * focal_loss_fn(out, labels.long()) + loss_weights[1] * dice_loss_fn(out, labels.long())) for i, out in enumerate(outputs))
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg, lr=optimizer.param_groups[0]['lr'])
            
        current_mIoU = evaluate(net, val_loader, device, loss_fns, loss_weights)
        scheduler.step(current_mIoU)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Model saved.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epochs. Best mIoU: {best_mIoU:.4f}.")
        
        if patience_counter >= args.patience:
            logging.info("--- Early stopping triggered ---")
            break

if __name__ == '__main__':
    main()