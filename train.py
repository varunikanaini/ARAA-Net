# /kaggle/working/ARAA-Net/train.py
# --- FINAL COMPLETE & WORKING VERSION with Freeze-and-Tune + K-Fold ---

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

# --- Import All Modules ---
import config
from lasa_unet_model import LASA_Unet
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# --- Loss Functions (Unchanged) ---
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

# --- Backbone Freezing/Unfreezing Helpers ---
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    logging.info("--- Encoder FROZEN. Training Decoder + Custom Modules ---")

def unfreeze_backbone(model):
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = True
    logging.info("--- Encoder UN-FROZEN. Fine-tuning all layers ---")

# --- Argument Parsing (Complete) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet with Two-Phase Strategy')
    
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--backbone', type=str, default='vgg19', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--lasa-kernels', type=int, nargs='+', default=[1, 3, 5, 7])
    parser.add_argument('--deep-supervision-weights', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--focal-loss-weight', type=float, default=0.5)
    parser.add_argument('--dice-loss-weight', type=float, default=1.5)
    parser.add_argument('--scheduler-type', type=str, default='CosineAnnealingWarmRestarts', choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--scheduler-T0', type=int, default=15)
    parser.add_argument('--scheduler-T-mult', type=int, default=2)
    parser.add_argument('--scheduler-patience', type=int, default=5)
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--run-kfold', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--fine-tune-epochs', type=int, default=40)
    
    args = parser.parse_args()
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path = dataset_info['path']
    args.num_classes = dataset_info['num_classes']
    backbone_res = config.get_backbone_resolution(args.backbone)
    args.scale_h, args.scale_w = backbone_res[0], backbone_res[1]
    
    for key, value in config.DEFAULT_ARGS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            
    return args

# --- Logging ---
def setup_logging(log_dir, filename='training.log'):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating", fold_num=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes)
    loss_recorder = AvgMeter()
    log_prefix = f"Fold {fold_num} " if fold_num is not None else ""

    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{log_prefix}{mode}", leave=False):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs)
            final_pred = outputs[-1]
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                focal_loss = focal_loss_fn(pred_output, labels.long())
                dice_loss = dice_loss_fn(pred_output, labels.long())
                combined_loss = (args.focal_loss_weight * focal_loss) + (args.dice_loss_weight * dice_loss)
                total_loss += args.deep_supervision_weights[i] * combined_loss
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    global_acc, _, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info(f"--- {log_prefix}{mode} Summary ---")
    logging.info(f"  Loss: {loss_recorder.avg:.4f}, OA: {global_acc.item():.4f}, mIoU: {mIoU:.4f}")
    
    if mode == "Validating": net.train()
    return mIoU

# --- Collate Function ---
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Main Function ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = f"{args.backbone}_FreezeTune_LASA{'_'.join(map(str, args.lasa_kernels))}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    check_mkdir(base_exp_path)
    setup_logging(base_exp_path, filename='main_training.log')

    logging.info(f"Starting experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")

    all_train_dataset = ImageFolder(os.path.join(args.dataset_path, 'train'), args.dataset_name, args, split='train')
    val_dataset = ImageFolder(os.path.join(args.dataset_path, 'val'), args.dataset_name, args, split='val')
    test_dataset = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, split='test')

    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    
    if args.test_only:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"FATAL: Checkpoint for testing not found at {checkpoint_path}")
            return
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Model loaded from {checkpoint_path}")
        evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing")
        return

    if args.run_kfold:
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_dataset)):
            fold_num = fold + 1
            fold_exp_path = os.path.join(base_exp_path, f'fold_{fold_num}')
            check_mkdir(fold_exp_path)
            setup_logging(fold_exp_path, filename=f'fold_{fold_num}_training.log')
            logging.info(f"--- Starting Fold {fold_num}/{args.k_folds} ---")
            
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(all_train_dataset, batch_size=args.batch_size, sampler=train_subsampler, num_workers=args.num_workers, collate_fn=custom_collate_fn)
            val_loader = DataLoader(all_train_dataset, batch_size=1, sampler=val_subsampler, num_workers=args.num_workers, collate_fn=custom_collate_fn)

            fold_net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
            optimizer = optim.Adam(fold_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=args.scheduler_T_mult, eta_min=1e-6)
            
            freeze_backbone(fold_net) # Start with frozen backbone for each fold

            best_mIoU_fold, patience_counter = 0.0, 0
            best_checkpoint_path = os.path.join(fold_exp_path, 'best_checkpoint.pth')
            
            for epoch in range(args.epochs):
                if epoch == args.fine_tune_epochs:
                    unfreeze_backbone(fold_net)
                    new_lr = args.lr / 10.0
                    optimizer = optim.Adam(fold_net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=args.scheduler_T_mult, eta_min=1e-6)
                    logging.info(f"Fold {fold_num}: Switched to Phase 2. New LR: {new_lr}")

                # Training Step
                fold_net.train()
                loss_recorder = AvgMeter()
                train_iterator = tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1}/{args.epochs}", leave=False)
                for data in train_iterator:
                    if data is None: continue
                    inputs, labels = data['image'].to(device), data['label'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = fold_net(inputs)
                    total_loss = 0
                    for head_idx, pred_output in enumerate(outputs):
                        f_loss = focal_loss_fn(pred_output, labels.long())
                        d_loss = dice_loss_fn(pred_output, labels.long())
                        total_loss += args.deep_supervision_weights[head_idx] * ((args.focal_loss_weight * f_loss) + (args.dice_loss_weight * d_loss))
                    total_loss.backward()
                    optimizer.step()
                    loss_recorder.update(total_loss.item(), inputs.size(0))
                
                # Validation Step
                current_mIoU = evaluate_model(fold_net, val_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating", fold_num=fold_num)
                scheduler.step()
                
                if current_mIoU > best_mIoU_fold:
                    best_mIoU_fold = current_mIoU
                    patience_counter = 0
                    torch.save(fold_net.state_dict(), best_checkpoint_path)
                    logging.info(f"Fold {fold_num}: ✅ New best mIoU: {best_mIoU_fold:.4f}. Model saved.")
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    logging.info(f"Fold {fold_num}: Early stopping triggered.")
                    break
        return # End after K-Fold

    # --- Standard Training (No K-Fold) ---
    train_loader = DataLoader(all_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=args.scheduler_T_mult, eta_min=1e-6)
    
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    best_checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
    
    if start_epoch < args.fine_tune_epochs:
        freeze_backbone(net)
    else:
        unfreeze_backbone(net)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.fine_tune_epochs:
            unfreeze_backbone(net)
            new_lr = args.lr / 10.0
            optimizer = optim.Adam(net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=args.scheduler_T_mult, eta_min=1e-6)
            logging.info(f"--- Switched to Phase 2. New LR: {new_lr} ---")
            
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, data in enumerate(train_iterator):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)
            total_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                f_loss = focal_loss_fn(pred_output, labels.long())
                d_loss = dice_loss_fn(pred_output, labels.long())
                total_loss += args.deep_supervision_weights[head_idx] * ((args.focal_loss_weight * f_loss) + (args.dice_loss_weight * d_loss))
            total_loss.backward()
            optimizer.step()
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, args)
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