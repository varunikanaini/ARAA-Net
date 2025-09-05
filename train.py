#!/usr/bin/env python3
import os
import time
import sys
import logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Make sure this project path is correct for your environment
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
from config import DATA_ROOT, CKPT_ROOT
from datasets import ImageFolder
import loss as loss_module
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with LASA, Focal Loss, and dynamic datasets')
    # --- CHANGE 1: ADDED --dataset-name ARGUMENT ---
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Name of the dataset to use')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose the backbone model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    # These args are now passed to the dataset class, ensuring alignment
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to')
    
    # Using a try-except block to handle running in environments like Kaggle notebooks
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # Fallback for notebooks
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    # Clear existing handlers to prevent duplicate logs in notebooks
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def validate(net, test_loader, device):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- Validation mIoU: {mIoU:.4f} ---")
    net.train() # Set model back to training mode
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)

    # --- CHANGE 3: IMPLEMENT NEW OUTPUT DIRECTORY FORMAT ---
    exp_name = f"lasa_mfr_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}_{args.backbone}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training with arguments: {args}")
    logging.info(f"Experiment name: {exp_name}")
    logging.info(f"Using device: {device}")

    # --- Construct dataset paths dynamically ---
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    cod_training_root = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # The ImageFolder now takes the 'args' object to get its parameters
    train_set = ImageFolder(cod_training_root, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(val_path, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images.")

    net = daseg(backbone_name=args.backbone).to(device)
    
    optimizer = optim.Adam([
        {'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr}, 
        {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
    ])

    # --- CHANGE 2: INTEGRATE FOCAL LOSS ---
    structure_loss_fn = loss_module.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss_module.IOU().to(device)
    focal_loss_fn = loss_module.FocalLoss(alpha=0.25, gamma=2).to(device) # Standard Focal Loss values
    def bce_iou_loss(pred, target): return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint_path):
        logging.info(f"Resuming from checkpoint: {latest_checkpoint_path}")
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")

    total_iterations = len(train_loader) * args.epochs
    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, data in enumerate(train_iterator):
            curr_iter = epoch * len(train_loader) + i
            lr_decay = (1 - curr_iter / total_iterations) ** args.lr_decay
            optimizer.param_groups[0]['lr'] = 2 * args.lr * lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * lr_decay
            
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            p4, p3, p2, p1, p0 = net(inputs)
            
            binary_labels = labels.unsqueeze(1).float()
            loss_1 = bce_iou_loss(p1, binary_labels)
            loss_2 = structure_loss_fn(p2, binary_labels)
            loss_3 = structure_loss_fn(p3, binary_labels)
            loss_4 = structure_loss_fn(p4, binary_labels)
            # Use Focal Loss for the final prediction
            loss_0 = focal_loss_fn(p0, labels.long())
            
            total_loss = loss_1 + loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg, lr=optimizer.param_groups[1]['lr'])
            
        logging.info(f"Epoch {epoch+1} Train | Average Loss: {loss_recorder.avg:.4f}")
        current_mIoU = validate(net, test_loader, device)

        is_best = current_mIoU > best_mIoU
        if is_best:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

if __name__ == '__main__':
    main()