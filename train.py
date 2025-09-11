#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12
@author: XuWang
This script is based on the user's proven-fast template, with added flexibility and bug fixes.
"""
import datetime
import time
import os
import sys
import logging
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# --- Setup Project Path and Imports ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from config import DATA_ROOT, CKPT_ROOT
from datasets import ImageFolder
# --- FIX 2: Correct the import path to not use 'models' subfolder ---
from daseg import daseg
import loss as loss_module
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with LASA integration')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
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
    net.train()
    return mIoU

# --- FIX 1: Add 'device' as an argument to the train function ---
def train(net, optimizer, start_epoch, train_loader, test_loader, writer, log_path, checkpoint_path, args, device):
    net.train()
    total_iterations = args.epochs * len(train_loader)
    curr_iter = start_epoch * len(train_loader)
    
    patience_counter = 0
    best_mIoU = 0.0

    # Loss functions are defined inside train(), giving them access to the 'device' variable
    structure_loss = loss_module.structure_loss().to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    iou_loss = loss_module.IOU().to(device)
    last_criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    def bce_iou_loss(pred, target): return bce_loss(pred, target) + iou_loss(pred, target)

    for epoch in range(start_epoch, args.epochs):
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, data in enumerate(train_iterator):
            current_iter_num = curr_iter + i
            lr_decay = (1 - float(current_iter_num) / float(total_iterations)) ** args.lr_decay
            optimizer.param_groups[0]['lr'] = 2 * args.lr * lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * lr_decay

            inputs, labels = data['image'].to(device), data['label'].to(device)

            optimizer.zero_grad()
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
            
            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1).float())
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1).float())
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1).float())
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1).float())
            loss_0 = last_criterion(predict0, labels.long())
            total_loss = loss_1 + loss_2 + 2 * loss_3 + 4 * loss_4 + 10*loss_0
            
            total_loss.backward()
            optimizer.step()
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg, lr=optimizer.param_groups[1]['lr'])

        curr_iter += len(train_loader)
        logging.info(f"Epoch {epoch+1} Train | Average Loss: {loss_recorder.avg:.4f}")
        current_mIoU = validate(net, test_loader, device)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
            # The checkpoint path is passed in directly, fixing the previous bug
            torch.save({'epoch': epoch, 'model_state_dict': net.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, checkpoint_path)
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}_{args.backbone}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training with arguments: {args}")

    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    cod_training_root = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    train_set = ImageFolder(cod_training_root, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(val_path, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    net = daseg(backbone_name=args.backbone).to(device)
    net = nn.DataParallel(net)
    
    optimizer = optim.Adam([{'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr}, {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}])

    start_epoch = 0
    best_mIoU = 0.0 # Initialize best_mIoU
    best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')

    if os.path.exists(best_checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        
        # Load model state
        net.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume epoch and best mIoU
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        best_mIoU = checkpoint['best_mIoU']
        logging.info(f"Loaded checkpoint: Epoch {start_epoch-1}, Best mIoU: {best_mIoU:.4f}")
    else:
        logging.info("No checkpoint found, starting training from scratch.")
    
    writer = SummaryWriter(log_dir=os.path.join(exp_path, 'log'), comment=exp_name)
    
    # Pass best_mIoU to the train function if you want to use it for initial comparison
    # You might also want to modify the train function to accept and use the loaded best_mIoU
    train(net, optimizer, start_epoch, train_loader, test_loader, writer, os.path.join(exp_path, 'log.txt'), best_checkpoint_path, args, device)
    
    writer.close()

if __name__ == '__main__':
    main()