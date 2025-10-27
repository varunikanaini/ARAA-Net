#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import datetime
import time
import os
import argparse 
import logging 
from collections import OrderedDict 
import torch.utils.data.dataloader 
import sys
import json
from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

# --- UNCHANGED IMPORTS ---
from config import backbone_path, DATASET_PATHS 
from datasets import ImageFolder, make_dataset
from misc import AvgMeter, check_mkdir
from daseg import daseg
import loss
from seg_utils import ConfusionMatrix

cudnn.benchmark = True
torch.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================================
# --- ADDED: Definition for setup_logging ---
# This function was called but not defined in the previous version. This corrects the error.
# =========================================================================================
def setup_logging(log_dir, filename='training.log'):
    """Configures the logging for a specific fold."""
    # Clear any existing handlers to prevent duplicate logs in different files
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, filename)),
                            logging.StreamHandler(sys.stdout) # Log to console as well
                        ])

# --- UNCHANGED: Loss Functions and Collate ---
structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255) 

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- ORIGINAL `validate` FUNCTION (LOGIC UNCHANGED) ---
def validate(net, val_loader, epoch): 
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    val_iterator = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch} (Val)")
    
    with torch.no_grad():
        for data in val_iterator:
            if data is None: continue
            inputs, labels, _ = data['image'], data['label'], data['name'] 
            inputs, labels = inputs.to(device), labels.to(device)
            *_, predict0 = net(inputs)
            confmat.update(labels.flatten(), predict0.argmax(1).flatten())
        
    _, _, class_iou, _, mDice = confmat.compute()
    val_miou = np.mean(class_iou.cpu().numpy())
    logging.info(f'--- Validation Results (Epoch {epoch}) --- Mean IoU: {val_miou:.4f}, Mean Dice: {mDice:.4f}') 
    net.train() 
    return val_miou

# --- ORIGINAL `train` FUNCTION (LOGIC UNCHANGED) ---
def train(net, optimizer, args, train_loader, val_loader, fold_exp_path, start_epoch, initial_best_miou, initial_patience):
    writer = SummaryWriter(log_dir=os.path.join(fold_exp_path, 'log'))
    total_iterations = args['epoch_num'] * len(train_loader)
    best_mIoU, patience_counter = initial_best_miou, initial_patience
    curr_iter = (start_epoch - 1) * len(train_loader) + 1

    for epoch in range(start_epoch, args['epoch_num'] + 1):
        loss_record = AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args['epoch_num']} (Train)")
        
        for data in train_iterator:
            if data is None: continue
            
            # Your Original Poly LR Schedule
            base_lr = args['lr'] * (1 - float(curr_iter) / float(total_iterations)) ** args['lr_decay']
            optimizer.param_groups[0]['lr'] = 2 * base_lr
            optimizer.param_groups[1]['lr'] = 1 * base_lr

            # Your Original Forward/Backward Pass
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1))
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1))
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1))
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1))       
            loss_0 = last_criterion(predict0, labels.long())
            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), inputs.size(0))
            curr_iter += 1
        
        # Your Original Validation and Checkpointing Logic
        current_val_mIoU = validate(net, val_loader, epoch) 
        writer.add_scalar('val/miou', current_val_mIoU, epoch)

        if current_val_mIoU > best_mIoU:
            best_mIoU = current_val_mIoU
            patience_counter = 0 
            torch.save(net.module.state_dict(), os.path.join(fold_exp_path, 'best.pth'))
            logging.info(f"Epoch {epoch}: Saved best model with mIoU: {best_mIoU:.5f}")
        else:
            patience_counter += 1
        
        torch.save({
            'epoch': epoch, 'model_state_dict': net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mIoU': best_mIoU, 'patience_counter': patience_counter
        }, os.path.join(fold_exp_path, 'latest_checkpoint.pth'))
        
        if patience_counter >= args['patience']:
            logging.info("Early stopping triggered.")
            break
            
    writer.close()
    return best_mIoU

# --- ORIGINAL `main` FUNCTION (Now `run_training_process`) ---
def run_training_process(args, train_loader, val_loader, fold_exp_path):
    net = daseg(backbone_path).train().to(device)
    optimizer = optim.Adam([
        {'params': [p for n, p in net.named_parameters() if n.endswith('bias')], 'lr': 2 * args['lr']},
        {'params': [p for n, p in net.named_parameters() if not n.endswith('bias')], 'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ]) if args['optimizer'] == 'Adam' else optim.SGD([
        {'params': [p for n, p in net.named_parameters() if n.endswith('bias')], 'lr': 2 * args['lr']},
        {'params': [p for n, p in net.named_parameters() if not n.endswith('bias')], 'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    start_epoch, best_mIoU, patience_counter = 1, 0.0, 0
    latest_ckpt_path = os.path.join(fold_exp_path, 'latest_checkpoint.pth')
    if args['resume'] and os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resumed from epoch {start_epoch}. Best mIoU: {best_mIoU:.4f}.")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
    
    net = nn.DataParallel(net) 
    return train(net, optimizer, args, train_loader, val_loader, fold_exp_path, start_epoch, best_mIoU, patience_counter)

# --- NEW: K-Fold Orchestration Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANet Training with K-Fold')
    parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--k-folds', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    args = vars(parser.parse_args())

    exp_name = 'DANet_' + args['dataset']
    base_exp_path = os.path.join('./ckpt', exp_name)
    check_mkdir(base_exp_path)
    setup_logging(base_exp_path, 'main_training_log.log')
    logging.info(f"Starting experiment: '{exp_name}' with arguments: {args}")

    if args['k_folds'] > 1:
        train_path = DATASET_PATHS[f"{args['dataset']}_train"]
        val_path = DATASET_PATHS[f"{args['dataset']}_test"]
        all_imgs = np.array(make_dataset(train_path, f"{args['dataset']}_train") + make_dataset(val_path, f"{args['dataset']}_test"))
        kf = KFold(n_splits=args['k_folds'], shuffle=True, random_state=args['random_state'])
        kfold_state_path = os.path.join(base_exp_path, 'kfold_state.json')
        start_fold, all_fold_metrics = 0, {}

        if args['resume'] and os.path.exists(kfold_state_path):
            with open(kfold_state_path, 'r') as f:
                state = json.load(f)
                start_fold = state.get('next_fold_to_run', 0)
                all_fold_metrics = state.get('all_fold_metrics', {})
            logging.info(f"Resuming k-fold process from fold {start_fold}.")

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_imgs)):
            if fold_idx < start_fold: continue
            
            fold_exp_path = os.path.join(base_exp_path, f"fold_{fold_idx}")
            check_mkdir(fold_exp_path)
            setup_logging(fold_exp_path, f'fold_{fold_idx}_training.log')
            with open(kfold_state_path, 'w') as f: json.dump({'next_fold_to_run': fold_idx, 'all_fold_metrics': all_fold_metrics}, f)
            
            train_loader = DataLoader(ImageFolder(root=None, dataset_name=args['dataset'], split='train', imgs=all_imgs[train_indices].tolist()), batch_size=args['train_batch_size'], num_workers=0, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(ImageFolder(root=None, dataset_name=args['dataset'], split='val', imgs=all_imgs[val_indices].tolist()), batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
            
            best_fold_mIoU = run_training_process(args, train_loader, val_loader, fold_exp_path)
            all_fold_metrics[f'fold_{fold_idx}'] = best_fold_mIoU
            with open(kfold_state_path, 'w') as f: json.dump({'next_fold_to_run': fold_idx + 1, 'all_fold_metrics': all_fold_metrics}, f)
        
        logging.info(f"K-Fold training finished. Avg Val mIoU: {np.mean(list(all_fold_metrics.values())):.4f}")
    else:
        fold_exp_path = os.path.join(base_exp_path, "fold_0")
        check_mkdir(fold_exp_path)
        setup_logging(fold_exp_path, 'fold_0_training.log')
        train_loader = DataLoader(ImageFolder(DATASET_PATHS[f"{args['dataset']}_train"], f"{args['dataset']}_train", split='train'), batch_size=args['train_batch_size'], num_workers=0, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(ImageFolder(DATASET_PATHS[f"{args['dataset']}_test"], f"{args['dataset']}_test", split='val'), batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
        run_training_process(args, train_loader, val_loader, fold_exp_path)