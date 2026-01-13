#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime, time, os, argparse, logging, sys, json
from collections import OrderedDict
from sklearn.model_selection import KFold
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import config
from datasets import ImageFolder, make_dataset
from misc import AvgMeter, check_mkdir
from daseg import daseg
import loss
from seg_utils import ConfusionMatrix

cudnn.benchmark = True
torch.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_dir, filename='training.log'):
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(log_dir, filename)), logging.StreamHandler(sys.stdout)])

# Losses
structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + iou_out

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def validate(net, val_loader, epoch, num_classes):
    net.eval()
    confmat = ConfusionMatrix(num_classes=num_classes)
    val_iter = tqdm(val_loader, desc=f"Epoch {epoch} (Val)")
    with torch.no_grad():
        for data in val_iter:
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            *_, predict0 = net(inputs)
            confmat.update(labels.flatten(), predict0.argmax(1).flatten())
    _, _, class_iou, _, mDice = confmat.compute()
    val_miou = np.mean(class_iou.cpu().numpy())
    logging.info(f'--- Validation Results (Epoch {epoch}) --- mIoU: {val_miou:.4f}, Dice: {mDice:.4f}')
    net.train()
    return val_miou

def train(net, optimizer, args, train_loader, val_loader, fold_exp_path, start_epoch, best_mIoU, patience, num_classes):
    total_iter = args['epoch_num'] * len(train_loader)
    curr_iter = (start_epoch - 1) * len(train_loader) + 1
    
    # Adaptive Criterion
    # For binary, use BCE+IoU. For multiclass, use CE.
    is_multiclass = num_classes > 2
    if is_multiclass:
        criterion_ce = nn.CrossEntropyLoss(ignore_index=255).to(device)
    
    for epoch in range(start_epoch, args['epoch_num'] + 1):
        loss_rec = AvgMeter()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args['epoch_num']} (Train)")
        
        for data in train_iter:
            if data is None: continue
            
            # LR Scheduler
            base_lr = args['lr'] * (1 - float(curr_iter) / float(total_iter)) ** args['lr_decay']
            optimizer.param_groups[0]['lr'] = 2 * base_lr
            optimizer.param_groups[1]['lr'] = 1 * base_lr
            
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            
            preds = net(inputs) # Returns list of predictions (p1, p2, p3, p4, p0)
            
            if is_multiclass:
                # Use Cross Entropy for all auxiliary outputs
                # Labels must be Long Tensor [B, H, W]
                target = labels.long()
                loss = 0
                weights = [1, 1, 2, 4, 10]
                for pred, w in zip(preds, weights):
                    loss += w * criterion_ce(pred, target)
            else:
                # Use BCE+IoU and Structure Loss
                # Labels must be [B, 1, H, W]
                target = labels.unsqueeze(1)
                p1, p2, p3, p4, p0 = preds
                loss = 1*bce_iou_loss(p1, target) + 1*structure_loss(p2, target) + \
                       2*structure_loss(p3, target) + 4*structure_loss(p4, target) + \
                       10*nn.CrossEntropyLoss(ignore_index=255)(p0, labels.long())

            loss.backward()
            optimizer.step()
            loss_rec.update(loss.item(), inputs.size(0))
            curr_iter += 1

        logging.info(f'--- Epoch {epoch} Summary --- Loss: {loss_rec.avg:.4f}')
        val_miou = validate(net, val_loader, epoch, num_classes)
        
        if val_miou > best_mIoU:
            best_mIoU, patience = val_miou, 0
            torch.save(net.module.state_dict(), os.path.join(fold_exp_path, 'best.pth'))
            logging.info(f"New Best mIoU: {best_mIoU:.5f}")
        else:
            patience += 1
            
        torch.save({'epoch': epoch, 'model_state_dict': net.module.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU, 
                    'patience_counter': patience}, os.path.join(fold_exp_path, 'latest_checkpoint.pth'))
        
        if patience >= args['patience']:
            logging.info("Early stopping."); break
            
    return best_mIoU

def run_training_process(args, train_loader, val_loader, fold_exp_path, num_classes):
    net = daseg(config.backbone_path, num_classes=num_classes).train().to(device)
    
    # Optimizer params split
    params_bias = [p for n, p in net.named_parameters() if n.endswith('bias')]
    params_weight = [p for n, p in net.named_parameters() if not n.endswith('bias')]
    
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam([{'params': params_bias, 'lr': 2*args['lr']}, 
                                {'params': params_weight, 'lr': args['lr'], 'weight_decay': args['weight_decay']}])
    else:
        optimizer = optim.SGD([{'params': params_bias, 'lr': 2*args['lr']}, 
                               {'params': params_weight, 'lr': args['lr'], 'weight_decay': args['weight_decay']}], 
                              momentum=args['momentum'])
                              
    start_epoch, best_mIoU, patience = 1, 0.0, 0
    latest_ckpt = os.path.join(fold_exp_path, 'latest_checkpoint.pth')
    
    if args['resume'] and os.path.exists(latest_ckpt):
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch, best_mIoU, patience = ckpt['epoch']+1, ckpt.get('best_mIoU', 0.0), ckpt.get('patience_counter', 0)
            logging.info(f"Resumed from epoch {start_epoch}, Best mIoU: {best_mIoU:.4f}")
        except: logging.error("Checkpoint load failed, starting fresh.")
            
    net = nn.DataParallel(net)
    return train(net, optimizer, args, train_loader, val_loader, fold_exp_path, start_epoch, best_mIoU, patience, num_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--k-folds', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    args = vars(parser.parse_args())

    exp_name = 'DANet_' + args['dataset']
    base_exp_path = os.path.join('./ckpt', exp_name)
    check_mkdir(base_exp_path)
    setup_logging(base_exp_path, 'main_training_log.log')
    logging.info(f"Starting {exp_name} with {args}")
    
    dataset_cfg = config.DATASET_CONFIG[args['dataset']]
    num_classes = dataset_cfg.get('num_classes', 2)
    
    # Data Split
    imgs = np.array(make_dataset(dataset_cfg['path'], args['dataset'], split='all'), dtype=object)
    if len(imgs) == 0: raise ValueError(f"No images found for {args['dataset']}")
    
    kf = KFold(n_splits=args['k_folds'], shuffle=True, random_state=args['random_state'])
    state_path = os.path.join(base_exp_path, 'kfold_state.json')
    start_fold, all_metrics = 0, {}
    
    if args['resume'] and os.path.exists(state_path):
        with open(state_path, 'r') as f:
            st = json.load(f)
            start_fold, all_metrics = st.get('next_fold', 0), st.get('metrics', {})
            
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(imgs)):
        if fold_idx < start_fold: continue
        
        fold_path = os.path.join(base_exp_path, f"fold_{fold_idx}")
        check_mkdir(fold_path)
        setup_logging(fold_path, f'fold_{fold_idx}_log.txt')
        
        with open(state_path, 'w') as f: json.dump({'next_fold': fold_idx, 'metrics': all_metrics}, f)
        
        train_dl = DataLoader(ImageFolder(None, args['dataset'], 'train', imgs[tr_idx].tolist()), 
                              batch_size=args['train_batch_size'], shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
        val_dl = DataLoader(ImageFolder(None, args['dataset'], 'val', imgs[val_idx].tolist()), 
                            batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
                            
        best = run_training_process(args, train_dl, val_dl, fold_path, num_classes)
        all_metrics[f'fold_{fold_idx}'] = best
        
        with open(state_path, 'w') as f: json.dump({'next_fold': fold_idx+1, 'metrics': all_metrics}, f)
        
    logging.info(f"Finished. Avg mIoU: {np.mean(list(all_metrics.values())):.4f}")