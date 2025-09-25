#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import datetime
import time
import os
import argparse # Import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

# Removed unused import joint_transforms
from config import backbone_path, DATASET_PATHS # Import DATASET_PATHS from config
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from daseg import daseg
# Removed unused import DARConv2d
import loss

from seg_utils import ConfusionMatrix

cudnn.benchmark = True

torch.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add argparse for dataset selection
parser = argparse.ArgumentParser(description='DANet Training')
parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis',
                    help='Dataset to use for training (TSRS_RSNA-Epiphysis, JSRT, COVID19_Radiography, CVC-ClinicDB)')
args_parser = parser.parse_args()


# Dynamic root paths based on selected dataset
# For simplicity, assuming the same root for train/test splits for new datasets if a dedicated split is not defined
if args_parser.dataset == 'TSRS_RSNA-Epiphysis':
    train_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_train']
    test_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_test']
    train_dataset_name = 'TSRS_RSNA-Epiphysis_train'
    test_dataset_name = 'TSRS_RSNA-Epiphysis_test'
elif args_parser.dataset == 'JSRT':
    # JSRT might not have a dedicated train/test split, use the whole for both for now.
    # User might need to implement a split in ImageFolder if required.
    train_root_path = DATASET_PATHS['JSRT']
    test_root_path = DATASET_PATHS['JSRT'] 
    train_dataset_name = 'JSRT'
    test_dataset_name = 'JSRT'
elif args_parser.dataset == 'COVID19_Radiography':
    train_root_path = DATASET_PATHS['COVID19_Radiography']
    test_root_path = DATASET_PATHS['COVID19_Radiography']
    train_dataset_name = 'COVID19_Radiography'
    test_dataset_name = 'COVID19_Radiography'
elif args_parser.dataset == 'CVC-ClinicDB':
    train_root_path = DATASET_PATHS['CVC-ClinicDB']
    test_root_path = DATASET_PATHS['CVC-ClinicDB']
    train_dataset_name = 'CVC-ClinicDB'
    test_dataset_name = 'CVC-ClinicDB'
else:
    raise ValueError(f"Unsupported dataset: {args_parser.dataset}")

ckpt_path = './ckpt'
exp_name = 'DANet_' + args_parser.dataset # Append dataset name to experiment name for separate checkpoints

args = {
    'epoch_num': 100, # Changed from 1000 to 100 as per paper
    'train_batch_size': 10, # Kept as 10 as per paper
    'last_epoch': 0,
    'lr': 1e-3, # Kept as 1e-3 as per paper
    'lr_decay': 0.9,
    'weight_decay': 5e-4, # Kept as 5e-4 as per paper
    'momentum': 0.9,
    'snapshot': '',
    'scale_w': 576,
    'scale_h': 896,
    'poly_train': True,
    'optimizer': 'Adam', # Kept as Adam as per paper
}

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name,'log.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Prepare Data Set.
# Pass dataset_name to ImageFolder, transforms handled internally.
train_set = ImageFolder(train_root_path, train_dataset_name, split='train')
print(f"Train set ({train_dataset_name}): {train_set.__len__()} images")
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)

test_set = ImageFolder(test_root_path, test_dataset_name, split='test') # For validation during training
print(f"Test set ({test_dataset_name}) for validation: {test_set.__len__()} images")
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

total_iterations = args['epoch_num'] * len(train_loader)
print("total_iterations =", total_iterations)

# Loss functions (kept as per original functionality, acknowledging paper's mention of CrossEntropy)
structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255) # CrossEntropy for final prediction
print("iou_loss =", iou_loss)

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

def train(net, optimizer):
    net.train()
    curr_iter = 1
    
    best_mIoU = 0
    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        # Reset AvgMeters and ConfusionMatrix for each epoch
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_0_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        confmat = ConfusionMatrix(num_classes=2)

        train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args['epoch_num']} (Train)")
        
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_iterations)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data['image'], data['label']
            batch_size = inputs.size(0)
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)

            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1))
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1))
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1))
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1))       
            loss_0 = last_criterion(predict0, labels.long())

            # Combined loss as per original functionality, with emphasis on CrossEntropy (loss_0)
            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0

            loss.backward()
            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            
            # Update confusion matrix
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
            
            global_acc, class_acc, class_iou, FWIoU, mDice = confmat.compute()

            if curr_iter % 10 == 0:
                writer.add_scalar('loss/total', loss_record.avg, curr_iter)
                writer.add_scalar('loss/bce_iou_1', loss_1_record.avg, curr_iter)
                writer.add_scalar('loss/structure_2', loss_2_record.avg, curr_iter)
                writer.add_scalar('loss/structure_3', loss_3_record.avg, curr_iter)
                writer.add_scalar('loss/structure_4', loss_4_record.avg, curr_iter)
                writer.add_scalar('loss/crossentropy_0', loss_0_record.avg, curr_iter)
                writer.add_scalar('metrics/train_miou', np.mean(class_iou.cpu().numpy()), curr_iter)
                writer.add_scalar('metrics/train_mdice', mDice, curr_iter)


            log_str = f"Epoch {epoch}/{args['epoch_num']}, Iter {curr_iter}, LR: {base_lr:.6f}, Loss: {loss_record.avg:.5f}, mIoU: {np.mean(class_iou.cpu().numpy()):.5f}, mDice: {mDice:.5f}"
            train_iterator.set_description(log_str)
            open(log_path, 'a').write(log_str + '\n')

            curr_iter += 1
        
        # Validation after each epoch
        current_mIoU = validate(net, epoch) # Pass epoch for logging
        writer.add_scalar('metrics/val_miou', current_mIoU, epoch)

        if best_mIoU < current_mIoU:
            best_mIoU = current_mIoU
            print(f"Saving best model with mIoU: {best_mIoU:.5f}")
            # Save state_dict of the original model, not DataParallel wrapper
            if isinstance(net, nn.DataParallel):
                torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, 'best.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'best.pth'))
            
        # Also save epoch-specific checkpoints for potential later loading
        # Save state_dict of the original model
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
            
        print(f"Epoch {epoch} finished. Best mIoU so far: {best_mIoU:.5f}")


def validate(net, epoch): # Removed optimizer argument as it's not used in validation
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_record, loss_0_record = AvgMeter(), AvgMeter()

    test_iterator = tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch}/{args['epoch_num']} (Val)")
    for data in test_iterator:
        inputs, labels, _ = data['image'], data['label'], data['name'] # Unpack 'name' but don't use it
        
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
    
            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1))
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1))
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1))
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1))
            
            loss_0 = last_criterion(predict0, labels.long())
            
            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0
    
            loss_record.update(loss.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
        
            global_acc, class_acc, class_iou, FWIoU, mDice = confmat.compute()
            
            log_str = f"Epoch {epoch}/{args['epoch_num']}, Val Loss: {loss_record.avg:.5f}, Val mIoU: {np.mean(class_iou.cpu().numpy()):.5f}, Val mDice: {mDice:.5f}"
            test_iterator.set_description(log_str)
            # No need to write to log_path inside this loop, only final epoch metrics.

    # Compute final metrics for the epoch's validation
    global_acc, class_acc, class_iou, FWIoU, mDice = confmat.compute()
    global_acc = global_acc.item()
    class_acc = class_acc.cpu().numpy()
    class_iou = class_iou.cpu().numpy()
    FWIoU = FWIoU.cpu().numpy()
    
    val_log_str = (
        f'--- Validation Results (Epoch {epoch}) ---\n'
        f'Global Acc: {global_acc:.4f}\n'
        f'Class Acc: {class_acc}\n'
        f'Class IoU: {class_iou}\n'
        f'Mean IoU: {np.mean(class_iou):.4f}\n'
        f'FWIoU: {FWIoU:.4f}\n'
        f'Mean Dice: {mDice:.4f}\n'
    )
    print(val_log_str)
    open(log_path, 'a').write(val_log_str + '\n') # Write full validation results to log file

    net.train() # Set net back to train mode after validation
    return np.mean(class_iou)


def main():
    print("Args:", args_parser.__dict__)
    
    net = daseg(backbone_path).train()
    net = net.to(device)

    # Optimizer configuration as per paper (Adam)
    if args['optimizer'] == 'Adam':
        print("Using Adam optimizer")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name.endswith('bias')], # Targets bias parameters
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if not name.endswith('bias')], # Targets non-bias parameters
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else: # Fallback to SGD if specified, though paper uses Adam
        print("Using SGD optimizer")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name.endswith('bias')],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if not name.endswith('bias')],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print(f'Training Resumes From snapshot: {args["snapshot"]}')
        model_path = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))
            args['last_epoch'] = int(args['snapshot']) # Update last_epoch to resume correctly
            print(f"Resuming training from epoch {args['last_epoch']}")
        else:
            print(f"Snapshot not found at {model_path}. Starting from scratch.")

    net = nn.DataParallel(net) # Wrap model in DataParallel *after* loading state_dict
    
    # Write initial args to log file
    with open(log_path, 'w') as f:
        f.write(str(args_parser.__dict__) + '\n') # Log argparse arguments
        f.write(str(args) + '\n\n') # Log main args dictionary
        f.write(f"Training dataset: {train_dataset_name}, Test dataset for validation: {test_dataset_name}\n\n")

    train(net, optimizer)
    writer.close()


if __name__ == '__main__':
    main()