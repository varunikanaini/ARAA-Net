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
import logging # Import logging
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

from config import backbone_path, DATASET_PATHS 
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from daseg import daseg
import loss

from seg_utils import ConfusionMatrix

cudnn.benchmark = True

torch.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add argparse for dataset selection
parser = argparse.ArgumentParser(description='DANet Training')
parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis',
                    help='Dataset to use for training (TSRS_RSNA-Epiphysis, JSRT, COVID19_Radiography, CVC-ClinicDB)')
parser.add_argument('--patience', type=int, default=20,
                    help='Number of epochs to wait for improvement before early stopping')
args_parser = parser.parse_args()


# Dynamic root paths based on selected dataset
if args_parser.dataset == 'TSRS_RSNA-Epiphysis':
    train_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_train']
    test_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_test']
    train_dataset_name = 'TSRS_RSNA-Epiphysis_train'
    test_dataset_name = 'TSRS_RSNA-Epiphysis_test'
elif args_parser.dataset == 'JSRT':
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
exp_name = 'DANet_' + args_parser.dataset 

# Path setup and logging configuration
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)

# Configure logging
log_filename = os.path.join(ckpt_path, exp_name, 'training.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)


args = {
    'epoch_num': 100, 
    'train_batch_size': 10, 
    'last_epoch': 0,
    'lr': 1e-3, 
    'lr_decay': 0.9,
    'weight_decay': 5e-4, 
    'momentum': 0.9,
    'snapshot': '',
    'scale_w': 576,
    'scale_h': 896,
    'poly_train': True,
    'optimizer': 'Adam', 
    'patience': args_parser.patience # Use patience from argparse
}

# Log initial arguments
logger.info(f"Command-line arguments: {args_parser.__dict__}")
logger.info(f"Training arguments: {args}")
logger.info(f"Training dataset: {train_dataset_name}, Validation dataset: {test_dataset_name}")

# Prepare Data Set.
train_set = ImageFolder(train_root_path, train_dataset_name, split='train')
logger.info(f"Train set ({train_dataset_name}): {train_set.__len__()} images")
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)

test_set = ImageFolder(test_root_path, test_dataset_name, split='test') 
logger.info(f"Validation set ({test_dataset_name}): {test_set.__len__()} images")
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

total_iterations = args['epoch_num'] * len(train_loader)
logger.info(f"Total training iterations: {total_iterations}")

# loss function
structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255) 
logger.info(f"Using BCE+IOU for intermediate losses and CrossEntropy for final prediction.")

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

def train(net, optimizer):
    net.train()
    curr_iter = 1
    
    best_mIoU = -1.0 # Initialize with a low value for comparison
    patience_counter = 0 # Initialize patience counter

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
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

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0

            loss.backward()
            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
            
            global_acc, class_acc, class_iou, FWIoU, mDice = confmat.compute() # Compute metrics
            current_train_miou = np.mean(class_iou.cpu().numpy())

            if curr_iter % 10 == 0:
                writer.add_scalar('train/loss_total', loss_record.avg, curr_iter)
                writer.add_scalar('train/loss_bce_iou_1', loss_1_record.avg, curr_iter)
                writer.add_scalar('train/loss_structure_2', loss_2_record.avg, curr_iter)
                writer.add_scalar('train/loss_structure_3', loss_3_record.avg, curr_iter)
                writer.add_scalar('train/loss_structure_4', loss_4_record.avg, curr_iter)
                writer.add_scalar('train/loss_crossentropy_0', loss_0_record.avg, curr_iter)
                writer.add_scalar('train/miou', current_train_miou, curr_iter)
                writer.add_scalar('train/mdice', mDice, curr_iter)
                writer.add_scalar('train/learning_rate', base_lr, curr_iter)

            log_str = f"Epoch: {epoch:03d}/{args['epoch_num']}, Iter: {curr_iter:06d}/{total_iterations}, LR: {base_lr:.6f}, Total_Loss: {loss_record.avg:.5f}, CE_Loss: {loss_0_record.avg:.5f}, Train_mIoU: {current_train_miou:.5f}, Train_mDice: {mDice:.5f}"
            train_iterator.set_description(log_str)
            logger.info(log_str) # Log to file and console

            curr_iter += 1
        
        # Validation after each epoch
        current_val_mIoU = validate(net, epoch) # Pass epoch for logging
        writer.add_scalar('val/miou', current_val_mIoU, epoch)
        logger.info(f"Epoch {epoch} validation mIoU: {current_val_mIoU:.5f}")

        # Checkpoint: Save best model
        if current_val_mIoU > best_mIoU:
            best_mIoU = current_val_mIoU
            patience_counter = 0 # Reset patience if improvement
            checkpoint_path = os.path.join(ckpt_path, exp_name, 'best.pth')
            # Save state_dict of the original model, not DataParallel wrapper
            if isinstance(net, nn.DataParallel):
                torch.save(net.module.state_dict(), checkpoint_path)
            else:
                torch.save(net.state_dict(), checkpoint_path)
            logger.info(f"Epoch {epoch}: Saved best model with mIoU: {best_mIoU:.5f} to {checkpoint_path}")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch}: Validation mIoU did not improve. Patience counter: {patience_counter}/{args['patience']}")
            
        # Checkpoint: Save latest model at the end of every epoch
        latest_checkpoint_path = os.path.join(ckpt_path, exp_name, f'{epoch}.pth')
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.state_dict(), latest_checkpoint_path)
        else:
            torch.save(net.state_dict(), latest_checkpoint_path)
        logger.info(f"Epoch {epoch}: Saved latest model to {latest_checkpoint_path}")

        # Early stopping check
        if patience_counter >= args['patience']:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement. Best mIoU: {best_mIoU:.5f}")
            break
            
        logger.info(f"Epoch {epoch} finished. Current best mIoU: {best_mIoU:.5f}")


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
    logger.info(val_log_str) # Log to file and console

    net.train() # Set net back to train mode after validation
    return np.mean(class_iou)


def main():
    logger.info("Starting training process.")
    
    net = daseg(backbone_path).train()
    net = net.to(device)

    # Optimizer configuration as per paper (Adam)
    if args['optimizer'] == 'Adam':
        logger.info("Using Adam optimizer")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name.endswith('bias')], 
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if not name.endswith('bias')], 
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else: 
        logger.info("Using SGD optimizer")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name.endswith('bias')],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if not name.endswith('bias')],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        logger.info(f'Training Resumes From snapshot: {args["snapshot"]}')
        model_path = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Remove 'module.' prefix if the model was saved from DataParallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            args['last_epoch'] = int(args['snapshot']) 
            logger.info(f"Resuming training from epoch {args['last_epoch']}")
        else:
            logger.warning(f"Snapshot not found at {model_path}. Starting from scratch.")

    net = nn.DataParallel(net) 
    
    train(net, optimizer)
    writer.close()
    logger.info("Training process completed.")


if __name__ == '__main__':
    main()