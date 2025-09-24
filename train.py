#!/usr/bin/env python3
import os
import time
import logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sys
sys.path.append('/kaggle/working/ARAA-Net/')

from daseg import daseg
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING
from datasets import ImageFolder, DATASET_CONFIGS
from datasets import make_dataset as make_full_dataset_list
import joint_transforms # Keep if custom_transforms relies on it, though ImageFolder handles most
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import shutil
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with multi-backbone support and multiple dataset support')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for training')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    
    # Arguments directly from your requested command
    parser.add_argument('--epoch-num', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--snapshot', type=str, default='', help='Path to snapshot for resuming (relative to ckpt_path)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    
    # Image transformation related arguments (required by ImageFolder)
    parser.add_argument('--scale-h', type=int, default=576, help='Height images were resized to for ImageFolder transforms')
    parser.add_argument('--scale-w', type=int, default=896, help='Width images were resized to for ImageFolder transforms')
    parser.add_argument('--crop-size-h', type=int, default=576, help='Height images were cropped to for ImageFolder transforms.')
    parser.add_argument('--crop-size-w', type=int, default=576, help='Width images were cropped to for ImageFolder transforms.')

    # These are specific to CenterAmplification which was in a previous ImageFolder reference, 
    # but not explicitly used in the ImageFolder provided above, but kept as args for compatibility 
    # if ImageFolder were modified to use them later. They will be ignored in the current ImageFolder.
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder (if CenterAmplification used).')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder (if CenterAmplification used).')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder (if CenterAmplification used).')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder (if CenterAmplification used).')
    
    # Programmatic splitting ratios (used for COVID-19_Radiography)
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio for programmatic splitting.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio for programmatic splitting.')
    
    # Deep supervision weights (these are actively used by the daseg model's loss function)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[1.0, 1.0, 2.0, 4.0, 10.0],
                        help='Weights for deep supervision losses for predict_1 to predict_0 (total 5 values).')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
        
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")

    return args

# Global loss functions (defined once)
structure_loss_fn = None
bce_loss_fn = None
iou_loss_fn = None
ce_loss_fn = None

def bce_iou_loss(pred, target):
    global bce_loss_fn, iou_loss_fn
    return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)


def validate(net, test_loader, device, writer=None, curr_iter=None, args=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long()
            
            # Use deep supervision weights from args for validation loss calculation
            loss_1 = bce_iou_loss(predict_1, binary_labels) * args.deep_supervision_weights[0]
            loss_2 = structure_loss_fn(predict_2, binary_labels) * args.deep_supervision_weights[1]
            loss_3 = structure_loss_fn(predict_3, binary_labels) * args.deep_supervision_weights[2]
            loss_4 = structure_loss_fn(predict_4, binary_labels) * args.deep_supervision_weights[3]
            loss_0 = ce_loss_fn(predict_0, ce_labels) * args.deep_supervision_weights[4]
            
            total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_0
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), predict_0.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info("\n--- Validation Results ---")
    logging.info(f"global_acc = {global_acc.item():.4f}")
    logging.info(f"class_acc  = {class_acc.cpu().numpy()}")
    logging.info(f"class_iou  = {class_iou.cpu().numpy()}")
    logging.info(f"mIoU       = {mIoU:.4f}")
    logging.info(f"FWIoU      = {fwiou.item():.4f}")
    logging.info(f"mDice      = {mDice:.4f}")
    logging.info(f"Validation Loss = {loss_recorder.avg:.4f}")
    logging.info("--------------------------")
    
    if writer and curr_iter is not None:
        writer.add_scalar('validation/mIoU', mIoU, curr_iter)
        writer.add_scalar('validation/loss', loss_recorder.avg, curr_iter)
    
    net.train()
    return mIoU

def main():
    global structure_loss_fn, bce_loss_fn, iou_loss_fn, ce_loss_fn
    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2021)
        torch.backends.cudnn.benchmark = True
    np.random.seed(2021)

    exp_name = f"{args.backbone}_ARAA-Net_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    
    vis_path = os.path.join(exp_path, 'log')
    check_mkdir(vis_path)
    writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(exp_path, 'training.log')), logging.StreamHandler()])
    logging.info(f"Starting Training for experiment '{exp_name}'")
    logging.info(f"Arguments: {args}")
    logging.info(f"Using device: {device}")

    # --- Determine the base root for the dataset (download if KaggleHub) ---
    dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
    base_dataset_root = None

    if dataset_info and dataset_info['id']:
        base_dataset_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
        if not base_dataset_root:
            logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (KaggleHub): {base_dataset_root}")
    elif dataset_info and dataset_info['local_dir_name']:
        base_dataset_root = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
        if not os.path.exists(base_dataset_root):
            logging.error(f"Local dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (local): {base_dataset_root}")
    else:
        base_dataset_root = os.path.join(DATA_ROOT, args.dataset_name)
        if not os.path.exists(base_dataset_root):
            logging.error(f"Dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (default): {base_dataset_root}")

    # --- Data Loading and Splitting Logic ---
    dataset_config = DATASET_CONFIGS.get(args.dataset_name)
    if not dataset_config:
        logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
        sys.exit(1)

    train_image_mask_list = []
    val_image_mask_list = []

    if dataset_config['has_predefined_splits']:
        logging.info(f"Using predefined splits for dataset '{args.dataset_name}'.")
        train_full_path = os.path.join(base_dataset_root, 'train')
        val_full_path = os.path.join(base_dataset_root, 'val')

        train_image_mask_list = make_full_dataset_list(train_full_path, args.dataset_name, split_name='train')
        val_image_mask_list = make_full_dataset_list(val_full_path, args.dataset_name, split_name='val')

        if not train_image_mask_list:
            logging.error(f"No training data found in '{train_full_path}'. Exiting.")
            sys.exit(1)
        if not val_image_mask_list:
            logging.error(f"No validation data found in '{val_full_path}'. Exiting.")
            sys.exit(1)

    else:
        logging.info(f"Performing programmatic splitting for dataset '{args.dataset_name}'.")
        full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
        
        if not full_image_mask_list:
            logging.error(f"No data found for programmatic splitting in '{base_dataset_root}'. Exiting.")
            sys.exit(1)

        total_len = len(full_image_mask_list)
        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        test_len = total_len - train_len - val_len

        train_image_mask_list, val_image_mask_list, _ = random_split(
            full_image_mask_list, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
        
        logging.info(f"Programmatic split: Total {total_len}, Train {len(train_image_mask_list)}, Val {len(val_image_mask_list)}")

    train_set = ImageFolder(train_image_mask_list, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    
    test_set = ImageFolder(val_image_mask_list, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images.")
    
    net = daseg(backbone_name=args.backbone).to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net = nn.DataParallel(net)
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else:
        optimizer = optim.SGD([
            {'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ], momentum=args.momentum)

    structure_loss_fn = loss.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device)

    start_epoch = 0
    best_mIoU = 0.0
    latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    
    if os.path.exists(latest_ckpt_path):
        logging.info(f"Resuming from checkpoint: {latest_ckpt_path}")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt['model_state_dict']
            if 'module.' in list(state_dict.keys())[0]:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch, best_mIoU = 0, 0.0

    total_iterations = len(train_loader) * args.epoch_num
    
    try:
        for epoch in range(start_epoch, args.epoch_num):
            net.train()
            loss_recorder = AvgMeter()
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch_num} [Train]")
            for i, data in enumerate(train_iterator):
                curr_iter = epoch * len(train_loader) + i
                base_lr = args.lr * (1 - curr_iter / total_iterations) ** args.lr_decay
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = base_lr
            
                inputs, labels = data['image'].to(device), data['label'].to(device)
                binary_labels = labels.unsqueeze(1).float()
                ce_labels = labels.long()
                optimizer.zero_grad(set_to_none=True)
                predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
                
                loss_1 = bce_iou_loss(predict_1, binary_labels) * args.deep_supervision_weights[0]
                loss_2 = structure_loss_fn(predict_2, binary_labels) * args.deep_supervision_weights[1]
                loss_3 = structure_loss_fn(predict_3, binary_labels) * args.deep_supervision_weights[2]
                loss_4 = structure_loss_fn(predict_4, binary_labels) * args.deep_supervision_weights[3]
                loss_0 = ce_loss_fn(predict_0, ce_labels) * args.deep_supervision_weights[4]
                
                total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_0
                total_loss.backward()
                optimizer.step()
                loss_recorder.update(total_loss.item(), inputs.size(0))
                
                writer.add_scalar('train/loss', total_loss.item(), curr_iter)
                writer.add_scalar('train/loss_1', loss_1.item(), curr_iter)
                writer.add_scalar('train/loss_2', loss_2.item(), curr_iter)
                writer.add_scalar('train/loss_3', loss_3.item(), curr_iter)
                writer.add_scalar('train/loss_4', loss_4.item(), curr_iter)
                writer.add_scalar('train/loss_0', loss_0.item(), curr_iter)
                
                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{base_lr:.6f}")
                
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    current_mIoU = validate(net, test_loader, device, writer, curr_iter, args)
                    logging.info(f"Iteration {curr_iter}: mIoU = {current_mIoU:.4f}")

            current_mIoU = validate(net, test_loader, device, writer, (epoch + 1) * len(train_loader), args)
            
            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU
                checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
                torch.save(net.state_dict(), checkpoint_path)
                logging.info(f"✅ New best model saved at {checkpoint_path} with mIoU: {best_mIoU:.4f}")
                shutil.copy(checkpoint_path, f'/kaggle/working/best_checkpoint_{exp_name}.pth')
            
            checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, checkpoint_path)
            logging.info(f"Saved latest checkpoint to {checkpoint_path}")
            shutil.copy(checkpoint_path, f'/kaggle/working/latest_checkpoint_{exp_name}.pth')
            
    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")
        writer.close()

if __name__ == '__main__':
    main()