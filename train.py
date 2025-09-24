# /kaggle/working/ARAA-Net/train.py

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

# Ensure project path is in sys.path
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import necessary components from your project structure
from daseg import daseg
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING
from datasets import ImageFolder, DATASET_CONFIGS
from datasets import make_dataset as make_full_dataset_list
import custom_transforms as tr
import loss # Assuming loss.py contains structure_loss, IOU
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import shutil
import torch.nn.functional as F # Ensure F is imported here for loss.py's internal usage


def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with multi-backbone support and dynamic dataset/transforms')
    
    # --- JSRT-specific defaults for clarity ---
    parser.add_argument('--dataset-name', type=str, default='JSRT', # Default to JSRT
                        choices=list(DATASET_CONFIGS.keys()), help='Name of the dataset to train on')
    parser.add_argument('--backbone', type=str, default='resnet50', # Default to resnet50
                        choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    # --- End JSRT-specific defaults ---

    parser.add_argument('--epoch-num', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--snapshot', type=str, default='', help='Path to snapshot for resuming (relative to ckpt_path)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio for datasets without predefined splits.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio for datasets without predefined splits.')
    
    # Dummy args for ImageFolder compatibility. These values are overridden by dataset_config['transform_params']
    # in main(), but they need to exist in the Namespace initially.
    parser.add_argument('--scale-h', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--scale-w', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--crop-size-h', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--crop-size-w', type=int, default=448, help='Dummy for ImageFolder init, actual from config')


    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # For notebook compatibility
    
    return args

def validate(net, test_loader, device, bce_iou_loss_fn, structure_loss_fn, ce_loss_fn, writer=None, curr_iter=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
            binary_labels = labels.unsqueeze(1).float() # (B, 1, H, W)
            ce_labels = labels.long() # (B, H, W)
            
            loss_1 = bce_iou_loss_fn(predict_1, binary_labels) 
            loss_2 = structure_loss_fn(predict_2, binary_labels)
            loss_3 = structure_loss_fn(predict_3, binary_labels)
            loss_4 = structure_loss_fn(predict_4, binary_labels)
            loss_0 = ce_loss_fn(predict_0, ce_labels)
            
            loss_total = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
            loss_recorder.update(loss_total.item(), inputs.size(0))
            
            confmat.update(labels.flatten(), predict_0.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info("\n--- Validation Results ---")
    logging.info(f"global_acc = {global_acc.item():.4f}")
    logging.info(f"class_acc  = {class_acc}")
    logging.info(f"class_iou  = {class_iou}")
    logging.info(f"mIoU       = {mIoU:.4f}")
    logging.info(f"FWIoU      = {fwiou.item():.4f}")
    logging.info(f"mDice      = {mDice:.4f}")
    logging.info(f"Validation Loss = {loss_recorder.avg:.4f}")
    logging.info("--------------------------")
    
    if writer and curr_iter is not None:
        writer.add_scalar('validation/mIoU', mIoU, curr_iter)
        writer.add_scalar('validation/loss', loss_recorder.avg, curr_iter)
    
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2021)
        torch.backends.cudnn.benchmark = True

    exp_name = f"{args.backbone}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    
    vis_path = os.path.join(exp_path, 'log')
    check_mkdir(vis_path)
    writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
    
    log_file_path = os.path.join(exp_path, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"Starting Training with Arguments: {args}")
    logging.info(f"Using device: {device}")

    dataset_config = DATASET_CONFIGS.get(args.dataset_name)
    if not dataset_config:
        logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
        sys.exit(1)

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

    train_image_mask_list = []
    val_image_mask_list = []
    
    if dataset_config['has_predefined_splits']:
        train_root = os.path.join(base_dataset_root, 'train')
        val_root = os.path.join(base_dataset_root, 'val')
        train_image_mask_list = make_full_dataset_list(train_root, args.dataset_name, split_name='train')
        val_image_mask_list = make_full_dataset_list(val_root, args.dataset_name, split_name='val')
        logging.info(f"Using predefined splits: Train {len(train_image_mask_list)}, Val {len(val_image_mask_list)}")
    else:
        full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
        if not full_image_mask_list:
            logging.error(f"No data found for dataset '{args.dataset_name}' for programmatic splitting. Exiting.")
            sys.exit(1)

        total_len = len(full_image_mask_list)
        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        test_len = total_len - train_len - val_len

        g = torch.Generator().manual_seed(42)
        train_image_mask_list, val_image_mask_list, _ = random_split(
            full_image_mask_list, [train_len, val_len, test_len], generator=g)
        logging.info(f"Programmatic split: Total {total_len}, Train {len(train_image_mask_list)}, Val {len(val_image_mask_list)}")

    args.scale_w = dataset_config['transform_params']['resize_w']
    args.scale_h = dataset_config['transform_params']['resize_h']
    args.crop_size_h = dataset_config['transform_params']['crop_size_h']
    args.crop_size_w = dataset_config['transform_params']['crop_size_w']

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

    def bce_iou_loss_fn_wrapper(pred, target): return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    start_epoch = 0
    best_mIoU = 0.0
    latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if args.snapshot:
        snapshot_path = os.path.join(CKPT_ROOT, exp_name, args.snapshot + '.pth')
        if os.path.exists(snapshot_path):
            logging.info(f"Resuming from snapshot: {snapshot_path}")
            try:
                ckpt = torch.load(snapshot_path, map_location=device, weights_only=False) 
                state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

                if 'module.' in list(state_dict.keys())[0] and not isinstance(net, nn.DataParallel):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                elif not 'module.' in list(state_dict.keys())[0] and isinstance(net, nn.DataParallel):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                
                net.load_state_dict(state_dict)
                
                if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    start_epoch = ckpt['epoch'] + 1
                    best_mIoU = ckpt.get('best_mIoU', 0.0)
                logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}")
            except Exception as e:
                logging.error(f"Could not load snapshot: {e}. Starting from scratch.")
                start_epoch, best_mIoU = 0, 0.0
    elif os.path.exists(latest_ckpt_path):
        logging.info(f"Resuming from checkpoint: {latest_ckpt_path}")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt['model_state_dict']

            if 'module.' in list(state_dict.keys())[0] and not isinstance(net, nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif not 'module.' in list(state_dict.keys())[0] and isinstance(net, nn.DataParallel):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}

            net.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch, best_mIoU = 0, 0.0

    total_iterations = len(train_loader) * args.epoch_num
    loss_recorder = AvgMeter()
    
    try:
        for epoch in range(start_epoch, args.epoch_num):
            net.train()
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch_num} [Train]")
            for i, data in enumerate(train_iterator):
                curr_iter = epoch * len(train_loader) + i
                base_lr = args.lr * (1 - curr_iter / total_iterations) ** args.lr_decay
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = base_lr
            
                inputs, labels = data['image'].to(device), data['label'].to(device)
                binary_labels = labels.unsqueeze(1).float() # (B, 1, H, W)
                ce_labels = labels.long() # (B, H, W)
                optimizer.zero_grad(set_to_none=True)
                
                predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
                
                loss_1 = bce_iou_loss_fn_wrapper(predict_1, binary_labels)
                loss_2 = structure_loss_fn(predict_2, binary_labels)
                loss_3 = structure_loss_fn(predict_3, binary_labels)
                loss_4 = structure_loss_fn(predict_4, binary_labels)
                loss_0 = ce_loss_fn(predict_0, ce_labels)
                
                total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
                total_loss.backward()
                optimizer.step()
                loss_recorder.update(total_loss.item(), inputs.size(0))
                
                writer.add_scalar('train/loss', total_loss.item(), curr_iter)
                writer.add_scalar('train/loss_1', loss_1.item(), curr_iter)
                writer.add_scalar('train/loss_2', loss_2.item(), curr_iter)
                writer.add_scalar('train/loss_3', loss_3.item(), curr_iter)
                writer.add_scalar('train/loss_4', loss_4.item(), curr_iter)
                writer.add_scalar('train/loss_0', loss_0.item(), curr_iter)
                writer.add_scalar('train/lr', base_lr, curr_iter)
                
                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{base_lr:.6f}")
                
            # --- VALIDATE ONLY AT THE END OF EACH EPOCH ---
            # This call happens *after* the entire training loop for the epoch has completed.
            # We pass the final curr_iter for TensorBoard logging.
            current_mIoU = validate(net, test_loader, device, bce_iou_loss_fn_wrapper, structure_loss_fn, ce_loss_fn, writer, curr_iter)
            logging.info(f"Epoch {epoch+1} End Validation: mIoU = {current_mIoU:.4f}")


            # Save best model
            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU
                best_model_path = os.path.join(exp_path, 'best_checkpoint.pth')
                torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, best_model_path) # Save full checkpoint for best
                logging.info(f"✅ New best model saved at {best_model_path} with mIoU: {best_mIoU:.4f}")
                # Persist to Kaggle output
                shutil.copy(best_model_path, '/kaggle/working/best_checkpoint.pth')
            
            # Save latest checkpoint (always overwrites previous latest)
            latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, latest_checkpoint_path) # Save full checkpoint for latest
            logging.info(f"Saved latest checkpoint to {latest_checkpoint_path}")
            # Persist to Kaggle output
            shutil.copy(latest_checkpoint_path, '/kaggle/working/latest_checkpoint.pth')
            
    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")
        writer.close()

if __name__ == '__main__':
    main()