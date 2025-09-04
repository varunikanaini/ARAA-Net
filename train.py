#!/usr/bin/env python3
import os
import time
import logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sys
sys.path.append('/kaggle/working/ARAA-Net/')
from daseg import daseg
from config import DATA_ROOT, CKPT_ROOT # Import DATA_ROOT and CKPT_ROOT
from datasets import ImageFolder
import joint_transforms
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import shutil

# Declare loss functions as global here so they can be accessed by validate()
# Their initialization will happen within main().
structure_loss_fn = None
bce_loss_fn = None
iou_loss_fn = None
focal_loss_fn = None
bce_iou_loss = None # This will hold the combined BCE+IoU loss function

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with multi-backbone support')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Name of the dataset to use (e.g., TSRS_RSNA-Epiphysis, TSRS_RSNA-Articular-Surface)') # Added dataset-name arg
    parser.add_argument('--epoch-num', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement before early stopping') # Added patience arg
    parser.add_argument('--snapshot', type=str, default='', help='Path to snapshot for resuming (relative to exp_path)') # Corrected comment
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to for training')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to for training')
    parser.add_argument('--crop-size', type=int, default=576, help='Crop size used during training (square)')
    return parser.parse_args()

def validate(net, test_loader, device, writer=None, curr_iter=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        # tqdm for validation to show progress, but separate from epoch train tqdm
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            # Fetch predictions (need to execute the model to get them)
            predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs) 
            
            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long()
            
            # These global variables are now accessible
            loss_1 = bce_iou_loss(predict_1, binary_labels)
            loss_2 = structure_loss_fn(predict_2, binary_labels)
            loss_3 = structure_loss_fn(predict_3, binary_labels)
            loss_4 = structure_loss_fn(predict_4, binary_labels)
            loss_0 = focal_loss_fn(predict_0, ce_labels) 
            total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), predict_0.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    # Format validation results as a string to be printed and logged
    results_string = (
        f"\n--- Validation Results (mIoU: {mIoU:.4f}) ---\n"
        f"global_acc = {global_acc.item():.4f}\n"
        f"class_acc  = {class_acc}\n"
        f"class_iou  = {class_iou}\n"
        f"mIoU       = {mIoU:.4f}\n"
        f"FWIoU      = {fwiou.item():.4f}\n"
        f"mDice      = {mDice:.4f}\n"
        f"Validation Loss = {loss_recorder.avg:.4f}\n"
        f"--------------------------"
    )
    
    # Log results
    logging.info(results_string)
    
    if writer and curr_iter is not None:
        writer.add_scalar('validation/mIoU', mIoU, curr_iter)
        writer.add_scalar('validation/loss', loss_recorder.avg, curr_iter)
    
    return mIoU, results_string # Return both mIoU and the formatted string

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2021)
        torch.backends.cudnn.benchmark = True

    # Construct dataset paths dynamically based on args.dataset_name
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    cod_training_root = os.path.join(dataset_path, 'train')
    test_path_val = os.path.join(dataset_path, 'val') 

    # Set Kaggle-compatible checkpoint path
    # New experiment name format: backbone_name_ULD_datasetname
    exp_name = f"{args.backbone}_ULD_{args.dataset_name}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    
    # Initialize TensorBoard
    vis_path = os.path.join(exp_path, 'log')
    check_mkdir(vis_path)
    writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
    
    # Setup logging to file and console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(exp_path, 'training.log')),
                            logging.StreamHandler(sys.stdout) # Explicitly set stream handler to stdout
                        ])
    logging.info(f"Starting Training with Arguments: {args}")
    logging.info(f"Using device: {device}")

    # Adjust input size for inception_v3 if not explicitly overridden by args
    if args.backbone == 'inception_v3' and (args.scale_h == 896 or args.scale_w == 576 or args.crop_size == 576):
        logging.info("InceptionV3 detected, overriding scale/crop to 299x299 as per common practice.")
        args.scale_h = 299
        args.scale_w = 299
        args.crop_size = 299
    
    # Data Transformations & Dataloaders
    train_set = ImageFolder(cod_training_root, split='train', scale_h=args.scale_h, scale_w=args.scale_w, crop_size=args.crop_size)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    
    test_set = ImageFolder(test_path_val, split='val', scale_h=args.scale_h, scale_w=args.scale_w, crop_size=args.crop_size)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images for dataset '{args.dataset_name}'.")
    
    net = daseg(backbone_name=args.backbone).to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net = nn.DataParallel(net)
    
    # Optimizer
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

    # Loss Functions: Assign to global variables
    global structure_loss_fn, bce_loss_fn, iou_loss_fn, focal_loss_fn, bce_iou_loss
    structure_loss_fn = loss.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device)
    focal_loss_fn = loss.FocalLoss(alpha=1, gamma=2, reduction='mean', ignore_index=255).to(device)
    
    # Define the combined loss function globally after its components are global
    bce_iou_loss = lambda pred, target: bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    # Checkpoint Resuming Logic
    start_epoch = 0
    best_mIoU = 0.0
    epochs_no_improve = 0 # Initialize counter for early stopping
    
    latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if args.snapshot:
        snapshot_path = os.path.join(exp_path, args.snapshot + '.pth') 
        if os.path.exists(snapshot_path):
            logging.info(f"Resuming from snapshot: {snapshot_path}")
            try:
                ckpt = torch.load(snapshot_path, map_location=device, weights_only=False)
                state_dict = ckpt if args.snapshot else ckpt['model_state_dict']
                if 'module.' in list(state_dict.keys())[0]:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                net.load_state_dict(state_dict)
                if not args.snapshot: # if it's a full checkpoint (not just state_dict)
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    start_epoch = ckpt['epoch'] + 1
                    best_mIoU = ckpt.get('best_mIoU', 0.0)
                    epochs_no_improve = ckpt.get('epochs_no_improve', 0) # Load epochs_no_improve
                logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}, epochs_no_improve: {epochs_no_improve}")
            except Exception as e:
                logging.error(f"Could not load snapshot: {e}. Starting from scratch.")
                start_epoch, best_mIoU, epochs_no_improve = 0, 0.0, 0
    elif os.path.exists(latest_ckpt_path):
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
            epochs_no_improve = ckpt.get('epochs_no_improve', 0) # Load epochs_no_improve
            logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}, epochs_no_improve: {epochs_no_improve}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch, best_mIoU, epochs_no_improve = 0, 0.0, 0

    # Training Loop
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
                binary_labels = labels.unsqueeze(1).float()
                ce_labels = labels.long() 
                optimizer.zero_grad(set_to_none=True)
                predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
                
                loss_1 = bce_iou_loss(predict_1, binary_labels)
                loss_2 = structure_loss_fn(predict_2, binary_labels)
                loss_3 = structure_loss_fn(predict_3, binary_labels)
                loss_4 = structure_loss_fn(predict_4, binary_labels)
                loss_0 = focal_loss_fn(predict_0, ce_labels) 
                
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
                
                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{base_lr:.6f}")
                
            # Validate at the end of each epoch
            current_mIoU, validation_results_string = validate(net, test_loader, device, writer, curr_iter)
            
            # Print validation results string directly to console
            print(validation_results_string)

            # Early stopping logic
            if current_mIoU > best_mIoU:
                logging.info(f"✅ Epoch {epoch+1}: mIoU improved from {best_mIoU:.4f} to {current_mIoU:.4f}. Resetting patience.")
                best_mIoU = current_mIoU
                epochs_no_improve = 0 # Reset counter
                checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
                torch.save(net.state_dict(), checkpoint_path)
                shutil.copy(checkpoint_path, '/kaggle/working/best_checkpoint.pth')
            else:
                epochs_no_improve += 1
                print(f"⚠️ Epoch {epoch+1}: No improvement in mIoU for {epochs_no_improve} epoch(s). Best mIoU remains {best_mIoU:.4f}.") # Print to screen
                logging.warning(f"Epoch {epoch+1}: No improvement in mIoU for {epochs_no_improve} epoch(s). Best mIoU remains {best_mIoU:.4f}.") # Log to file
                
            # Save latest checkpoint with early stopping state
            checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                        'best_mIoU': best_mIoU, 'epochs_no_improve': epochs_no_improve}, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            shutil.copy(checkpoint_path, '/kaggle/working/latest_checkpoint.pth')

            if epochs_no_improve >= args.patience:
                print(f"🛑 Early stopping triggered after {args.patience} epochs without improvement. Training finished.") # Print to screen
                logging.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break
            
    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")
        writer.close()

if __name__ == '__main__':
    main()