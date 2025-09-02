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
import shutil

# Ensure the project path is in sys.path
sys.path.append('/kaggle/working/ARAA-Net/')

from daseg import daseg
from config import cod_training_root, test_path, PSEUDO_LABEL_CONF_THRESHOLD, EMA_DECAY_RATE 
from datasets import ImageFolder 
import joint_transforms 
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import torch.nn.functional as F 

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with multi-backbone support')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    parser.add_argument('--epoch-num', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=5, help='Batch size for training') 
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    # --- ADDED: patience for early stopping ---
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement before early stopping')
    # --- MODIFIED: snapshot now implies loading only model weights (e.g., for pre-training) ---
    parser.add_argument('--snapshot', type=str, default='', help='Path to snapshot (model weights only, relative to exp_path)')
    # --- ADDED: explicit resume flag for full training state ---
    parser.add_argument('--resume', action='store_true', help='Resume full training state (optimizer, epoch, best_mIoU) from latest checkpoint')
    
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers') 
    return parser.parse_args()

def validate(net, test_loader, device, writer=None, curr_iter=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2) 
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred0 = net(inputs) 
            
            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long() 
            
            val_loss = ce_loss_fn(pred0, ce_labels) 
            loss_recorder.update(val_loss.item(), inputs.size(0))
            
            confmat.update(labels.flatten(), pred0.argmax(1).flatten())
            
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

def update_teacher_weights(student_model, teacher_model, alpha):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2021)
        torch.backends.cudnn.benchmark = True

    ckpt_path = '/kaggle/working/ARAA-Net/ckpt' # Corrected base path for checkpoints
    exp_name = args.backbone + "_teacher_student" 
    exp_path = os.path.join(ckpt_path, exp_name)
    check_mkdir(exp_path)
    
    vis_path = os.path.join(exp_path, 'log')
    check_mkdir(vis_path)
    writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
    
    log_file_path = os.path.join(exp_path, 'training.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stderr) 
        ]
    )

    logging.info(f"Starting Training with Teacher-Student framework, Backbone: {args.backbone}, Arguments: {args}")
    logging.info(f"Using device: {device}")
    logging.info(f"Pseudo-label confidence threshold: {PSEUDO_LABEL_CONF_THRESHOLD}")
    logging.info(f"EMA decay rate: {EMA_DECAY_RATE}")

    scale_h_target = 256 
    scale_w_target = 256 
    crop_size_target = 256 
    
    if args.backbone == 'inception_v3':
        scale_h_target = 256 
        scale_w_target = 256 
        crop_size_target = 256 

    train_set = ImageFolder(cod_training_root, split='train', 
                            scale_h=scale_h_target, scale_w=scale_w_target, crop_size=crop_size_target) 
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    
    test_set = ImageFolder(test_path, split='val',
                           scale_h=scale_h_target, scale_w=scale_w_target, crop_size=scale_h_target) 
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images.")
    logging.info(f"Training image dimensions (H, W): ({scale_h_target}, {scale_w_target}), Crop size: {crop_size_target}")

    net_student = daseg(backbone_name=args.backbone).to(device)
    net_teacher = daseg(backbone_name=args.backbone).to(device)
    
    net_teacher.load_state_dict(net_student.state_dict())
    net_teacher.eval() 
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net_student = nn.DataParallel(net_student)
        net_teacher = nn.DataParallel(net_teacher) 

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': [p for n, p in net_student.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net_student.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else: 
        optimizer = optim.SGD([
            {'params': [p for n, p in net_student.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net_student.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ], momentum=args.momentum)

    global structure_loss_fn, bce_iou_loss_fn, ce_loss_fn 
    structure_loss_fn = loss.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device) 
    
    def bce_iou_loss_fn(pred, target): 
        return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    start_epoch = 0
    best_mIoU = 0.0
    patience_counter = 0 # Initialize patience counter

    # --- ENHANCED CHECKPOINT LOADING LOGIC ---
    exp_latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    kaggle_working_latest_ckpt_path = '/kaggle/working/latest_checkpoint.pth' # Kaggle persistence path
    
    # 1. Prioritize loading explicit snapshot for pre-training (model weights only)
    if args.snapshot:
        snapshot_full_path = os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')
        if not os.path.exists(snapshot_full_path):
            logging.warning(f"Snapshot '{snapshot_full_path}' not found. Starting from scratch (or attempting resume).")
        else:
            logging.info(f"Loading model weights from snapshot: {snapshot_full_path}")
            try:
                state_dict_to_load = torch.load(snapshot_full_path, map_location=device, weights_only=True)
                if 'module.' in list(state_dict_to_load.keys())[0]:
                    state_dict_to_load = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}
                net_student.load_state_dict(state_dict_to_load)
                net_teacher.load_state_dict(state_dict_to_load)
                logging.info("Snapshot loaded successfully. Training will start from epoch 0 with new optimizer state.")
            except Exception as e:
                logging.error(f"Error loading snapshot '{snapshot_full_path}': {e}. Starting from scratch.")
    
    # 2. Then, attempt to resume full training state (optimizer, epoch, best_mIoU)
    # This happens if '--resume' is specified, OR if no snapshot was loaded AND a latest checkpoint exists.
    # It prioritizes the experiment-specific path, then the Kaggle-persisted root path.
    if args.resume or (not args.snapshot and (os.path.exists(exp_latest_ckpt_path) or os.path.exists(kaggle_working_latest_ckpt_path))):
        
        checkpoint_to_load_path = None
        if os.path.exists(exp_latest_ckpt_path):
            checkpoint_to_load_path = exp_latest_ckpt_path
            logging.info(f"Attempting to resume from primary checkpoint: {exp_latest_ckpt_path}")
        elif os.path.exists(kaggle_working_latest_ckpt_path):
            checkpoint_to_load_path = kaggle_working_latest_ckpt_path
            logging.info(f"Attempting to resume from Kaggle-persisted fallback checkpoint: {kaggle_working_latest_ckpt_path}")
            
            # If loaded from fallback, copy it to the experiment dir for consistency
            try:
                if not os.path.exists(exp_latest_ckpt_path): # Only copy if not already there
                    shutil.copy(kaggle_working_latest_ckpt_path, exp_latest_ckpt_path)
                    logging.info(f"Copied fallback checkpoint to experiment path: {exp_latest_ckpt_path}")
            except Exception as e:
                logging.warning(f"Could not copy fallback checkpoint to experiment path: {e}")

        if checkpoint_to_load_path and os.path.exists(checkpoint_to_load_path): # Check again if path exists after potential copy
            try:
                ckpt = torch.load(checkpoint_to_load_path, map_location=device, weights_only=False)
                state_dict = ckpt['model_state_dict']
                if 'module.' in list(state_dict.keys())[0]:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                net_student.load_state_dict(state_dict)
                net_teacher.load_state_dict(state_dict)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                best_mIoU = ckpt.get('best_mIoU', 0.0)
                patience_counter = ckpt.get('patience_counter', 0) # Load patience counter if exists
                logging.info(f"Resumed training from epoch: {start_epoch}, best_mIoU: {best_mIoU:.4f}, patience_counter: {patience_counter}")
            except Exception as e:
                logging.error(f"Error resuming full training state from '{checkpoint_to_load_path}': {e}. Starting from scratch.")
                start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
        else:
            logging.info("No valid checkpoint found for resuming. Starting training from scratch.")
    else: # If no snapshot and no resume was intended
        logging.info("No snapshot or resume specified. Starting training from scratch.")
    # --- END ENHANCED CHECKPOINT LOADING ---


    total_iterations = len(train_loader) * args.epoch_num
    loss_recorder = AvgMeter()
    
    try:
        for epoch in range(start_epoch, args.epoch_num):
            net_student.train() 
            net_teacher.eval()  
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch_num} [Train]")
            for i, data in enumerate(train_iterator):
                curr_iter = epoch * len(train_loader) + i
                
                current_lr = args.lr 
                
                optimizer.param_groups[0]['lr'] = 2 * current_lr
                optimizer.param_groups[1]['lr'] = current_lr
            
                inputs, original_labels = data['image'].to(device), data['label'].to(device)
                
                with torch.no_grad():
                    _, _, _, _, teacher_pred0 = net_teacher(inputs)
                    
                    teacher_probs = F.softmax(teacher_pred0, dim=1) 
                    teacher_mask_classes = teacher_probs.argmax(dim=1) 
                    foreground_confidence = teacher_probs[:, 1, :, :] 
                    
                    mined_pseudo_labels = torch.zeros_like(original_labels, dtype=torch.float, device=device)
                    
                    reliable_foreground_pixels = (teacher_mask_classes == 1) & (foreground_confidence > PSEUDO_LABEL_CONF_THRESHOLD)
                    is_original_gt_positive = (original_labels > 0) 
                    newly_mined_pixels = reliable_foreground_pixels & (~is_original_gt_positive)
                    
                    mined_pseudo_labels[newly_mined_pixels] = 1.0
                    
                    Y_combined_segmentation = torch.max(is_original_gt_positive.float(), mined_pseudo_labels)
                    
                binary_combined_labels = Y_combined_segmentation.unsqueeze(1).float()
                ce_combined_labels = Y_combined_segmentation.long()
                
                optimizer.zero_grad(set_to_none=True)
                
                predict_1, predict_2, predict_3, predict_4, predict_0 = net_student(inputs)
                
                loss_1 = bce_iou_loss_fn(predict_1, binary_combined_labels)
                loss_2 = structure_loss_fn(predict_2, binary_combined_labels)
                loss_3 = structure_loss_fn(predict_3, binary_combined_labels)
                loss_4 = structure_loss_fn(predict_4, binary_combined_labels)
                loss_0 = ce_loss_fn(predict_0, ce_combined_labels) 

                total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
                
                total_loss.backward()
                optimizer.step()
                
                update_teacher_weights(net_student, net_teacher, EMA_DECAY_RATE)
                
                loss_recorder.update(total_loss.item(), inputs.size(0))
                
                writer.add_scalar('train/loss', total_loss.item(), curr_iter)
                writer.add_scalar('train/loss_1', loss_1.item(), curr_iter)
                writer.add_scalar('train/loss_2', loss_2.item(), curr_iter)
                writer.add_scalar('train/loss_3', loss_3.item(), curr_iter)
                writer.add_scalar('train/loss_4', loss_4.item(), curr_iter)
                writer.add_scalar('train/loss_0', loss_0.item(), curr_iter)
                writer.add_scalar('train/lr', current_lr, curr_iter) 

                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{current_lr:.6f}")
                
            current_mIoU = validate(net_student, test_loader, device, writer, curr_iter) 
            
            # --- Saving Checkpoints and Early Stopping ---
            is_best = current_mIoU > best_mIoU 
            if is_best:
                best_mIoU = current_mIoU
                patience_counter = 0 # Reset patience if improvement
                checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
                if isinstance(net_student, nn.DataParallel):
                    torch.save(net_student.module.state_dict(), checkpoint_path)
                else:
                    torch.save(net_student.state_dict(), checkpoint_path)
                logging.info(f"✅ Epoch {epoch+1}: New best mIoU: {best_mIoU:.4f}. Saving best model.")
                # Also copy to /kaggle/working/ for easy access and persistence
                shutil.copy(checkpoint_path, '/kaggle/working/best_checkpoint.pth') 
            else:
                patience_counter += 1
                logging.info(f"⚠️ Epoch {epoch+1}: No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")
            
            # Always save the latest checkpoint
            latest_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': (net_student.module.state_dict() if isinstance(net_student, nn.DataParallel) else net_student.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'patience_counter': patience_counter # Save patience counter for resuming
            }
            checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
            torch.save(latest_checkpoint_data, checkpoint_path)
            logging.info(f"Saved latest checkpoint to {checkpoint_path}")
            shutil.copy(checkpoint_path, '/kaggle/working/latest_checkpoint.pth') 
            
            # Check for early stopping
            if patience_counter >= args.patience: 
                logging.info("Early stopping triggered due to no improvement.")
                break # Exit the training loop

    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")
        writer.close()

if __name__ == '__main__':
    main()