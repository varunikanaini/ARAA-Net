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
from config import cod_training_root, test_path, PSEUDO_LABEL_CONF_THRESHOLD, EMA_DECAY_RATE # Import new params
from datasets import ImageFolder
import joint_transforms
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import torch.nn.functional as F # For softmax and interpolation in pseudo-labeling

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with multi-backbone support')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    parser.add_argument('--epoch-num', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    # Removed --lr-decay as paper suggests constant LR for T/S, or will be handled in a simplified way
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--snapshot', type=str, default='', help='Path to snapshot for resuming (relative to ckpt_path)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    return parser.parse_args()

def validate(net, test_loader, device, writer=None, curr_iter=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            # The network outputs 5 predictions. For validation, we typically evaluate the final one (pred0)
            _, _, _, _, pred0 = net(inputs) 
            
            # Loss calculation for validation, consistent with training
            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long() # Assuming labels are 0 or 1, and CrossEntropy takes long()
            
            # We compute a combined loss for validation to track overall performance, 
            # similar to the total_loss in training.
            # However, since pred1-4 are intermediate, we only use pred0 for final metrics (mIoU etc.)
            # For validation loss, it's safer to use just the final prediction (pred0) with relevant loss.
            # Or, if you want full consistency, apply all losses. Let's use final CE loss for simplicity here.
            
            # For validation, we use original labels
            val_loss = ce_loss_fn(pred0, ce_labels) # Use only the most refined prediction for validation loss
            loss_recorder.update(val_loss.item(), inputs.size(0))
            
            # For metrics, we use the final prediction (pred0)
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

# Helper function to update teacher parameters via EMA
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

    # Set Kaggle-compatible checkpoint path
    ckpt_path = '/kaggle/working/ckpt'
    exp_name = args.backbone + '_teacher_student' # Append suffix for T/S runs
    exp_path = os.path.join(ckpt_path, exp_name)
    check_mkdir(exp_path)
    
    # Initialize TensorBoard
    vis_path = os.path.join(exp_path, 'log')
    check_mkdir(vis_path)
    writer = SummaryWriter(log_dir=vis_path, comment=exp_name)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(exp_path, 'training.log')), logging.StreamHandler()])
    logging.info(f"Starting Training with Teacher-Student framework, Backbone: {args.backbone}, Arguments: {args}")
    logging.info(f"Using device: {device}")
    logging.info(f"Pseudo-label confidence threshold: {PSEUDO_LABEL_CONF_THRESHOLD}")
    logging.info(f"EMA decay rate: {EMA_DECAY_RATE}")

    # Adjust input size for inception_v3
    scale_h = 299 if args.backbone == 'inception_v3' else 896
    scale_w = 299 if args.backbone == 'inception_v3' else 576
    
    # Data Transformations & Dataloaders
    # Note: custom_transforms are applied in ImageFolder.transform_tr/val
    # joint_transforms are applied before custom_transforms for global augmentations.
    # The paper mentions random resize, which your custom_transforms already include for training (RandomScaleCrop/RandomCrop).
    # For simplicity, we'll keep the existing transform logic and ensure it's compatible.
    
    # Original joint_transform
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((scale_h, scale_w)), # Resize to base for potential cropping
        joint_transforms.RandomCrop((299 if args.backbone == 'inception_v3' else 576, 299 if args.backbone == 'inception_v3' else 576), pad_if_needed=True, lbl_fill=0) # Changed lbl_fill to 0 for background
    ])
    val_joint_transform = joint_transforms.Compose([joint_transforms.Resize((scale_h, scale_w))])
    
    # NOTE: The current `ImageFolder` expects `joint_transform` (PIL operations)
    # and then `transform` (ToTensor, Normalize).
    # We will pass a `None` to `joint_transform` in ImageFolder to avoid double application
    # and apply all necessary transforms within `transform_tr` and `transform_val` which use `custom_transforms`.
    # This also allows for consistent normalization, etc., applied to both image and label as needed by custom_transforms.
    train_set = ImageFolder(cod_training_root, split='train') # Transforms are handled internally by ImageFolder
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(test_path, split='val') # Transforms are handled internally by ImageFolder
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images.")
    
    # --- Instantiate Student and Teacher Models ---
    net_student = daseg(backbone_name=args.backbone).to(device)
    net_teacher = daseg(backbone_name=args.backbone).to(device)
    
    # Initialize teacher weights with student weights and set to eval mode
    net_teacher.load_state_dict(net_student.state_dict())
    net_teacher.eval() # Teacher is always in evaluation mode and does not get gradients
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net_student = nn.DataParallel(net_student)
        net_teacher = nn.DataParallel(net_teacher) # Wrap teacher too if student is DP

    # Optimizer (only for student)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': [p for n, p in net_student.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net_student.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else: # SGD
        optimizer = optim.SGD([
            {'params': [p for n, p in net_student.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
            {'params': [p for n, p in net_student.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ], momentum=args.momentum)

    # Loss Functions
    global structure_loss_fn, bce_iou_loss_fn, ce_loss_fn # Changed bce_iou_loss to bce_iou_loss_fn
    structure_loss_fn = loss.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device) # Keep ignore_index for original GT potentially
    
    # Define a combined BCE+IOU loss function
    def bce_iou_loss_fn(pred, target): 
        return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    # Checkpoint Resuming Logic
    start_epoch = 0
    best_mIoU = 0.0
    latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if args.snapshot:
        snapshot_path = os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')
        if os.path.exists(snapshot_path):
            logging.info(f"Resuming from snapshot: {snapshot_path}")
            try:
                ckpt = torch.load(snapshot_path, map_location=device, weights_only=False)
                state_dict = ckpt if args.snapshot else ckpt['model_state_dict']
                if 'module.' in list(state_dict.keys())[0]:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                net_student.load_state_dict(state_dict)
                net_teacher.load_state_dict(state_dict) # Also load teacher state
                if not args.snapshot:
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
            if 'module.' in list(state_dict.keys())[0]:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            net_student.load_state_dict(state_dict)
            net_teacher.load_state_dict(state_dict) # Also load teacher state
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            logging.info(f"Loaded epoch: {start_epoch}, best_mIoU: {best_mIoU}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch, best_mIoU = 0, 0.0

    # Training Loop
    total_iterations = len(train_loader) * args.epoch_num
    loss_recorder = AvgMeter()
    
    try:
        for epoch in range(start_epoch, args.epoch_num):
            net_student.train() # Student is in training mode
            net_teacher.eval()  # Teacher is always in eval mode
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch_num} [Train]")
            for i, data in enumerate(train_iterator):
                curr_iter = epoch * len(train_loader) + i
                
                # --- Learning Rate Schedule: Paper suggests constant LR for T/S ---
                # Comment out or simplify LR decay if following paper's suggestion
                # current_lr = args.lr # A simpler constant LR
                current_lr = args.lr * (1 - curr_iter / total_iterations) ** args.lr # Original decay if you still want it
                optimizer.param_groups[0]['lr'] = 2 * current_lr
                optimizer.param_groups[1]['lr'] = current_lr
            
                inputs, original_labels = data['image'].to(device), data['label'].to(device)
                
                # --- Teacher Inference & Pseudo-Label Generation ---
                with torch.no_grad():
                    # Teacher provides predictions
                    # The network returns pred4, pred3, pred2, pred1, pred0
                    # pred0 is the final prediction for CE_Loss, which is [B, 2, H, W]
                    _, _, _, _, teacher_pred0 = net_teacher(inputs)
                    
                    # Get softmax probabilities from teacher's final prediction (pred0)
                    teacher_probs = F.softmax(teacher_pred0, dim=1) # [B, 2, H, W]
                    
                    # Get the predicted class (0 for background, 1 for foreground)
                    teacher_mask_classes = teacher_probs.argmax(dim=1) # [B, H, W]
                    
                    # Get confidence for the foreground class (class 1)
                    foreground_confidence = teacher_probs[:, 1, :, :] # [B, H, W]
                    
                    # Initialize pseudo-labels with zeros
                    mined_pseudo_labels = torch.zeros_like(original_labels, dtype=torch.float, device=device)
                    
                    # Identify reliable foreground pixels from teacher
                    reliable_foreground_pixels = (teacher_mask_classes == 1) & (foreground_confidence > PSEUDO_LABEL_CONF_THRESHOLD)
                    
                    # Identify existing positive labels from original ground truth
                    is_original_gt_positive = (original_labels > 0) # Assuming 0 is background
                    
                    # Pixels that are reliably foreground by teacher AND NOT positive in original GT
                    # These are the *missing* annotations the teacher found
                    newly_mined_pixels = reliable_foreground_pixels & (~is_original_gt_positive)
                    
                    mined_pseudo_labels[newly_mined_pixels] = 1.0
                    
                    # Combine original ground truth with newly mined pseudo-labels
                    # Use torch.max to ensure if either is positive, the combined label is positive
                    # Y_combined will be [B, H, W]
                    Y_combined_segmentation = torch.max(is_original_gt_positive.float(), mined_pseudo_labels)
                    
                # --- Prepare Labels for Student Loss Calculation ---
                # For BCEWithLogitsLoss and IOU, we need [B, 1, H, W] float tensors
                binary_combined_labels = Y_combined_segmentation.unsqueeze(1).float()
                # For CrossEntropyLoss, we need [B, H, W] long tensors
                ce_combined_labels = Y_combined_segmentation.long()
                
                optimizer.zero_grad(set_to_none=True)
                
                # Student makes predictions
                predict_1, predict_2, predict_3, predict_4, predict_0 = net_student(inputs)
                
                # Calculate losses using the combined labels (original + pseudo)
                loss_1 = bce_iou_loss_fn(predict_1, binary_combined_labels)
                loss_2 = structure_loss_fn(predict_2, binary_combined_labels)
                loss_3 = structure_loss_fn(predict_3, binary_combined_labels)
                loss_4 = structure_loss_fn(predict_4, binary_combined_labels)
                loss_0 = ce_loss_fn(predict_0, ce_combined_labels) # Use CE loss for the final prediction

                # Original weighting of losses
                total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
                
                total_loss.backward()
                optimizer.step()
                
                # --- Update Teacher Weights (EMA) ---
                update_teacher_weights(net_student, net_teacher, EMA_DECAY_RATE)
                
                loss_recorder.update(total_loss.item(), inputs.size(0))
                
                writer.add_scalar('train/loss', total_loss.item(), curr_iter)
                writer.add_scalar('train/loss_1', loss_1.item(), curr_iter)
                writer.add_scalar('train/loss_2', loss_2.item(), curr_iter)
                writer.add_scalar('train/loss_3', loss_3.item(), curr_iter)
                writer.add_scalar('train/loss_4', loss_4.item(), curr_iter)
                writer.add_scalar('train/loss_0', loss_0.item(), curr_iter)
                writer.add_scalar('train/lr', current_lr, curr_iter) # Log current LR

                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{current_lr:.6f}")
                
                # Validate every N iterations (e.g., 100 or at end of epoch)
                # Reduced frequency for validation inside loop to avoid slowing down too much.
                # A full epoch validation is performed after the loop.
                if (i + 1) % 100 == 0: # Validate less frequently within epoch
                    current_mIoU = validate(net_student, test_loader, device, writer, curr_iter)
                    logging.info(f"Iteration {curr_iter}: mIoU = {current_mIoU:.4f}")
                    net_student.train() # Set back to train mode

            # --- Epoch End Validation ---
            current_mIoU = validate(net_student, test_loader, device, writer, curr_iter) # Validate at epoch end
            
            # --- Checkpoint Saving ---
            # Save the student model (which is the one we want to deploy)
            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU
                checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
                # Save student's state dict
                if isinstance(net_student, nn.DataParallel):
                    torch.save(net_student.module.state_dict(), checkpoint_path)
                else:
                    torch.save(net_student.state_dict(), checkpoint_path)
                logging.info(f"✅ New best model saved at {checkpoint_path} with mIoU: {best_mIoU:.4f}")
                shutil.copy(checkpoint_path, '/kaggle/working/best_checkpoint.pth') # Persist to Kaggle output
            
            # Save latest checkpoint (for resuming)
            latest_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': (net_student.module.state_dict() if isinstance(net_student, nn.DataParallel) else net_student.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_mIoU
            }
            checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
            torch.save(latest_checkpoint_data, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            shutil.copy(checkpoint_path, '/kaggle/working/latest_checkpoint.pth') # Persist to Kaggle output
            
    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")
        writer.close()

if __name__ == '__main__':
    main()