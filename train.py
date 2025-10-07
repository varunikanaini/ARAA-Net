# train.py
import os
import time
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.nn.functional as F

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import config 
from lasa_unet_model import LASA_Unet 
from datasets import ImageFolder, make_dataset, IMAGE_EXTENSIONS, MASK_EXTENSIONS
from config import CKPT_ROOT, DATASET_CONFIG, BACKBONE_CHANNELS, DEFAULT_ARGS # Import config
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        if num_classes > 1:
            # Multi-class, one-vs-all approach for Dice Loss
            # We typically apply Dice Loss per class and average, or focus on foreground.
            # Here, assuming num_classes is for the final output layer.
            # If your target is binary (0/1), this should be handled correctly.
            # For multi-class segmentation, this needs careful implementation.
            # Let's assume num_classes=2 for binary case as in original code.
            # If num_classes > 2, you might need a different DiceLoss implementation for multi-class.
            if num_classes == 2:
                pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1) # Extract foreground prob
                true_oh = (targets == 1).float().unsqueeze(1) # Binary mask for foreground
            else: # General multi-class Dice (requires averaging or specific class focus)
                # This is a simplified version for binary/multi-class where we want to average Dice
                # It might need adjustment if specific class Dice is required.
                true_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
                pred_probs = F.softmax(inputs, dim=1)
        else: # Binary case where output is sigmoid (num_classes=1)
            pred_probs = F.sigmoid(inputs).unsqueeze(1)
            true_oh = targets.float().unsqueeze(1)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            # Ensure mask is broadcastable to pred_probs and true_oh shapes
            if mask.ndim == 3: # If mask is (N, H, W), unsqueeze for channels
                mask = mask.unsqueeze(1)
            pred_probs = pred_probs * mask
            true_oh = true_oh * mask
        
        # Flatten for easier calculation
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice
        
        # Reduction logic
        if self.reduction == 'mean':
            return loss # Dice loss is often averaged over the batch directly
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] # Scale by batch size
        else: # 'none' or other values
            return loss 

# train.py (Modified get_args function)

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Multi-Dataset and Multi-Backbone Support')
    
    # --- Explicitly add arguments, using DEFAULT_ARGS for their default values ---

    # --- Dataset Selection ---
    parser.add_argument('--dataset-name', type=str, default=DEFAULT_ARGS['dataset_name'],
                        choices=DATASET_CONFIG.keys(), help='Name of the dataset to use')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'], help='Dataset split to load')

    # --- Backbone Selection ---
    # This is the problematic one. We want it to be explicitly defined here.
    # The 'choices' should ideally come from config.BACKBONE_CHANNELS.keys()
    parser.add_argument('--backbone', type=str, default=DEFAULT_ARGS['backbone'],
                        choices=BACKBONE_CHANNELS.keys(), help='Backbone architecture to use')

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=DEFAULT_ARGS['epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_ARGS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_ARGS['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_ARGS['weight_decay'])
    parser.add_argument('--patience', type=int, default=DEFAULT_ARGS['patience'],
                        help='Patience for early stopping based on validation mIoU.')

    # --- Image Preprocessing ---
    parser.add_argument('--scale-h', type=int, default=DEFAULT_ARGS['scale_h'], help='Nominal height for resizing')
    parser.add_argument('--scale-w', type=int, default=DEFAULT_ARGS['scale_w'], help='Nominal width for resizing')
    
    # --- LASA Module Arguments ---
    # If lasa_kernels is a key in DEFAULT_ARGS, add it here explicitly.
    # Assuming it's handled by the loop or needs to be added if not in DEFAULT_ARGS.
    # If it IS in DEFAULT_ARGS:
    parser.add_argument('--lasa-kernels', type=int, default=DEFAULT_ARGS.get('lasa_kernels'), nargs='+',
                        help='Kernel sizes for LASA module')


    # --- Deep Supervision Weights ---
    # If it's in DEFAULT_ARGS, add it explicitly.
    parser.add_argument('--deep-supervision-weights', type=float, default=DEFAULT_ARGS.get('deep_supervision_weights'), nargs='+',
                        help='Weights for deep supervision outputs')

    # --- Loss Function Parameters ---
    parser.add_argument('--focal-alpha', type=float, default=DEFAULT_ARGS['focal_alpha'], help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=DEFAULT_ARGS['focal_gamma'], help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=DEFAULT_ARGS['focal_loss_weight'], help='Weight for Focal Loss component.')
    parser.add_argument('--dice-loss-weight', type=float, default=DEFAULT_ARGS['dice_loss_weight'], help='Weight for Dice Loss component.')

    # --- Data Augmentation Parameters ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=DEFAULT_ARGS['min_lesion_area_pixels'], help='Min lesion area for CenterAmplification.')
    parser.add_argument('--expansion-factor', type=float, default=DEFAULT_ARGS['expansion_factor'], help='Expansion factor for CenterAmplification.')
    parser.add_argument('--min-bbox-h', type=int, default=DEFAULT_ARGS['min_bbox_h'], help='Min bbox height for CenterAmplification.')
    parser.add_argument('--min-bbox-w', type=int, default=DEFAULT_ARGS['min_bbox_w'], help='Min bbox width for CenterAmplification.')
    parser.add_argument('--wavelet-type', type=str, default=DEFAULT_ARGS['wavelet_type'], help='Wavelet type for DWT contrast enhancement.')
    parser.add_argument('--wavelet-level', type=int, default=DEFAULT_ARGS['wavelet_level'], help='DWT decomposition level.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=DEFAULT_ARGS['wavelet_detail_scale'], help='Scaling factor for DWT detail coefficients.')

    # --- Scheduler Parameters ---
    parser.add_argument('--scheduler-patience', type=int, default=DEFAULT_ARGS['scheduler_patience'], help='Patience for ReduceLROnPlateau.')
    parser.add_argument('--scheduler-factor', type=float, default=DEFAULT_ARGS['scheduler_factor'], help='Factor for ReduceLROnPlateau.')
    parser.add_argument('--scheduler-min-lr', type=float, default=DEFAULT_ARGS['scheduler_min_lr'], help='Minimum learning rate for the scheduler.')

    # --- Control Flow ---
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')

    # --- Add num_workers argument explicitly ---
    parser.add_argument('--num-workers', type=int, default=DEFAULT_ARGS['num_workers'], help='Number of data loading workers.')
    
    # --- Parse Arguments ---
    # Use parse_args([]) for environments like notebooks where sys.exit might be too harsh
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # Fallback for notebooks
    
    # Post-parsing validation and adjustments
    # Ensure deep supervision weights have the correct length (5 for this model)
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    # Get dataset specific config and update args if necessary
    try:
        dataset_info = config.get_dataset_info(args.dataset_name)
        args.dataset_path = dataset_info['path']
        args.dataset_structure = dataset_info['structure']
        args.num_classes = dataset_info['num_classes']
        args.image_ext = dataset_info.get('image_ext', IMAGE_EXTENSIONS)
        args.mask_ext = dataset_info.get('mask_ext', MASK_EXTENSIONS)
        args.dataset_subfolders = dataset_info.get('subfolders')
    except ValueError as e:
        parser.error(str(e))
    
    # If --backbone is not provided and default is also not in BACKBONE_CHANNELS, handle it.
    # This should not happen if DEFAULT_ARGS['backbone'] is valid.
    if args.backbone not in BACKBONE_CHANNELS:
        parser.error(f"The specified backbone '{args.backbone}' is not supported. Supported backbones are: {list(BACKBONE_CHANNELS.keys())}")

    return args

# --- Logging Setup ---
def setup_logging(log_dir, filename='training.log'):
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to prevent duplicate logs if script is re-run in an interactive session
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating"):
    """
    Evaluates the model on a given data_loader. Returns mIoU and average loss.
    """
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes) # Use num_classes from args
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{mode}", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs) 
            final_pred = outputs[-1] # The last output is always the final prediction
            
            total_loss = 0
            # Calculate combined loss across all heads using deep supervision weights
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            # Update confusion matrix with the final prediction
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA (Overall Accuracy): {global_acc.item():.4f}")
    logging.info(f"  mIoU (Mean IoU): {mIoU:.4f}")
    logging.info(f"  FWIoU (Frequency Weighted IoU): {fwiou.item():.4f}")
    logging.info(f"  Dice (Mean Dice Coefficient): {mDice:.4f}")
    
    if mode == "Validating": 
        net.train() # Set back to train mode
    return mIoU

# --- Custom Collate Function ---
# Needed if ImageFolder might return None for corrupted samples
def custom_collate_fn(batch):
    # Filter out None samples that might be returned if a file failed to load
    batch = [item for item in batch if item is not None] 
    if not batch: # If the batch is empty after filtering
        return None
    # Use default collate for the remaining valid samples
    return torch.utils.data.dataloader.default_collate(batch) 

# --- Main Training Function ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Experiment Naming and Logging Setup ---
    # Dynamically create experiment name based on dataset, backbone, and key parameters
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels)) if args.lasa_kernels else "nolasa"
    exp_name_parts = [
        args.backbone,
        f"LASA{lasa_kernels_str}",
        f"DSW{'_'.join(map(str, args.deep_supervision_weights))}", # Example: DSW0.2_0.4_0.6_0.8_1.0
        f"FLW{args.focal_loss_weight}_DLW{args.dice_loss_weight}",
        args.dataset_name.replace('TSRS_RSNA-', '').lower(),
    ]
    exp_name = "_".join(exp_name_parts)
    
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path) # Setup logging for this experiment

    logging.info(f"Starting training for experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")

    # --- Instantiate Model ---
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    
    # --- Instantiate Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

    # --- Resuming Training ---
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    
    if args.resume and os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming training from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")
            args.resume = False # Reset resume flag if loading fails

    # --- Data Loading ---
    # Create datasets and dataloaders.
    # We can support multiple datasets by loading them separately and then concatenating.
    
    # For simplicity, we'll start by assuming training on a single dataset specified by --dataset-name.
    # If you need to train on multiple datasets simultaneously, you'd use ConcatDataset.
    
    # Get dataset info for the chosen dataset
    dataset_info = config.get_dataset_info(args.dataset_name)
    
    # Load training data
    train_data_path = os.path.join(dataset_info['path'], 'train') # Assuming 'train' split exists
    if not os.path.exists(train_data_path):
        logging.warning(f"Train split not found at '{train_data_path}'. This might be an issue if --test-only is not set.")
        # If not in test-only mode, this is an error.
        if not args.test_only:
             raise FileNotFoundError(f"Train split not found for dataset '{args.dataset_name}'. Expected at: {train_data_path}")
        train_set = None # Set to None if not needed (e.g., test_only mode)
    else:
        train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
        logging.info(f"Loaded {len(train_set)} training images from '{args.dataset_name}' split 'train'.")

    # Load validation data
    val_data_path = os.path.join(dataset_info['path'], 'val') # Assuming 'val' split exists
    if not os.path.exists(val_data_path):
        logging.warning(f"Validation split not found at '{val_data_path}'. Validation will be skipped if not in test-only mode.")
        val_set = None
    else:
        val_set = ImageFolder(val_data_path, args.dataset_name, args, split='val')
        val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
        logging.info(f"Loaded {len(val_set)} validation images from '{args.dataset_name}' split 'val'.")

    # --- Test-Only Mode ---
    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{exp_path}'. Please run training first or specify correct path.")
            sys.exit(1)

        try:
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"✅ Model loaded successfully from {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
            sys.exit(1)

        # Determine test data path. If 'test' split exists, use it. Otherwise, use 'val'.
        test_split_name = 'test'
        test_data_path = os.path.join(dataset_info['path'], test_split_name)
        if not os.path.exists(test_data_path):
            logging.warning(f"Test split not found at '{test_data_path}'. Falling back to 'val' split for testing.")
            test_split_name = 'val'
            test_data_path = os.path.join(dataset_info['path'], test_split_name)
            if not os.path.exists(test_data_path):
                logging.error(f"❌ ERROR: Neither 'test' nor 'val' split found for dataset '{args.dataset_name}'. Cannot proceed with testing.")
                sys.exit(1)
        
        test_set = ImageFolder(test_data_path, args.dataset_name, args, split=test_split_name)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
        logging.info(f"Loaded {len(test_set)} images for testing from '{args.dataset_name}' split '{test_split_name}'.")

        # Run evaluation on the test set
        test_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing")
        
        # Log Final Test Results
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"\n\n--- FINAL TEST RESULTS ({timestamp}) ---")
        logging.info(f"Model: LASA-Unet with {args.backbone} backbone and LASA Kernels: {args.lasa_kernels}")
        logging.info(f"Dataset: {args.dataset_name} (evaluated on '{test_split_name}' split)")
        logging.info(f"Image scale for test: ({args.scale_h}, {args.scale_w})")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        logging.info("---------------------------------")
        logging.info("✅ Testing completed.")
        return 

    # --- Training Loop ---
    if train_set is None or val_set is None:
        logging.error("Training or Validation dataset could not be loaded. Exiting.")
        sys.exit(1)

    for epoch in range(start_epoch, args.epochs):
        net.train() 
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for i, data in enumerate(train_iterator):
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Iter {i}: Skipping empty training batch.")
                continue

            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True) 
            
            outputs = net(inputs)
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        # --- Validation Step ---
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating")

        # --- Scheduler Step ---
        scheduler.step(current_mIoU)

        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0 
            torch.save(net.state_dict(), best_checkpoint_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model to '{best_checkpoint_path}'.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ Validation mIoU did not improve for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save latest checkpoint regardless of improvement
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        # --- Early Stopping ---
        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered after {args.patience} epochs of no improvement.")
            break

    logging.info("Training finished.")

if __name__ == '__main__':
    main()