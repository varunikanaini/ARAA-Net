# /kaggle/working/ARAA-Net/train_lasa_vgg.py

import os
import time
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS
# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet # <<< Ensure this imports the updated model
from datasets import ImageFolder, mixup_data # <<< Import mixup_data from datasets.py
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Ensure inputs are float and targets are long
        inputs = inputs.float()
        targets = targets.long()

        # Calculate cross-entropy loss per element
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            # Create a mask to exclude ignored_index pixels
            mask = (targets != self.ignore_index).float()
            # Sum the focal loss only for non-ignored pixels and divide by the count of non-ignored pixels
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6) # Add epsilon for stability
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
        # Ensure inputs are float and targets are long
        inputs = inputs.float()
        targets = targets.long()

        # Determine if multi-class or binary output
        if inputs.shape[1] > 1: # Multi-class case (expects num_classes output from model)
            # Use softmax to get probabilities for class 1 (assuming class 1 is the target class)
            # If your model outputs probabilities directly, skip softmax.
            # Ensure your model outputs logits for cross_entropy and sigmoid/softmax for Dice
            pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :] # Probability of class 1
            # Convert target to one-hot encoding for class 1
            true_oh = (targets == 1).float() 
        else: # Binary case (expects sigmoid output)
            pred_probs = torch.sigmoid(inputs).squeeze(1) # Probability of class 1
            true_oh = targets.float() # Target should already be binary 0/1

        # Apply ignore_index mask if specified
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            # Apply mask to both predictions and targets, ensuring they have the same number of dimensions for element-wise ops
            # Add a dimension to the mask to match pred_probs/true_oh for broadcasting if necessary
            if mask.ndim == 3: # e.g., (B, H, W)
                mask = mask.unsqueeze(1) # Make it (B, 1, H, W) if pred_probs is (B, 1, H, W) or (B, 1, H, W) if pred_probs is (B, C, H, W)
            pred_probs = pred_probs * mask
            true_oh = true_oh * mask
        
        # Flatten for calculation
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice_score
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] # Multiply by batch size for sum reduction
        else:
            return loss

def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Augmentations')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 'DentalPanoramic', 'SixDiseasesChestXRay'], help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], help='Weights for deep supervision losses.')
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss.')
    
    parser.add_argument('--mixup-alpha', type=float, default=0.4, help='Alpha for MixUp beta distribution.')

    parser.add_argument('--min-lesion-area-pixels', type=int, default=576)
    parser.add_argument('--expansion-factor', type=float, default=1.5)
    parser.add_argument('--min-bbox-h', type=int, default=32)
    parser.add_argument('--min-bbox-w', type=int, default=32)
    parser.add_argument('--wavelet-type', type=str, default='haar')
    parser.add_argument('--wavelet-level', type=int, default=1)
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--scheduler-patience', type=int, default=5) 
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6)

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, deep_supervision_weights, focal_loss_weight, dice_loss_weight, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2) 
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{mode}", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch.")
                continue
            
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs) 
            final_pred_for_metrics = outputs[-1] 
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                combined_loss_per_head = (focal_loss_weight * current_focal_loss) + (dice_loss_weight * current_dice_loss)
                
                # Ensure index is within bounds for deep supervision weights
                if i < len(deep_supervision_weights):
                    total_loss += deep_supervision_weights[i] * combined_loss_per_head
                else:
                    logging.warning(f"Deep supervision weight index {i} out of bounds. Skipping weight for output {i}.")

            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred_for_metrics.argmax(1).flatten())
            
    # Compute metrics only if confmat has data
    if confmat.mat is not None:
        global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute() # Assuming compute returns these values
        mIoU = class_iou.mean().item()
        logging.info(f"--- {mode} mIoU: {mIoU:.4f} | {mode} Loss: {loss_recorder.avg:.4f} ---")
    else:
        mIoU = 0.0 # Default to 0 if no data processed
        logging.warning(f"--- {mode}: No data processed for metric calculation. Loss: {loss_recorder.avg:.4f} ---")
    
    if mode == "Validating": 
        net.train() # Return to train mode after validation
        
    return mIoU

# --- Custom Collate Function ---
# This function needs to be defined in a scope accessible by DataLoader,
# typically in the training script or imported.
def custom_collate_fn(batch):
    """
    Custom collate function to filter out None samples from the batch.
    Returns None if the entire batch becomes empty after filtering.
    """
    batch = [item for item in batch if item is not None] # Filter out None samples
    if not batch: 
        return None # Return None if the batch is empty after filtering
    # Use default_collate for the remaining valid samples
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(2024)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(2024)
        torch.cuda.deterministic = True # Potentially makes CUDA ops deterministic
        torch.cuda.benchmark = False # May need to set to True for performance if model structure is fixed
    np.random.seed(2024)

    # Construct experiment name and path
    exp_name = f"{args.backbone}_LASA_Enc_ASPP_Attn_MixUp_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training for '{exp_name}' with arguments: {args}")

    # --- Initialize Model ---
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(device)
    
    # --- Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- Test Only Mode ---
    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"Best checkpoint not found at {best_checkpoint_path}. Exiting.")
            sys.exit(1)
        try:
            # Load only the model state dict for testing
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"Loaded model from {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model state dict: {e}. Exiting.")
            sys.exit(1)

        # Determine test data path (using 'val' for evaluation in test-only mode based on original log)
        base_dataset_path = DATASET_PATHS.get(args.dataset_name)
        if not base_dataset_path:
             logging.error(f"Dataset '{args.dataset_name}' not found in DATASET_PATHS. Exiting.")
             sys.exit(1)

        if 'TSRS_RSNA' in args.dataset_name:
            # Using 'val' split for evaluation in test-only mode, as per previous logs. Adjust if 'test' is truly intended.
            eval_data_path = os.path.join(base_dataset_path, 'val') 
        else:
            # For other datasets, assume 'val' split is used for evaluation if not specified otherwise.
            # Or use 'test' if a dedicated test split exists and is meant for final eval.
            eval_data_path = os.path.join(base_dataset_path, 'val') 
        
        logging.info(f"Evaluating on {args.dataset_name} - evaluating split: 'val' (path: {eval_data_path})")
        try:
            test_set_for_eval = ImageFolder(eval_data_path, args.dataset_name, args, split='val') 
            test_loader_for_eval = DataLoader(test_set_for_eval, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
            
            if not test_set_for_eval:
                logging.warning("No data found for evaluation set. Exiting.")
                sys.exit(0)

            test_mIoU = evaluate_model(net, test_loader_for_eval, device, focal_loss_fn, dice_loss_fn, 
                                       args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
            logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            sys.exit(1)
        return # Exit after test-only run

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                               patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    
    # --- Resume Training ---
    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1 # Start from next epoch
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming training from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint from {latest_checkpoint_path}: {e}. Starting from scratch.")
            # Ensure best_mIoU and patience_counter are reset if loading fails
            best_mIoU, patience_counter = 0.0, 0


    # --- Dataset Loading ---
    base_dataset_path = DATASET_PATHS.get(args.dataset_name)
    if not base_dataset_path:
         logging.error(f"Dataset '{args.dataset_name}' not found in DATASET_PATHS. Exiting.")
         sys.exit(1)

    if 'TSRS_RSNA' in args.dataset_name:
        train_data_path = os.path.join(base_dataset_path, 'train')
        val_data_path = os.path.join(base_dataset_path, 'val') 
    else:
        # For other datasets, assume base_dataset_path is the root and ImageFolder will look for train/val/test subdirs
        # or perform programmatic splitting if root contains files directly.
        train_data_path = base_dataset_path
        val_data_path = base_dataset_path 

    logging.info(f"Loading training data from: {train_data_path}")
    train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train') 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    
    logging.info(f"Loading validation data from: {val_data_path}")
    val_set = ImageFolder(val_data_path, args.dataset_name, args, split='val') 
    test_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    if not train_set or len(train_set) == 0:
        logging.error("No training data found. Exiting.")
        sys.exit(1)
    if not val_set or len(val_set) == 0:
        logging.warning("No validation data found. Training will proceed without validation metrics.")
        
    logging.info(f"Found {len(train_set)} training images.")
    logging.info(f"Found {len(val_set)} validation images.")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        
        # Use tqdm for epoch progress and manual batch iteration if custom collate returns None
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
        
        batch_count = 0
        for i, data in enumerate(train_iterator):
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Batch {i}: Skipping empty batch.")
                continue
            
            inputs, labels = data['image'].to(device), data['label'].to(device)

            # Apply MixUp if enabled and batch size is sufficient
            if args.batch_size > 1 and args.mixup_alpha > 0:
                # Ensure inputs and labels are on the correct device before mixup
                if inputs.is_cuda: # Move targets to device if not already there
                    labels = labels.to(inputs.device)
                inputs, labels = mixup_data(inputs, labels, alpha=args.mixup_alpha)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = net(inputs) 
            
            total_loss = 0
            # Calculate loss using deep supervision outputs
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + (args.dice_loss_weight * current_dice_loss)
                
                if i < len(args.deep_supervision_weights):
                    total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
                else:
                    logging.warning(f"Deep supervision weight index {i} out of bounds. Skipping weight for output {i}.")
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            batch_count += 1
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        # --- Validation after Epoch ---
        if len(val_set) > 0: # Only validate if validation set is not empty
            current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                          args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")
            scheduler.step(current_mIoU)

            # --- Checkpointing ---
            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU
                patience_counter = 0
                torch.save(net.state_dict(), best_checkpoint_path)
                logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model to {best_checkpoint_path}.")
            else:
                patience_counter += 1
                logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")
        else:
            logging.warning(f"Epoch {epoch+1}: Validation set is empty. Skipping validation and scheduler step.")
            current_mIoU = 0.0 # Set to 0 if no validation

        # --- Save Latest Checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter,
            'current_mIoU': current_mIoU # Save current mIoU as well
        }, latest_checkpoint_path)
        logging.info(f"Saved latest checkpoint to {latest_checkpoint_path}")
        
        # --- Early Stopping ---
        if patience_counter >= args.patience and len(val_set) > 0: # Only early stop if validation was performed
            logging.info("Early stopping triggered due to no improvement.")
            break

    logging.info("Training finished.")
    # Optionally load best model again here if needed after loop ends

if __name__ == '__main__':
    main()