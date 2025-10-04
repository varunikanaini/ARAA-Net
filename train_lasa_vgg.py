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

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
import custom_transforms as tr # Import custom transforms

# ===================================================================
#      FOCAL LOSS CLASS
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# ===================================================================

# ===================================================================
#      DICE LOSS CLASS
# ===================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        if num_classes > 1:
            probs = F.softmax(inputs, dim=1)
            pred_probs = probs[:, 1, :, :].unsqueeze(1) 
            true_oh = (targets == 1).float().unsqueeze(1) 
        else: 
            pred_probs = torch.sigmoid(inputs)
            true_oh = targets.float().unsqueeze(1) 

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            pred_probs = pred_probs * mask.unsqueeze(1)
            true_oh = true_oh * mask.unsqueeze(1)
        
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        union = pred_probs.sum() + true_oh.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        loss = 1. - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] 
        else: # 'none'
            return loss 

# ===================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    
    # --- Dataset and Model Configuration ---
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of segmentation classes (e.g., 2 for background + foreground)')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping based on validation mIoU')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    
    # --- Input Image Configuration ---
    parser.add_argument('--scale-h', type=int, default=896, help='Nominal image height for resizing (overridden by DASEG fixed size)')
    parser.add_argument('--scale-w', type=int, default=576, help='Nominal image width for resizing (overridden by DASEG fixed size)')

    # --- Loss Function Configuration ---
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    parser.add_argument('--focal-alpha', type=float, default=0.5, 
                        help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, 
                        help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=3.0, # Increased Dice weight
                        help='Weight for Dice Loss component in combined loss (tuned for MIOU)')

    # --- Data Augmentation Parameters ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Minimum lesion area in pixels to trigger CenterAmplification')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Factor by which to expand the bounding box during CenterAmplification')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Minimum height of the expanded bounding box in pixels')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Minimum width of the expanded bounding box in pixels')
    
    # Wavelet params
    parser.add_argument('--wavelet-type', type=str, default='haar', 
                        help='Wavelet type for DWT-based contrast enhancement.')
    parser.add_argument('--wavelet-level', type=int, default=1, 
                        help='Decomposition level for DWT-based contrast enhancement.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, 
                        help='Scaling factor for detail coefficients in wavelet enhancement.')
                        
    # Cutout params
    parser.add_argument('--cutout-num-holes', type=int, default=1, help='Number of holes for Cutout augmentation')
    parser.add_argument('--cutout-max-h-size', type=int, default=64, help='Max height of cutout square')
    parser.add_argument('--cutout-max-w-size', type=int, default=64, help='Max width of cutout square')
    parser.add_argument('--cutout-fill-value', type=int, default=0, help='Fill value for cutout region (0 for black)')

    # --- Scheduler Configuration ---
    parser.add_argument('--scheduler-patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6, help='Minimum learning rate for scheduler')

    # --- Testing Configuration ---
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint and exit')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    # Validation of argument constraints
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    """Sets up logging to both file and console."""
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, deep_supervision_weights, focal_loss_weight, dice_loss_weight, mode="Validating"):
    """
    Evaluates the model on a given data_loader.
    Returns the mean IoU (mIoU) and the average loss.
    """
    net.eval()
    confmat = ConfusionMatrix(num_classes=net.num_classes)
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Evaluating ({mode})", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
                
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            
            outputs = net(inputs) 
            final_pred = outputs[-1]
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (focal_loss_weight * current_focal_loss) + \
                                         (dice_loss_weight * current_dice_loss)
                
                total_loss += deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    _, _, class_iou, _, _, mIoU_val = confmat.compute()
    
    logging.info(f"--- {mode} Results ---")
    logging.info(f"mIoU: {mIoU_val:.4f} | Avg Loss: {loss_recorder.avg:.4f}")
    logging.info(f"--------------------")
    
    if mode == "Validating": 
        net.train()
    return mIoU_val

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(42) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # --- Experiment Setup ---
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training for experiment: '{exp_name}'")
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {args}")

    # --- Model Initialization ---
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(device)
    logging.info(f"Model initialized: LASA-Unet with {args.backbone} backbone.")

    # --- Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255).to(device)
    dice_loss_fn = DiceLoss(smooth=1e-6, ignore_index=255).to(device)
    logging.info(f"Loss functions initialized: FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma}), DiceLoss (smooth={1e-6}).")

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)
    logging.info(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay}).")
    logging.info(f"LR Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience}, min_lr={args.scheduler_min_lr}).")

    # --- Resume Training ---
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path_save = os.path.join(exp_path, 'best_checkpoint.pth')

    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0) 
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming training from epoch {start_epoch}. Loaded checkpoint: {latest_checkpoint_path}")
            logging.info(f"Resumed best mIoU: {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not resume training from {latest_checkpoint_path}: {e}. Starting from scratch.")
            start_epoch = 0 
            best_mIoU = 0.0
            patience_counter = 0

    # --- Dataset Loading ---
    base_dataset_path = DATASET_PATHS[args.dataset_name]
    
    train_data_path = os.path.join(base_dataset_path, 'train') 
    val_data_path = os.path.join(base_dataset_path, 'val')   

    logging.info(f"Loading training data from: {train_data_path}")
    train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train') 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Loading validation data from: {val_data_path}")
    test_set = ImageFolder(val_data_path, args.dataset_name, args, split='val') 
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Number of training samples: {len(train_set)}")
    logging.info(f"Number of validation samples: {len(test_set)}")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        net.train() 
        loss_recorder = AvgMeter() 
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for i, data in enumerate(train_iterator):
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Iter {i}: Skipping empty training batch due to corrupted/missing samples.")
                continue

            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True) 
            
            outputs = net(inputs) 
            
            total_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[head_idx] * combined_loss_per_head
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        # --- Validation after each epoch ---
        current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        # --- Scheduler Step ---
        scheduler.step(current_mIoU)

        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0 
            torch.save(net.state_dict(), best_checkpoint_path_save)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model to {best_checkpoint_path_save}.")
        else:
            patience_counter += 1 
            logging.info(f"⚠️ Validation mIoU did not improve for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save the latest checkpoint (for resuming training)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        logging.info(f"Saved latest checkpoint to {latest_checkpoint_path}")
        
        # --- Early Stopping ---
        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered: Validation mIoU did not improve for {args.patience} epochs.")
            break

    logging.info("Training finished.")
    logging.info(f"Best validation mIoU achieved: {best_mIoU:.4f}")

if __name__ == '__main__':
    main()