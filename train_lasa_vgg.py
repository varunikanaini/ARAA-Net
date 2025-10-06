# train_unet.py (Renamed from train_lasa_vgg.py)
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
# Assuming the file is now named lasa_unet_model.py
from lasa_unet_model import LASA_Unet 
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

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
        # Handle multi-class segmentation by treating it as binary classification per class
        # For simplicity, assuming binary segmentation here (num_classes=2 means background and foreground)
        # If `num_classes > 1` logic is needed, it needs to be more general.
        # For now, assuming binary output or one-vs-all for multi-class.
        # This loss is typically applied to the probability output of the foreground class.
        
        # Assuming `inputs` are logits of shape (B, num_classes, H, W)
        # and `targets` are labels of shape (B, H, W)
        
        num_classes = inputs.shape[1]
        
        if num_classes == 2: # Binary segmentation (foreground vs background)
            # Sigmoid to get probabilities for the foreground class (class 1)
            pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1) 
            true_oh = (targets == 1).float().unsqueeze(1) 
        elif num_classes > 2: # Multi-class (one-vs-all approach for Dice)
            # One-hot encode targets
            true_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            # Softmax to get probabilities for each class
            pred_probs = F.softmax(inputs, dim=1)
            # For simplicity, we can calculate Dice per class and average, or one-vs-all.
            # Let's use one-vs-all for each class.
            # For this training loop, we'll assume num_classes=2 or adapt it.
            # For now, assuming binary or we are calculating Dice loss for the main foreground class (if applicable).
            # If input is (B, 2, H, W), then pred_probs[:, 1, :, :] is foreground prob.
            # If input is (B, C, H, W) with C>2, this needs more care.
            # Let's stick to binary for now.
            pass 

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            pred_probs = pred_probs * mask.unsqueeze(1)
            true_oh = true_oh * mask.unsqueeze(1)
        
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] 
        else:
            return loss 

# ===================================================================


def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    
    # Dataset arguments
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Name of the dataset')
    
    # Backbone arguments
    parser.add_argument('--backbone', type=str, default='vgg16', 
                        choices=['vgg16', 'resnet50', 'inception_v3', 'efficientnet_b0', 'efficientnet_b3'], 
                        help='Backbone architecture to use')
    
    # LASA Module arguments
    # L_list defines the kernel sizes [k1, k2, k3, k4] for the M=4 groups
    parser.add_argument('--lasa-kernels', nargs='+', type=int, default=[1, 3, 5, 7], 
                        help='List of kernel sizes for LASA module (M=4 groups). Example: --lasa-kernels 1 3 5 7')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6) 
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--weight-decay', type=float, default=0.0005) 
    parser.add_argument('--patience', type=int, default=20, 
                        help='Patience for early stopping based on validation mIoU.')
    parser.add_argument('--num-workers', type=int, default=2) 

    # Image preprocessing parameters (used by datasets.py custom transforms)
    parser.add_argument('--scale-h', type=int, default=896, help='Nominal height for resizing (overridden by DASEG fixed sizes)')
    parser.add_argument('--scale-w', type=int, default=576, help='Nominal width for resizing (overridden by DASEG fixed sizes)')
    
    # Deep Supervision weights
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision outputs (from lowest resolution to highest). Must have 5 values.')
    
    # Loss function parameters
    parser.add_argument('--focal-alpha', type=float, default=0.5, 
                        help='Alpha parameter for Focal Loss (balance between positive/negative).')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Gamma parameter for Focal Loss (focus on hard examples).')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, 
                        help='Weight for Focal Loss component in the combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, 
                        help='Weight for Dice Loss component in the combined loss.')
    
    # Data augmentation parameters (passed to ImageFolder)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Min lesion area for CenterAmplification.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Expansion factor for CenterAmplification bbox.')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Min bbox height for CenterAmplification.')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Min bbox width for CenterAmplification.')
    parser.add_argument('--wavelet-type', type=str, default='haar', 
                        help='Wavelet type for DWT contrast enhancement.')
    parser.add_argument('--wavelet-level', type=int, default=1, 
                        help='DWT decomposition level.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, 
                        help='Scaling factor for DWT detail coefficients.')

    # Scheduler parameters
    parser.add_argument('--scheduler-patience', type=int, default=5, 
                        help='Patience for ReduceLROnPlateau scheduler.')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, 
                        help='Factor for ReduceLROnPlateau scheduler.')
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6, 
                        help='Minimum learning rate for the scheduler.')

    # Control flow
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # For environments like notebooks where exit might be too harsh
    
    # Validate deep supervision weights length
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values (one for each deep supervision output + final). Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    # Clear existing handlers to prevent duplicate logs if script is re-run in an interactive session
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, deep_supervision_weights, focal_loss_weight, dice_loss_weight, mode="Validating"):
    """
    Evaluates the model on a given data_loader. Returns mIoU and average loss.
    """
    net.eval()
    # Use num_classes=2 for binary segmentation (background=0, foreground=1)
    confmat = ConfusionMatrix(num_classes=2) 
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{mode}", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            # Ensure labels are LongTensor for CrossEntropy and FocalLoss
            # For DiceLoss, it might need float, so we handle it inside the loss function if needed.
            
            outputs = net(inputs) 
            final_pred = outputs[-1] # The last output is the prediction from the highest resolution decoder stage
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                # Calculate loss for each output head (including deep supervision heads)
                current_focal_loss = focal_loss_fn(pred_output, labels.long()) # Focal loss requires Long targets
                current_dice_loss = dice_loss_fn(pred_output, labels.long()) # Dice loss also uses Long targets for one-hot encoding
                
                combined_loss_per_head = (focal_loss_weight * current_focal_loss) + \
                                         (dice_loss_weight * current_dice_loss)
                
                total_loss += deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            # Update confusion matrix with the final prediction (argmax for class prediction)
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    # Compute metrics from confusion matrix
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item() # Mean IoU over all classes (foreground IoU is often more relevant for binary)
    
    # Log metrics for the current evaluation phase
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  mIoU: {mIoU:.4f}")
    logging.info(f"  OA: {global_acc.item():.4f}")
    logging.info(f"  Dice: {mDice:.4f}")
    # Optionally log class-specific IoU and Acc
    # logging.info(f"  Class IoU: {class_iou.cpu().numpy()}")
    # logging.info(f"  Class Acc: {class_acc.cpu().numpy()}")
    
    if mode == "Validating": 
        net.train() # Set back to training mode if it was validation
    return mIoU

# Custom collate function to filter out None samples (from `datasets.py` returning None for corrupted images)
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] # Filter out any None samples returned by the dataset
    if not batch:
        return None # Return None if the batch is empty
    return torch.utils.data.dataloader.default_collate(batch) # Use default collate for the remaining valid samples

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    # Generate a more descriptive experiment name
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels))
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_Kernels{lasa_kernels_str}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training for experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}") # Log all arguments

    # Get base dataset path from config
    base_dataset_path = DATASET_PATHS[args.dataset_name]

    # Instantiate the model
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    
    # Instantiate loss functions
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- TEST ONLY MODE ---
    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{exp_path}'. Please run training first.")
            sys.exit(1)

        try:
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"✅ Model loaded successfully from {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
            sys.exit(1)

        # Determine test data path based on dataset name
        if 'TSRS_RSNA' in args.dataset_name:
            # For TSRS-like datasets, the test data is in the 'test' folder
            test_data_path = os.path.join(base_dataset_path, 'test') 
        else:
            # For other datasets, the ImageFolder class will handle internal splitting based on 'split' argument
            test_data_path = base_dataset_path
        
        # Initialize test dataset and dataloader
        # Use split='test' to ensure it loads from the designated test folder/split
        test_set = ImageFolder(test_data_path, args.dataset_name, args, split='test') 
        test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
        logging.info(f"Loaded {len(test_set)} images for testing from '{args.dataset_name}'.")

        # Evaluate the model
        test_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                   args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
        logging.info(f"--- FINAL TEST RESULTS ---")
        logging.info(f"Dataset: {args.dataset_name}, Backbone: {args.backbone}, LASA Kernels: {args.lasa_kernels}")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        return # Exit after testing

    # --- TRAINING MODE ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ReduceLROnPlateau scheduler for learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    
    # Try to resume from a previous training session
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
            logging.info(f"Resuming training from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")

    # Determine data paths based on dataset name
    if 'TSRS_RSNA' in args.dataset_name:
        train_data_path = os.path.join(base_dataset_path, 'train')
        val_data_path = os.path.join(base_dataset_path, 'val') 
    else:
        # For other datasets, ImageFolder will handle internal splitting if needed,
        # but we still need to point to the root data directory.
        train_data_path = base_dataset_path
        val_data_path = base_dataset_path 

    # Initialize datasets and dataloaders
    train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train') 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    
    # For validation, we use the 'val' split from the ImageFolder dataset class.
    # If the dataset doesn't have explicit train/val/test folders (e.g., JSRT), 
    # ImageFolder will split it internally.
    val_set = ImageFolder(val_data_path, args.dataset_name, args, split='val') 
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Found {len(train_set)} training images.")
    logging.info(f"Found {len(val_set)} validation images.")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        net.train() # Set model to training mode
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for data in train_iterator:
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Iter: Skipping empty training batch due to corrupted/missing samples.")
                continue

            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True) # Clear gradients efficiently
            
            outputs = net(inputs)
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            total_loss.backward() # Backpropagate
            optimizer.step() # Update weights
            
            loss_recorder.update(total_loss.item(), inputs.size(0)) # Record loss
            train_iterator.set_postfix(loss=loss_recorder.avg) # Update progress bar
            
        # --- Validation Phase ---
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        # Adjust learning rate based on validation mIoU
        scheduler.step(current_mIoU)

        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0 # Reset patience counter
            torch.save(net.state_dict(), best_checkpoint_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model to '{best_checkpoint_path}'.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ Validation mIoU did not improve for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save latest checkpoint to allow resuming training
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        # Early stopping condition
        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered after {args.patience} epochs of no improvement.")
            break

if __name__ == '__main__':
    main()