# train.py (Updated for new datasets.py structure and None handling)

import sys
import os
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import time
import datetime
from sklearn.model_selection import KFold
import copy
import cv2

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Config and Other Modules ---
import config 
from lasa_unet_model import LASA_Unet 
from datasets import ImageFolder, make_dataset, IMAGE_EXTENSIONS, MASK_EXTENSIONS # Import make_dataset from datasets
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# --- Loss Functions (same as before) ---
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
            if num_classes == 2:
                pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1) 
                true_oh = (targets == 1).float().unsqueeze(1) 
            else: 
                true_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
                pred_probs = F.softmax(inputs, dim=1)
        else: 
            pred_probs = F.sigmoid(inputs).unsqueeze(1)
            true_oh = targets.float().unsqueeze(1)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            if mask.ndim == 3: mask = mask.unsqueeze(1)
            pred_probs = pred_probs * mask
            true_oh = true_oh * mask
        
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice
        
        if self.reduction == 'mean': return loss
        elif self.reduction == 'sum': return loss * inputs.shape[0] 
        else: return loss 

# --- Backbone Freezing/Unfreezing Helper Functions (same as before) ---
def freeze_backbone(model, backbone_name):
    """Freezes parameters of the backbone encoder."""
    frozen_layers = []
    if backbone_name == 'vgg16' or backbone_name == 'vgg19': # Include VGG19
        frozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name == 'resnet50':
        frozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name == 'inception_v3':
        frozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name.startswith('efficientnet'):
        frozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    else:
        logging.warning(f"Backbone '{backbone_name}' not recognized for freezing. No layers frozen.")
        return

    for layer_name in frozen_layers:
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = False
    logging.info(f"Backbone '{backbone_name}' frozen for Phase 1 training.")

def unfreeze_backbone(model, backbone_name):
    """Unfreezes parameters of the backbone encoder."""
    unfrozen_layers = []
    if backbone_name == 'vgg16' or backbone_name == 'vgg19': # Include VGG19
        unfrozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name == 'resnet50':
        unfrozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name == 'inception_v3':
        unfrozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    elif backbone_name.startswith('efficientnet'):
        unfrozen_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'bottleneck_layer']
    else:
        logging.warning(f"Backbone '{backbone_name}' not recognized for unfreezing. No layers unfrozen.")
        return

    for layer_name in unfrozen_layers:
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True
    logging.info(f"Backbone '{backbone_name}' unfrozen for Phase 2 training.")

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Multi-Dataset, Multi-Backbone, and K-Fold Cross-Validation')
    
    # --- Dataset Selection ---
    dataset_choices = list(config.DATASET_CONFIG.keys())
    parser.add_argument('--dataset-name', type=str, default=config.DEFAULT_ARGS['dataset_name'],
                        choices=dataset_choices, help='Name of the dataset to use')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'], help='Dataset split to load (used if not running k-fold)')

    # --- Backbone Selection ---
    backbone_choices = list(config.BACKBONE_CHANNELS.keys())
    parser.add_argument('--backbone', type=str, default=config.DEFAULT_ARGS['backbone'],
                        choices=backbone_choices, help='Backbone architecture to use')

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_ARGS['epochs'])
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_ARGS['batch_size'])
    parser.add_argument('--lr', type=float, default=config.DEFAULT_ARGS['lr'])
    parser.add_argument('--weight-decay', type=float, default=config.DEFAULT_ARGS['weight_decay'])
    parser.add_argument('--patience', type=int, default=config.DEFAULT_ARGS['patience'],
                        help='Patience for early stopping based on validation mIoU.')

    # --- Image Preprocessing ---
    parser.add_argument('--scale-h', type=int, help='Height for resizing (adjusted based on backbone)')
    parser.add_argument('--scale-w', type=int, help='Width for resizing (adjusted based on backbone)')
    
    # --- LASA Module Arguments ---
    parser.add_argument('--lasa-kernels', type=int, default=config.DEFAULT_ARGS.get('lasa_kernels'), nargs='+',
                        help='Kernel sizes for LASA module')

    # --- Deep Supervision Weights ---
    parser.add_argument('--deep-supervision-weights', type=float, default=config.DEFAULT_ARGS.get('deep_supervision_weights'), nargs='+',
                        help='Weights for deep supervision outputs')

    # --- Loss Function Parameters ---
    parser.add_argument('--focal-alpha', type=float, default=config.DEFAULT_ARGS['focal_alpha'], help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=config.DEFAULT_ARGS['focal_gamma'], help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=config.DEFAULT_ARGS['focal_loss_weight'], help='Weight for Focal Loss component.')
    parser.add_argument('--dice-loss-weight', type=float, default=config.DEFAULT_ARGS['dice_loss_weight'], help='Weight for Dice Loss component.')

    # --- Data Augmentation Parameters ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=config.DEFAULT_ARGS['min_lesion_area_pixels'], help='Min lesion area for CenterAmplification.')
    parser.add_argument('--expansion-factor', type=float, default=config.DEFAULT_ARGS['expansion_factor'], help='Expansion factor for CenterAmplification.')
    parser.add_argument('--min-bbox-h', type=int, default=config.DEFAULT_ARGS['min_bbox_h'], help='Min bbox height for CenterAmplification.')
    parser.add_argument('--min-bbox-w', type=int, default=config.DEFAULT_ARGS['min_bbox_w'], help='Min bbox width for CenterAmplification.')
    parser.add_argument('--wavelet-type', type=str, default=config.DEFAULT_ARGS['wavelet_type'], help='Wavelet type for DWT contrast enhancement.')
    parser.add_argument('--wavelet-level', type=int, default=config.DEFAULT_ARGS['wavelet_level'], help='DWT decomposition level.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=config.DEFAULT_ARGS['wavelet_detail_scale'], help='Scaling factor for DWT detail coefficients.')

    # --- Scheduler Parameters ---
    parser.add_argument('--scheduler-type', type=str, default=config.DEFAULT_ARGS['scheduler_type'], choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'], help='Learning rate scheduler type.')
    parser.add_argument('--scheduler-patience', type=int, default=config.DEFAULT_ARGS['scheduler_patience'], help='Patience for ReduceLROnPlateau.')
    parser.add_argument('--scheduler-factor', type=float, default=config.DEFAULT_ARGS['scheduler_factor'], help='Factor for ReduceLROnPlateau.')
    parser.add_argument('--scheduler-min-lr', type=float, default=config.DEFAULT_ARGS['scheduler_min_lr'], help='Minimum learning rate for the scheduler.')
    parser.add_argument('--scheduler-T0', type=int, default=config.DEFAULT_ARGS.get('scheduler_T0', 10), help='T_0 for CosineAnnealingWarmRestarts.')
    parser.add_argument('--scheduler-T-mult', type=int, default=config.DEFAULT_ARGS.get('scheduler_T_mult', 2), 
                        help='T_mult for CosineAnnealingWarmRestarts (must be integer).')

    # --- Control Flow ---
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')

    # --- Fine-tuning Control ---
    parser.add_argument('--fine-tune-epochs', type=int, default=config.DEFAULT_ARGS['fine_tune_epochs'], 
                        help='Number of epochs to freeze backbone (Phase 1). Set to 0 for end-to-end training.')

    # --- K-Fold Cross-Validation ---
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for K-Fold cross-validation.')
    parser.add_argument('--run-kfold', action='store_true', help='Enable K-Fold cross-validation.')

    # --- Add num_workers argument explicitly ---
    parser.add_argument('--num-workers', type=int, default=config.DEFAULT_ARGS['num_workers'], help='Number of data loading workers.')
    
    # --- Parse Arguments ---
    args = parser.parse_args()
    
    # --- Post-parsing validation and adjustments ---
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    # Get dataset specific config and update args
    try:
        dataset_info = config.DATASET_CONFIG[args.dataset_name] 
        args.dataset_path = dataset_info['path']
        args.dataset_structure = dataset_info['structure']
        args.num_classes = dataset_info['num_classes']
        args.image_ext = dataset_info.get('image_ext', IMAGE_EXTENSIONS)
        args.mask_ext = dataset_info.get('mask_ext', MASK_EXTENSIONS)
        
        args.DATASET_CONFIG = config.DATASET_CONFIG # Make config available to ImageFolder

    except KeyError: 
        parser.error(f"Dataset '{args.dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")
    except Exception as e: 
        parser.error(f"Error accessing dataset configuration for '{args.dataset_name}': {e}")

    # Validate selected backbone
    if args.backbone not in config.BACKBONE_CHANNELS:
        parser.error(f"The specified backbone '{args.backbone}' is not supported. Supported backbones are: {list(config.BACKBONE_CHANNELS.keys())}")

    # --- Dynamically set scale_h and scale_w based on the selected backbone ---
    backbone_h, backbone_w = config.get_backbone_resolution(args.backbone)
    args.scale_h = backbone_h
    args.scale_w = backbone_w
    logging.info(f"Set input resolution to {args.scale_h}x{args.scale_w} based on backbone '{args.backbone}'.")

    return args

# --- Logging Setup ---
def setup_logging(log_dir, filename='training.log'):
    # Clear existing handlers to prevent duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def clean_mask(mask, min_area_threshold=100):
    """
    Removes small, noisy connected components from a binary mask.
    
    Args:
        mask (torch.Tensor): A binary mask tensor of shape (H, W).
        min_area_threshold (int): The minimum number of pixels for a component to be kept.
        
    Returns:
        torch.Tensor: The cleaned binary mask.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    # Create a new mask to store the cleaned result
    cleaned_mask_np = np.zeros_like(mask_np)
    
    # Iterate through components, keeping the ones larger than the threshold
    # Note: component 0 is always the background, so we start from 1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            cleaned_mask_np[labels == i] = 1
            
    return torch.from_numpy(cleaned_mask_np).to(mask.device)

# In /kaggle/working/ARAA-Net/train.py

# --- REPLACE YOUR ENTIRE evaluate_model FUNCTION WITH THIS ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating", fold_num=None):
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes)
    loss_recorder = AvgMeter()
    
    log_prefix = f"Fold {fold_num} " if fold_num is not None else ""

    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{log_prefix}{mode}", leave=False):
            if data is None:
                logging.warning(f"{log_prefix}Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            # --- 1. Get Predictions (with TTA for final mask) ---
            # Prediction on the original image
            outputs_original = net(inputs)
            
            # Prediction on the horizontally flipped image
            flipped_inputs = torch.flip(inputs, dims=[3])
            outputs_flipped_raw = net(flipped_inputs)
            
            # Un-flip the flipped outputs to align with the original
            outputs_flipped = [torch.flip(out, dims=[3]) for out in outputs_flipped_raw]

            # --- 2. Calculate Loss (using original outputs for consistency) ---
            total_loss = 0  # <--- INITIALIZE THE VARIABLE HERE
            for i, pred_output in enumerate(outputs_original):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))

            # --- 3. Create Final Mask using Averaged TTA Predictions ---
            final_pred_original = outputs_original[-1]
            final_pred_flipped = outputs_flipped[-1]
            final_pred_averaged = (final_pred_original + final_pred_flipped) / 2.0
            
            # Get the class indices from the averaged predictions
            pred_mask_tta = final_pred_averaged.argmax(dim=1)

            # --- 4. Apply Post-Processing (Clean Mask) ---
            # Clean each mask in the batch
            cleaned_preds = [clean_mask(mask, min_area_threshold=150) for mask in pred_mask_tta]
            final_pred_cleaned = torch.stack(cleaned_preds)
            
            # --- 5. Update Confusion Matrix with the best possible mask ---
            confmat.update(labels.flatten(), final_pred_cleaned.flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info(f"--- {log_prefix}{mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA (Overall Accuracy): {global_acc.item():.4f}")
    logging.info(f"  mIoU (Mean IoU): {mIoU:.4f}")
    logging.info(f"  FWIoU (Frequency Weighted IoU): {fwiou.item():.4f}")
    logging.info(f"  Dice (Mean Dice Coefficient): {mDice:.4f}")
    
    if mode == "Validating":
        net.train() # Set back to train mode
    return mIoU


def custom_collate_fn(batch):
    """
    Custom collate function to handle None values returned by ImageFolder
    when a sample fails to load.
    """
    batch = [item for item in batch if item is not None] 
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch) 

# --- Main Training Function ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Experiment Naming and Logging Setup ---
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels)) if args.lasa_kernels else "nolasa"
    exp_name_parts = [
        args.backbone,
        f"LASA{lasa_kernels_str}",
        f"DSW{'_'.join(map(str, args.deep_supervision_weights))}",
        f"FLW{args.focal_loss_weight}_DLW{args.dice_loss_weight}",
        args.dataset_name.replace('TSRS_RSNA-', '').lower(),
    ]
    exp_name = "_".join(exp_name_parts)
    
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name) 
    check_mkdir(base_exp_path)
    
    # Setup root logging
    setup_logging(base_exp_path, filename='main_training.log') 

    logging.info(f"Starting experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")

    # --- Dataset Loading ---
    all_train_dataset = None
    val_dataset = None
    test_dataset = None # For test-only mode

    # Load training data
    if not args.test_only: 
        train_data_path = os.path.join(args.dataset_path, 'train') 
        if not os.path.exists(train_data_path):
            logging.error(f"Train split not found at '{train_data_path}'. Cannot proceed with training.")
            sys.exit(1)
        else:
            # Load all training data for K-Fold splitting or single training run
            all_train_dataset = ImageFolder(train_data_path, args.dataset_name, args, split='train', kfold_mode=True) 
            logging.info(f"Loaded {len(all_train_dataset)} training images for potential K-Fold split.")

    # Load validation data (used if not running k-fold, or within k-fold if specified)
    val_data_path = os.path.join(args.dataset_path, 'val') 
    if not os.path.exists(val_data_path):
        logging.warning(f"Validation split not found at '{val_data_path}'. Validation will be skipped or K-Fold split will be used.")
    else:
        val_dataset = ImageFolder(val_data_path, args.dataset_name, args, split='val')
        logging.info(f"Loaded {len(val_dataset)} validation images from separate 'val' split.")

    # Load test data (used only in test-only mode)
    test_data_path = os.path.join(args.dataset_path, 'test')
    if os.path.exists(test_data_path):
        test_dataset = ImageFolder(test_data_path, args.dataset_name, args, split='test')
        logging.info(f"Loaded {len(test_dataset)} test images.")
    else:
        logging.warning(f"Test split not found at '{test_data_path}'. Test-only mode will fall back to validation set if available.")


    # --- Test-Only Mode ---
    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        
        checkpoint_path_to_load = None
        if args.run_kfold:
            last_fold_best_ckpt = os.path.join(base_exp_path, f'fold_{args.k_folds}', 'best_checkpoint.pth')
            if os.path.exists(last_fold_best_ckpt):
                checkpoint_path_to_load = last_fold_best_ckpt
                logging.info(f"Using best checkpoint from last fold for testing: {last_fold_best_ckpt}")
            else:
                logging.warning("Could not find best checkpoint from the last fold. Trying main best checkpoint.")
        
        if checkpoint_path_to_load is None:
            main_best_checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
            if os.path.exists(main_best_checkpoint_path):
                checkpoint_path_to_load = main_best_checkpoint_path
                logging.info(f"Using main best checkpoint for testing: {main_best_checkpoint_path}")
            else:
                logging.error(f"❌ ERROR: No suitable 'best_checkpoint.pth' found. Cannot run test-only mode.")
                sys.exit(1)
        
        # --- Load Model for Testing ---
        net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
        focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
        dice_loss_fn = DiceLoss().to(device)
        
        try:
            net.load_state_dict(torch.load(checkpoint_path_to_load, map_location=device))
            logging.info(f"✅ Model loaded successfully from {checkpoint_path_to_load}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint {checkpoint_path_to_load}: {e}. Exiting.")
            sys.exit(1)

        # --- Determine Test Data Loader ---
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
            logging.info(f"Using dedicated test set ({len(test_dataset)} images) for evaluation.")
        elif val_dataset: # Fallback to validation set if test set is not available
            test_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
            logging.warning("Dedicated test set not found. Falling back to using the validation set for testing.")
        else:
            logging.error("❌ ERROR: No test or validation data available for evaluation. Cannot proceed.")
            sys.exit(1)

        # --- Run Evaluation ---
        test_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"\n\n--- FINAL TEST RESULTS ({timestamp}) ---")
        logging.info(f"Model: LASA-Unet ({args.backbone} backbone, LASA Kernels: {args.lasa_kernels})")
        logging.info(f"Dataset: {args.dataset_name} (evaluated on dedicated test set or fallback)")
        logging.info(f"Image scale for test: ({args.scale_h}, {args.scale_w})")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        logging.info("---------------------------------")
        logging.info("✅ Testing completed.")
        return # Exit after test-only run

    # --- K-Fold Cross-Validation Logic ---
    if args.run_kfold and all_train_dataset is not None:
        logging.info(f"Starting K-Fold Cross-Validation with {args.k_folds} folds.")
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42) # Ensure reproducibility
        
        fold_metrics = [] # Stores best mIoU for each fold
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_dataset)):
            fold_num = fold + 1
            fold_exp_path = os.path.join(base_exp_path, f'fold_{fold_num}')
            check_mkdir(fold_exp_path)
            
            # Setup logging for the current fold
            setup_logging(fold_exp_path, filename=f'fold_{fold_num}_training.log') 
            logging.info(f"--- Starting Fold {fold_num}/{args.k_folds} ---")
            
            # Create data samplers and loaders for the current fold
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                all_train_dataset, 
                batch_size=args.batch_size, 
                sampler=train_subsampler, 
                num_workers=args.num_workers, 
                pin_memory=True, 
                collate_fn=custom_collate_fn
            )
            
            # Use the separate validation dataset if available, otherwise use the fold's validation split
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=args.num_workers, 
                    pin_memory=True, 
                    collate_fn=custom_collate_fn
                )
                logging.info(f"Fold {fold_num}: Using separate 'val' dataset for validation.")
            else:
                val_loader = DataLoader(
                    all_train_dataset, 
                    batch_size=1, 
                    sampler=val_subsampler, 
                    num_workers=args.num_workers, 
                    shuffle=False, 
                    pin_memory=True, 
                    collate_fn=custom_collate_fn
                )
                logging.info(f"Fold {fold_num}: Using split from training data for validation.")

            # --- Instantiate Model, Optimizer, Scheduler for each fold ---
            net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
            focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
            dice_loss_fn = DiceLoss().to(device)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.scheduler_type == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
            elif args.scheduler_type == 'CosineAnnealingWarmRestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=int(args.scheduler_T_mult), eta_min=args.scheduler_min_lr)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

            # --- Fine-tuning setup for the fold ---
            if args.fine_tune_epochs > 0:
                freeze_backbone(net, args.backbone)

            # --- Training Loop for the fold ---
            start_epoch, best_mIoU_fold, patience_counter = 0, 0.0, 0
            latest_checkpoint_path = os.path.join(fold_exp_path, 'latest_checkpoint.pth')
            best_checkpoint_path = os.path.join(fold_exp_path, 'best_checkpoint.pth')

            for epoch in range(start_epoch, args.epochs):
                # --- Phase Transition ---
                if args.fine_tune_epochs > 0 and epoch == args.fine_tune_epochs:
                    logging.info(f"Fold {fold_num}: Transitioning to Phase 2: Unfreezing backbone at Epoch {epoch}")
                    unfreeze_backbone(net, args.backbone)
                    
                    new_lr = args.lr / 5.0 
                    logging.info(f"Fold {fold_num}: Adjusting LR for Phase 2 to: {new_lr:.6f}")
                    optimizer = optim.Adam(net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
                    
                    if args.scheduler_type == 'ReduceLROnPlateau':
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
                    elif args.scheduler_type == 'CosineAnnealingWarmRestarts':
                        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=int(args.scheduler_T_mult), eta_min=args.scheduler_min_lr)
                    else:
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
                    logging.info("Optimizer and scheduler re-initialized for Phase 2.")

                net.train() 
                loss_recorder = AvgMeter()
                train_iterator = tqdm(train_loader, desc=f"Fold {fold_num}/{args.k_folds} Epoch {epoch+1}/{args.epochs}", leave=False)
                
                for i, data in enumerate(train_iterator):
                    if data is None: 
                        logging.warning(f"Fold {fold_num} Epoch {epoch+1}, Iter {i}: Skipping empty training batch.")
                        continue

                    inputs, labels = data['image'].to(device), data['label'].to(device)
                    
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
                    
                # --- Validation Step ---
                current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating", fold_num=fold_num)

                # --- Scheduler Step ---
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_mIoU)
                elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step() 
                
                # --- Checkpointing ---
                if current_mIoU > best_mIoU_fold:
                    best_mIoU_fold = current_mIoU
                    patience_counter = 0 
                    torch.save(net.state_dict(), best_checkpoint_path)
                    logging.info(f"Fold {fold_num}: ✅ New best mIoU: {best_mIoU_fold:.4f}. Saving best model to '{best_checkpoint_path}'.")
                else:
                    patience_counter += 1
                    logging.info(f"Fold {fold_num}: ⚠️ Validation mIoU did not improve for {patience_counter} epoch(s). Best mIoU: {best_mIoU_fold:.4f}.")

                # Save latest checkpoint regardless of improvement
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'best_mIoU': best_mIoU_fold,
                    'patience_counter': patience_counter
                }, latest_checkpoint_path)
                
                # --- Early Stopping ---
                if patience_counter >= args.patience:
                    logging.info(f"Fold {fold_num}: Early stopping triggered after {args.patience} epochs of no improvement.")
                    break
            
            fold_metrics.append(best_mIoU_fold) # Store the best mIoU for this fold
        
        # --- Summarize K-Fold Results ---
        avg_mIoU = np.mean(fold_metrics)
        std_mIoU = np.std(fold_metrics)
        logging.info(f"\n\n--- K-Fold Cross-Validation Summary ---")
        logging.info(f"Experiment Name: {exp_name}")
        logging.info(f"Backbone: {args.backbone}, Dataset: {args.dataset_name}, K-Folds: {args.k_folds}")
        for i, metric in enumerate(fold_metrics):
            logging.info(f"  Fold {i+1}/{args.k_folds} Best mIoU: {metric:.4f}")
        logging.info(f"  Average mIoU across {args.k_folds} folds: {avg_mIoU:.4f} +/- {std_mIoU:.4f}")
        logging.info("------------------------------------")
        logging.info("✅ K-Fold Cross-Validation completed.")

        return # Exit after K-Fold training is complete

    # --- Standard Training Loop (if not running K-Fold) ---
    if all_train_dataset is None: # Should not happen if test_only is false and train data exists
         logging.error("No training dataset available for standard training. Exiting.")
         sys.exit(1)

    logging.info("Starting standard training (K-Fold not enabled).")
    
    # Setup logging for the main experiment path
    setup_logging(base_exp_path, filename='training.log') 
    
    train_loader = DataLoader(all_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    
    # Use the separate validation dataset if available, otherwise fallback to train split if kfold was NOT run
    validation_loader = None
    if val_dataset is not None:
        validation_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    elif not args.run_kfold and len(all_train_dataset) > 0: # Fallback if no separate val set and not running kfold
        logging.warning("No separate validation set found. Using training data for validation.")
        validation_loader = DataLoader(all_train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    if validation_loader is None:
        logging.error("Validation data is not available. Cannot proceed with training.")
        sys.exit(1)

    # --- Instantiate Model, Optimizer, Scheduler for standard training ---
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
    elif args.scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=int(args.scheduler_T_mult), eta_min=args.scheduler_min_lr)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

    # --- Fine-tuning setup for standard training ---
    if args.fine_tune_epochs > 0:
        freeze_backbone(net, args.backbone)

    # --- Resuming Training (for standard training) ---
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(base_exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
    
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
            logging.info(f"Resuming standard training from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
            
            if args.fine_tune_epochs > 0:
                if start_epoch < args.fine_tune_epochs:
                    freeze_backbone(net, args.backbone)
                else: 
                    unfreeze_backbone(net, args.backbone)
                    logging.info(f"Resuming into or past fine-tuning phase. Backbone is {'unfrozen' if net.encoder1.weight.requires_grad else 'frozen'}.")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")
            args.resume = False 
    
    # --- Initial Freezing if not resuming or resuming before fine-tune phase (standard training) ---
    if args.fine_tune_epochs > 0 and start_epoch < args.fine_tune_epochs:
        freeze_backbone(net, args.backbone)

    # --- Main Training Loop (Standard) ---
    for epoch in range(start_epoch, args.epochs):
        # --- Phase Transition ---
        if args.fine_tune_epochs > 0 and epoch == args.fine_tune_epochs:
            logging.info(f"--- Transitioning to Phase 2: Unfreezing backbone at Epoch {epoch} ---")
            unfreeze_backbone(net, args.backbone)
            
            new_lr = args.lr / 5.0 
            logging.info(f"Adjusting LR for Phase 2 to: {new_lr:.6f}")
            optimizer = optim.Adam(net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
            
            if args.scheduler_type == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
            elif args.scheduler_type == 'CosineAnnealingWarmRestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=int(args.scheduler_T_mult), eta_min=args.scheduler_min_lr)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
            logging.info("Optimizer and scheduler re-initialized for Phase 2.")

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
            
        # --- Validation Step ---
        current_mIoU = evaluate_model(net, validation_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating")

        # --- Scheduler Step ---
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_mIoU)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step() 
        
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

    logging.info("Standard training finished.")

if __name__ == '__main__':
    main() 