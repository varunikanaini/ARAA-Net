# evaluate.py (or run train.py with --test-only)
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import argparse
import logging

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Model, Config, Dataset, Utilities, Losses ---
from lasa_unet_model import LASA_Unet # Use the generalized LASA_Unet
from datasets import ImageFolder # Use the flexible ImageFolder
from config import CKPT_ROOT, DATASET_CONFIG # Import config
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter
# Import loss functions for consistent loss calculation
# Assuming they are defined in a separate file or directly available in train script's scope
# For this example, let's assume they are available in the train script and re-import or redefine.
# For simplicity, we'll redefine them here.
# from train_unet import FocalLoss, DiceLoss # If you want to import from train script

# Redefining losses for standalone evaluation script:
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

# --- Argument Parsing ---
def get_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate LASA-Unet Model')
    
    # --- Dynamic Config Loading ---
    # Load default args from config.py to ensure consistency with training
    # We need to include all relevant args from train.py that affect model/data loading
    for key, value in DEFAULT_ARGS.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', type=type(value[0]) if value else str, default=value, nargs='+', help=f'List of {key} (default: {value})')
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key} (default: {value})')

    # --- Dataset Selection ---
    parser.add_argument('--dataset-name', type=str, default=DEFAULT_ARGS['dataset_name'],
                        choices=DATASET_CONFIG.keys(), help='Name of the dataset for evaluation')
    # Specify the split to evaluate on (usually 'test', but can be 'val')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'], help='Dataset split to load for evaluation')

    # --- Backbone Selection ---
    parser.add_argument('--backbone', type=str, default=DEFAULT_ARGS['backbone'],
                        choices=BACKBONE_CHANNELS.keys(), help='Backbone architecture used')

    # --- Control Flow ---
    # Note: --test-only is typically used in train script to invoke evaluation.
    # This script is primarily for evaluation, so it's implied.

    # --- Parse Arguments ---
    args = parser.parse_args()
    
    # Post-parsing validation and adjustments
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
    
    # Ensure deep supervision weights have the correct length
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    return args

# --- Logging Setup ---
def setup_logging_eval(log_dir, filename='evaluation_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing"):
    """
    Evaluates the model on a given data_loader. Returns mIoU and average loss.
    Logs OA, mIoU, FWIoU, and Dice metrics to the console and log file.
    """
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes) 
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{mode}", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs) 
            final_pred = outputs[-1] 
            
            total_loss = 0
            # Calculate combined loss across all heads using deep supervision weights
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    # Log all metrics
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA (Overall Accuracy): {global_acc.item():.4f}")
    logging.info(f"  mIoU (Mean IoU): {mIoU:.4f}")
    logging.info(f"  FWIoU (Frequency Weighted IoU): {fwiou.item():.4f}")
    logging.info(f"  Dice (Mean Dice Coefficient): {mDice:.4f}")
    logging.info(f"  Class IoU: {class_iou.cpu().numpy()}")
    logging.info(f"  Class Accuracy: {class_acc.cpu().numpy()}")
    
    return mIoU

# --- Main Evaluation Function ---
def main():
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Experiment Naming and Logging Setup ---
    # Construct experiment name based on selected dataset and backbone for finding checkpoints
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels)) if args.lasa_kernels else "nolasa"
    exp_name_parts = [
        args.backbone,
        f"LASA{lasa_kernels_str}",
        f"DSW{'_'.join(map(str, args.deep_supervision_weights))}",
        f"FLW{args.focal_loss_weight}_DLW{args.dice_loss_weight}",
        args.dataset_name.replace('TSRS_RSNA-', '').lower(),
    ]
    exp_name = "_".join(exp_name_parts)
    
    log_dir = os.path.join(CKPT_ROOT, exp_name) 
    check_mkdir(log_dir) # Ensure log directory exists
    # Use a specific filename for evaluation logs
    setup_logging_eval(log_dir=log_dir, filename=f'evaluation_{args.dataset_name}_{args.split}_{args.backbone}.log')

    logging.info(f"Starting evaluation for experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")

    # --- Instantiate Model ---
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    
    # --- Load Best Checkpoint ---
    best_checkpoint_path = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(best_checkpoint_path):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please ensure training was completed for this experiment.")
        sys.exit(1)

    try:
        net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        logging.info(f"✅ Model loaded successfully from {best_checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
        sys.exit(1)

    # --- Prepare Dataset and DataLoader ---
    # Get dataset info to correctly determine path and structure
    dataset_info = config.get_dataset_info(args.dataset_name)
    
    # Determine the correct path for the specified split
    # This logic will vary based on dataset structure.
    # Using the flexible ImageFolder which relies on dataset_info.
    
    # Path to the specific split directory (e.g., /path/to/dataset/test)
    split_data_path = os.path.join(dataset_info['path'], args.split)
    
    # Fallback logic if the specified split directory doesn't exist
    if not os.path.exists(split_data_path):
        logging.warning(f"Split directory '{args.split}' not found at '{split_data_path}'. Trying 'val' split as fallback.")
        fallback_split = 'val'
        split_data_path = os.path.join(dataset_info['path'], fallback_split)
        if not os.path.exists(split_data_path):
            logging.error(f"❌ ERROR: Neither '{args.split}' nor '{fallback_split}' split found for dataset '{args.dataset_name}'.")
            sys.exit(1)
        logging.info(f"Using fallback split: '{fallback_split}'")
        args.split = fallback_split # Update args.split to reflect the actual split used
    
    # Instantiate the dataset and dataloader
    test_set = ImageFolder(split_data_path, args.dataset_name, args, split=args.split)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    logging.info(f"Loaded {len(test_set)} images for evaluation from '{args.dataset_name}' split '{args.split}'.")

    # --- Instantiate Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- Run Evaluation ---
    test_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Testing")
    
    # --- Log Final Results ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"\n\n--- FINAL EVALUATION RESULTS ({timestamp}) ---")
    logging.info(f"Model: LASA-Unet with {args.backbone} backbone and LASA Kernels: {args.lasa_kernels}")
    logging.info(f"Dataset: {args.dataset_name} (evaluated on '{args.split}' split)")
    logging.info(f"Image scale for eval: ({args.scale_h}, {args.scale_w})")
    logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
    logging.info("---------------------------------")
    logging.info("✅ Evaluation completed.")

if __name__ == '__main__':
    main()