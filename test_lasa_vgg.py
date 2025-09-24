# /kaggle/working/ARAA-Net/test_lasa_vgg.py (MODIFIED for robust dataset loading and programmatic splitting)
import sys
import os
import torch
from torch.utils.data import DataLoader, random_split # <<< MODIFIED: Import random_split
from tqdm import tqdm
import numpy as np
import datetime
import argparse
import logging

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet 
from datasets import ImageFolder, DATASET_CONFIGS # <<< MODIFIED: Import DATASET_CONFIGS, make_dataset
from datasets import make_dataset as make_full_dataset_list # <<< NEW: Rename to avoid conflict with ImageFolder's internal make_dataset (if any)
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter 
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING 

from train_lasa_vgg import FocalLoss, DiceLoss 


def get_test_args():
    parser = argparse.ArgumentParser(description='Test LASA-Unet Model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Articular-Surface',
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for training')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture used for training')
    parser.add_argument('--scale-h', type=int, default=448, help='Height images were resized to')
    parser.add_argument('--scale-w', type=int, default=448, help='Width images were resized to')

    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component in combined loss.')

    # CenterAmplification args (needed for ImageFolder to instantiate correctly, even if not used in test split)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')

    # Programmatic splitting ratios (needed for consistency if programmatic split was used in training)
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Dummy arg for programmatic split consistency.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Dummy arg for programmatic split consistency.')

    # Add num_workers argument
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes for data loading.') # ADD THIS LINE

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")

    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Construct the correct experiment name to find the checkpoint ---
    EXP_NAME = f"{args.backbone}_LASA_Unet_FocalDice_DS_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    check_mkdir(log_dir) # Ensure log directory exists
    setup_logging(log_dir) # Setup logging for this specific test run

    logging.info(f"Starting FINAL TESTING for experiment '{EXP_NAME}'")
    logging.info(f"Arguments: {args}")

    # --- Determine the base root for the dataset (download if KaggleHub) ---
    dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
    base_dataset_root = None

    if dataset_info and dataset_info['id']: # If it's a KaggleHub dataset
        base_dataset_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
        if not base_dataset_root:
            logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (KaggleHub): {base_dataset_root}")
    elif dataset_info and dataset_info['local_dir_name']: # For local datasets like KOA
        base_dataset_root = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
        if not os.path.exists(base_dataset_root):
            logging.error(f"Local dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (local): {base_dataset_root}")
    else: # For TSRS_RSNA datasets (default)
        base_dataset_root = os.path.join(DATA_ROOT, args.dataset_name)
        if not os.path.exists(base_dataset_root):
            logging.error(f"TSRS_RSNA dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (TSRS_RSNA default): {base_dataset_root}")
    # --- END dataset root determination ---

    # --- Data Loading and Splitting Logic for Testing ---
    dataset_config = DATASET_CONFIGS.get(args.dataset_name)
    if not dataset_config:
        logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
        sys.exit(1)

    test_image_mask_list = []

    if dataset_config['has_predefined_splits']:
        # Datasets with pre-defined 'train', 'val', 'test' folders (e.g., RSNA, KOA)
        logging.info(f"Using predefined splits for dataset '{args.dataset_name}' for testing.")
        test_path_defined = os.path.join(base_dataset_root, 'test') # Look for a 'test' folder first
        val_path_defined = os.path.join(base_dataset_root, 'val') # Fallback to 'val' if 'test' not found

        if os.path.exists(test_path_defined) and os.listdir(test_path_defined):
            test_image_mask_list = make_full_dataset_list(test_path_defined, args.dataset_name, split_name='test')
            logging.info(f"Using 'test' split from predefined folder: {test_path_defined}")
        elif os.path.exists(val_path_defined) and os.listdir(val_path_defined):
            test_image_mask_list = make_full_dataset_list(val_path_defined, args.dataset_name, split_name='val')
            logging.warning(f"No explicit 'test' split folder found for '{args.dataset_name}'. Using 'val' split from '{val_path_defined}' for testing.")
        else:
            logging.error(f"Neither 'test' nor 'val' split folders found at '{base_dataset_root}'. Exiting.")
            sys.exit(1)

    else:
        # Datasets requiring programmatic splitting (e.g., COVID-19_Radiography, JSRT)
        logging.info(f"Using programmatic splitting for dataset '{args.dataset_name}' for testing.")
        full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
        
        if not full_image_mask_list:
            logging.error(f"No data found for programmatic splitting in '{base_dataset_root}'. Exiting.")
            sys.exit(1)

        # Re-create the same random split as during training for consistency
        total_len = len(full_image_mask_list)
        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        test_len = total_len - train_len - val_len

        _, _, test_image_mask_list = random_split(
            full_image_mask_list, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
        
        logging.info(f"Programmatic split for test: Total {total_len}, Test {len(test_image_mask_list)}")

    # --- Instantiate DataLoader for Testing ---
    test_set = ImageFolder(test_image_mask_list, args, split='test') # <<< MODIFIED: Pass image_mask_list
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    logging.info(f"Found {len(test_set)} testing images for dataset '{args.dataset_name}'.")

    # --- Load the BEST trained model ---
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1)

    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(DEVICE)
    net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
    net.eval()
    logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")

    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(DEVICE)
    dice_loss_fn = DiceLoss().to(DEVICE)

    # --- Run Evaluation ---
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            
            outputs = net(inputs) 
            final_pred = outputs[-1] 
            
            total_loss = 0
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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(
        f"\n\n--- Final Test Results ({timestamp}) ---\n"
        f"Model: LASA-Unet with {args.backbone} backbone\n"
        f"Experiment Name: {EXP_NAME}\n"
        f"Dataset: {args.dataset_name} (evaluated on test split)\n" 
        f"Image scale for test: ({args.scale_h}, {args.scale_w})\n" 
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU (mIoU) = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"FWIoU           = {fwiou.item():.4f}\n"
        f"Class IoU       = {class_iou.cpu().numpy()}\n" 
        f"Class Accuracy  = {class_acc.cpu().numpy()}\n" 
        f"Combined Loss (Avg) = {loss_recorder.avg:.4f}\n" 
        f"--------------------------------------------------\n"
    )
    logging.info("✅ Final testing completed.")

if __name__ == '__main__':
    main()