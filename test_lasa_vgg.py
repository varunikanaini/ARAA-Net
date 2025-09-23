# /kaggle/working/ARAA-Net/test_lasa_vgg.py (MODIFIED for new datasets and KaggleHub download)
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

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet 
from datasets import ImageFolder, DATASET_CONFIGS # <<< MODIFIED: Import DATASET_CONFIGS
from misc import check_mkdir, AvgMeter 
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING # <<< MODIFIED IMPORTS

from train_lasa_vgg import FocalLoss, DiceLoss 


def get_test_args():
    parser = argparse.ArgumentParser(description='Test LASA-Unet Model')
    # <<< MODIFIED: Use DATASET_CONFIGS.keys() for choices >>>
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Articular-Surface', 
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for training')
    # <<< END MODIFIED >>>
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
    # This MUST match the naming convention used in train_lasa_vgg.py
    EXP_NAME = f"{args.backbone}_LASA_Unet_FocalDice_DS_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    check_mkdir(log_dir) # Ensure log directory exists
    setup_logging(log_dir) # Setup logging for this specific test run

    logging.info(f"Starting FINAL TESTING for experiment '{EXP_NAME}'")
    logging.info(f"Arguments: {args}")

    # --- NEW: Handle KaggleHub dataset download and placement for testing ---
    dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
    if dataset_info and dataset_info['id']:
        downloaded_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
        if not downloaded_root:
            logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
            sys.exit(1)
        base_dataset_root_for_splits = downloaded_root
    elif dataset_info and dataset_info['local_dir_name']:
        base_dataset_root_for_splits = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
        if not os.path.exists(base_dataset_root_for_splits):
            logging.error(f"Local dataset directory not found at '{base_dataset_root_for_splits}'. Please place it there. Exiting.")
            sys.exit(1)
    else:
        base_dataset_root_for_splits = os.path.join(DATA_ROOT, args.dataset_name)
        if not os.path.exists(base_dataset_root_for_splits):
            logging.error(f"TSRS_RSNA dataset directory not found at '{base_dataset_root_for_splits}'. Please place it there. Exiting.")
            sys.exit(1)
    
    # Use 'test' split if available, otherwise 'val'
    test_data_path = os.path.join(base_dataset_root_for_splits, 'test')
    if not os.path.exists(test_data_path):
        logging.warning(f"Test split not found at '{test_data_path}'. Falling back to 'val' split for testing.")
        test_data_path = os.path.join(base_dataset_root_for_splits, 'val')
        if not os.path.exists(test_data_path):
            logging.error(f"❌ ERROR: Neither 'test' nor 'val' split data found for testing at '{base_dataset_root_for_splits}'.")
            sys.exit(1)
    logging.info(f"Using data from: {test_data_path}")
    # --- END NEW ---


    test_set = ImageFolder(test_data_path, args, split='test') # Use split='test' for transform consistency
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    logging.info(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    # --- Load the BEST trained model ---
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1)

    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(DEVICE)
    net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
    net.eval()
    logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")

    # --- Initialize Loss Functions (needed for consistent loss calculation in evaluation) ---
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

    # --- Compute and log all metrics from the confusion matrix ---
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(
        f"\n\n--- Final Test Results ({timestamp}) ---\n"
        f"Model: LASA-Unet with {args.backbone} backbone\n"
        f"Experiment Name: {EXP_NAME}\n"
        f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n" # Note: may be 'val' if 'test' not found
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