# /kaggle/working/ARAA-Net/test.py

# --- STEP 1: SETUP THE ENVIRONMENT ---
import sys
import os
import torch
import torch.nn as nn # Added for loss functions
import torch.nn.functional as F # Added for loss functions and model interpolation
from torch.utils.data import DataLoader, random_split # Added random_split
from torchvision import transforms
from tqdm.notebook import tqdm
import numpy as np
import datetime # Added for timestamp in results
import argparse # For command line arguments
import logging # For structured logging

# Ensure project path is in sys.path
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import necessary components from your project structure
from daseg import daseg
from datasets import ImageFolder, DATASET_CONFIGS # Import DATASET_CONFIGS and ImageFolder
from datasets import make_dataset as make_full_dataset_list # Rename to avoid conflict
# import joint_transforms # Not strictly needed if ImageFolder handles all transforms internally
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter # Added AvgMeter

# Import loss functions (assuming they are in 'loss.py')
import loss # Assuming loss.py contains structure_loss, IOU, BCEWithLogitsLoss

from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING

print("✅ Environment setup complete.")

# --- STEP 2: DEFINE PARAMETERS (USING ARGPARSE WITH JSRT DEFAULTS) ---
def get_test_args():
    parser = argparse.ArgumentParser(description='Test ARAA-Net Model for specific dataset')
    parser.add_argument('--dataset-name', type=str, default='JSRT', # Default to JSRT
                        choices=list(DATASET_CONFIGS.keys()), help='Name of the dataset to test on')
    parser.add_argument('--backbone', type=str, default='vgg16', # Default to vgg16 for JSRT
                        choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Backbone architecture used for training')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    
    # Dummy args for ImageFolder compatibility, actual values from dataset_config['transform_params'] will override
    # These defaults are set to match JSRT config for convenience
    parser.add_argument('--scale-h', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--scale-w', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--crop-size-h', type=int, default=448, help='Dummy for ImageFolder init, actual from config')
    parser.add_argument('--crop-size-w', type=int, default=448, help='Dummy for ImageFolder init, actual from config')

    # Programmatic splitting ratios (needed for consistency with training split if no predefined test split)
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio used during training.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio used during training.')

    # Loss weights are not used for evaluation metrics, but might be needed for consistency with loss function init
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component.')
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], help='Weights for deep supervision losses.')

    # Dummy args for CenterAmplification in ImageFolder, if present
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # For notebook execution if no args are passed
    return args

args = get_test_args() # Parse arguments
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STEP 3: PREPARE FOR LOGGING ---
EXP_NAME = f"{args.backbone}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'testing_results.log')

# Setup logging to file and console
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler) # Clear previous handlers
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])

logging.info(f"Starting FINAL EVALUATION for experiment '{EXP_NAME}'")
logging.info(f"Arguments: {vars(args)}") # Log all arguments


# --- STEP 4: LOAD THE DATA ---
logging.info("\n--- Loading Test Data ---")

# Determine base_dataset_root (download if KaggleHub, or local path)
dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
base_dataset_root = None

if dataset_info and dataset_info['id']: # If it's a KaggleHub dataset
    base_dataset_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
    if not base_dataset_root:
        logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
        sys.exit(1)
    logging.info(f"Base dataset root (KaggleHub): {base_dataset_root}")
elif dataset_info and dataset_info['local_dir_name']: # For local datasets
    base_dataset_root = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
    if not os.path.exists(base_dataset_root):
        logging.error(f"Local dataset directory not found at '{base_dataset_root}'. Exiting.")
        sys.exit(1)
    logging.info(f"Base dataset root (local): {base_dataset_root}")
else: # Default for datasets without explicit mapping, assume default structure in DATA_ROOT
    base_dataset_root = os.path.join(DATA_ROOT, args.dataset_name)
    if not os.path.exists(base_dataset_root):
        logging.error(f"Dataset directory not found at '{base_dataset_root}'. Exiting.")
        sys.exit(1)
    logging.info(f"Base dataset root (default): {base_dataset_root}")


# Get all image/mask pairs based on dataset config
dataset_config = DATASET_CONFIGS.get(args.dataset_name)
if not dataset_config:
    logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
    sys.exit(1)

test_image_mask_list = []

if dataset_config['has_predefined_splits']:
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
    logging.info(f"Using programmatic splitting for dataset '{args.dataset_name}' for testing.")
    full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
    
    if not full_image_mask_list:
        logging.error(f"No data found for programmatic splitting in '{base_dataset_root}'. Exiting.")
        sys.exit(1)

    total_len = len(full_image_mask_list)
    train_len = int(args.train_ratio * total_len)
    val_len = int(args.val_ratio * total_len)
    test_len = total_len - train_len - val_len

    g = torch.Generator().manual_seed(42) # Consistent seed for splitting
    _, _, test_image_mask_list = random_split(
        full_image_mask_list, [train_len, val_len, test_len], generator=g)
    
    logging.info(f"Programmatic split for test: Total {total_len}, Test {len(test_image_mask_list)}")

# Update args with specific transform_params from dataset_config for ImageFolder
args.scale_w = dataset_config['transform_params']['resize_w']
args.scale_h = dataset_config['transform_params']['resize_h']
args.crop_size_h = dataset_config['transform_params']['crop_size_h']
args.crop_size_w = dataset_config['transform_params']['crop_size_w']

test_set = ImageFolder(test_image_mask_list, args, split='test')
test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False) # pin_memory=True if CUDA
logging.info(f"Found {len(test_set)} testing images for dataset '{args.dataset_name}'.")

# Add a check to prevent crashing if no images are found
if len(test_set) == 0:
    logging.error("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
    sys.exit(1)


# --- STEP 5: LOAD THE TRAINED MODEL ---
logging.info(f"\n--- Loading Trained {args.backbone} Model ---")
checkpoint_path = os.path.join(CKPT_ROOT, EXP_NAME, 'best_checkpoint.pth')

if not os.path.exists(checkpoint_path):
    logging.error(f"❌ ERROR: Checkpoint not found at '{checkpoint_path}'")
    sys.exit(1)
else:
    net = daseg(backbone_name=args.backbone).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle DataParallel prefix if checkpoint was saved from a DataParallel model
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    
    net.eval()
    logging.info("✅ Model loaded successfully.")

    # --- STEP 6: INITIALIZE LOSS FUNCTIONS FOR METRIC CALCULATION ---
    # These are only for reporting loss metrics during test, not for training.
    structure_loss_fn = loss.structure_loss().to(DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)
    iou_loss_fn = loss.IOU().to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(DEVICE) # Assuming 255 is ignore index for CE loss
    
    def bce_iou_loss_fn_wrapper(pred, target):
        return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    # --- STEP 7: RUN EVALUATION ---
    logging.info("\n--- Running Evaluation ---")
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter() # To record average test loss

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            
            # daseg returns 5 predictions
            predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
            
            # --- Calculate loss for reporting during testing ---
            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long()

            loss_1 = bce_iou_loss_fn_wrapper(predict_1.squeeze(1), binary_labels.squeeze(1)) # Squeeze for BCEWithLogitsLoss
            loss_2 = structure_loss_fn(predict_2, binary_labels)
            loss_3 = structure_loss_fn(predict_3, binary_labels)
            loss_4 = structure_loss_fn(predict_4, binary_labels)
            loss_0 = ce_loss_fn(predict_0, ce_labels)

            # Sum with assumed deep supervision weights (or just use predict_0 loss for overall test loss)
            # These weights are hardcoded here for testing, match training if different
            # Note: The deep supervision weights from args are not directly used in the test loss calculation here.
            # If you want to use them for a weighted test loss, you'd add:
            # total_loss = (args.deep_supervision_weights[0] * loss_1_component) + ...
            total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0 # Matches your original train.py weighting
            loss_recorder.update(total_loss.item(), inputs.size(0))

            # Use final prediction (predict_0) for mIoU calculation
            confmat.update(labels.flatten(), predict_0.argmax(1).flatten())
            
    # --- STEP 8: COMPUTE, PRINT, AND SAVE RESULTS ---
    logging.info("\n--- Final Test Results ---")
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_text = (
        f"------ Final Test Results for {EXP_NAME} ({timestamp}) ------\n"
        f"Model: ARAA-Net with {args.backbone} backbone\n"
        f"Dataset: {args.dataset_name} (evaluated on test split)\n"
        f"Image scale for test: ({args.scale_h}, {args.scale_w})\n"
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU (mIoU) = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"FWIoU           = {fwiou.item():.4f}\n"
        f"Class IoU       = {class_iou.cpu().numpy()}\n"
        f"Class Accuracy  = {class_acc.cpu().numpy()}\n"
        f"Combined Test Loss (Avg) = {loss_recorder.avg:.4f}\n"
        f"--------------------------------------------------\n"
    )
    
    # Print to screen and save to file
    logging.info(results_text)
    with open(log_file_path, 'a') as f: # Use 'a' for append to not overwrite prior logs if any
        f.write(results_text)
    logging.info(f"✅ Results successfully saved to: {log_file_path}")

if __name__ == '__main__':
    # Example usage:
    # To run with default JSRT settings:
    # !python /kaggle/working/ARAA-Net/test.py

    # To run with TSRS_RSNA-Epiphysis:
    # !python /kaggle/working/ARAA-Net/test.py --dataset-name TSRS_RSNA-Epiphysis --backbone resnet50
    # (Note: You'll need a 'best_checkpoint.pth' for resnet50_tsrs-epiphysis in ckpt/resnet50_tsrs-epiphysis/)

    main()