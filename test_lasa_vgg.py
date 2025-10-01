# /kaggle/working/ARAA-Net/test_lasa_vgg.py

print("--- EXECUTING TEST_LASA_VGG.PY ---") # DIAGNOSTIC LINE
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
from lasa_vgg_model import LASA_Unet # <<< Ensure this imports the updated model
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter 
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 

# Import loss functions for consistent loss calculation if logging loss during test
from train_lasa_vgg import FocalLoss, DiceLoss 


def get_test_args():
    parser = argparse.ArgumentParser(description='Test LASA-Unet Model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Dataset used for testing')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture used for testing')
    parser.add_argument('--scale-h', type=int, default=896, help='Height images were nominally resized to (internal logic overrides for DASEG alignment)')
    parser.add_argument('--scale-w', type=int, default=576, help='Width images were nominally resized to (internal logic overrides for DASEG alignment)')
    
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component in combined loss.')

    # Dummy arguments to match ImageFolder's requirements if not used by the test script directly
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-type', type=str, default='haar', help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-level', type=int, default=1, help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, help='Dummy arg for ImageFolder.')

    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle cases where arguments might not be fully provided (e.g., in some notebook environments)
        # Provide default values if parsing fails partially, or re-raise if critical
        # For simplicity here, we'll re-parse with empty list to get defaults if needed.
        args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to avoid duplicate logs if script is re-run
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Custom collate function to filter out None samples
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] # Filter out None samples
    if not batch: # If the batch becomes empty after filtering
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct the correct experiment name to find the checkpoint
    # Updated exp_name to include ASPP for clarity in logging and checkpoint finding
    exp_name = f"{args.backbone}_LASA_Unet_ASPP_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(log_dir) 
    setup_logging(log_dir) 

    logging.info(f"Starting FINAL TESTING for experiment '{exp_name}'")
    logging.info(f"Arguments: {args}")

    # Get base dataset path from config
    if args.dataset_name not in DATASET_PATHS:
        logging.error(f"❌ ERROR: Dataset '{args.dataset_name}' not found in DATASET_PATHS in config.py.")
        sys.exit(1)
    base_dataset_path = DATASET_PATHS[args.dataset_name]

    # --- Path Construction for Test Data ---
    # For TSRS_RSNA datasets, the 'root' passed to ImageFolder (and make_dataset)
    # needs to point to the directory containing the actual image files for the split.
    # The make_dataset function in datasets.py has been updated to look for masks
    # relative to this `root` path (e.g., looking in `root_of_split_labels`).
    if 'TSRS_RSNA' in args.dataset_name:
        # The `root` for ImageFolder will be the specific split directory (e.g., 'test')
        test_data_path = os.path.join(base_dataset_path, 'test') 
    else:
        # For other datasets, if they are pre-split, `base_dataset_path` might be a parent dir
        # and ImageFolder might look for train/val/test subdirs.
        # If test_data_path needs to be a specific split like 'test', adjust here.
        # Assuming 'test' split dir exists within base_dataset_path for non-TSRS datasets as well.
        test_data_path = os.path.join(base_dataset_path, 'test') 

    # Check if the determined test_data_path actually exists before proceeding
    if not os.path.isdir(test_data_path):
        logging.error(f"❌ ERROR: Test data directory not found at: '{test_data_path}'. "
                      f"Please verify the DATASET_PATHS in config.py and the directory structure. "
                      f"Expected images in '{test_data_path}'.")
        sys.exit(1) # Exit if the test directory doesn't exist

    # Initialize the dataset. ImageFolder will load data from `test_data_path` based on the `split='test'` argument.
    try:
        test_set = ImageFolder(test_data_path, args.dataset_name, args, split='test') 
    except RuntimeError as e:
        logging.error(f"❌ ERROR initializing ImageFolder: {e}")
        sys.exit(1)

    # --- End Path Construction ---

    logging.info(f"Found {len(test_set)} testing images for dataset '{args.dataset_name}'.")
    
    # If no images were loaded, exit gracefully
    if len(test_set) == 0:
        logging.warning("No testing images were loaded. Exiting.")
        sys.exit(0) # Exit successfully if no test data, as it's not an error per se.

    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load the BEST trained model
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1)

    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(DEVICE)
    net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
    net.eval()
    logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")

    # Initialize Loss Functions (for consistent loss calculation in evaluation)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(DEVICE)
    dice_loss_fn = DiceLoss().to(DEVICE)

    # --- Run Evaluation ---
    # Initialize ConfusionMatrix only if there are images to process
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            if data is None: 
                logging.warning(f"Skipping empty testing batch due to corrupted/missing samples.")
                continue
            
            # Ensure data is on the correct device
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            
            outputs = net(inputs) # Model returns tuple of outputs
            final_pred_for_metrics = outputs[-1] # Use final output for metrics
            
            total_loss = 0
            # Calculate total loss for logging purposes, using deep supervision weights
            for i, pred_output in enumerate(outputs):
                # Ensure labels are long for loss functions if they are not already
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                # Ensure index is within bounds for deep supervision weights
                if i < len(args.deep_supervision_weights):
                    total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
                else:
                    logging.warning(f"Deep supervision weight index {i} out of bounds. Skipping weight for output {i}.")
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            # Update confusion matrix with final predictions
            # Argmax to get class index, flatten for confmat
            confmat.update(labels.flatten(), final_pred_for_metrics.argmax(1).flatten())

    # --- Compute and Log Metrics ---
    # Only compute if confmat has been updated (i.e., at least one batch was processed)
    if confmat.mat is not None: # Check if confmat was actually initialized/updated
        global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
        mIoU = class_iou.mean().item()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logging.info(
            f"\n\n--- Final Test Results ({timestamp}) ---\n"
            f"Model: LASA-Unet with ASPP and {args.backbone} backbone\n" # Updated model description
            f"Experiment Name: {exp_name}\n"
            f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n" # Changed to 'test' for clarity
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
    else:
        logging.warning("Confusion matrix could not be computed as no data was processed.")

    logging.info("✅ Final testing completed.")

if __name__ == '__main__':
    main()