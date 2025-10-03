# /kaggle/working/ARAA-Net/test_lasa_vgg.py (FINAL & CORRECTED - VGG16 only)
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
from lasa_vgg_model import LASA_Unet 
from datasets import ImageFolder # Import the updated datasets module
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter 
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 

# Import loss functions for consistent loss calculation if logging loss during test
# These are needed if you want to report the test loss, otherwise can be omitted
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
    parser.add_argument('--num-classes', type=int, default=2, help='Number of segmentation classes (e.g., 2 for background + foreground)')
    
    # Image size for testing (should match training input size for fair comparison)
    parser.add_argument('--scale-h', type=int, default=896, help='Nominal image height for resizing (overridden by DASEG fixed size)')
    parser.add_argument('--scale-w', type=int, default=576, help='Nominal image width for resizing (overridden by DASEG fixed size)')
    
    # Parameters for loss calculation (if reporting test loss)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.5, help='Weight for Dice Loss component in combined loss.') # Match train setting

    # Dummy arguments needed by ImageFolder, ensure they match train args if possible
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
        # Useful for notebooks or if script is run without args to avoid exiting
        args = parser.parse_args([])
    
    # Validate arguments
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    """Configures logging for the testing script."""
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler) # Clear existing handlers
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[
                            logging.FileHandler(log_file),  # Log to file
                            logging.StreamHandler()         # Log to console
                        ])

# Custom collate function to filter out None samples
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct experiment name and log directory
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(log_dir) 
    setup_logging(log_dir) 

    logging.info(f"Starting FINAL TESTING for experiment '{exp_name}'")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Arguments: {args}")

    # --- Load Dataset ---
    # Get base dataset path from config
    base_dataset_path = DATASET_PATHS[args.dataset_name]

    # Determine the correct path and split for testing
    if 'TSRS_RSNA' in args.dataset_name:
        # For TSRS_RSNA, explicitly use the 'test' folder within the dataset path
        test_data_path = os.path.join(base_dataset_path, 'test')
    else:
        # For other datasets, ImageFolder will handle splitting (or use 'test' if explicit)
        # Ensure your ImageFolder logic correctly uses the 'test' split
        test_data_path = base_dataset_path
        
    # Instantiate the test dataset and loader, using split='test'
    # Note: ImageFolder will internally split if dataset_name is not TSRS_RSNA and explicit test folder isn't found
    test_set = ImageFolder(test_data_path, args.dataset_name, args, split='test') 
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    logging.info(f"Loaded {len(test_set)} testing images for dataset '{args.dataset_name}'.")

    # --- Load Model ---
    # Construct the path to the best model checkpoint
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1) # Exit if the model checkpoint is not found

    # Instantiate the model
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(DEVICE)
    # Load the state dictionary from the best checkpoint
    try:
        net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
        logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")
    except Exception as e:
        logging.error(f"Error loading model from checkpoint {checkpoint_to_load}: {e}. Exiting.")
        sys.exit(1)

    # Initialize Loss Functions (used for calculating test loss if needed, and for consistency)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255).to(DEVICE)
    dice_loss_fn = DiceLoss(smooth=1e-6, ignore_index=255).to(DEVICE)

    # --- Run Evaluation ---
    confmat = ConfusionMatrix(num_classes=args.num_classes) # Initialize confusion matrix
    loss_recorder = AvgMeter() # Initialize loss meter

    logging.info("Starting evaluation on the test set...")
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through the test data loader
        for data in tqdm(test_loader, desc="Testing", leave=False):
            if data is None: # Handle potential None batches
                logging.warning("Skipping empty testing batch due to corrupted/missing samples.")
                continue
                
            inputs = data['image'].to(DEVICE)
            labels = data['label'].to(DEVICE)
            
            # Forward pass
            outputs = net(inputs) 
            final_pred = outputs[-1] # Get the final prediction (from the highest resolution decoder stage)
            
            # Calculate total loss if reporting test loss (using the same weighted scheme as training)
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0)) # Record average loss
            
            # Update confusion matrix with predicted class and ground truth
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())

    # --- Compute and Log Final Metrics ---
    # Get computed metrics from the confusion matrix
    global_acc, class_acc, class_iou, fwiou, mDice, mIoU = confmat.compute()
    
    # Convert tensors to CPU numpy arrays for easier display if needed
    class_acc_np = class_acc.cpu().numpy()
    class_iou_np = class_iou.cpu().numpy()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(
        f"\n\n--- Final Test Results ({timestamp}) ---\n"
        f"Model: LASA-Unet with {args.backbone} backbone\n"
        f"Experiment Name: {exp_name}\n"
        f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n" 
        f"Image scale for test: ({args.scale_h}, {args.scale_w})\n" 
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU (mIoU) = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"FWIoU           = {fwiou.item():.4f}\n"
        f"Class IoU       = {class_iou_np}\n" 
        f"Class Accuracy  = {class_acc_np}\n" 
        f"Combined Loss (Avg) = {loss_recorder.avg:.4f}\n" 
        f"--------------------------------------------------\n"
    )
    logging.info("✅ Final testing completed.")

if __name__ == '__main__':
    main()
