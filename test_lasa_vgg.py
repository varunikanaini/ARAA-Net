# /kaggle/working/ARAA-Net/test_lasa_vgg.py (Updated with arguments to find correct checkpoint)
# --- EXECUTING TEST_LASA_VGG.PY ---
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
from lasa_vgg_model import LASA_Unet # <<< IMPORT THE UPDATED MODEL
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter 
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 

# Import loss functions for consistent loss calculation
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
    
    # --- LASA Module Tuning Arguments (to match training config for checkpoint finding) ---
    parser.add_argument('--lasa-M', type=int, default=4, help='Number of groups (M) for LASA module. Default: 4.')
    parser.add_argument('--lasa-L', nargs='+', type=int, default=[5, 7, 9, 11], 
                        help='List of square side lengths (L) for LASA module. Default: [5, 7, 9, 11].')

    # --- Deep Supervision Weights (for consistent loss calculation during eval) ---
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses. Must have 5 values.')
    
    # --- Loss Function Parameters (for consistent loss calculation during eval) ---
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component in combined loss.')

    # --- Other potential arguments (dummy for consistency, not used for actual processing here) ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg.')
    parser.add_argument('--wavelet-type', type=str, default='haar', help='Dummy arg.')
    parser.add_argument('--wavelet-level', type=int, default=1, help='Dummy arg.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, help='Dummy arg.')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    # Basic validation for test arguments
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Custom collate function to filter out None samples
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct the correct experiment name to find the checkpoint
    # Use the same naming convention as in train_lasa_vgg.py
    exp_name_parts = [
        args.backbone,
        f"LASA_M{args.lasa_M}_L{'_'.join(map(str, args.lasa_L))}", # Include LASA params
        "FocalDice",
        "DS",
        "WaveletHE"
    ]
    dataset_name_clean = args.dataset_name.replace('TSRS_RSNA-', '').lower()
    exp_name = "_".join(exp_name_parts) + f"_{dataset_name_clean}"
    
    log_dir = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(log_dir) 
    setup_logging(log_dir) 

    logging.info(f"Starting FINAL TESTING for experiment '{exp_name}'")
    logging.info(f"Arguments used for testing: {args}")

    try:
        base_dataset_path = DATASET_PATHS[args.dataset_name]
    except KeyError:
        logging.error(f"Dataset '{args.dataset_name}' not found in DATASET_PATHS configuration.")
        sys.exit(1)

    # --- Data Loading for Testing ---
    # For TSRS datasets, we expect the 'val' directory to contain the test data.
    # For other datasets, we expect a 'test' directory if it exists.
    if 'TSRS_RSNA' in args.dataset_name:
        test_data_path = os.path.join(base_dataset_path, 'val') 
        # Pass 'val' as the split to ImageFolder, because for TSRS, the 'val' dir contains the test data.
        test_split_arg = 'val' 
    else:
        test_data_path = base_dataset_path
        # For non-TSRS datasets, explicitly request the 'test' split.
        test_split_arg = 'test' 
        
    test_set = ImageFolder(test_data_path, args.dataset_name, args, split=test_split_arg) 
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False, collate_fn=custom_collate_fn)
    
    if not test_set:
        logging.error(f"Test dataset is empty for '{args.dataset_name}'. Check dataset path and split configuration.")
        sys.exit(1)
    logging.info(f"Found {len(test_set)} testing images for dataset '{args.dataset_name}'.")

    # Load the BEST trained model
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this experiment configuration.")
        sys.exit(1)

    # Initialize the model with the specified backbone and LASA parameters used during training
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone, 
                    lasa_M=args.lasa_M, lasa_L=args.lasa_L).to(DEVICE)
    
    try:
        net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
        logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")
    except Exception as e:
        logging.error(f"Error loading model state dict: {e}. Exiting.")
        sys.exit(1)

    net.eval() # Set model to evaluation mode

    # Initialize Loss Functions (for consistent loss calculation in evaluation)
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(DEVICE)
    dice_loss_fn = DiceLoss().to(DEVICE)

    # Run Evaluation
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            if data is None: 
                logging.warning(f"Skipping empty testing batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            
            outputs = net(inputs) 
            final_pred = outputs[-1] 
            
            # Calculate total loss for logging purposes
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                pred_output = pred_output.to(labels.device) # Ensure same device
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            # Update confusion matrix
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())

    # Compute and log all metrics from the confusion matrix
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(
        f"\n\n--- Final Test Results ({timestamp}) ---\n"
        f"Model: LASA-Unet with {args.backbone} backbone\n"
        f"Experiment Name: {exp_name}\n"
        f"Dataset: {args.dataset_name} (evaluated on '{test_split_arg}' split)\n" 
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