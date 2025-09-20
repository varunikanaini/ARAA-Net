# /kaggle/working/ARAA-Net/test_lasa_vgg.py (FINAL VERSION with DETAILED METRICS)
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
from lasa_vgg_model import LASA_Unet # Use the generalized LASA_Unet
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir
from config import DATA_ROOT, CKPT_ROOT
from train_lasa_vgg import FocalLoss, DiceLoss # Import loss functions for consistent loss calculation


def get_test_args():
    parser = argparse.ArgumentParser(description='Test LASA-Unet Model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Dataset used for training')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture used for training')
    parser.add_argument('--scale-h', type=int, default=448, help='Height images were resized to')
    parser.add_argument('--scale-w', type=int, default=448, help='Width images were resized to')
    
    # Deep Supervision weights (needed for loss calculation in evaluate_model if you log loss)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    # Focal Loss specific hyperparameters
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    # Dice Loss specific hyperparameters
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component in combined loss.')


    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to avoid duplicate log entries
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Construct the correct experiment name to find the checkpoint ---
    # Must match the training script's naming convention
    EXP_NAME = f"{args.backbone}_LASA_Unet_FocalDice_DS_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    check_mkdir(log_dir) # Ensure log directory exists
    setup_logging(log_dir) # Setup logging for this specific test run

    logging.info(f"Starting FINAL TESTING for experiment '{EXP_NAME}'")
    logging.info(f"Arguments: {args}")

    # --- Load the 'test' split of the data ---
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    # Assuming your dataset has a 'test' subdirectory. If not, use 'val' instead.
    test_data_path = os.path.join(dataset_path, 'test') 

    if not os.path.exists(test_data_path):
        logging.error(f"❌ ERROR: Test data not found at '{test_data_path}'. Please check dataset structure or specify 'val' split if no 'test' exists.")
        sys.exit(1)

    test_set = ImageFolder(test_data_path, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    logging.info(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    # --- Load the BEST trained model ---
    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1)

    # Instantiate the model with the correct backbone
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(DEVICE)
    net.load_state_dict(torch.load(checkpoint_to_load, map_location=DEVICE))
    net.eval()
    logging.info(f"✅ Model loaded successfully from best checkpoint: {checkpoint_to_load}")

    # --- Initialize Loss Functions (needed for consistent loss calculation in evaluation) ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(DEVICE)
    dice_loss_fn = DiceLoss().to(DEVICE)

    # --- Run Evaluation (re-using the evaluate_model function from train_lasa_vgg.py) ---
    # We copy relevant parts of evaluate_model here or define it separately
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
        f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n" 
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