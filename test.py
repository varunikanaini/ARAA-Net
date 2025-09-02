# /kaggle/working/ARAA-Net/test_LASA.py (FINAL VERSION FOR LASA MODEL - ALIGNED)
# NOTE: Renamed to test_LASA.py to distinguish from Teacher-Student test script

import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import argparse

# --- Add project path to run script from anywhere ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
print(f"Project path set to: {project_path}")

from daseg import daseg
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir

print("✅ Environment setup complete.")

# --- STEP 1: DEFINE PARAMETERS ---
BACKBONE_TO_TEST = 'resnet50' # Default value if not overridden by CLI args
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'
DATA_ROOT = '/kaggle/working/ARAA-Net/data'
# This MUST match the experiment name from train.py for LASA
EXP_NAME = BACKBONE_TO_TEST + "_with_LASA"

# --- NEW: Define argparse for test script to match train.py's image scaling defaults ---
def get_test_args_parser():
    parser = argparse.ArgumentParser(description='Test ARAA-Net LASA model')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone (must match trained model)')
    # These defaults MUST match what was used for training (from train.py's get_args for LASA)
    # Assuming LASA also used 896x576 by default
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to for testing')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to for testing')
    parser.add_argument('--crop-size', type=int, default=576, help='Crop size used during training/testing (square)')
    return parser

# --- CRITICAL FIX: Handle kernel arguments for argparse ---
try:
    test_args = get_test_args_parser().parse_args()
except SystemExit:
    test_args = get_test_args_parser().parse_args([])
# --- END CRITICAL FIX ---


# Override BACKBONE_TO_TEST to ensure consistency if run with command-line arg
BACKBONE_TO_TEST = test_args.backbone

# Dynamically set EXP_NAME based on the (potentially updated) backbone
EXP_NAME = BACKBONE_TO_TEST + "_with_LASA"

# --- STEP 2: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'final_testing_results.log')
print(f"Results will be appended to: {log_file_path}")

# --- STEP 3: LOAD THE DATA ---
print("\n--- Loading Test Data ---")
TEST_DATASET_NAME = 'TSRS_RSNA-Epiphysis'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)

# --- CRITICAL CHANGE: Use 'test' subfolder instead of 'val' ---
test_data_path = os.path.join(dataset_path, 'test') # Use the dedicated test set
# --- END CRITICAL CHANGE ---

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
    print("Please make sure the 'test' subfolder exists inside your dataset directory.")
else:
    # --- Use test_args for image dimensions ---
    scale_h_val = test_args.scale_h
    scale_w_val = test_args.scale_w
    crop_size_val = test_args.crop_size

    # Add a warning for InceptionV3 if it's not set to 299x299
    if BACKBONE_TO_TEST == 'inception_v3' and (scale_h_val != 299 or scale_w_val != 299 or crop_size_val != 299):
        print(f"⚠️ Warning: Using InceptionV3 with scale ({scale_h_val}, {scale_w_val}) and crop {crop_size_val}. "
              "Inception models often expect 299x299 input. Ensure these match training settings.")

    # The split parameter in ImageFolder is just a label, 'val' for test here means using val transforms (no random augmentations)
    test_set = ImageFolder(
        test_data_path,
        split='val',
        scale_h=scale_h_val,
        scale_w=scale_w_val,
        crop_size=crop_size_val
    )
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")
    print(f"Test image dimensions (H, W): ({scale_h_val}, {scale_w_val}), Crop size: {crop_size_val}")


    if len(test_set) == 0:
        print("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
    else:
        # --- STEP 4: LOAD THE TRAINED MODEL (ROBUST LOGIC) ---
        print(f"\n--- Loading Trained {EXP_NAME} Model ---")
        best_checkpoint_path = os.path.join(log_dir, 'best_checkpoint.pth')
        latest_checkpoint_path = os.path.join(log_dir, 'latest_checkpoint.pth')

        checkpoint_to_load = None
        if os.path.exists(best_checkpoint_path):
            print(f"Found 'best_checkpoint.pth'. Loading this version for final evaluation.")
            checkpoint_to_load = best_checkpoint_path
        elif os.path.exists(latest_checkpoint_path):
            print(f"Could not find 'best_checkpoint.pth'.")
            print(f"Falling back to 'latest_checkpoint.pth'. Note: This may not be the best performing model.")
            checkpoint_to_load = latest_checkpoint_path
        else:
            print(f"❌ ERROR: No checkpoint found at all in '{log_dir}'")
            print("Please run the training script first.")

        if checkpoint_to_load:
            net = daseg(backbone_name=BACKBONE_TO_TEST).to(DEVICE)

            state_dict_or_ckpt = torch.load(checkpoint_to_load, map_location=DEVICE)

            if 'model_state_dict' in state_dict_or_ckpt:
                state_dict = state_dict_or_ckpt['model_state_dict']
            else:
                state_dict = state_dict_or_ckpt

            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                net.load_state_dict(new_state_dict)
            else:
                net.load_state_dict(state_dict) # Corrected: load_state_dict

            net.eval()
            print("✅ Model loaded successfully.")

            # --- STEP 5: RUN EVALUATION ---
            print("\n--- Running Evaluation ---")
            confmat = ConfusionMatrix(num_classes=2)
            with torch.no_grad():
                for data in tqdm(test_loader, desc="Evaluating"):
                    inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
                    _, _, _, _, pred = net(inputs)
                    confmat.update(labels.flatten(), pred.argmax(1).flatten())

            # --- STEP 6: COMPUTE, PRINT, AND SAVE RESULTS ---
            print("\n--- Final Test Results ---")
            global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
            mIoU = class_iou.mean().item()

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results_text = (
                f"\n\n------ Test run at: {timestamp} ------\n"
                f"Model evaluated: {os.path.basename(checkpoint_to_load)}\n"
                f"Backbone: {BACKBONE_TO_TEST}\n"
                f"Experiment Name: {EXP_NAME}\n"
                f"Dataset: {TEST_DATASET_NAME} (evaluated on 'test' split)\n"
                f"Image scale for test: ({scale_h_val}, {scale_w_val}), Crop: {crop_size_val}\n"
                f"--------------------------------------------------\n"
                f"Global Accuracy = {global_acc.item():.4f}\n"
                f"Mean IoU        = {mIoU:.4f}\n"
                f"Mean Dice       = {mDice:.4f}\n"
                f"FWIoU           = {fwiou.item():.4f}\n"
                f"Class IoU       = {class_iou}\n"
                f"Class Accuracy  = {class_acc}\n"
                f"--------------------------------------------------\n"
            )

            print(results_text)

            with open(log_file_path, 'a') as f:
                f.write(results_text)
            print(f"✅ Results successfully appended to: {log_file_path}")