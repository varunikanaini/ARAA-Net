# /kaggle/working/araa/ARAA-Net/test.py (FINAL VERSION FOR LASA MODEL)

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
BACKBONE_TO_TEST = 'resnet50' # Or 'resnet101', 'vgg16', etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'
DATA_ROOT = '/kaggle/working/ARAA-Net/data'
EXP_NAME = BACKBONE_TO_TEST + "_with_LASA" # This MUST match the experiment name from train.py

# --- STEP 2: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'final_testing_results.log')
print(f"Results will be appended to: {log_file_path}")

# --- STEP 3: LOAD THE DATA ---
print("\n--- Loading Test Data ---")
TEST_DATASET_NAME = 'TSRS_RSNA-Articular-Surface'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)
test_data_path = os.path.join(dataset_path, 'val') # Using the validation set for testing

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
else:
    # We need a dummy args object to pass to the dataset, matching the train script's defaults
    dataset_args = argparse.Namespace(scale_h=896, scale_w=576)
    
    # The new ImageFolder call is much simpler
    test_set = ImageFolder(test_data_path, args=dataset_args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")

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

            # Handle both formats: latest checkpoint (dict) and best checkpoint (state_dict only)
            if 'model_state_dict' in state_dict_or_ckpt:
                state_dict = state_dict_or_ckpt['model_state_dict']
            else:
                state_dict = state_dict_or_ckpt
            
            net.load_state_dict(state_dict)
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
                f"Dataset: {TEST_DATASET_NAME}\n"
                f"--------------------------------------------------\n"
                f"Global Accuracy = {global_acc.item():.4f}\n"
                f"Mean IoU        = {mIoU:.4f}\n"
                f"Mean Dice       = {mDice:.4f}\n"
                f"FWIoU           = {fwiou.item():.4f}\n"
                f"Class IoU       = {class_iou}\n"
                f"--------------------------------------------------\n"
            )
            
            print(results_text)
            
            # --- APPEND TO FILE INSTEAD OF OVERWRITING ---
            with open(log_file_path, 'a') as f:
                f.write(results_text)
            print(f"✅ Results successfully appended to: {log_file_path}")