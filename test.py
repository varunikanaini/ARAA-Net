# /kaggle/working/ARAA-Net/test.py (FINAL VERSION FOR TEACHER-STUDENT MODEL)

import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import argparse # Keep argparse for consistent definition of parameters

# --- Add project path to run script from anywhere ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
print(f"Project path set to: {project_path}")

from daseg import daseg
from datasets import ImageFolder # Ensure datasets.py has been modified as above
from seg_utils import ConfusionMatrix
from misc import check_mkdir

print("✅ Environment setup complete.")

# --- STEP 1: DEFINE PARAMETERS ---
BACKBONE_TO_TEST = 'resnet50' # Or 'resnet101', 'vgg16', 'inception_v3', etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'
DATA_ROOT = '/kaggle/working/ARAA-Net/data'
# This MUST match the experiment name used in train.py for Teacher-Student
EXP_NAME = BACKBONE_TO_TEST + "_teacher_student" 

# --- STEP 2: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'final_testing_results.log')
print(f"Results will be appended to: {log_file_path}")

# --- STEP 3: LOAD THE DATA ---
print("\n--- Loading Test Data ---")
TEST_DATASET_NAME = 'TSRS_RSNA-Epiphysis'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)
test_data_path = os.path.join(dataset_path, 'val') # Using the validation set for testing

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
else:
    # Determine image dimensions based on backbone, consistent with train.py
    scale_h_val = 299 if BACKBONE_TO_TEST == 'inception_v3' else 896
    scale_w_val = 299 if BACKBONE_TO_TEST == 'inception_v3' else 576
    crop_size_val = 299 if BACKBONE_TO_TEST == 'inception_v3' else 576 # Assuming crop_size for validation is same as target size
    
    # NEW: Pass the dynamic scale parameters to ImageFolder
    test_set = ImageFolder(
        test_data_path,
        split='val',
        scale_h=scale_h_val,
        scale_w=scale_w_val,
        crop_size=crop_size_val # This might not be strictly used by transform_val, but for consistency
    )
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

            # Handle both formats: latest checkpoint (dict with 'model_state_dict') and best checkpoint (state_dict only)
            if 'model_state_dict' in state_dict_or_ckpt:
                state_dict = state_dict_or_ckpt['model_state_dict']
            else:
                state_dict = state_dict_or_ckpt
            
            # Remove 'module.' prefix if the model was saved with DataParallel but loaded without it
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                net.load_state_dict(new_state_dict)
            else:
                net.load_state_dict(state_dict)

            net.eval() # Set model to evaluation mode
            print("✅ Model loaded successfully.")

            # --- STEP 5: RUN EVALUATION ---
            print("\n--- Running Evaluation ---")
            confmat = ConfusionMatrix(num_classes=2) # Assuming binary segmentation
            with torch.no_grad():
                for data in tqdm(test_loader, desc="Evaluating"):
                    inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
                    # The daseg model returns 5 predictions; pred0 is the final one
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
                f"Dataset: {TEST_DATASET_NAME}\n"
                f"--------------------------------------------------\n"
                f"Global Accuracy = {global_acc.item():.4f}\n"
                f"Mean IoU        = {mIoU:.4f}\n"
                f"Mean Dice       = {mDice:.4f}\n"
                f"FWIoU           = {fwiou.item():.4f}\n"
                f"Class IoU       = {class_iou}\n" # Raw per-class IoU values
                f"Class Accuracy  = {class_acc}\n" # Raw per-class accuracy values
                f"--------------------------------------------------\n"
            )
            
            print(results_text)
            
            # --- APPEND TO FILE INSTEAD OF OVERWRITING ---
            with open(log_file_path, 'a') as f:
                f.write(results_text)
            print(f"✅ Results successfully appended to: {log_file_path}")