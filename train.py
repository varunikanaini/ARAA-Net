# /kaggle/working/araa/ARAA-Net/test.py (CORRECTED)

import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from daseg import daseg
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir

# Add the project's code to the Python path
project_path = '/kaggle/working/araa/ARAA-Net/'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print("✅ Environment setup complete.")

# --- STEP 1: DEFINE PARAMETERS ---
BACKBONE_TO_TEST = 'resnet50' # Or 'resnet101', etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_ROOT = '/kaggle/working/araa/ARAA-Net/ckpt'
DATA_ROOT = '/kaggle/working/araa/ARAA-Net/data'
EXP_NAME = BACKBONE_TO_TEST + "_with_LASA" # Match the training experiment name

# --- STEP 2: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'final_testing_results.log')
print(f"Results will be saved to: {log_file_path}")

# --- STEP 3: LOAD THE DATA ---
print("\n--- Loading Test Data ---")
TEST_DATASET_NAME = 'TSRS_RSNA-Epiphysis'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)
test_data_path = os.path.join(dataset_path, 'val') # Use the validation set for testing

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
else:
    # --- THIS IS THE KEY CORRECTION ---
    # We now instantiate the ImageFolder by just specifying the split.
    test_set = ImageFolder(test_data_path, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")
    # ------------------------------------

    if len(test_set) == 0:
        print("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
    else:
        # --- STEP 4: LOAD THE TRAINED MODEL ---
        print(f"\n--- Loading Trained {EXP_NAME} Model ---")
        checkpoint_path = os.path.join(CKPT_ROOT, EXP_NAME, 'best_checkpoint.pth')

        if not os.path.exists(checkpoint_path):
            print(f"❌ ERROR: Checkpoint not found at '{checkpoint_path}'")
        else:
            net = daseg(backbone_name=BACKBONE_TO_TEST).to(DEVICE)
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            
            # This logic to handle 'module.' prefix is good, keep it.
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                net.load_state_dict(new_state_dict)
            else:
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

            results_text = (
                f"------ Final Test Results for {EXP_NAME} on {TEST_DATASET_NAME} ------\n"
                f"global_acc = {global_acc.item():.4f}\n"
                f"class_acc  = {class_acc.cpu().numpy()}\n"
                f"class_iou  = {class_iou.cpu().numpy()}\n"
                f"mIoU       = {mIoU:.4f}\n"
                f"FWIoU      = {fwiou.item():.4f}\n"
                f"mDice      = {mDice:.4f}\n"
                f"--------------------------------------------------------------------\n"
            )
            
            print(results_text)
            with open(log_file_path, 'w') as f:
                f.write(results_text)
            print(f"✅ Results successfully saved to: {log_file_path}")