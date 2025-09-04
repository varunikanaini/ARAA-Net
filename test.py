import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import argparse 

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
print(f"Project path set to: {project_path}")

from daseg import daseg
from datasets import ImageFolder 
from seg_utils import ConfusionMatrix
from misc import check_mkdir
from config import DATA_ROOT, CKPT_ROOT # Import DATA_ROOT and CKPT_ROOT

print("✅ Environment setup complete.")

BACKBONE_TO_TEST = 'resnet50' # Default value if not overridden by CLI args
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_args_parser(): 
    parser = argparse.ArgumentParser(description='Test ARAA-Net model')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone (must match trained model)')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Name of the dataset to use (e.g., TSRS_RSNA-Epiphysis, TSRS_RSNA-Articular-Surface)') # Added dataset-name arg
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to for testing')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to for testing')
    parser.add_argument('--crop-size', type=int, default=576, help='Crop size used during training/testing (square)') 
    return parser

try:
    test_args = get_test_args_parser().parse_args()
except SystemExit: 
    test_args = get_test_args_parser().parse_args([])


BACKBONE_TO_TEST = test_args.backbone

# New experiment name format: backbone_name_ULD_datasetname
EXP_NAME = f"{BACKBONE_TO_TEST}_ULD_{test_args.dataset_name}" 

log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'final_testing_results.log')
print(f"Results will be appended to: {log_file_path}")

print("\n--- Loading Test Data ---")
dataset_path = os.path.join(DATA_ROOT, test_args.dataset_name) # Use args.dataset_name

# Explicitly use the 'test' subfolder
test_data_path = os.path.join(dataset_path, 'test') 

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
    print("Please make sure the 'test' subfolder exists inside your dataset directory.")
else:
    scale_h_val = test_args.scale_h
    scale_w_val = test_args.scale_w
    # Adjust input size for inception_v3 if not explicitly overridden by args
    if BACKBONE_TO_TEST == 'inception_v3' and (scale_h_val == 896 or scale_w_val == 576): # Check current defaults
        print("InceptionV3 detected, overriding scale to 299x299 for testing as per common practice.")
        scale_h_val = 299
        scale_w_val = 299
    
    test_set = ImageFolder(
        test_data_path,
        split='test', 
        scale_h=scale_h_val,
        scale_w=scale_w_val,
        crop_size=test_args.crop_size 
    )
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}' for dataset '{test_args.dataset_name}'.")
    print(f"Test image dimensions (H, W) used for resize: ({scale_h_val}, {scale_w_val})")


    if len(test_set) == 0:
        print("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
    else:
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
                f"Backbone: {BACKBONE_TO_TEST}\n"
                f"Experiment Name: {EXP_NAME}\n"
                f"Dataset: {test_args.dataset_name} (evaluated on 'test' split)\n" 
                f"Image scale for test: ({scale_h_val}, {scale_w_val})\n" 
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