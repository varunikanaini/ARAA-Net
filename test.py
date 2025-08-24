import sys, os
# Add the project's code to the Python path so we can import its files
sys.path.append('/kaggle/working/ARAA-Net/')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import numpy as np
from daseg import daseg
from datasets import ImageFolder
import joint_transforms
from seg_utils import ConfusionMatrix
from misc import check_mkdir

# Add the project's code to the Python path so we can import its files
project_path = '/kaggle/working/ARAA-Net/'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print("✅ Environment setup complete.")

# --- STEP 2: DEFINE PARAMETERS ---
# Make sure this matches the backbone you trained
BACKBONE_TO_TEST = 'inception_v3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All paths point to your /kaggle/working/ directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# --- STEP 3: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, BACKBONE_TO_TEST)
check_mkdir(log_dir) # Ensure the directory exists
log_file_path = os.path.join(log_dir, 'testing_results.log')
print(f"Results will be saved to: {log_file_path}")

# --- STEP 4: LOAD THE DATA ---
print("\n--- Loading Test Data ---")
# IMPORTANT: Ensure this folder name is correct and exists inside '/kaggle/working/ARAA-Net/data/'
TEST_DATASET_NAME = 'TSRS_RSNA-Epiphysis'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)

if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Dataset not found at '{dataset_path}'")
    print(f"Please make sure the folder '{TEST_DATASET_NAME}' exists inside '{DATA_ROOT}'.")
else:
    # Define transformations for the test set
    test_joint_transform = joint_transforms.Compose([joint_transforms.Resize((576, 576))])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    target_transform = transforms.ToTensor()

    # Create the dataset and dataloader
    test_set = ImageFolder(dataset_path, joint_transform=test_joint_transform, transform=img_transform, target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{TEST_DATASET_NAME}'.")

    # --- STEP 5: LOAD THE TRAINED MODEL ---
    print(f"\n--- Loading Trained {BACKBONE_TO_TEST} Model ---")
    checkpoint_path = os.path.join(CKPT_ROOT, BACKBONE_TO_TEST, 'best_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at '{checkpoint_path}'")
        print("Please ensure your training has completed at least one epoch of validation and saved a checkpoint.")
    else:
        net = daseg(backbone_name=BACKBONE_TO_TEST).to(DEVICE)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle models saved with a 'module.' prefix (from nn.DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(state_dict)
        
        net.eval()
        print("✅ Model loaded successfully.")

        # --- STEP 6: RUN EVALUATION ---
        print("\n--- Running Evaluation ---")
        confmat = ConfusionMatrix(num_classes=2)
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
                _, _, _, _, pred = net(inputs)
                confmat.update(labels.flatten(), pred.argmax(1).flatten())
                
        # --- STEP 7: COMPUTE, PRINT, AND SAVE RESULTS ---
        print("\n--- Final Test Results ---")
        global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
        mIoU = class_iou.mean().item()

        # Improved results formatting for cleaner output
        results_text = (
            f"------ Final Test Results for {BACKBONE_TO_TEST} on {TEST_DATASET_NAME} ------\n"
            f"Global Accuracy = {global_acc.item():.4f}\n"
            f"Class Accuracy  = {class_acc}\n"
            f"Class IoU       = {class_iou}\n"
            f"Mean IoU (mIoU) = {mIoU:.4f}\n"
            f"FWIoU           = {fwiou.item():.4f}\n"
            f"Mean Dice       = {mDice:.4f}\n"
            f"--------------------------------------------------------------------\n"
        )
        
        print(results_text)
        
        with open(log_file_path, 'w') as f:
            f.write(results_text)
        print(f"✅ Results successfully saved to: {log_file_path}")
