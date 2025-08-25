import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import numpy as np
import numpy as np
from daseg import daseg
from datasets import ImageFolder
import joint_transforms
from seg_utils import ConfusionMatrix
from misc import check_mkdir

# Add the project's code to the Python path
project_path = '/kaggle/working/ARAA-Net/'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print("✅ Environment setup complete.")

# --- STEP 2: DEFINE PARAMETERS ---
BACKBONE_TO_TEST = 'inception_v3' # Or 'resnet50', etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RESTORED: CORRECTED PATHS ---
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'
# This matches the structure shown in your file tree screenshot
DATA_ROOT = '/kaggle/working/araa/ARAA-Net/data'

# --- STEP 3: PREPARE FOR LOGGING ---
log_dir = os.path.join(CKPT_ROOT, BACKBONE_TO_TEST)
check_mkdir(log_dir)
log_file_path = os.path.join(log_dir, 'testing_results.log')
print(f"Results will be saved to: {log_file_path}")

# --- STEP 4: LOAD THE DATA (WITH ORIGINAL PATH LOGIC) ---
print("\n--- Loading Test Data ---")
TEST_DATASET_NAME = 'TSRS_RSNA-Articular-Surface'
dataset_path = os.path.join(DATA_ROOT, TEST_DATASET_NAME)

# --- RESTORED: LOOKING FOR 'val' SUBFOLDER ---
test_data_path = os.path.join(dataset_path, 'val')

if not os.path.exists(test_data_path):
    print(f"❌ ERROR: Test data not found at '{test_data_path}'")
    print("Please make sure the 'val' subfolder exists inside your dataset directory.")
else:
    # --- RESTORED: ORIGINAL TRANSFORM LOGIC ---
    test_joint_transform = joint_transforms.Compose([joint_transforms.Resize((896, 576))])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    target_transform = transforms.ToTensor()

    test_set = ImageFolder(test_data_path, joint_transform=test_joint_transform, transform=img_transform, target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    # Add a check to prevent crashing if no images are found
    if len(test_set) == 0:
        print("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
    else:
        # --- STEP 5: LOAD THE TRAINED MODEL ---
        print(f"\n--- Loading Trained {BACKBONE_TO_TEST} Model ---")
        checkpoint_path = os.path.join(CKPT_ROOT, BACKBONE_TO_TEST, 'best_checkpoint.pth')

        if not os.path.exists(checkpoint_path):
            print(f"❌ ERROR: Checkpoint not found at '{checkpoint_path}'")
        else:
            net = daseg(backbone_name=BACKBONE_TO_TEST).to(DEVICE)
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            
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
                    
            # --- STEP 7: COMPUTE, PRINT, AND SAVE RESULTS (WITH ORIGINAL FORMAT) ---
            print("\n--- Final Test Results ---")
            global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
            mIoU = class_iou.mean().item()

            results_text = (
                f"------ Final Test Results for {BACKBONE_TO_TEST} on {TEST_DATASET_NAME} ------\\n"
                f"global_acc = {global_acc.item():.4f}\\n"
                f"class_acc  = {class_acc.cpu().numpy()}\\n"
                f"class_iou  = {class_iou.cpu().numpy()}\\n"
                f"mIoU       = {mIoU:.4f}\\n"
                f"FWIoU      = {fwiou.item():.4f}\\n"
                f"mDice      = {mDice:.4f}\\n"
                f"--------------------------------------------------------------------\\n"
            )
            
            # Print to screen and save to file
            print(results_text.replace('\\n', '\\n'))
            with open(log_file_path, 'w') as f:
                f.write(results_text.replace('\\n', '\\n'))
            print(f"✅ Results successfully saved to: {log_file_path}")
