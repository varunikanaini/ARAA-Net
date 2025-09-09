# /kaggle/working/ARAA-Net/test_lasa_vgg.py
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import datetime

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_VGG_Unet
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir
from config import DATA_ROOT, CKPT_ROOT

def get_test_args():
    parser = argparse.ArgumentParser(description='Test Standalone LASA-VGG-Unet Model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Dataset used for training')
    # These scale args must match what the model was trained with
    parser.add_argument('--scale-h', type=int, default=448, help='Height images were resized to')
    parser.add_argument('--scale-w', type=int, default=448, help='Width images were resized to')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Construct the correct experiment name to find the checkpoint ---
    EXP_NAME = f"standalone_LASA_VGG16_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    log_file_path = os.path.join(log_dir, 'final_testing_results.log')
    print(f"Results for experiment '{EXP_NAME}' will be saved to: {log_file_path}")

    # --- Load the 'test' split of the data ---
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    test_data_path = os.path.join(dataset_path, 'test')

    if not os.path.exists(test_data_path):
        print(f"❌ ERROR: Test data not found at '{test_data_path}'")
        return

    test_set = ImageFolder(test_data_path, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    # --- Load the BEST trained model ---
    best_checkpoint_path = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(best_checkpoint_path):
        print(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first.")
        return

    net = LASA_VGG_Unet(num_classes=2).to(DEVICE)
    # The 'best_checkpoint.pth' for this model saves the raw state_dict
    net.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    net.eval()
    print("✅ Model loaded successfully from best checkpoint.")

    # --- Run Evaluation ---
    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())

    # --- Compute, Print, and Save Results ---
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_text = (
        f"\n\n------ Test run at: {timestamp} ------\n"
        f"Model: {EXP_NAME}\n"
        f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n"
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU        = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"--------------------------------------------------\n"
    )
    print(results_text)
    with open(log_file_path, 'a') as f:
        f.write(results_text)
    print(f"✅ Results appended to: {log_file_path}")

if __name__ == '__main__':
    main()