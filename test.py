# /kaggle/working/ARAA-Net/test.py (FINAL VERSION)
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

from daseg import daseg
from datasets import ImageFolder
from config import DATA_ROOT, CKPT_ROOT
from seg_utils import ConfusionMatrix
from misc import check_mkdir

def get_test_args():
    parser = argparse.ArgumentParser(description='Test ARAA-Net with LASA model')
    # --- Args must match train.py to load the correct model and data ---
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Name of the dataset used for training')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone (must match trained model)')
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to for testing')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to for testing')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CHANGE 3: USE NEW DIRECTORY FORMAT ---
    EXP_NAME = f"lasa_mfr_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}_{args.backbone}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    log_file_path = os.path.join(log_dir, 'final_testing_results.log')
    print(f"Results for experiment '{EXP_NAME}' will be saved to: {log_file_path}")

    # --- CHANGE 5: USE 'test' SPLIT FOR DATA ---
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    test_data_path = os.path.join(dataset_path, 'test')

    if not os.path.exists(test_data_path):
        print(f"❌ ERROR: Test data not found at '{test_data_path}'")
        return

    # Pass args to ImageFolder so it uses the correct scaling
    test_set = ImageFolder(test_data_path, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    if len(test_set) == 0:
        return

    # --- Load the BEST performing checkpoint ---
    best_checkpoint_path = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(best_checkpoint_path):
        print(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'")
        return

    net = daseg(backbone_name=args.backbone).to(DEVICE)
    
    state_dict = torch.load(best_checkpoint_path, map_location=DEVICE)
    # Handle DataParallel wrapper if present
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    
    net.eval()
    print("✅ Model loaded successfully from best checkpoint.")

    # --- Run Evaluation ---
    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())

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
        f"FWIoU           = {fwiou.item():.4f}\n" # Added this line
        f"--------------------------------------------------\n"
    )
    print(results_text)
    with open(log_file_path, 'a') as f:
        f.write(results_text)
    print(f"✅ Results appended to: {log_file_path}")

if __name__ == '__main__':
    main()