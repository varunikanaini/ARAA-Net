# /kaggle/working/ARAA-Net/test.py (FINAL 'lasa' BRANCH VERSION)
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import argparse

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg # Using 'models' subfolder
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir
from config import DATA_ROOT, CKPT_ROOT

def get_test_args():
    parser = argparse.ArgumentParser(description='Test ARAA-Net with LASA model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CHANGE 2: USE NEW DIRECTORY FORMAT ---
    EXP_NAME = f"LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}_{args.backbone}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    log_file_path = os.path.join(log_dir, 'final_testing_results.log')
    print(f"Results for experiment '{EXP_NAME}' will be saved to: {log_file_path}")

    # --- CHANGE 3: USE 'test' SPLIT FOR DATA ---
    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    test_data_path = os.path.join(dataset_path, 'test')

    if not os.path.exists(test_data_path):
        print(f"❌ ERROR: Test data not found at '{test_data_path}'")
        return

    test_set = ImageFolder(test_data_path, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
    print(f"Found {len(test_set)} testing images in '{test_data_path}'.")

    checkpoint_to_load = os.path.join(log_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_to_load):
        print(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'")
        return

    net = daseg(backbone_name=args.backbone).to(DEVICE)
    state_dict = torch.load(checkpoint_to_load, map_location=DEVICE)
    
    # Handle DataParallel wrapper from training
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    
    net.eval()
    print("✅ Model loaded successfully from best checkpoint.")

    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())

    # --- CHANGE 4: PRINT ALL DETAILED METRICS ---
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_text = (
        f"\n\n------ Test run at: {timestamp} ------\n"
        f"Model evaluated: {os.path.basename(checkpoint_to_load)}\n"
        f"Backbone: {args.backbone}\n"
        f"Experiment Name: {EXP_NAME}\n"
        f"Dataset: {args.dataset_name} (evaluated on 'test' split)\n" 
        f"Image scale for test: ({args.scale_h}, {args.scale_w})\n" 
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU        = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"FWIoU           = {fwiou.item():.4f}\n"
        f"Class IoU       = {class_iou.cpu().numpy()}\n" 
        f"Class Accuracy  = {class_acc.cpu().numpy()}\n" 
        f"--------------------------------------------------\n"
    )
    print(results_text)
    with open(log_file_path, 'a') as f: f.write(results_text)
    print(f"✅ Results appended to: {log_file_path}")

if __name__ == '__main__':
    main()