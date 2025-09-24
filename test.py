# --- FINAL EVALUATION WORKFLOW (DEFINITIVE & CORRECTED) ---

# --- STEP 1: SETUP THE ENVIRONMENT ---
import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import datetime
import argparse
import logging

# Add the project's code to the Python path
project_path = '/kaggle/working/ARAA-Net/'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
from datasets import ImageFolder, DATASET_CONFIGS
from datasets import make_dataset as make_full_dataset_list
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING

import loss
from torch import nn
structure_loss_fn = loss.structure_loss()
bce_loss_fn = nn.BCEWithLogitsLoss()
iou_loss_fn = loss.IOU()
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
def bce_iou_loss(pred, target): return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)


def get_test_args():
    parser = argparse.ArgumentParser(description='Test ARAA-Net Model with multi-backbone and multi-dataset support')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis',
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for testing')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Backbone architecture used for training')
    
    # Image transformation related arguments (required by ImageFolder)
    parser.add_argument('--scale-h', type=int, default=576, help='Height images were resized to for ImageFolder transforms')
    parser.add_argument('--scale-w', type=int, default=896, help='Width images were resized to for ImageFolder transforms')
    parser.add_argument('--crop-size-h', type=int, default=576, help='Height images were cropped to for ImageFolder transforms.')
    parser.add_argument('--crop-size-w', type=int, default=576, help='Width images were cropped to for ImageFolder transforms.')

    # These are specific to CenterAmplification which was in a previous ImageFolder reference.
    # Kept as args for compatibility/future use, but largely ignored by current ImageFolder test transforms.
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')

    # Programmatic splitting ratios (needed for consistency if programmatic split was used in training)
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Dummy arg for programmatic split consistency.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Dummy arg for programmatic split consistency.')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of worker processes for data loading.')

    # Deep supervision weights (these are actively used by the daseg model's loss function for reporting)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[1.0, 1.0, 2.0, 4.0, 10.0],
                        help='Weights for deep supervision losses for predict_1 to predict_0 (total 5 values).')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
        
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")

    return args

def setup_logging(log_dir, filename='final_testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def main():
    args = get_test_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("✅ Environment setup complete.")

    EXP_NAME = f"{args.backbone}_ARAA-Net_{args.dataset-name.replace('TSRS_RSNA-', '').lower()}"
    log_dir = os.path.join(CKPT_ROOT, EXP_NAME)
    check_mkdir(log_dir)
    setup_logging(log_dir)

    logging.info(f"Starting FINAL TESTING for experiment '{EXP_NAME}'")
    logging.info(f"Arguments: {args}")

    # --- Determine the base root for the dataset (download if KaggleHub) ---
    dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
    base_dataset_root = None

    if dataset_info and dataset_info['id']:
        base_dataset_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
        if not base_dataset_root:
            logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (KaggleHub): {base_dataset_root}")
    elif dataset_info and dataset_info['local_dir_name']:
        base_dataset_root = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
        if not os.path.exists(base_dataset_root):
            logging.error(f"Local dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (local): {base_dataset_root}")
    else:
        base_dataset_root = os.path.join(DATA_ROOT, args.dataset_name)
        if not os.path.exists(base_dataset_root):
            logging.error(f"Dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (default): {base_dataset_root}")

    # --- Data Loading and Splitting Logic for Testing ---
    logging.info("\n--- Loading Test Data ---")
    dataset_config = DATASET_CONFIGS.get(args.dataset_name)
    if not dataset_config:
        logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
        sys.exit(1)

    test_image_mask_list = []

    if dataset_config['has_predefined_splits']:
        logging.info(f"Using predefined splits for dataset '{args.dataset_name}' for testing.")
        test_path_defined = os.path.join(base_dataset_root, 'test')
        val_path_defined = os.path.join(base_dataset_root, 'val')

        if os.path.exists(test_path_defined) and os.listdir(test_path_defined):
            test_image_mask_list = make_full_dataset_list(test_path_defined, args.dataset_name, split_name='test')
            logging.info(f"Using 'test' split from predefined folder: {test_path_defined}")
        elif os.path.exists(val_path_defined) and os.listdir(val_path_defined):
            test_image_mask_list = make_full_dataset_list(val_path_defined, args.dataset_name, split_name='val')
            logging.warning(f"No explicit 'test' split folder found for '{args.dataset_name}'. Using 'val' split from '{val_path_defined}' for testing.")
        else:
            logging.error(f"Neither 'test' nor 'val' split folders found at '{base_dataset_root}'. Exiting.")
            sys.exit(1)

    else:
        logging.info(f"Using programmatic splitting for dataset '{args.dataset_name}' for testing.")
        full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
        
        if not full_image_mask_list:
            logging.error(f"No data found for programmatic splitting in '{base_dataset_root}'. Exiting.")
            sys.exit(1)

        total_len = len(full_image_mask_list)
        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        test_len = total_len - train_len - val_len

        _, _, test_image_mask_list = random_split(
            full_image_mask_list, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
        
        logging.info(f"Programmatic split for test: Total {total_len}, Test {len(test_image_mask_list)}")

    test_set = ImageFolder(test_image_mask_list, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    logging.info(f"Found {len(test_set)} testing images for dataset '{args.dataset_name}'.")

    if len(test_set) == 0:
        logging.error("❌ ERROR: The dataloader found 0 images. Cannot proceed with evaluation.")
        sys.exit(1)

    logging.info(f"\n--- Loading Trained {args.backbone} Model ---")
    checkpoint_path = os.path.join(log_dir, 'best_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{log_dir}'. Please run training first for this backbone/dataset combination.")
        sys.exit(1)

    net = daseg(backbone_name=args.backbone).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    
    net.eval()
    logging.info("✅ Model loaded successfully.")

    logging.info("\n--- Running Evaluation ---")
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            inputs, labels = data['image'].to(DEVICE), data['label'].to(DEVICE)
            
            predict_1, predict_2, predict_3, predict_4, predict_0 = net(inputs)
            final_pred = predict_0

            binary_labels = labels.unsqueeze(1).float()
            ce_labels = labels.long()
            
            loss_1 = bce_iou_loss(predict_1, binary_labels) * args.deep_supervision_weights[0]
            loss_2 = structure_loss_fn(predict_2, binary_labels) * args.deep_supervision_weights[1]
            loss_3 = structure_loss_fn(predict_3, binary_labels) * args.deep_supervision_weights[2]
            loss_4 = structure_loss_fn(predict_4, binary_labels) * args.deep_supervision_weights[3]
            loss_0 = ce_loss_fn(predict_0, ce_labels) * args.deep_supervision_weights[4]
            
            total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_0
            loss_recorder.update(total_loss.item(), inputs.size(0))
            
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    logging.info("\n--- Final Test Results ---")
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_text = (
        f"\n\n--- Final Test Results ({timestamp}) ---\n"
        f"Model: ARAA-Net with {args.backbone} backbone\n"
        f"Experiment Name: {EXP_NAME}\n"
        f"Dataset: {args.dataset_name} (evaluated on test split)\n" 
        f"Image scale for test: ({args.scale_h}, {args.scale_w})\n" 
        f"Crop size for test: ({args.crop_size_h}, {args.crop_size_w})\n"
        f"--------------------------------------------------\n"
        f"Global Accuracy = {global_acc.item():.4f}\n"
        f"Mean IoU (mIoU) = {mIoU:.4f}\n"
        f"Mean Dice       = {mDice:.4f}\n"
        f"FWIoU           = {fwiou.item():.4f}\n"
        f"Class IoU       = {class_iou.cpu().numpy()}\n" 
        f"Class Accuracy  = {class_acc.cpu().numpy()}\n" 
        f"Combined Loss (Avg) = {loss_recorder.avg:.4f}\n"
        f"--------------------------------------------------\n"
    )
    
    log_file_path = os.path.join(log_dir, 'final_testing_results.log')
    logging.info(results_text)
    with open(log_file_path, 'w') as f:
        f.write(results_text)
    logging.info(f"✅ Results successfully saved to: {log_file_path}")

if __name__ == '__main__':
    main()