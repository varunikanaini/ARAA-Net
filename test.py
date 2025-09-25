#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import time
import datetime
import os
import argparse # Import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from config import backbone_path, DATASET_PATHS # Import DATASET_PATHS from config

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from collections import OrderedDict

from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from daseg import daseg
import loss

from seg_utils import ConfusionMatrix

cudnn.benchmark = True

torch.manual_seed(2021)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add argparse for dataset selection and model snapshot
parser = argparse.ArgumentParser(description='DANet Testing')
parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis',
                    help='Dataset to use for testing (TSRS_RSNA-Epiphysis, JSRT, COVID19_Radiography, CVC-ClinicDB)')
parser.add_argument('--snapshot', type=str, required=True,
                    help='Path to the trained model snapshot (e.g., ckpt/DANet_TSRS_RSNA-Epiphysis/best.pth)')
args_parser = parser.parse_args()


# Dynamic root paths based on selected dataset
if args_parser.dataset == 'TSRS_RSNA-Epiphysis':
    test_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_test']
    test_dataset_name = 'TSRS_RSNA-Epiphysis_test'
elif args_parser.dataset == 'JSRT':
    test_root_path = DATASET_PATHS['JSRT']
    test_dataset_name = 'JSRT'
elif args_parser.dataset == 'COVID19_Radiography':
    test_root_path = DATASET_PATHS['COVID19_Radiography']
    test_dataset_name = 'COVID19_Radiography'
elif args_parser.dataset == 'CVC-ClinicDB':
    test_root_path = DATASET_PATHS['CVC-ClinicDB']
    test_dataset_name = 'CVC-ClinicDB'
else:
    raise ValueError(f"Unsupported dataset: {args_parser.dataset}")


ckpt_path = './ckpt'
exp_name = 'DANet_' + args_parser.dataset # Match experiment name with training
results_path = os.path.join('./results', exp_name)
check_mkdir(results_path)
log_path = os.path.join(results_path, 'test_log.txt')


args = {
    'scale_w': 576,
    'scale_h': 896,
    'save_results': True # Whether to save predicted masks
}

print(torch.__version__)

# Loss functions for consistency, though not used for optimizing in test.py
# Kept as per original functionality, as they define bce_iou_loss
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
# last_criterion is not explicitly used for calculation here but part of original code context
last_criterion = nn.CrossEntropyLoss(ignore_index=255) 

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss


# Prepare Data Set.
test_set = ImageFolder(test_root_path, test_dataset_name, split='test')
print(f"Test set ({test_dataset_name}): {test_set.__len__()} images")
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)


def evaluate(net):
    net.eval()
    curr_iter = 1
    start_time = time.time()

    confmat = ConfusionMatrix(num_classes=2)
    
    test_iterator = tqdm(test_loader, total=len(test_loader), desc="Testing")
    for data in test_iterator:
        inputs, labels, name_tuple = data['image'], data['label'], data['name']
        
        # name_tuple is (img_path, gt_path), extract only the image filename
        pname = os.path.basename(name_tuple[0][0]) 
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # Only the final prediction `predict0` is directly used for metrics and saving.
            # Other predictions are computed but not used further in test.
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
            
            pred_mask = predict0.argmax(1) # Get the predicted class mask (0 or 1)
            
            # Update confusion matrix for metrics calculation
            confmat.update(labels.flatten(), pred_mask.flatten())
            
            if args['save_results']:
                # Convert predicted mask to 0-255 grayscale image for saving
                binary_mask_for_save = (pred_mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
                mask_image = Image.fromarray(binary_mask_for_save, mode='L') # 'L' for grayscale image
                
                # Save the mask in a 'masks' subdirectory within the results folder
                save_dir = os.path.join(results_path, 'masks')
                check_mkdir(save_dir)
                # Ensure consistent file extension (.png)
                base_filename_no_ext = os.path.splitext(pname)[0]
                mask_image.save(os.path.join(save_dir, base_filename_no_ext + '.png'))
        
        # Update progress bar description
        _, _, class_iou, _, mDice = confmat.compute() # Get current metrics
        log_str = f"Testing: Iter {curr_iter}/{len(test_loader)}, mIoU: {np.mean(class_iou.cpu().numpy()):.5f}, mDice: {mDice:.5f}"
        test_iterator.set_description(log_str)
        curr_iter += 1

    # Compute and print final metrics after iterating through the entire test set
    global_acc, class_acc, class_iou, FWIoU, mDice = confmat.compute()
    global_acc = global_acc.item()
    class_acc = class_acc.cpu().numpy()
    class_iou = class_iou.cpu().numpy()
    FWIoU = FWIoU.cpu().numpy()
    
    final_log_str = (
        f'--- Final Testing Results ({test_dataset_name}) ---\n'
        f'Model: {args_parser.snapshot}\n'
        f'Global Acc: {global_acc:.4f}\n'
        f'Class Acc: {class_acc}\n'
        f'Class IoU: {class_iou}\n'
        f'Mean IoU: {np.mean(class_iou):.4f}\n'
        f'FWIoU: {FWIoU:.4f}\n'
        f'Mean Dice: {mDice:.4f}\n'
    )
    print(final_log_str)
    # Write final results to a log file, overwriting previous content
    with open(log_path, 'w') as f:
        f.write(final_log_str + '\n')

    return np.mean(class_iou)


def main():
    print("Args:", args_parser.__dict__)
    
    net = daseg(backbone_path).to(device)

    # Load the specified snapshot
    model_snapshot_path = args_parser.snapshot
    if not os.path.exists(model_snapshot_path):
        raise FileNotFoundError(f"Model snapshot not found at: {model_snapshot_path}")
        
    print(f"Loading model from: {model_snapshot_path}")
    state_dict = torch.load(model_snapshot_path, map_location=device)
    
    # Remove 'module.' prefix if the model was saved from DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    
    net.eval() # Set to evaluation mode

    start = time.time()
    evaluate(net) # Call the evaluation function
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()