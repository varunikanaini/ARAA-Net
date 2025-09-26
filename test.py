#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import time
import datetime
import os
import argparse 
from collections import OrderedDict
import logging 
from PIL import Image 
import torch.utils.data.dataloader 
import sys # ADDED: Import sys for stdout flushing


from config import backbone_path, DATASET_PATHS 

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils

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
                    help='Dataset to use for testing (TSRS_RSNA-Epiphysis, JSRT, COVID19_Radiography, CVC-ClinicDB, DentalPanoramic, SixDiseasesChestXRay)') 
parser.add_argument('--snapshot', type=str, required=True,
                    help='Path to the trained model snapshot (e.g., ckpt/DANet_TSRS_RSNA-Epiphysis/best.pth)')
parser.add_argument('--batch_size', type=int, default=1, # Default to 1 for testing
                    help='Batch size for testing')
parser.add_argument('--scale_w', type=int, default=576,
                    help='Width to scale input images to (Note: Actual transform sizes are fixed to 576x896 as per paper).')
parser.add_argument('--scale_h', type=int, default=896,
                    help='Height to scale input images to (Note: Actual transform sizes are fixed to 576x896 as per paper).')
parser.add_argument('--save_results', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Whether to save predicted masks')
args_parser = parser.parse_args()


# Dynamic root paths based on selected dataset
if args_parser.dataset == 'TSRS_RSNA-Epiphysis':
    test_root_path = DATASET_PATHS['TSRS_RSNA-Epiphysis_test']
    test_dataset_name = 'TSRS_RSNA-Epiphysis_test'
elif args_parser.dataset in ['JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 'DentalPanoramic', 'SixDiseasesChestXRay']:
    test_root_path = DATASET_PATHS[args_parser.dataset]
    test_dataset_name = args_parser.dataset
else:
    raise ValueError(f"Unsupported dataset: {args_parser.dataset}")

ckpt_path = './ckpt'
exp_name = 'DANet_' + args_parser.dataset 
results_path = os.path.join('./results', exp_name)
check_mkdir(results_path)
log_path = os.path.join(results_path, 'test_log.txt')

# Configure logging for test.py
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_path),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Use parsed arguments directly
args = vars(args_parser)

logger.info(f"Test arguments: {args}")
logger.info(f"Testing dataset: {test_dataset_name}")

bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255) 

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

# Custom collate function to handle None samples
def custom_collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch: # If the entire batch was corrupted/skipped
        return None # Indicate an empty batch
    return torch.utils.data.dataloader.default_collate(batch)

# Prepare Data Set.
test_set = ImageFolder(test_root_path, test_dataset_name, split='test')
logger.info(f"Test set ({test_dataset_name}): {test_set.__len__()} images")
test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=0, shuffle=False, collate_fn=custom_collate_fn) # Use custom collate_fn


def evaluate(net):
    net.eval()
    curr_iter = 1

    confmat = ConfusionMatrix(num_classes=2)
    
    test_iterator = tqdm(test_loader, total=len(test_loader), desc="Testing")
    for data in test_iterator:
        if data is None: # Skip if collate_fn returned None (entire batch was invalid)
            logger.warning(f"Testing: Iter {curr_iter}: Skipping empty test batch due to corrupted/missing samples.")
            continue

        inputs, labels, name_tuple = data['image'], data['label'], data['name']
        
        pname = os.path.basename(name_tuple[0][0]) 
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
            
            pred_mask = predict0.argmax(1) 
            
            confmat.update(labels.flatten(), pred_mask.flatten())
            
            if args['save_results']:
                binary_mask_for_save = (pred_mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
                mask_image = Image.fromarray(binary_mask_for_save, mode='L') 
                
                save_dir = os.path.join(results_path, 'masks')
                check_mkdir(save_dir)
                base_filename_no_ext = os.path.splitext(pname)[0]
                mask_image.save(os.path.join(save_dir, base_filename_no_ext + '.png'))
        
        _, _, class_iou, _, mDice = confmat.compute() 
        log_str = f"Testing: Iter {curr_iter}/{len(test_loader)}, mIoU: {np.mean(class_iou.cpu().numpy()):.5f}, mDice: {mDice:.5f}"
        test_iterator.set_description(log_str)
        curr_iter += 1

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
    logger.info(final_log_str) 
    print(final_log_str)      # Explicit print to guarantee console output
    sys.stdout.flush()         # ADDED: Force flush stdout

    return np.mean(class_iou)


def main():
    logger.info("Starting testing process.")
    
    net = daseg(backbone_path).to(device)

    model_snapshot_path = args_parser.snapshot
    if not os.path.exists(model_snapshot_path):
        raise FileNotFoundError(f"Model snapshot not found at: {model_snapshot_path}")
        
    logger.info(f"Loading model from: {model_snapshot_path}")
    state_dict = torch.load(model_snapshot_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    
    net.eval() 

    start = time.time()
    evaluate(net) 
    end = time.time()
    logger.info("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    logger.info("Testing process completed.")
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start))))) # Explicit print for total time
    sys.stdout.flush() # ADDED: Force flush stdout

if __name__ == '__main__':
    main()