#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import time, datetime, os, argparse, logging, sys
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

# --- UNCHANGED IMPORTS ---
from config import backbone_path, DATASET_PATHS
from datasets import ImageFolder
from misc import check_mkdir
from daseg import daseg
from seg_utils import ConfusionMatrix

cudnn.benchmark = True
torch.manual_seed(2021)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# =========================================================================================
# --- NEW: `evaluate_fold` FUNCTION ---
# This function evaluates a single model on the test set.
# =========================================================================================
def evaluate_fold(net, test_loader, fold_idx):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    test_iterator = tqdm(test_loader, total=len(test_loader), desc=f"Testing Fold {fold_idx}")
    for data in test_iterator:
        if data is None: continue
        inputs, labels, _ = data['image'], data['label'], data['name']
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            *_, predict0 = net(inputs)
            pred_mask = predict0.argmax(1)
            confmat.update(labels.flatten(), pred_mask.flatten())
    
    global_acc, _, class_iou, FWIoU, mDice = confmat.compute()
    miou = np.mean(class_iou.cpu().numpy())
    logging.info(f"Fold {fold_idx} Results -> mIoU: {miou:.4f}, Dice: {mDice:.4f}, OA: {global_acc.item():.4f}")
    return {'miou': miou, 'dice': mDice, 'oa': global_acc.item(), 'fwiou': FWIoU.item()}

# =========================================================================================
# --- MAIN TESTING ORCHESTRATION ---
# =========================================================================================
def main():
    parser = argparse.ArgumentParser(description='DANet K-Fold Testing')
    parser.add_argument('--exp-name', type=str, required=True, help='e.g., DANet_TSRS_RSNA-Epiphysis')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--k-folds', type=int, required=True, help='The number of folds that were trained.')
    args = parser.parse_args()

    base_exp_path = os.path.join('./ckpt', args.exp_name)
    log_path = os.path.join(base_exp_path, 'final_test_log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info(f"Starting final testing for experiment: {args.exp_name}")
    
    # --- Load Test Dataset (used for all folds) ---
    test_root_path = DATASET_PATHS[f"{args.dataset}_test"]
    test_set = ImageFolder(test_root_path, f"{args.dataset}_test", split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
    
    all_metrics = {'miou': [], 'dice': [], 'oa': [], 'fwiou': []}
    
    for fold_idx in range(args.k_folds):
        logging.info("-" * 50)
        net = daseg(backbone_path).to(device)
        model_path = os.path.join(base_exp_path, f"fold_{fold_idx}", 'best.pth')
        
        if not os.path.exists(model_path):
            logging.warning(f"Checkpoint for fold {fold_idx} not found at {model_path}. Skipping.")
            continue
            
        logging.info(f"Loading model for Fold {fold_idx} from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict([(k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
        
        fold_metrics = evaluate_fold(net, test_loader, fold_idx)
        for key in all_metrics:
            all_metrics[key].append(fold_metrics[key])

    # --- Final Summary ---
    logging.info("\n" + "=" * 50)
    logging.info(f"Final K-Fold Test Summary ({args.k_folds} folds)")
    logging.info(f"Mean IoU (mIoU): {np.mean(all_metrics['miou']):.4f} ± {np.std(all_metrics['miou']):.4f}")
    logging.info(f"Dice Score:      {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
    logging.info(f"Overall Acc (OA):{np.mean(all_metrics['oa']):.4f} ± {np.std(all_metrics['oa']):.4f}")
    logging.info(f"FW-IoU:          {np.mean(all_metrics['fwiou']):.4f} ± {np.std(all_metrics['fwiou']):.4f}")
    logging.info("=" * 50)

if __name__ == '__main__':
    main()