#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, cv2, argparse
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

# --- Setup Project Path and Imports ---
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import config
from daseg import daseg
from datasets import ImageFolder
from misc import check_mkdir

# --- Helper Functions ---
def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalizes a tensor image with mean and standard deviation."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def apply_color_map(mask_np, color_map):
    """Converts a class index mask to an RGB image."""
    height, width = mask_np.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        colored_mask[mask_np == class_idx] = color
    return colored_mask

# --- Main Visualization Logic ---
def main():
    parser = argparse.ArgumentParser(description='Visualize DANet Results')
    parser.add_argument('--exp-name', type=str, required=True, help='e.g., DANet_TSRS_RSNA-Epiphysis')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset being visualized.')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to visualize.')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to save.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COLOR_MAP = np.array([[0, 0, 0], [255, 0, 0]]) # Black for BG, Red for class 1

    # --- Load Model ---
    model = daseg(config.backbone_path).to(device)
    checkpoint_path = os.path.join('./ckpt', args.exp_name, f"fold_{args.fold}", 'best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict([(k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items()])
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded successfully from {checkpoint_path}")

    # === Use the new robust data loading method ===
    dataset_cfg = config.DATASET_CONFIG[args.dataset]
    test_data_path = dataset_cfg['path']
    test_dataset = ImageFolder(root=test_data_path, dataset_name=args.dataset, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- Run Visualization ---
    output_dir = os.path.join('./ckpt', args.exp_name, f"fold_{args.fold}", "visualizations")
    check_mkdir(output_dir)
    print(f"Saving visualizations to: {output_dir}")

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc="Visualizing")):
            if i >= args.num_images: break
            if sample is None: continue

            image_tensor = sample['image'].to(device)
            label_tensor = sample['label']
            image_name = os.path.basename(sample['name'][0][0])
            
            # --- Inference ---
            *_, prediction_output = model(image_tensor)
            prediction_tensor = prediction_output.argmax(1).squeeze(0).cpu()

            # --- Prepare Images for Plotting ---
            original_img_tensor = denormalize(image_tensor.squeeze(0).cpu())
            original_img_np = np.transpose(original_img_tensor.numpy(), (1, 2, 0))
            original_img_np = np.clip(original_img_np * 255, 0, 255).astype(np.uint8)

            gt_mask_np = label_tensor.squeeze(0).numpy().astype(np.uint8)
            h, w, _ = original_img_np.shape
            gt_mask_resized = cv2.resize(gt_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_colored = apply_color_map(gt_mask_resized, COLOR_MAP)

            prediction_np = prediction_tensor.numpy().astype(np.uint8)
            pred_mask_resized = cv2.resize(prediction_np, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_colored = apply_color_map(pred_mask_resized, COLOR_MAP)
            
            overlay = cv2.addWeighted(original_img_np, 0.6, pred_colored, 0.4, 0)
            
            # --- Plotting ---
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes[0].imshow(original_img_np); axes[0].set_title(f"Original: {image_name}"); axes[0].axis('off')
            axes[1].imshow(gt_colored); axes[1].set_title("Ground Truth"); axes[1].axis('off')
            axes[2].imshow(pred_colored); axes[2].set_title("Prediction"); axes[2].axis('off')
            axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

if __name__ == '__main__':
    main()