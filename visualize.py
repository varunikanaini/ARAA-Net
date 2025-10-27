# /path/to/your/project/visualize.py (New file)

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import cv2
import argparse
from tqdm import tqdm
from collections import OrderedDict

from torch.utils.data import DataLoader

# --- Setup Project Path ---
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
from config import backbone_path, DATASET_PATHS
from daseg import daseg
from datasets import ImageFolder
from misc import check_mkdir

# --- Helper Functions ---
def apply_color_map(mask_np, color_map):
    height, width = mask_np.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        colored_mask[mask_np == class_idx] = color
    return colored_mask

# --- Main Visualization Logic ---
def main():
    parser = argparse.ArgumentParser(description='Visualize DANet Results')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment (e.g., DANet_TSRS_RSNA-Epiphysis)')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0, help='Fold number to visualize.')
    parser.add_argument('--num-images', type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Color Map Definition ---
    COLOR_MAP = np.array([[0, 0, 0], [255, 0, 0]]) # Black for BG, Red for class 1

    # --- Load Model ---
    model = daseg(backbone_path).to(device)
    checkpoint_path = os.path.join('./ckpt', args.exp_name, f"fold_{args.fold}", 'best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict([(k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items()])
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded successfully from {checkpoint_path}")

    # --- Load Dataset ---
    test_data_path = DATASET_PATHS[f"{args.dataset}_test"]
    test_dataset = ImageFolder(root=test_data_path, dataset_name=f"{args.dataset}_test", split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- Run Visualization ---
    output_dir = os.path.join('./ckpt', args.exp_name, f"fold_{args.fold}", "visualizations")
    check_mkdir(output_dir)
    print(f"Saving visualizations to: {output_dir}")
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc="Visualizing")):
            if i >= args.num_images: break
            if sample is None: continue

            image_pil = sample['image'] # Assuming ToTensor is in the dataset transform
            label_np = sample['label'].squeeze(0).numpy().astype(np.uint8)
            image_name = os.path.basename(sample['name'][0][0])
            
            # --- Recreate input tensor from PIL for model ---
            # This is to get the original image before normalization for display
            display_img_np = np.array(image_pil) 
            
            # Re-apply the validation transform for model input
            from torchvision import transforms
            import custom_transforms as tr
            val_transform = transforms.Compose([
                tr.FixedResize(576, 896),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
            transformed_sample = val_transform({'image': image_pil, 'label': Image.fromarray(label_np)})
            image_tensor = transformed_sample['image'].unsqueeze(0).to(device)

            # --- Inference and Post-processing ---
            *_, prediction_output = model(image_tensor)
            prediction = prediction_output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Resize prediction to match original label size for comparison if needed
            h, w = label_np.shape
            prediction_resized = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)

            gt_colored = apply_color_map(label_np, COLOR_MAP)
            pred_colored = apply_color_map(prediction_resized, COLOR_MAP)
            overlay = cv2.addWeighted(display_img_np, 0.6, pred_colored, 0.4, 0)

            # --- Plotting ---
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes[0].imshow(display_img_np); axes[0].set_title(f"Original: {image_name}"); axes[0].axis('off')
            axes[1].imshow(gt_colored); axes[1].set_title("Ground Truth"); axes[1].axis('off')
            axes[2].imshow(pred_colored); axes[2].set_title("Prediction"); axes[2].axis('off')
            axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.png"), dpi=150)
            plt.close(fig)

if __name__ == '__main__':
    main()