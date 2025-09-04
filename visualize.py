# /kaggle/working/ARAA-Net/visualize.py
# !python visualize.py --image_index 25 --backbone resnet50

import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Add project path to run script from anywhere ---
import sys
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
from datasets import ImageFolder
from config import test_path # Assuming test_path points to the 'val' directory inside the dataset
from misc import check_mkdir

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone used for training')
    parser.add_argument('--image_index', type=int, default=15, help='Index of the validation image to test')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoint.pth', help='Name of the checkpoint file to use (e.g., best_checkpoint.pth or latest_checkpoint.pth)')
    args = parser.parse_args() # Use parse_args() here as all args are known

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # exp_name should match how it's saved in train.py (which is just args.backbone)
    exp_name = args.backbone 
    
    # --- 1. Load the Model ---
    model_dir = os.path.join('/kaggle/working/ckpt', exp_name) # Adjusted base path for checkpoints
    model_path = os.path.join(model_dir, args.ckpt_name)

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Checkpoint not found at {model_path}")
        print(f"Please ensure the checkpoint exists in {model_dir}")
        return

    net = daseg(backbone_name=args.backbone).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint formats (full dict or just state_dict)
    if 'model_state_dict' in state_dict:
        # If saved with 'model_state_dict', extract it
        loaded_state_dict = state_dict['model_state_dict']
    else:
        # Otherwise, assume state_dict is the model's state_dict directly
        loaded_state_dict = state_dict

    # Handle DataParallel prefix if present
    if list(loaded_state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in loaded_state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(loaded_state_dict)

    net.eval()
    print(f"✅ Model loaded from {model_path}")

    # --- 2. Load the Dataset and a Specific Image ---
    # ImageFolder's transform_val handles fixed resizing and normalization internally.
    # No need for dummy args object.
    test_set = ImageFolder(test_path, split='val')

    if args.image_index >= len(test_set) or args.image_index < 0:
        print(f"❌ ERROR: Image index {args.image_index} is out of bounds. Dataset has {len(test_set)} images (indices 0 to {len(test_set)-1}).")
        return

    sample = test_set[args.image_index]
    image_tensor = sample['image'].unsqueeze(0).to(device) # Add batch dimension
    label_tensor = sample['label']
    
    # image_name is a tuple (img_path, gt_path), extract base filename
    original_img_path = sample['name'][0]
    image_base_name = os.path.basename(original_img_path)

    print(f"✅ Visualizing image: {image_base_name} (index {args.image_index})")

    # --- 3. Run Inference ---
    with torch.no_grad():
        _, _, _, _, pred_logits = net(image_tensor)
    
    # Process prediction: apply argmax and move to CPU
    # Squeeze the batch dimension and convert to numpy
    prediction_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # --- 4. Prepare Images for Display ---
    # Un-normalize the original image tensor for viewing
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) # Clip values to [0, 1] for correct display

    ground_truth_mask = label_tensor.numpy().astype(np.uint8)

    # --- 5. Plot the Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(prediction_mask, cmap='gray')
    axes[2].set_title("Model's Prediction")
    axes[2].axis('off')

    # Save the figure
    output_dir = os.path.join('/kaggle/working/visual_results', exp_name) # Changed output dir for Kaggle persistence
    check_mkdir(output_dir)
    save_path = os.path.join(output_dir, f"result_index_{args.image_index}_{image_base_name}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()