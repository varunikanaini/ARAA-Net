# /kaggle/working/ARAA-Net/visualize.py
# Example usage: !python visualize.py --image_index 25 --backbone resnet50 --split test --dataset-name TSRS_RSNA-Epiphysis

import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 


# --- Add project path to run script from anywhere ---
import sys
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
from datasets import ImageFolder
from config import DATA_ROOT, CKPT_ROOT # Import CKPT_ROOT
from misc import check_mkdir

# Define default normalization parameters (must match training/validation)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Backbone used for training')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Name of the dataset to use') # Added dataset-name arg
    parser.add_argument('--image_index', type=int, default=15, help='Index of the validation/test image to visualize')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoint.pth', help='Name of the checkpoint file to use (e.g., best_checkpoint.pth or latest_checkpoint.pth)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which dataset split to visualize from') # Changed default to 'test'
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to for visualization')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to for visualization')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # New experiment name format: backbone_name_ULD_datasetname
    exp_name = f"{args.backbone}_ULD_{args.dataset_name}" 
    
    # --- 1. Load the Model ---
    model_dir = os.path.join(CKPT_ROOT, exp_name) 
    model_path = os.path.join(model_dir, args.ckpt_name)

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Checkpoint not found at {model_path}")
        print(f"Please ensure the checkpoint exists in {model_dir}")
        return

    net = daseg(backbone_name=args.backbone).to(device)
    state_dict_or_ckpt = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in state_dict_or_ckpt:
        loaded_state_dict = state_dict_or_ckpt['model_state_dict']
    else:
        loaded_state_dict = state_dict_or_ckpt

    if list(loaded_state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in loaded_state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(loaded_state_dict)

    net.eval()
    print(f"✅ Model loaded from {model_path}")

    # --- 2. Load the Dataset and a Specific Image ---
    # Adjust input size for inception_v3 if not explicitly overridden by args
    scale_h_val = args.scale_h
    scale_w_val = args.scale_w
    if args.backbone == 'inception_v3' and (scale_h_val == 896 or scale_w_val == 576):
        print("InceptionV3 detected, overriding scale to 299x299 for visualization.")
        scale_h_val = 299
        scale_w_val = 299

    # Construct the correct data path for the specified split and dataset
    dataset_base_path = os.path.join(DATA_ROOT, args.dataset_name, args.split)
    
    if not os.path.exists(dataset_base_path):
        print(f"❌ ERROR: Dataset split '{args.split}' for '{args.dataset_name}' not found at '{dataset_base_path}'.")
        print("Please ensure the specified split folder exists.")
        return

    test_set = ImageFolder(
        dataset_base_path,
        split=args.split, # Uses the split specified in argparse
        scale_h=scale_h_val, # Use dynamically determined scale
        scale_w=scale_w_val  # Use dynamically determined scale
    )

    if args.image_index >= len(test_set) or args.image_index < 0:
        print(f"❌ ERROR: Image index {args.image_index} is out of bounds. Dataset has {len(test_set)} images (indices 0 to {len(test_set)-1}).")
        return

    sample = test_set[args.image_index]
    image_tensor = sample['image'].unsqueeze(0).to(device) # Add batch dimension
    label_tensor = sample['label']
    
    original_img_path = sample['name'][0]
    image_base_name = os.path.basename(original_img_path)

    print(f"✅ Visualizing image: {image_base_name} (index {args.image_index}) from '{args.split}' split of '{args.dataset_name}'")

    # --- 3. Run Inference ---
    with torch.no_grad():
        _, _, _, _, pred_logits = net(image_tensor)
    
    prediction_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # --- 4. Prepare Images for Display ---
    # Un-normalize the original image tensor for viewing
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) 

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

    fig.suptitle(f"Prediction for {image_base_name} (Index {args.image_index}, Split: {args.split}, Dataset: {args.dataset_name})", fontsize=16)

    output_dir = os.path.join('/kaggle/working/visual_results', exp_name) 
    check_mkdir(output_dir)
    save_path = os.path.join(output_dir, f"{args.split}_index_{args.image_index}_{image_base_name}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()