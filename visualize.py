# /kaggle/working/araa/ARAA-Net/visualize.py
# !python visualize.py --image_index 25
import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Add project path to run script from anywhere ---
import sys
project_path = '/kaggle/working/araa/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
from datasets import ImageFolder
from config import test_path
from misc import check_mkdir

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone used for training')
    parser.add_argument('--image_index', type=int, default=15, help='Index of the validation image to test')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoint.pth', help='Name of the checkpoint file to use (e.g., best_checkpoint.pth or latest_checkpoint.pth)')
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = args.backbone + "_with_LASA"
    
    # --- 1. Load the Model ---
    model_path = os.path.join('/kaggle/working/araa/ARAA-Net/ckpt', exp_name, args.ckpt_name)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Checkpoint not found at {model_path}")
        return

    net = daseg(backbone_name=args.backbone).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint formats
    if 'model_state_dict' in state_dict:
        net.load_state_dict(state_dict['model_state_dict'])
    else:
        net.load_state_dict(state_dict)
    net.eval()
    print(f"✅ Model loaded from {model_path}")

    # --- 2. Load the Dataset and a Specific Image ---
    # We need a dummy args object for the dataset class
    dataset_args = argparse.Namespace(scale_h=896, scale_w=576)
    test_set = ImageFolder(test_path, args=dataset_args, split='val')

    if args.image_index >= len(test_set):
        print(f"❌ ERROR: Image index {args.image_index} is out of bounds. Dataset has {len(test_set)} images.")
        return

    sample = test_set[args.image_index]
    image_tensor = sample['image'].unsqueeze(0).to(device) # Add batch dimension
    label_tensor = sample['label']
    image_name = sample['name']
    print(f"✅ Visualizing image: {image_name} (index {args.image_index})")

    # --- 3. Run Inference ---
    with torch.no_grad():
        _, _, _, _, pred_logits = net(image_tensor)
    
    # Process prediction: apply argmax and move to CPU
    prediction_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # --- 4. Prepare Images for Display ---
    # Un-normalize the original image tensor for viewing
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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

    # Save the figure
    output_dir = os.path.join('/kaggle/working/araa/ARAA-Net/visual_results', exp_name)
    check_mkdir(output_dir)
    save_path = os.path.join(output_dir, f"result_index_{args.image_index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()