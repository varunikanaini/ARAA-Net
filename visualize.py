import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import logging
from torchvision import transforms

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from lasa_vgg_model import LASA_Unet # Use generalized LASA_Unet
from datasets import ImageFolder
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS # Updated: Import DATASET_PATHS
from misc import check_mkdir

def setup_logging_visualize(log_dir, filename='visualization.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Custom collate function to filter out None samples (from `datasets.py` returning None for corrupted images)
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    parser = argparse.ArgumentParser(description='Visualize LASA-Unet predictions')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=['TSRS_RSNA-Epiphysis', 'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 'DentalPanoramic', 'SixDiseasesChestXRay'], # Updated choices
                        help='Dataset used for training')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture used for training')
    parser.add_argument('--image-index', type=int, default=0, help='Index of the test image to visualize (0-indexed)') 
    parser.add_argument('--scale-h', type=int, default=448, help='Height images were resized to')
    parser.add_argument('--scale-w', type=int, default=448, help='Width images were resized to')
    
    # Dummy args for ImageFolder to instantiate correctly (these are training-only but must be parsed)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')

    # Wavelet Preprocessing args (needed for ImageFolder to instantiate correctly)
    parser.add_argument('--wavelet-type', type=str, default='haar', help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-level', type=int, default=1, help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct the experiment name to find the checkpoint
    # This MUST match the naming convention used in train_lasa_vgg.py
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    output_dir = os.path.join(CKPT_ROOT, 'visual_results', exp_name)
    check_mkdir(output_dir)
    setup_logging_visualize(output_dir) # Setup logging for this specific visualization run

    logging.info(f"Starting visualization for experiment '{exp_name}'")
    logging.info(f"Arguments: {args}")

    model_path = os.path.join(CKPT_ROOT, exp_name, 'best_checkpoint.pth')
    if not os.path.exists(model_path):
        logging.error(f"❌ ERROR: Checkpoint not found at {model_path}. Please train a model first.")
        sys.exit(1)

    # Instantiate the model with the correct backbone
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    logging.info(f"✅ Model loaded from {model_path}")

    # Updated: Get dataset_path from DATASET_PATHS
    dataset_root_for_current_run = DATASET_PATHS[args.dataset_name]
            
    # Updated: ImageFolder now takes dataset_name
    test_set = ImageFolder(dataset_root_for_current_run, args.dataset_name, args, split='test') 

    if len(test_set) == 0:
        logging.error(f"❌ ERROR: No images found for dataset '{args.dataset_name}' with 'test' split. Cannot visualize.")
        sys.exit(1)

    if args.image_index >= len(test_set) or args.image_index < 0:
        logging.error(f"❌ ERROR: Image index {args.image_index} is out of bounds. Dataset has {len(test_set)} images. Defaulting to index 0.")
        args.image_index = 0 # Default to 0 if out of bounds
        
    sample = test_set[args.image_index]
    
    if sample is None: # Handle case where ImageFolder returns None for a corrupted image
        logging.error(f"❌ ERROR: Failed to load image at index {args.image_index} for visualization. It might be corrupted.")
        sys.exit(1)

    image_tensor = sample['image'].unsqueeze(0).to(device)
    label_tensor = sample['label']
    
    image_name = sample['name'] if 'name' in sample else f"image_idx_{args.image_index}"
    logging.info(f"✅ Visualizing image: {image_name} (index {args.image_index})")

    with torch.no_grad():
        outputs = net(image_tensor)
        pred_logits = outputs[-1] # Get the final prediction from deep supervision outputs
    
    prediction_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # Denormalize image for display
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    ground_truth_mask = label_tensor.numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np); axes[0].set_title(f'Original Image ({image_name})'); axes[0].axis('off')
    axes[1].imshow(ground_truth_mask, cmap='gray'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
    axes[2].imshow(prediction_mask, cmap='gray'); axes[2].set_title("Model's Prediction"); axes[2].axis('off')

    # Save path includes experiment name and image identifier
    save_path = os.path.join(output_dir, f"visual_result_{image_name}_index_{args.image_index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"✅ Visualization saved to {save_path}")
    # plt.show() # Uncomment if you want to display plot in interactive environments

if __name__ == '__main__':
    main()