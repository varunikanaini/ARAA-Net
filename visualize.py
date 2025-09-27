# /kaggle/working/ARAA-Net/visualize.py (FINAL & CORRECTED)
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
from config import DATA_ROOT, CKPT_ROOT # <<< FIXED: Import CKPT_ROOT from config
from misc import check_mkdir

def setup_logging_visualize(log_dir, filename='visualization.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def main():
    parser = argparse.ArgumentParser(description='Visualize LASA-Unet predictions')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 
                            'jsrt-247-image-lung-segmentation-mask-dataset', 
                            'covid19-radiography-database', 
                            'CVC-ClinicDB', 
                            'dental_panoramic_xrays', 
                            'Dataset' # For SixDiseasesChestXRay
                        ], help='Dataset used for training')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture used for training')
    parser.add_argument('--image-index', type=int, default=0, help='Index of the test image to visualize (0-indexed)') # Default to 0 for first image
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
    # Replaced long dataset names with shorter aliases for folder naming
    exp_name_dataset_alias = args.dataset_name.replace('TSRS_RSNA-', '').lower()
    exp_name_dataset_alias = exp_name_dataset_alias.replace('jsrt-247-image-lung-segmentation-mask-dataset', 'jsrt')
    exp_name_dataset_alias = exp_name_dataset_alias.replace('covid19-radiography-database', 'covid')
    exp_name_dataset_alias = exp_name_dataset_alias.replace('CVC-ClinicDB', 'cvc')
    exp_name_dataset_alias = exp_name_dataset_alias.replace('dental_panoramic_xrays', 'dental')
    exp_name_dataset_alias = exp_name_dataset_alias.replace('Dataset', 'sixdiseases') # For SixDiseasesChestXRay

    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{exp_name_dataset_alias}"
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

    # Load image from the 'test' split (or 'val' if no 'test' folder)
    dataset_base_path = os.path.join(DATA_ROOT, args.dataset_name)

    if args.dataset_name == 'TSRS_RSNA-Epiphysis':
        test_data_path = os.path.join(dataset_base_path, 'test')
        if not os.path.exists(test_data_path):
            logging.warning(f"Test data for {args.dataset_name} not found at '{test_data_path}'. Using 'val' split for visualization.")
            test_data_path = os.path.join(dataset_base_path, 'val')
            if not os.path.exists(test_data_path):
                logging.error(f"❌ ERROR: Neither 'test' nor 'val' data found for visualization for {args.dataset_name} at '{test_data_path}'.")
                sys.exit(1)
    else:
        # For new datasets, ImageFolder will handle internal structure and programmatic splits
        test_data_path = dataset_base_path # Pass the base path, ImageFolder does the split
            
    test_set = ImageFolder(test_data_path, args, split='test') # Use split='test' for transform consistency

    if args.image_index >= len(test_set) or args.image_index < 0:
        logging.error(f"❌ ERROR: Image index {args.image_index} is out of bounds. Dataset has {len(test_set)} images for {args.dataset_name}.")
        sys.exit(1)

    sample = test_set[args.image_index]
    if sample is None:
        logging.error(f"❌ ERROR: Sample at index {args.image_index} is corrupted or could not be loaded. Exiting.")
        sys.exit(1)

    image_tensor = sample['image'].unsqueeze(0).to(device)
    label_tensor = sample['label']
    
    # Ensure 'name' is retrieved correctly, it's a string
    image_name = sample['name']
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
    save_path = os.path.join(output_dir, f"visual_result_{image_name.replace('.png', '').replace('.jpg', '')}_index_{args.image_index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"✅ Visualization saved to {save_path}")
    plt.show() # Uncomment if you want to display plot in interactive environments

if __name__ == '__main__':
    main()