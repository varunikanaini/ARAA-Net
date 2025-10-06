# test_unet.py (or test_lasa_vgg.py)

import torch
import argparse
import time
import os
import sys
import logging
import datetime

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
# Make sure this import path is correct based on your file structure
from lasa_unet_model import LASA_Unet 
from misc import check_mkdir, AvgMeter
from config import CKPT_ROOT, DATASET_PATHS, DATA_ROOT
from datasets import ImageFolder
from seg_utils import ConfusionMatrix

# Import loss functions
from train_unet import FocalLoss, DiceLoss 

# --- CRITICAL: Ensure DataLoader is imported ---
from torch.utils.data import DataLoader 
# ------------------------------------------------

# --- Argument Parsing ---
def get_test_args():
    parser = argparse.ArgumentParser(description='Test LASA-Unet Model')
    
    # --- Dataset Arguments ---
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Dataset used for testing')
    
    # --- Backbone Arguments ---
    parser.add_argument('--backbone', type=str, default='vgg16', 
                        choices=['vgg16', 'resnet50', 'inception_v3', 'efficientnet_b0', 'efficientnet_b3'], 
                        help='Backbone architecture used for testing')
    
    # --- LASA Kernel Arguments ---
    parser.add_argument('--lasa-kernels', nargs='+', type=int, default=[1, 3, 5, 7], 
                        help='List of kernel sizes for LASA module (M=4 groups). Example: --lasa-kernels 1 3 5 7')

    # --- Preprocessing / Input Arguments ---
    parser.add_argument('--scale-h', type=int, default=896, help='Nominal height for resizing (internal logic overrides for DASEG alignment)')
    parser.add_argument('--scale-w', type=int, default=576, help='Nominal width for resizing (internal logic overrides for DASEG alignment)')
    
    # --- Loss arguments (optional for testing, but good to keep for consistency) ---
    parser.add_argument('--focal-alpha', type=float, default=0.5, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, help='Weight for Focal Loss component in combined loss.')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='Weight for Dice Loss component in combined loss.')
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses.')

    # --- Data Augmentation parameters (passed to ImageFolder, needed for constructor) ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, help='Dummy arg for ImageFolder.')
    parser.add_argument('--expansion-factor', type=float, default=1.5, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-h', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--min-bbox-w', type=int, default=32, help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-type', type=str, default='haar', help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-level', type=int, default=1, help='Dummy arg for ImageFolder.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, help='Dummy arg for ImageFolder.')

    # --- Control Flow ---
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.') # This is kept for consistency, though the script IS the test script.

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values. Got {len(args.deep_supervision_weights)}")
    
    return args

# --- Logging Setup ---
def setup_logging_test(log_dir, filename='testing_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, deep_supervision_weights, focal_loss_weight, dice_loss_weight, mode="Testing"):
    """
    Evaluates the model on a given data_loader. Returns mIoU and average loss.
    Also logs OA, mIoU, FWIoU, and Dice metrics to the console.
    """
    net.eval()
    confmat = ConfusionMatrix(num_classes=2) 
    loss_recorder = AvgMeter()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"{mode}", leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs = net(inputs) 
            final_pred = outputs[-1] 
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (focal_loss_weight * current_focal_loss) + \
                                         (dice_loss_weight * current_dice_loss)
                
                total_loss += deep_supervision_weights[i] * combined_loss_per_head
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), final_pred.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    # --- Log all metrics to the console ---
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA (Overall Accuracy): {global_acc.item():.4f}")
    logging.info(f"  mIoU (Mean IoU): {mIoU:.4f}")
    logging.info(f"  FWIoU (Frequency Weighted IoU): {fwiou.item():.4f}")
    logging.info(f"  Dice (Mean Dice Coefficient): {mDice:.4f}")
    # ------------------------------------
    
    if mode == "Validating": 
        net.train() 
    return mIoU

# --- Custom Collate Function ---
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] 
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    args = get_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels))
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_Kernels{lasa_kernels_str}_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    
    # Define exp_path correctly
    exp_path = os.path.join(CKPT_ROOT, exp_name) 
    check_mkdir(exp_path)
    setup_logging_test(log_dir=exp_path, filename=f'testing_{args.backbone}_{args.dataset_name}.log')

    logging.info(f"Starting testing for experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")

    base_dataset_path = DATASET_PATHS[args.dataset_name]

    # Instantiate the model
    # Corrected: Pass lasa_kernels to the model and remove 'pretrained=True'
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
    
    # Instantiate loss functions
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- Load the BEST trained model ---
    logging.info("Attempting to load the best checkpoint.")
    best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    if not os.path.exists(best_checkpoint_path):
        logging.error(f"❌ ERROR: 'best_checkpoint.pth' not found in '{exp_path}'. Please ensure training was completed and the checkpoint was saved.")
        sys.exit(1)

    try:
        net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        logging.info(f"✅ Model loaded successfully from {best_checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
        sys.exit(1)

    # --- Prepare Test Dataset and DataLoader ---
    if 'TSRS_RSNA' in args.dataset_name:
        test_data_path = os.path.join(base_dataset_path, 'test') 
    else:
        test_data_path = base_dataset_path
    
    # Initialize test dataset and dataloader
    test_set = ImageFolder(test_data_path, args.dataset_name, args, split='test') 
    # Corrected: DataLoader is now imported, so this should work.
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    logging.info(f"Loaded {len(test_set)} images for testing from dataset '{args.dataset_name}' split '{args.split}'.")

    # --- Run Evaluation ---
    # The '--test-only' argument is implicitly handled by running this script directly.
    # If you were using a combined train/test script, you'd check args.test_only here.
    test_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                               args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
    
    # --- Log Final Results ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"\n\n--- FINAL TEST RESULTS ({timestamp}) ---")
    logging.info(f"Model: LASA-Unet with {args.backbone} backbone and LASA Kernels: {args.lasa_kernels}")
    logging.info(f"Dataset: {args.dataset_name} (evaluated on 'test' split)")
    logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
    logging.info("---------------------------------")
    logging.info("✅ Testing completed.")

if __name__ == '__main__':
    main()