# /kaggle/working/ARAA-Net/train_lasa_vgg.py (MODIFIED for robust dataset loading and programmatic splitting)
import os
import time
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split # <<< MODIFIED: Import random_split
from tqdm import tqdm
import torch.nn.functional as F

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet 
from config import DATA_ROOT, CKPT_ROOT, download_and_extract_kaggle_dataset, KAGGLE_DATASET_MAPPING
from datasets import ImageFolder, DATASET_CONFIGS # <<< MODIFIED: Import DATASET_CONFIGS, make_dataset
from datasets import make_dataset as make_full_dataset_list # <<< NEW: Rename to avoid conflict with ImageFolder's internal make_dataset (if any)
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# ... (FocalLoss and DiceLoss classes are unchanged) ...


def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=list(DATASET_CONFIGS.keys()), help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=448, help='Resize height for input images')
    parser.add_argument('--scale-w', type=int, default=448, help='Resize width for input images')
    
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    
    parser.add_argument('--focal-alpha', type=float, default=0.5, 
                        help='Alpha parameter for Focal Loss. Balance between positive/negative examples.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Gamma parameter for Focal Loss. Focus on hard examples.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, 
                        help='Weight for Focal Loss component in combined loss.')

    parser.add_argument('--dice-loss-weight', type=float, default=1.0, 
                        help='Weight for Dice Loss component in combined loss.')
    
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Minimum lesion area in pixels to trigger CenterAmplification (STS-Net default 576)')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Factor by which to expand the bounding box during CenterAmplification')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Minimum height of the expanded bounding box in pixels for CenterAmplification')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Minimum width of the expanded bounding box in pixels for CenterAmplification')

    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')

    parser.add_argument('--scheduler-patience', type=int, default=5, 
                        help='Number of epochs with no improvement after which learning rate will be reduced.')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, 
                        help='Factor by which the learning rate will be reduced.')
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6, 
                        help='Minimum learning rate.')
    
    # New args for programmatic splitting ratios
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio for datasets without predefined splits.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio for datasets without predefined splits.')


    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, deep_supervision_weights, focal_loss_weight, dice_loss_weight, mode="Validating"):
    """
    Evaluates the model on a given data_loader.
    """
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
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
            
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- {mode} mIoU: {mIoU:.4f} | {mode} Loss: {loss_recorder.avg:.4f} ---")
    if mode == "Validating": 
        net.train()
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}" 
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting operation for '{exp_name}' with arguments: {args}")

    # --- Determine the base root for the dataset (download if KaggleHub) ---
    dataset_info = KAGGLE_DATASET_MAPPING.get(args.dataset_name)
    base_dataset_root = None

    if dataset_info and dataset_info['id']: # If it's a KaggleHub dataset
        base_dataset_root = download_and_extract_kaggle_dataset(dataset_info['id'], DATA_ROOT)
        if not base_dataset_root:
            logging.error(f"Failed to prepare dataset '{args.dataset_name}'. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (KaggleHub): {base_dataset_root}")
    elif dataset_info and dataset_info['local_dir_name']: # For local datasets like KOA
        base_dataset_root = os.path.join(DATA_ROOT, dataset_info['local_dir_name'])
        if not os.path.exists(base_dataset_root):
            logging.error(f"Local dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (local): {base_dataset_root}")
    else: # For TSRS_RSNA datasets (default)
        base_dataset_root = os.path.join(DATA_ROOT, args.dataset_name)
        if not os.path.exists(base_dataset_root):
            logging.error(f"TSRS_RSNA dataset directory not found at '{base_dataset_root}'. Please place it there. Exiting.")
            sys.exit(1)
        logging.info(f"Base dataset root (TSRS_RSNA default): {base_dataset_root}")
    # --- END dataset root determination ---

    # --- Data Loading and Splitting Logic ---
    dataset_config = DATASET_CONFIGS.get(args.dataset_name)
    if not dataset_config: # Should not happen due to argparse choices, but defensive
        logging.error(f"Config for dataset '{args.dataset_name}' not found. Exiting.")
        sys.exit(1)

    full_image_mask_list = []
    train_image_mask_list = []
    val_image_mask_list = []
    test_image_mask_list = [] # For consistency, if we had a fixed test set for programmatic splits

    if dataset_config['has_predefined_splits']:
        # Datasets with pre-defined 'train', 'val', 'test' folders (e.g., RSNA, KOA)
        logging.info(f"Using predefined splits for dataset '{args.dataset_name}'.")
        train_path = os.path.join(base_dataset_root, 'train')
        val_path = os.path.join(base_dataset_root, 'val')
        test_path_defined = os.path.join(base_dataset_root, 'test') # If a 'test' folder exists

        train_image_mask_list = make_full_dataset_list(train_path, args.dataset_name, split_name='train')
        val_image_mask_list = make_full_dataset_list(val_path, args.dataset_name, split_name='val')
        # If test path exists and is not empty, use it. Otherwise, val_set might also be the test_set
        if os.path.exists(test_path_defined) and os.listdir(test_path_defined):
             test_image_mask_list = make_full_dataset_list(test_path_defined, args.dataset_name, split_name='test')
        else:
             logging.warning(f"No explicit 'test' split folder found for '{args.dataset_name}'. Validation set will be used for final testing if --test-only is used without specific test_path.")

    else:
        # Datasets requiring programmatic splitting (e.g., COVID-19_Radiography, JSRT)
        logging.info(f"Performing programmatic splitting for dataset '{args.dataset_name}'.")
        full_image_mask_list = make_full_dataset_list(base_dataset_root, args.dataset_name, split_name='all')
        
        if not full_image_mask_list:
            logging.error(f"No data found for programmatic splitting in '{base_dataset_root}'. Exiting.")
            sys.exit(1)

        # Calculate lengths for train, val, test splits
        total_len = len(full_image_mask_list)
        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        test_len = total_len - train_len - val_len # Remaining for test

        # Perform random split
        # Use a generator for reproducibility if desired, for now let random_split use its default rng
        train_image_mask_list, val_image_mask_list, test_image_mask_list = random_split(
            full_image_mask_list, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
        
        logging.info(f"Programmatic split: Total {total_len}, Train {len(train_image_mask_list)}, Val {len(val_image_mask_list)}, Test {len(test_image_mask_list)}")

    # --- Instantiate DataLoaders ---
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(device)
    
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"Best checkpoint not found at {best_checkpoint_path}. Please train a model first or specify correct path.")
            sys.exit(1)

        try:
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"Loaded model from {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}")
            sys.exit(1)

        # For test-only, use the test_image_mask_list for evaluation
        test_set_for_eval = ImageFolder(test_image_mask_list, args, split='test') # <<< MODIFIED: Pass image_mask_list
        test_loader_for_eval = DataLoader(test_set_for_eval, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

        test_mIoU = evaluate_model(net, test_loader_for_eval, device, focal_loss_fn, dice_loss_fn, 
                                   args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        return 

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    
    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")

    train_set = ImageFolder(train_image_mask_list, args, split='train') # <<< MODIFIED: Pass image_mask_list
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_set = ImageFolder(val_image_mask_list, args, split='val') # <<< MODIFIED: Pass image_mask_list
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True) 


    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data in train_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            outputs = net(inputs)
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        scheduler.step(current_mIoU)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered due to no improvement.")
            break

if __name__ == '__main__':
    main()