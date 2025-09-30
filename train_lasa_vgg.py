# /kaggle/working/ARAA-Net/train_lasa_vgg.py (Updated for LASA tuning, loss tuning, and gradient accumulation)
import os
import time
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Standalone Model and Utilities ---
from lasa_vgg_model import LASA_Unet # <<< IMPORT THE UPDATED MODEL
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS 
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir, AvgMeter 

# Import loss functions for consistent loss calculation
from train_lasa_vgg import FocalLoss, DiceLoss 


def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', 
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface', 
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB', 
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1, help='Initial batch size per GPU. Will be adjusted by gradient accumulation.')
    parser.add_argument('--grad-accum-steps', type=int, default=4, help='Number of batches to accumulate gradients over. Effective batch size = batch_size * grad_accum_steps.')
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--weight-decay', type=float, default=0.0005) 
    parser.add_argument('--patience', type=int, default=20) 
    parser.add_argument('--num-workers', type=int, default=2) 
    parser.add_argument('--scale-h', type=int, default=896, help='Height images were nominally resized to (internal logic overrides for DASEG alignment)')
    parser.add_argument('--scale-w', type=int, default=576, help='Width images were nominally resized to (internal logic overrides for DASEG alignment)')
    
    # --- LASA Module Tuning Arguments ---
    parser.add_argument('--lasa-M', type=int, default=4, help='Number of groups (M) for LASA module. Default: 4.')
    parser.add_argument('--lasa-L', nargs='+', type=int, default=[5, 7, 9, 11], 
                        help='List of square side lengths (L) for LASA module. Default: [5, 7, 9, 11].')

    # --- Deep Supervision Weights ---
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1/final) output. Must have 5 values.')
    
    # --- Loss Function Parameters ---
    parser.add_argument('--focal-alpha', type=float, default=0.5, 
                        help='Alpha parameter for Focal Loss. Balance between positive/negative examples.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Gamma parameter for Focal Loss. Focus on hard examples.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, 
                        help='Weight for Focal Loss component in combined loss.')

    parser.add_argument('--dice-loss-weight', type=float, default=1.0, 
                        help='Weight for Dice Loss component in combined loss.')
    
    # --- Other potential arguments (kept from original for completeness) ---
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Minimum lesion area in pixels to trigger CenterAmplification (STS-Net default 576)')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Factor by which to expand the bounding box during CenterAmplification')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Minimum height of the expanded bounding box in pixels for CenterAmplification')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Minimum width of the expanded bounding box in pixels for CenterAmplification')

    parser.add_argument('--wavelet-type', type=str, default='haar', 
                        help='Wavelet type for DWT-based contrast enhancement (e.g., haar, db1, db2).')
    parser.add_argument('--wavelet-level', type=int, default=1, 
                        help='Decomposition level for DWT-based contrast enhancement.')
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5, 
                        help='Scaling factor for detail coefficients in wavelet enhancement.')

    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')

    parser.add_argument('--scheduler-patience', type=int, default=5) 
    parser.add_argument('--scheduler-factor', type=float, default=0.5) 
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6) 

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    
    # --- Argument Validation ---
    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")
    
    if args.lasa_M <= 0:
        parser.error("LASA module parameter M must be a positive integer.")
    
    if not args.lasa_L or any(l <= 0 or l % 2 == 0 for l in args.lasa_L):
        parser.error("LASA module parameter L must be a non-empty list of positive odd integers.")

    if len(args.lasa_L) != args.lasa_M:
        parser.error(f"Number of LASA side lengths (L={len(args.lasa_L)}) must match the number of groups (M={args.lasa_M}).")

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
    
    focal_loss_fn.to(device)
    dice_loss_fn.to(device)

    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None: 
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = net(inputs) 
            final_pred = outputs[-1] 
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                pred_output = pred_output.to(labels.device) # Ensure same device

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

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    # Construct experiment name including LASA parameters for better tracking
    exp_name_parts = [
        args.backbone,
        f"LASA_M{args.lasa_M}_L{'_'.join(map(str, args.lasa_L))}",
        "FocalDice",
        "DS",
        "WaveletHE"
    ]
    dataset_name_clean = args.dataset_name.replace('TSRS_RSNA-', '').lower()
    exp_name = "_".join(exp_name_parts) + f"_{dataset_name_clean}"
    
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training for experiment '{exp_name}' with arguments: {args}")

    try:
        base_dataset_path = DATASET_PATHS[args.dataset_name]
    except KeyError:
        logging.error(f"Dataset '{args.dataset_name}' not found in DATASET_PATHS configuration.")
        sys.exit(1)

    # --- Initialize Model ---
    # Pass LASA parameters to the model constructor
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_M=args.lasa_M, lasa_L=args.lasa_L).to(device)
    
    # --- Initialize Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device) 

    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"Best checkpoint not found at {best_checkpoint_path}. Please run training first for this experiment configuration.")
            sys.exit(1)

        try:
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"Loaded model from {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
            sys.exit(1)

        if 'TSRS_RSNA' in args.dataset_name:
            test_data_path = os.path.join(base_dataset_path, 'val') 
        else:
            test_data_path = base_dataset_path
        
        # Use 'test' split as requested by user. ImageFolder in datasets.py will handle it.
        test_set_for_eval = ImageFolder(test_data_path, args.dataset_name, args, split='test') 
        test_loader_for_eval = DataLoader(test_set_for_eval, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

        if not test_set_for_eval:
            logging.error(f"Test dataset is empty for '{args.dataset_name}'. Check dataset path and split configuration.")
            sys.exit(1)

        test_mIoU = evaluate_model(net, test_loader_for_eval, device, focal_loss_fn, dice_loss_fn, 
                                   args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        return 

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    
    # --- Resume Training Logic ---
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
            start_epoch, best_mIoU, patience_counter = 0, 0.0, 0

    # --- Data Loading ---
    if 'TSRS_RSNA' in args.dataset_name:
        train_data_path = os.path.join(base_dataset_path, 'train')
        val_data_path = os.path.join(base_dataset_path, 'val') 
    else:
        train_data_path = base_dataset_path
        val_data_path = base_dataset_path 

    train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train') 
    # Calculate effective batch size for dataloader
    effective_batch_size = args.batch_size * args.grad_accum_steps
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    
    val_set = ImageFolder(val_data_path, args.dataset_name, args, split='val') 
    test_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    if not train_set:
        logging.error(f"Training dataset is empty for '{args.dataset_name}'. Check dataset path and split configuration.")
        sys.exit(1)
    if not val_set:
        logging.error(f"Validation dataset is empty for '{args.dataset_name}'. Check dataset path and split configuration.")
        sys.exit(1)

    logging.info(f"Found {len(train_set)} training images for dataset '{args.dataset_name}'.")
    logging.info(f"Found {len(val_set)} validation images for dataset '{args.dataset_name}'.")
    logging.info(f"Using batch size: {args.batch_size}, Grad Accumulation Steps: {args.grad_accum_steps}, Effective Batch Size: {effective_batch_size}")


    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        optimizer.zero_grad(set_to_none=True) # Initialize zero gradients for the first batch
        
        for i, data in enumerate(train_iterator):
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Batch {i}: Skipping empty training batch due to corrupted/missing samples.")
                continue

            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            
            outputs = net(inputs)
            
            total_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                pred_output = pred_output.to(labels.device) # Ensure same device

                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[head_idx] * combined_loss_per_head
            
            # --- Gradient Accumulation ---
            # Scale loss by gradient accumulation steps
            scaled_loss = total_loss / args.grad_accum_steps
            scaled_loss.backward() # Backpropagate the scaled loss

            # Check if it's time to update weights
            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.step() # Update weights
                optimizer.zero_grad(set_to_none=True) # Reset gradients for the next accumulation cycle
            
            # Update loss recorder with the *original* total loss for the batch
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg, lr=optimizer.param_groups[0]['lr'])
            
        # --- Validation Step ---
        # Evaluate on the validation set
        current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        # --- Scheduler Step ---
        scheduler.step(current_mIoU)

        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save latest checkpoint (for resuming training)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        # --- Early Stopping ---
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered due to no improvement.")
            break

    logging.info(f"Training finished. Best mIoU achieved: {best_mIoU:.4f}")


if __name__ == '__main__':
    main()