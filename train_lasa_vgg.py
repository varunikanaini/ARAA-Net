# train_lasa_vgg.py
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
from lasa_vgg_model import LASA_Unet
from config import DATA_ROOT, CKPT_ROOT, DATASET_PATHS
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# ===================================================================
#      FOCAL LOSS CLASS
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Ensure targets are long type for cross_entropy
        targets = targets.long()
        
        # If ignore_index is provided, we need to handle it.
        # For binary segmentation with class 0 as ignore, we can adapt.
        # Here, assuming ignore_index is for background class, and we are calculating loss for foreground.
        # For simplicity, we rely on `reduction='none'` and manual masking.

        # Apply cross_entropy per pixel, then apply weights
        # `F.cross_entropy` expects logits (raw scores)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss) # Probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            # Apply mask only if ignore_index was used
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index).float()
                return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ===================================================================

# ===================================================================
#      DICE LOSS CLASS
# ===================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are logits, targets are class indices
        num_classes = inputs.shape[1]
        
        # For binary segmentation, we typically use sigmoid on the output for class 1
        # and then calculate Dice based on the predicted probability of class 1 vs. ground truth class 1.
        if num_classes == 2:
            # Get predicted probabilities for class 1 (foreground)
            pred_probs = torch.sigmoid(inputs[:, 1, :, :]).unsqueeze(1) # Shape: (B, 1, H, W)
            # Create one-hot encoded ground truth for class 1
            true_oh = (targets == 1).float().unsqueeze(1) # Shape: (B, 1, H, W)
        else: # For multi-class, though typically binary for DiceLoss in this context
            pred_probs = F.softmax(inputs, dim=1)
            true_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float() # Convert to (B, C, H, W)

        # Apply ignore_index mask if provided
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float().unsqueeze(1) # Shape: (B, 1, H, W)
            pred_probs = pred_probs * mask
            true_oh = true_oh * mask
        
        # Flatten for easier calculation
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice_score
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            # Note: Summing loss might be tricky if batch size varies significantly or if ignore_index is used heavily.
            # Mean is generally more robust.
            return loss * inputs.shape[0] # Approximate scaling by batch size
        else:
            return loss
# ===================================================================


def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis',
                        choices=[
                            'TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface',
                            'JSRT', 'COVID19_Radiography', 'CVC-ClinicDB',
                            'DentalPanoramic', 'SixDiseasesChestXRay'
                        ], help='Name of the dataset')
    # --- MODIFIED ---
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['vgg16', 'resnet50', 'inception_v3', 'efficientnet_b0', 'efficientnet_b3'], # Added new choices
                        help='Backbone architecture to use')
    # --- END MODIFIED ---
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=896, help='Height images were nominally resized to (internal logic overrides for DASEG alignment)')
    parser.add_argument('--scale-w', type=int, default=576, help='Width images were nominally resized to (internal logic overrides for DASEG alignment)')

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

    # Augmentation parameters passed to ImageFolder
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

    if len(args.deep_supervision_weights) != 5:
        parser.error(f"deep-supervision-weights must have 5 values for the 5 outputs. Got {len(args.deep_supervision_weights)}")

    return args


def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    # Clear existing handlers
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
            if data is None:
                logging.warning(f"Skipping empty {mode} batch due to corrupted/missing samples.")
                continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            # Ensure inputs and labels are on the same device and have correct dtype for loss functions
            if inputs.dtype != torch.float32: inputs = inputs.float()
            if labels.dtype != torch.long: labels = labels.long() # Labels should be long for classification/CE loss

            outputs = net(inputs)
            final_pred = outputs[-1]

            total_loss = 0
            for i, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels)
                current_dice_loss = dice_loss_fn(pred_output, labels)

                combined_loss_per_head = (focal_loss_weight * current_focal_loss) + \
                                         (dice_loss_weight * current_dice_loss)

                total_loss += deep_supervision_weights[i] * combined_loss_per_head

            loss_recorder.update(total_loss.item(), inputs.size(0))

            # Flatten predictions and labels for confusion matrix update
            pred_labels = final_pred.argmax(1).flatten()
            confmat.update(labels.flatten(), pred_labels)

    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- {mode} mIoU: {mIoU:.4f} | {mode} Loss: {loss_recorder.avg:.4f} ---")
    if mode == "Validating":
        net.train() # Set back to train mode
    return mIoU

# Custom collate function to filter out None samples (from `datasets.py` returning None for corrupted images)
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting operation for '{exp_name}' with arguments: {args}")

    base_dataset_path = DATASET_PATHS[args.dataset_name]

    # --- MODIFIED ---
    # Instantiate the model with the correct backbone name
    # For InceptionV3/EfficientNet, pretrained=True is crucial for good feature extraction
    pretrained_weights = True if args.backbone not in ['vgg16', 'resnet50'] else True # Default to true for all backbones here
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone, pretrained=pretrained_weights).to(device)
    # --- END MODIFIED ---
    
    # ... (rest of the main function remains the same, e.g., loss functions, optimizer, data loaders) ...
    # Ensure the model instantiation is correct with the passed args.backbone
    
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=0).to(device)
    dice_loss_fn = DiceLoss(ignore_index=0).to(device)
    
    # ... (rest of training loop logic) ...

    if args.test_only:
        logging.info("Running in TEST ONLY mode.")
        best_checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
        if not os.path.exists(best_checkpoint_path):
            logging.error(f"Best checkpoint not found at {best_checkpoint_path}. Please run training first for this backbone/dataset combination.")
            sys.exit(1)

        try:
            net.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
            logging.info(f"✅ Model loaded successfully from best checkpoint: {best_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}. Exiting.")
            sys.exit(1)

        test_data_root = base_dataset_path
        if 'TSRS_RSNA' in args.dataset_name:
            test_data_root = os.path.join(base_dataset_path, 'test')
        
        if not os.path.exists(test_data_root):
            logging.error(f"❌ ERROR: Test data path does not exist: {test_data_root}. Please check your DATASET_PATHS and dataset organization.")
            sys.exit(1)

        test_set_for_eval = ImageFolder(test_data_root, args.dataset_name, args, split='test')
        test_loader_for_eval = DataLoader(test_set_for_eval, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

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

    train_data_root = base_dataset_path
    val_data_root = base_dataset_path
    if 'TSRS_RSNA' in args.dataset_name:
        train_data_root = os.path.join(base_dataset_path, 'train')
        val_data_root = os.path.join(base_dataset_path, 'val')

    if not os.path.exists(train_data_root):
        logging.error(f"❌ ERROR: Training data path does not exist: {train_data_root}.")
        sys.exit(1)
    if not os.path.exists(val_data_root):
        logging.error(f"❌ ERROR: Validation data path does not exist: {val_data_root}.")
        sys.exit(1)
        
    train_set = ImageFolder(train_data_root, args.dataset_name, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    test_set = ImageFolder(val_data_root, args.dataset_name, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Found {len(train_set)} training images for dataset '{args.dataset_name}'.")
    logging.info(f"Found {len(test_set)} validation images for dataset '{args.dataset_name}'.")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, data in enumerate(train_iterator):
            if data is None:
                logging.warning(f"Epoch {epoch+1}, Iter {i}: Skipping empty training batch due to corrupted/missing samples.")
                continue

            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = net(inputs)

            total_loss = 0
            for j, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())

                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)

                total_loss += args.deep_supervision_weights[j] * combined_loss_per_head

            total_loss.backward()
            optimizer.step()

            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)

        current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn,
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