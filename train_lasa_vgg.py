# /kaggle/working/ARAA-Net/train_lasa_vgg.py (Updated for Combined Loss, LR Scheduler, Deep Supervision)
import os
import time
import sys
import logging
import argparse
import torch
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
from config import DATA_ROOT, CKPT_ROOT 
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# ===================================================================
#      ✅ FOCAL LOSS CLASS - INCLUDED DIRECTLY IN THIS SCRIPT ✅
# ===================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification, included directly in the script.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ===================================================================

# ===================================================================
#      ✅ DICE LOSS CLASS - NEWLY ADDED ✅
# ===================================================================
class DiceLoss(nn.Module):
    """
    Dice Loss for binary or multi-class segmentation.
    Supports a single foreground class (target 1, background 0).
    """
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are logits (N, C, H, W)
        # targets are class indices (N, H, W)
        
        # Convert targets to one-hot encoding if needed (for C > 1)
        # For binary segmentation (C=2), we focus on the foreground class (index 1)
        num_classes = inputs.shape[1]
        
        if num_classes > 1:
            # Get probabilities for foreground class
            pred_probs = F.softmax(inputs, dim=1)[:, 1, :, :].unsqueeze(1) # (N, 1, H, W)
            true_oh = (targets == 1).float().unsqueeze(1) # One-hot for foreground (N, 1, H, W)
        else: # Single output channel for binary segmentation
            pred_probs = F.sigmoid(inputs)
            true_oh = targets.float().unsqueeze(1) # (N, 1, H, W)

        # Handle ignore_index (if present, mask it out)
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            pred_probs = pred_probs * mask.unsqueeze(1)
            true_oh = true_oh * mask.unsqueeze(1)
        
        # Flatten label and prediction tensors
        pred_probs = pred_probs.view(-1)
        true_oh = true_oh.view(-1)

        intersection = (pred_probs * true_oh).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + true_oh.sum() + self.smooth)
        
        loss = 1. - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * inputs.shape[0] # Scale by batch size for consistency
        else:
            return loss # Return per-sample loss if reduction is 'none' (though dice is typically aggregated)
# ===================================================================


def get_args():
    parser = argparse.ArgumentParser(description='Train LASA-Unet Model with Deep Supervision and Amplification')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=448, help='Resize height for input images')
    parser.add_argument('--scale-w', type=int, default=448, help='Resize width for input images')
    
    # Deep Supervision weights (5 outputs: d4, d3, d2, d1_aux, final)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Weights for deep supervision losses, from earliest (d4) to final (d1) output. Must have 5 values.')
    
    # Focal Loss specific hyperparameters
    parser.add_argument('--focal-alpha', type=float, default=0.5, 
                        help='Alpha parameter for Focal Loss. Balance between positive/negative examples.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Gamma parameter for Focal Loss. Focus on hard examples.')
    parser.add_argument('--focal-loss-weight', type=float, default=1.0, 
                        help='Weight for Focal Loss component in combined loss.')

    # Dice Loss specific hyperparameters
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, 
                        help='Weight for Dice Loss component in combined loss.')
    
    # Arguments for CenterAmplification
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576, 
                        help='Minimum lesion area in pixels to trigger CenterAmplification (STS-Net default 576)')
    parser.add_argument('--expansion-factor', type=float, default=1.5, 
                        help='Factor by which to expand the bounding box during CenterAmplification')
    parser.add_argument('--min-bbox-h', type=int, default=32, 
                        help='Minimum height of the expanded bounding box in pixels for CenterAmplification')
    parser.add_argument('--min-bbox-w', type=int, default=32, 
                        help='Minimum width of the expanded bounding box in pixels for CenterAmplification')

    # Test-only flag
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on the best saved checkpoint.')

    # Learning Rate Scheduler parameters
    parser.add_argument('--scheduler-patience', type=int, default=5, 
                        help='Number of epochs with no improvement after which learning rate will be reduced.')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, 
                        help='Factor by which the learning rate will be reduced.')
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6, 
                        help='Minimum learning rate.')


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
            final_pred = outputs[-1] # The last output is always the final one for evaluation metrics
            
            total_loss = 0
            for i, pred_output in enumerate(outputs):
                # Calculate combined loss for each deep supervision output
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
    if mode == "Validating": # Only set to train if we are in training loop
        net.train()
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    # Naming convention for experiment folder
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}" 
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting operation for '{exp_name}' with arguments: {args}")

    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # --- Instantiate the model ---
    net = LASA_Unet(num_classes=2, backbone_name=args.backbone).to(device)
    
    # --- Initialize Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    # --- Test-only Mode ---
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

        test_set_for_eval = ImageFolder(val_path, args, split='val') 
        test_loader_for_eval = DataLoader(test_set_for_eval, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

        test_mIoU = evaluate_model(net, test_loader_for_eval, device, focal_loss_fn, dice_loss_fn, 
                                   args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Testing")
        logging.info(f"Final Test mIoU: {test_mIoU:.4f}")
        return # Exit main function after testing

    # --- Training Mode ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    
    # Load latest checkpoint for resuming training
    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Ensure scheduler state is also loaded
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming from epoch {start_epoch}, best mIoU was {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")

    train_set = ImageFolder(train_path, args, split='train') 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(val_path, args, split='val') # This is for validation during training
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)


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
                # Calculate combined loss for each deep supervision output
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[i] * combined_loss_per_head
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        # Step the scheduler based on validation mIoU
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
            'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered due to no improvement.")
            break

if __name__ == '__main__':
    main()