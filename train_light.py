# /kaggle/working/ARAA-Net/train_light.py
# --- FINAL VERSION: MobileNetV2 + SE Blocks + Deep Supervision + CBAM + Boundary Module + Boundary Loss ---

import sys, os, logging, argparse, torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm
import torch.nn.functional as F

# --- Setup, Imports, Loss Functions ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path: sys.path.insert(0, project_path)

import config
from lasa_unet_model import LASA_Unet
from light_lasa_unet import Light_LASA_Unet # <-- Model with CBAM and Boundary Module
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir
from boundary_loss import BoundaryLoss # <-- Ensure this is imported

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2): 
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, i, t):
        ce = F.cross_entropy(i, t.long(), reduction='none')
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt)**self.gamma * ce
        return fl.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): 
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, i, t):
        p = F.softmax(i, dim=1)[:, 1] # Assuming binary segmentation, taking channel 1
        to = (t.float() == 1).float() # Target for foreground class 1
        
        inter = (p * to).sum()
        denominator = p.sum() + to.sum() + self.smooth
        return 1 - ((2. * inter + self.smooth) / denominator)

# --- Backbone Freezing ---
def freeze_backbone(model):
    for n, p in model.named_parameters():
        if 'encoder' in n: p.requires_grad = False
    logging.info("--- Encoder FROZEN ---")

def unfreeze_backbone(model):
    for n, p in model.named_parameters():
        if 'encoder' in n: p.requires_grad = True
    logging.info("--- Encoder UN-FROZEN ---")

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Models')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--backbone', type=str, default='vgg19', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lasa-kernels', type=int, nargs='+', default=[1, 3, 5, 7])
    parser.add_argument('--deep-supervision-weights', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--focal-loss-weight', type=float, default=0.5)
    parser.add_argument('--dice-loss-weight', type=float, default=1.5)
    parser.add_argument('--boundary-loss-weight', type=float, default=0.5) # Added boundary loss weight
    parser.add_argument('--scheduler-type', type=str, default='CosineAnnealingWarmRestarts', choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--scheduler-T0', type=int, default=15)
    parser.add_argument('--scheduler-T-mult', type=int, default=2)
    parser.add_argument('--scheduler-patience', type=int, default=10)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--fine-tune-epochs', type=int, default=40)
    
    args = parser.parse_args()
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    
    try:
        res_h, res_w = config.get_backbone_resolution(args.backbone)
    except AttributeError:
        res_h, res_w = 224, 224 
        logging.warning("config.get_backbone_resolution not found. Using default 224x224.")
    except Exception as e:
        logging.error(f"Error loading backbone resolution from config: {e}. Using default 224x224.")
        res_h, res_w = 224, 224

    args.scale_h, args.scale_w = res_h, res_w
    
    if hasattr(config, 'DEFAULT_ARGS'):
        for k, v in config.DEFAULT_ARGS.items():
            if not hasattr(args, k):
                setattr(args, k, v)
            
    return args

# --- Logging, Evaluation, Collate ---
def setup_logging(log_dir, filename='training.log'):
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    check_mkdir(log_dir)
    log_file_path = os.path.join(log_dir, filename)
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, boundary_loss_fn, args, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(args.num_classes)
    loss_recorder = AvgMeter() 

    # The model returns: [seg_d4, bound_d4, seg_d3, bound_d3, seg_d2, bound_d2, seg_d1, bound_d1, final_seg]
    num_seg_outputs = 5 # d4, d3, d2, d1, final_seg
    num_decoder_stages = 4 # d4, d3, d2, d1 (each has seg + bound output)

    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None: 
                continue 

            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            outputs_tuple = net(inputs)

            total_loss_batch = 0
            seg_output_idx = 0 # Index for deep_supervision_weights
            
            for i in range(num_decoder_stages):
                seg_pred = outputs_tuple[i * 2]         # Segmentation output
                boundary_pred = outputs_tuple[i * 2 + 1] # Boundary prediction

                seg_labels = labels.long()

                f_loss = focal_loss_fn(seg_pred, seg_labels)
                d_loss = dice_loss_fn(seg_pred, seg_labels)
                
                b_loss = torch.tensor(0.0, device=device) # Default to 0
                try:
                    b_loss = boundary_loss_fn(boundary_pred, seg_labels) 
                except RuntimeError as e:
                    logging.error(f"Error calculating boundary loss for stage {i}: {e}. Skipping boundary loss for this stage.")
                except IndexError as e:
                    logging.error(f"Index error calculating boundary loss for stage {i}: {e}. Skipping boundary loss.")

                weight_index = seg_output_idx
                seg_weight = args.deep_supervision_weights[weight_index] if weight_index < len(args.deep_supervision_weights) else 1.0

                segmentation_loss_component = seg_weight * (
                    (args.focal_loss_weight * f_loss) + 
                    (args.dice_loss_weight * d_loss)
                )
                boundary_loss_component = seg_weight * (args.boundary_loss_weight * b_loss) 
                
                total_loss_batch += segmentation_loss_component + boundary_loss_component
                
                seg_output_idx += 1

            # Handle the final segmentation output
            final_seg_pred = outputs_tuple[-1]
            f_loss_final = focal_loss_fn(final_seg_pred, seg_labels)
            d_loss_final = dice_loss_fn(final_seg_pred, seg_labels)
            
            final_seg_weight = args.deep_supervision_weights[-1] if args.deep_supervision_weights else 1.0
            total_loss_batch += final_seg_weight * ((args.focal_loss_weight * f_loss_final) + (args.dice_loss_weight * d_loss_final))
            
            loss_recorder.update(total_loss_batch.item(), inputs.size(0))
            
            confmat.update(labels.flatten(), final_seg_pred.argmax(1).flatten())
            
    acc, _, iou, fwiou, dice = confmat.compute()
    mIoU = iou.mean().item() 
    
    logging.info(f"--- {mode} Summary --- Loss: {loss_recorder.avg:.4f}, OA: {acc.item():.4f}, mIoU: {mIoU:.4f}")
    
    if mode == "Validating": 
        net.train() 
    return mIoU

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# --- Main Function ---
def main():
    args = get_args()
    device = torch.device("cuda")
    
    exp_name = f"{args.backbone}_FreezeTune_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    check_mkdir(base_exp_path) 
    setup_logging(base_exp_path, 'main_training.log')

    logging.info(f"Starting experiment: '{exp_name}'\nArguments: {vars(args)}")

    # --- Load Datasets ---
    try:
        train_ds = ImageFolder(os.path.join(args.dataset_path, 'train'), args.dataset_name, args, 'train')
        val_ds = ImageFolder(os.path.join(args.dataset_path, 'val'), args.dataset_name, args, 'val')
        test_ds = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, 'test')
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        return

    # --- Instantiate Model ---
    if args.backbone == 'mobilenet_v2':
        net = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=args.lasa_kernels).to(device)
        logging.info("Instantiated Lightweight MobileNetV2-based model.")
    else:
        net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels).to(device)
        logging.info(f"Instantiated {args.backbone}-based model.")
    
    # --- Initialize Losses ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    boundary_loss_fn = BoundaryLoss(device=device).to(device)

    # --- Test-Only Mode ---
    if args.test_only:
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        ckpt_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
        if not os.path.exists(ckpt_path): 
            logging.error(f"Checkpoint not found: {ckpt_path}. Cannot run test-only.")
            return
        
        try:
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            logging.info(f"Model loaded from {ckpt_path}")
        except Exception as e:
            logging.error(f"Error loading checkpoint {ckpt_path}: {e}")
            logging.error("Potential issue: Model architecture might have changed since checkpoint was saved.")
            return
            
        evaluate_model(net, loader, device, focal_loss_fn, dice_loss_fn, boundary_loss_fn, args, mode="Testing")
        return

    # --- DataLoaders for Training and Validation ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2, eta_min=1e-6)
    else: # ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.scheduler_patience)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    best_ckpt_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
    latest_ckpt_path = os.path.join(base_exp_path, 'latest_checkpoint.pth')
    
    # --- Resume Training ---
    if args.resume and os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming from epoch {start_epoch}. Best mIoU: {best_mIoU:.4f}.")
        except Exception as e:
            logging.error(f"Could not resume from {latest_ckpt_path}: {e}. Starting from scratch.")
            start_epoch = 0

    # --- Freeze Backbone ---
    if start_epoch < args.fine_tune_epochs:
        freeze_backbone(net)
    else:
        unfreeze_backbone(net)

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.fine_tune_epochs:
            unfreeze_backbone(net)
            # Adjust LR and re-initialize scheduler for fine-tuning phase
            new_lr = args.lr / 10.0 if args.lr > 1e-5 else args.lr 
            optimizer = optim.Adam(net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2, eta_min=1e-6) 
            logging.info(f"--- Switched to Phase 2. New LR: {new_lr} ---")
            
        net.train() # Set model to training mode
        loss_recorder = AvgMeter() # Meter for epoch loss
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in train_iterator:
            if data is None: 
                continue # Skip if sample loading failed

            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs_tuple = net(inputs) 

            total_loss = 0
            seg_output_idx = 0 
            
            num_decoder_stages = 4 
            
            for i in range(num_decoder_stages):
                seg_pred = outputs_tuple[i * 2]         
                boundary_pred = outputs_tuple[i * 2 + 1] 

                seg_labels = labels.long()

                f_loss = focal_loss_fn(seg_pred, seg_labels)
                d_loss = dice_loss_fn(seg_pred, seg_labels)
                
                b_loss = torch.tensor(0.0, device=device) 
                try:
                    b_loss = boundary_loss_fn(boundary_pred, seg_labels) 
                except RuntimeError as e:
                    logging.error(f"Error calculating boundary loss for stage {i}: {e}. Skipping boundary loss for this stage.")
                except IndexError as e:
                    logging.error(f"Index error calculating boundary loss for stage {i}: {e}. Skipping boundary loss.")

                weight_index = seg_output_idx
                seg_weight = args.deep_supervision_weights[weight_index] if weight_index < len(args.deep_supervision_weights) else 1.0

                segmentation_loss_component = seg_weight * (
                    (args.focal_loss_weight * f_loss) + 
                    (args.dice_loss_weight * d_loss)
                )
                boundary_loss_component = seg_weight * (args.boundary_loss_weight * b_loss) 
                
                total_loss += segmentation_loss_component + boundary_loss_component
                
                seg_output_idx += 1

            # Handle the final segmentation output
            final_seg_pred = outputs_tuple[-1]
            f_loss_final = focal_loss_fn(final_seg_pred, seg_labels)
            d_loss_final = dice_loss_fn(final_seg_pred, seg_labels)
            
            final_seg_weight = args.deep_supervision_weights[-1] if args.deep_supervision_weights else 1.0
            total_loss += final_seg_weight * ((args.focal_loss_weight * f_loss_final) + (args.dice_loss_weight * d_loss_final))
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        # --- Validation ---
        current_mIoU = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, boundary_loss_fn, args)
        
        # --- Scheduler Step ---
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_mIoU) 
        else:
            scheduler.step() 
        
        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0 
            torch.save(net.state_dict(), best_ckpt_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Model saved.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ mIoU did not improve for {patience_counter} epoch(s). Best: {best_mIoU:.4f}")
        
        torch.save({
            'epoch': epoch, 'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_mIoU': best_mIoU, 'patience_counter': patience_counter
        }, latest_ckpt_path)

        # --- Early Stopping ---
        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered after {patience_counter} epochs.")
            break
            
    logging.info("Training finished.")

if __name__ == '__main__':
    main()