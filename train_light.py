# /kaggle/working/ARAA-Net/train_light.py
# --- FINAL, COMPLETE & WORKING VERSION with RESUME LOGIC ---

import sys, os, logging, argparse, torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- Setup, Imports, Loss Functions ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path: sys.path.insert(0, project_path)
import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)[:, 1]
        targets_one_hot = (targets == 1).float()
        intersection = (probs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets_one_hot.sum() + self.smooth)
        return 1. - dice

# --- Argument Parsing (with --resume flag added) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train Lightweight LASA-Unet')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--focal-weight', type=float, default=0.5)
    parser.add_argument('--dice-weight', type=float, default=1.5)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint") # <-- RESUME FLAG
    parser.add_argument('--num-workers', type=int, default=2)
    
    args = parser.parse_args()
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path = dataset_info['path']
    args.num_classes = dataset_info['num_classes']
    args.scale_h, args.scale_w = 224, 224
    for key, value in config.DEFAULT_ARGS.items():
        if not hasattr(args, key): setattr(args, key, value)
    return args

# --- Logging, Evaluation, Collate ---
def setup_logging(log_dir, filename='training.log'):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def evaluate_model(net, data_loader, device, criterion, args, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(num_classes=args.num_classes)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss_recorder.update(loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), outputs.argmax(1).flatten())
            
    global_acc, _, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info(f"--- {mode} Summary ---")
    logging.info(f"  Average Loss: {loss_recorder.avg:.4f}")
    logging.info(f"  OA: {global_acc.item():.4f}, mIoU: {mIoU:.4f}, FWIoU: {fwiou.item():.4f}, Dice: {mDice:.4f}")
    
    if mode == "Validating": net.train()
    return mIoU

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Main Function ---
def main():
    args = get_args()
    device = torch.device("cuda")
    
    exp_name = f"mobilenetv2_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    check_mkdir(base_exp_path)
    setup_logging(base_exp_path)

    logging.info(f"Starting experiment: '{exp_name}'")
    logging.info(f"Arguments: {vars(args)}")

    train_dataset = ImageFolder(os.path.join(args.dataset_path, 'train'), args.dataset_name, args, split='train')
    val_dataset = ImageFolder(os.path.join(args.dataset_path, 'val'), args.dataset_name, args, split='val')
    test_dataset = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, split='test')

    net = Light_LASA_Unet(num_classes=args.num_classes).to(device)
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    def criterion(pred, target):
        return args.focal_weight * focal_loss(pred, target.long()) + args.dice_weight * dice_loss(pred, target.long())

    if args.test_only:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            return
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Model loaded from {checkpoint_path}")
        evaluate_model(net, test_loader, device, criterion, args, mode="Testing")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    best_checkpoint_path = os.path.join(base_exp_path, 'best_checkpoint.pth')
    latest_checkpoint_path = os.path.join(base_exp_path, 'latest_checkpoint.pth') # <-- DEFINE LATEST PATH

    # --- ADDED: RESUME LOGIC ---
    if args.resume and os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"✅ Resuming training from epoch {start_epoch}. Best mIoU was {best_mIoU:.4f}.")
        except Exception as e:
            logging.error(f"Could not load checkpoint for resuming: {e}. Starting from scratch.")
            start_epoch = 0 # Ensure we start from scratch on failure
    # ---------------------------

    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in train_iterator:
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_recorder.update(loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        current_mIoU = evaluate_model(net, val_loader, device, criterion, args)
        scheduler.step()
        
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), best_checkpoint_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ mIoU did not improve for {patience_counter} epoch(s). Best: {best_mIoU:.4f}")
        
        # --- ADDED: SAVE LATEST CHECKPOINT FOR RESUMING ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        # ----------------------------------------------------

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("Training finished.")

if __name__ == '__main__':
    main()