# /kaggle/working/ARAA-Net/train_lasa_vgg.py
import os
import time
import sys
import logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import the NEW standalone model ---
from lasa_vgg_model import LASA_VGG_Unet

from config import DATA_ROOT, CKPT_ROOT
from datasets import ImageFolder
import loss as loss_module
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

def get_args():
    parser = argparse.ArgumentParser(description='Train Standalone LASA-VGG-Unet Model')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=['TSRS_RSNA-Epiphysis', 'TSRS_RSNA-Articular-Surface'], help='Name of the dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=448, help='Resize height')
    parser.add_argument('--scale-w', type=int, default=448, help='Resize width')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def validate(net, test_loader, device, focal_loss_fn):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    loss_recorder = AvgMeter()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            # The model returns 5 identical outputs, we only need the last one for validation
            _, _, _, _, pred = net(inputs)
            loss = focal_loss_fn(pred, labels.long())
            loss_recorder.update(loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- Validation mIoU: {mIoU:.4f} | Validation Loss: {loss_recorder.avg:.4f} ---")
    net.train()
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed(2024)

    # Experiment name for this specific test
    exp_name = f"standalone_LASA_VGG16_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training with arguments: {args}")
    logging.info(f"Experiment name: {exp_name}")

    dataset_path = os.path.join(DATA_ROOT, args.dataset_name)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Pass args to the dataset class so it knows the image sizes
    train_set = ImageFolder(train_path, args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(val_path, args, split='val')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # --- Use the new, simpler model ---
    net = LASA_VGG_Unet(num_classes=2).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # --- Use only Focal Loss, as there is only one output ---
    focal_loss_fn = loss_module.FocalLoss(alpha=0.25, gamma=2).to(device)

    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint_path):
        # Resume logic here...
        pass
    
    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for data in train_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # The model returns 5 identical outputs, we only need the last one for loss
            _, _, _, _, pred = net(inputs)
            
            loss = focal_loss_fn(pred, labels.long())
            
            loss.backward()
            optimizer.step()
            
            loss_recorder.update(loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        current_mIoU = validate(net, test_loader, device, focal_loss_fn)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

if __name__ == '__main__':
    main()