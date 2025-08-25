#!/usr/bin/env python3
import os
import time
import logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# We need to tell the script where to find the other project files
import sys
sys.path.append('/kaggle/working/ARAA-Net/')
from daseg import daseg
from config import cod_training_root, test_path, CKPT_ROOT
from datasets import ImageFolder
import joint_transforms
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

def get_args():
    # --- RESTORED: ALL ORIGINAL ARGUMENTS ---
    parser = argparse.ArgumentParser(description='Train ARAA-Net with original logic')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose backbone')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    return parser.parse_args()

def validate(net, test_loader, device):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()

    logging.info("\\n--- Validation Results ---")
    logging.info(f"global_acc = {global_acc.item():.4f}")
    logging.info(f"class_acc  = {class_acc}")
    logging.info(f"class_iou  = {class_iou}")
    logging.info(f"mIoU       = {mIoU:.4f}")
    logging.info(f"FWIoU      = {fwiou.item():.4f}")
    logging.info(f"mDice      = {mDice:.4f}")
    logging.info("--------------------------")
    
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = args.backbone
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(exp_path, 'training.log')), logging.StreamHandler()])
    logging.info(f"--- Starting Training with Arguments: {args} ---")

    # Data Transformations & Dataloaders (using args for size)
    joint_transform = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(), joint_transforms.Resize((args.scale_h, args.scale_w))])
    val_joint_transform = joint_transforms.Compose([joint_transforms.Resize((args.scale_h, args.scale_w))])
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    train_set = ImageFolder(cod_training_root, joint_transform=joint_transform, transform=img_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_set = ImageFolder(test_path, joint_transform=val_joint_transform, transform=img_transform, target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False)
    
    net = daseg(backbone_name=args.backbone).to(device)
    
    # --- RESTORED: ORIGINAL OPTIMIZER LOGIC ---
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
                                {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}])
    else: # SGD
        optimizer = optim.SGD([{'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr},
                               {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}],
                              momentum=args.momentum)

    # Loss Functions
    structure_loss_fn = loss.structure_loss().to(device); bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device); ce_loss_fn = nn.CrossEntropyLoss().to(device)
    def bce_iou_loss(pred, target): return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    # Checkpoint Resuming Logic
    start_epoch = 0; best_mIoU = 0.0
    latest_ckpt_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if os.path.exists(latest_ckpt_path):
        logging.info(f"Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        net.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1; best_mIoU = ckpt.get('best_mIoU', 0.0)

    # --- RESTORED: EARLY STOPPING AND LR SCHEDULER SETUP ---
    patience_counter = 0
    total_iterations = len(train_loader) * args.epochs
    
    try:
        for epoch in range(start_epoch, args.epochs):
            net.train()
            loss_recorder = AvgMeter()
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            for i, data in enumerate(train_iterator):
                # --- RESTORED: POLYNOMIAL LR DECAY ---
                curr_iter = epoch * len(train_loader) + i
                lr_decay = (1 - curr_iter / total_iterations) ** args.lr_decay
                optimizer.param_groups[0]['lr'] = 2 * args.lr * lr_decay
                optimizer.param_groups[1]['lr'] = args.lr * lr_decay
            
                inputs, labels = data['image'].to(device), data['label'].to(device)
                binary_labels = labels.unsqueeze(1).float(); ce_labels = labels.long()
                optimizer.zero_grad()
                p4, p3, p2, p1, p0 = net(inputs)
                loss_1 = bce_iou_loss(p1, binary_labels); loss_2 = structure_loss_fn(p2, binary_labels)
                loss_3 = structure_loss_fn(p3, binary_labels); loss_4 = structure_loss_fn(p4, binary_labels)
                loss_0 = ce_loss_fn(p0, ce_labels)
                total_loss = loss_1 + loss_2 + 2*loss_3 + 4*loss_4 + 10*loss_0
                total_loss.backward(); optimizer.step()
                loss_recorder.update(total_loss.item(), inputs.size(0))
                train_iterator.set_postfix(loss=f'{loss_recorder.avg:.4f}', lr=f"{optimizer.param_groups[1]['lr']:.6f}")

            current_mIoU = validate(net, test_loader, device)
            
            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU
                patience_counter = 0 # Reset patience
                torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
                logging.info(f"✅ New best model saved at epoch {epoch+1} with mIoU: {best_mIoU:.4f}")
            else:
                patience_counter += 1 # Increment patience
            
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, latest_ckpt_path)

            # --- RESTORED: EARLY STOPPING LOGIC ---
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered after {args.patience} epochs with no improvement.")
                break
            
    finally:
        logging.info(f"--- Training Process Concluded --- Best mIoU achieved: {best_mIoU:.4f} ---")

if __name__ == '__main__':
    main()