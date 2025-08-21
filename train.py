#!/usr/bin/env python3
# train.py

import os
import time
import logging
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import ARAA_Net
from config import cod_training_root, test_path, CKPT_ROOT
from datasets import ImageFolder
import joint_transforms
import loss
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

def get_args():
    parser = argparse.ArgumentParser(description='Train ARAA-Net with original logic')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Exponent for polynomial LR decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to')
    parser.add_argument('--scale-w', type=int, default=576, ahelp='Width to resize images to')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Choose the backbone model')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    return parser.parse_args()

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2024)
        torch.backends.cudnn.benchmark = True

    exp_name = args.backbone
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)

    logging.info(f"Starting training with arguments: {args}")
    logging.info(f"Using device: {device}")

    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((args.scale_h, args.scale_w)),
        joint_transforms.RandomCrop((576, 576), pad_if_needed=True, lbl_fill=255)
    ])
    val_joint_transform = joint_transforms.Compose([joint_transforms.Resize((args.scale_h, args.scale_w))])
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_set = ImageFolder(test_path, val_joint_transform, img_transform, target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    logging.info(f"Found {len(train_set)} training images and {len(test_set)} validation images.")

    net = ARAA_Net(backbone_name=args.backbone, pretrained=True).to(device)
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr}, {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}])
    else: # SGD
        optimizer = optim.SGD([{'params': [p for n, p in net.named_parameters() if 'bias' in n], 'lr': 2 * args.lr}, {'params': [p for n, p in net.named_parameters() if 'bias' not in n], 'lr': args.lr, 'weight_decay': args.weight_decay}], momentum=args.momentum)

    structure_loss_fn = loss.structure_loss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    iou_loss_fn = loss.IOU().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device)

    def bce_iou_loss(pred, target):
        return bce_loss_fn(pred, target) + iou_loss_fn(pred, target)

    start_epoch, best_mIoU = 0, 0.0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint_path):
        logging.info(f"Resuming from checkpoint: {latest_checkpoint_path}")
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch, best_mIoU = 0, 0.0


    patience_counter = 0
    total_iterations = len(train_loader) * args.epochs

    for epoch in range(start_epoch, args.epochs):
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for i, data in enumerate(train_iterator):
            curr_iter = epoch * len(train_loader) + i
            lr_decay = (1 - curr_iter / total_iterations) ** args.lr_decay
            optimizer.param_groups[0]['lr'] = 2 * args.lr * lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * lr_decay
            
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)

            p4, p3, p2, p1, p0 = net(inputs)

            loss_1 = bce_iou_loss(p1, labels.unsqueeze(1).float())
            loss_2 = structure_loss_fn(p2, labels.unsqueeze(1).float())
            loss_3 = structure_loss_fn(p3, labels.unsqueeze(1).float())
            loss_4 = structure_loss_fn(p4, labels.unsqueeze(1).float())
            loss_0 = ce_loss_fn(p0, labels.long())
            total_loss = loss_1 + loss_2 + 2 * loss_3 + 4 * loss_4 + 10 * loss_0

            total_loss.backward()
            optimizer.step()

            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg, lr=optimizer.param_groups[1]['lr'])

        logging.info(f"Epoch {epoch+1} Train | Average Loss: {loss_recorder.avg:.4f}")
        current_mIoU = validate(net, test_loader, device)
        
        is_best = current_mIoU > best_mIoU
        if is_best:
            best_mIoU = current_mIoU
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ Epoch {epoch+1}: New best mIoU: {best_mIoU:.4f}. Saving checkpoint...")
        else:
            patience_counter += 1
            logging.info(f"⚠️ Epoch {epoch+1}: No improvement in mIoU for {patience_counter} epoch(s). Best mIoU remains {best_mIoU:.4f}.")

        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mIoU': best_mIoU}, latest_checkpoint_path)
        
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

def validate(net, test_loader, device):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    test_iterator = tqdm(test_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for data in test_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
            
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    
    logging.info("\n--- Validation Results ---")
    logging.info(f"global_acc = {global_acc.item():.4f}")
    logging.info(f"class_acc  = {class_acc}")
    logging.info(f"class_iou  = {class_iou}")
    logging.info(f"mIoU       = {mIoU:.4f}")
    logging.info(f"FWIoU      = {fwiou.item():.4f}")
    logging.info(f"mDice      = {mDice:.4f}")
    logging.info("--------------------------")
    
    return mIoU

if __name__ == '__main__':
    main()