#!/usr/bin/env python3
# test.py

import os
import time
import logging
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from model import ARAA_Net
from config import test_path, CKPT_ROOT
from datasets import ImageFolder
import joint_transforms
from seg_utils import ConfusionMatrix

def get_args():
    parser = argparse.ArgumentParser(description='Test ARAA-Net')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'],
                        help='Choose the backbone of the trained model')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    return parser.parse_args()

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'testing.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = args.backbone
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    setup_logging(exp_path)
    
    checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        logging.error(f"Best checkpoint not found at {checkpoint_path}")
        return

    logging.info(f"Loading best model from {checkpoint_path} for testing.")
    logging.info(f"Using device: {device}")

    test_transform = joint_transforms.Compose([joint_transforms.Resize((896, 576))])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    
    test_set = ImageFolder(test_path, test_transform, img_transform, target_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    logging.info(f"Found {len(test_set)} testing images.")

    net = ARAA_Net(backbone_name=args.backbone, pretrained=False).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    confmat = ConfusionMatrix(num_classes=2)
    test_iterator = tqdm(test_loader, desc="Testing")
    
    start_time = time.time()
    with torch.no_grad():
        for data in test_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
    end_time = time.time()
    
    global_acc, _, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean()

    logging.info("------ Test Results ------")
    logging.info(f"Overall Accuracy (OA): {global_acc.item():.4f}")
    logging.info(f"Mean IoU (mIoU): {mIoU:.4f}")
    logging.info(f"FWIoU: {fwiou.item():.4f}")
    logging.info(f"Mean Dice: {mDice:.4f}")
    logging.info(f"Class-wise IoU: {class_iou}")
    logging.info(f"Total Testing Time: {str(datetime.timedelta(seconds=int(end_time - start_time)))}")
    logging.info("--------------------------")

if __name__ == '__main__':
    main()