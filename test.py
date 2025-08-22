#!/usr/bin/env python3
# test.py (Final Version for Evaluation)

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

# This script uses the same model definition as train.py
from daseg import daseg
from config import test_path, CKPT_ROOT
from datasets import ImageFolder
import joint_transforms
from seg_utils import ConfusionMatrix

def get_args():
    # These are the only arguments needed for testing:
    # which model to test, and how to load the data.
    parser = argparse.ArgumentParser(description='Test the best trained ARAA-Net model')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'vgg16'],
                        help='Choose the backbone of the trained model to test')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--scale-h', type=int, default=896, help='Height to resize images to')
    parser.add_argument('--scale-w', type=int, default=576, help='Width to resize images to')
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
    
    # The script looks for the single best model saved by train.py
    checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        logging.error(f"Best checkpoint not found at {checkpoint_path}")
        logging.error("Please run train.py first to generate a checkpoint.")
        return

    logging.info(f"Loading best model from {checkpoint_path} for testing.")
    logging.info(f"Using device: {device}")

    # Data transforms should match the validation set from training
    test_joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((args.scale_h, args.scale_w))
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    
    test_set = ImageFolder(test_path, test_joint_transform, img_transform, target_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    logging.info(f"Found {len(test_set)} testing images.")

    # Initialize the same model architecture as used in training
    net = daseg(backbone_name=args.backbone).to(device)
    
    # Load the trained weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle checkpoints saved with or without DataParallel wrapper
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)

    net.eval()

    confmat = ConfusionMatrix(num_classes=2)
    test_iterator = tqdm(test_loader, desc="Testing")
    
    start_time = time.time()
    # No training happens here, so we use torch.no_grad() for speed
    with torch.no_grad():
        for data in test_iterator:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            _, _, _, _, pred = net(inputs)
            confmat.update(labels.flatten(), pred.argmax(1).flatten())
    end_time = time.time()
    
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()

    logging.info("\n------ Final Test Results ------")
    logging.info(f"global_acc = {global_acc.item():.4f}")
    logging.info(f"class_acc  = {class_acc.cpu().numpy()}")
    logging.info(f"class_iou  = {class_iou.cpu().numpy()}")
    logging.info(f"mIoU       = {mIoU:.4f}")
    logging.info(f"FWIoU      = {fwiou.item():.4f}")
    logging.info(f"mDice      = {mDice:.4f}")
    logging.info(f"Total Testing Time: {str(datetime.timedelta(seconds=int(end_time - start_time)))}")
    logging.info("------------------------------")

if __name__ == '__main__':
    main()