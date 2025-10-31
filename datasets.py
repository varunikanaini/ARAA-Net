#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang
"""
import os
import torch
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from torchvision import transforms
import custom_transforms as tr
import config  # Import the new, cleaner config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')

# ==================================================================================
# === NEW: A single, robust function to find all image/mask pairs in a directory ===
# ==================================================================================
def get_all_image_mask_pairs(root_path, dataset_name):
    """Finds all (image, mask) pairs based on the dataset's specific folder names."""
    all_pairs = []
    
    if dataset_name.startswith('TSRS_RSNA'):
        # This dataset is pre-split, so we search both train and test folders
        for split_folder in ['train', 'test']:
            image_dir = os.path.join(root_path, split_folder)
            mask_dir = os.path.join(root_path, f"{split_folder}_labels")
            if not (os.path.exists(image_dir) and os.path.exists(mask_dir)):
                continue
            img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
            for name in img_names:
                all_pairs.append((os.path.join(image_dir, name + '.jpg'), os.path.join(mask_dir, name + '.png')))

    elif dataset_name == 'JSRT':
        # This dataset has a flat structure
        image_dir = os.path.join(root_path, 'content', 'jsrt', 'cxr')
        mask_dir = os.path.join(root_path, 'content', 'jsrt', 'masks')
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.png')]
            for name in img_names:
                all_pairs.append((os.path.join(image_dir, name + '.png'), os.path.join(mask_dir, name + '.png')))
    
    # Add `elif` blocks here for other flat datasets like CVC-ClinicDB
    elif dataset_name == 'CVC-ClinicDB':
        image_dir = os.path.join(root_path, 'Original')
        mask_dir = os.path.join(root_path, 'Ground Truth')
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
             img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
             for name in img_names:
                all_pairs.append((os.path.join(image_dir, name + '.tif'), os.path.join(mask_dir, name + '.tif')))

    if not all_pairs:
        print(f"Warning: Found 0 image-mask pairs for '{dataset_name}' at root '{root_path}'. Check paths.")
    return all_pairs

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, split='train', imgs=None, val_size=0.1, test_size=0.1, random_state=42):
        self.split = split
        dataset_cfg = config.DATASET_CONFIG[dataset_name]

        if imgs is not None:
            # This path is for k-fold, where pre-split lists are passed in.
            self.imgs = imgs
        else:
            # This path is for standard train/val/test runs.
            structure = dataset_cfg['structure']
            
            if structure == 'PRE_SPLIT':
                # For pre-split data, just load the corresponding folder.
                image_dir = os.path.join(root, split)
                mask_dir = os.path.join(root, f"{split}_labels")
                if not (os.path.exists(image_dir) and os.path.exists(mask_dir)):
                    self.imgs = []
                else:
                    img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
                    self.imgs = [(os.path.join(image_dir, n + '.jpg'), os.path.join(mask_dir, n + '.png')) for n in img_names]

            elif structure == 'FLAT_SPLIT':
                # For flat data, load everything and then split it.
                all_pairs = get_all_image_mask_pairs(root, dataset_name)
                if not all_pairs:
                    self.imgs = []
                    return
                
                # Create a reproducible 80/10/10 split
                train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=random_state)
                val_proportion = val_size / (1 - test_size)
                train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_proportion, random_state=random_state)
                
                if split == 'train': self.imgs = train_pairs
                elif split == 'val': self.imgs = val_pairs
                elif split == 'test': self.imgs = test_pairs
                else: raise ValueError(f"Invalid split '{split}' for FLAT_SPLIT dataset.")

        if not self.imgs:
            print(f"Warning: The '{self.split}' split for '{dataset_name}' is empty.")

    # --- The rest of your ImageFolder class (getitem, transforms, etc.) remains UNCHANGED ---
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: {img_path}")
            if mask_np is None: raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}")
            if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_np)
            if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
            target = Image.fromarray(mask_np, mode='L')
            label = self.convert_label(target)
        except Exception as e:
            print(f"ERROR processing {img_path}. Skipping. Error: {e}")
            return None
        sample = {'image': img, 'label': label}
        if self.split == "train": return self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)
            sample['name'] = self.imgs[index]
            return sample
        
    def convert_label(self, label):
        label_array = np.array(label)
        if label_array.ndim == 3: label_array = label_array[:, :, 0]
        label_index = np.zeros_like(label_array, dtype='uint8')
        label_index[label_array > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self): return len(self.imgs)
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(), tr.FixedResize(576,896), tr.RandomCrop((576,576)),
            tr.RandomGaussianBlur(), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(576,896), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)