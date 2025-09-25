#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""

import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
import random # Import random for splitting

from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr


def make_dataset(root, dataset_name):
    img_list = []
    
    # Handling predefined split for TSRS_RSNA-Epiphysis
    if dataset_name == 'TSRS_RSNA-Epiphysis_train' or dataset_name == 'TSRS_RSNA-Epiphysis_test':
        image_path = root
        mask_path = root + '_labels'
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: {dataset_name} paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
        # This dataset is already split by folder, so we directly return its images
        return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    
    # For datasets that need programmatic splitting, gather all images first
    elif dataset_name == 'JSRT':
        image_path = os.path.join(root, 'content', 'jsrt', 'cxr')
        mask_path = os.path.join(root, 'content', 'jsrt', 'masks')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Did the download complete and extract correctly? Returning empty dataset.")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.png')]
        img_list = [(os.path.join(image_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        print(f"Attempting to load COVID19_Radiography from: {base_dataset_folder}")
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path):
                print(f"Warning: Image path {sub_image_path} not found for {sub_name}. Skipping this class.")
                continue
            if not os.path.exists(sub_mask_path):
                print(f"Warning: Mask path {sub_mask_path} not found for {sub_name}. Skipping this class. Please ensure masks are present if you intend to train/validate segmentation.")
                continue
            
            img_names = [os.path.splitext(f)[0] for f in os.listdir(sub_image_path) if f.lower().endswith(('.png', '.jpg'))]
            for img_name in img_names:
                original_img_path_png = os.path.join(sub_image_path, img_name + '.png')
                original_img_path_jpg = os.path.join(sub_image_path, img_name + '.jpg')
                
                if os.path.exists(original_img_path_png):
                    image_full_path = original_img_path_png
                elif os.path.exists(original_img_path_jpg):
                    image_full_path = original_img_path_jpg
                else:
                    continue 
                    
                mask_full_path = os.path.join(sub_mask_path, img_name + '.png') 
                
                if os.path.exists(mask_full_path):
                    img_list.append((image_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask {mask_full_path} not found for image {image_full_path}. Skipping image.")

    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'original')
        mask_path = os.path.join(root, 'ground truth')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: {image_path}, {mask_path}. Please ensure manual copy is correct. Returning empty dataset.")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.tif')]
        img_list = [(os.path.join(image_path, img_name + '.tif'), os.path.join(mask_path, img_name + '.tif')) for img_name in img_names]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    if not img_list:
        print(f"No images found for dataset: {dataset_name} at root: {root}")

    return img_list


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, joint_transform=None, transform=None, target_transform=None, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.label_mapping = {val: 1 if val > 0 else 0 for val in range(-1, 31)} 

        # Gather ALL image-mask pairs for programmatic splitting
        # For TSRS_RSNA-Epiphysis, make_dataset already returns a split specific list
        if 'TSRS_RSNA-Epiphysis' in dataset_name: # Handle predefined split datasets
            self.imgs = make_dataset(root, dataset_name)
        else: # For datasets that need programmatic splitting
            all_imgs = make_dataset(root, dataset_name)
            
            # Programmatic 80/10/10 split for JSRT, COVID19_Radiography, CVC-ClinicDB
            random.seed(42) # For reproducibility
            random.shuffle(all_imgs)
            
            total_size = len(all_imgs)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            # Test size is the rest
            
            if split == 'train':
                self.imgs = all_imgs[:train_size]
            elif split == 'val':
                self.imgs = all_imgs[train_size : train_size + val_size]
            elif split == 'test':
                self.imgs = all_imgs[train_size + val_size :]
            else:
                raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty.")

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample) 
        else: # split == "val" or "test"
            sample = self.transform_val(sample)
            sample['name'] = self.imgs[index] 
            return sample
        
    def convert_label(self, label):
        label_array = np.array(label)
        if label_array.ndim == 3 and label_array.shape[2] >= 1:
            label_array = label_array[:, :, 0] 
        elif label_array.ndim == 2:
            pass 
        else:
            print(f"Warning: Unexpected label array dimensions: {label_array.shape}. Assuming single channel.")
            
        label_index = np.zeros_like(label_array, dtype='uint8')
        label_index[label_array > 0] = 1 
        
        label_index = Image.fromarray(label_index, mode='P')
        return label_index

    def __len__(self):
        return len(self.imgs)
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(576,896), 
            tr.RandomCrop((576,576)), 
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
 
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(576,896), 
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)