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


from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
# from torchvision.datasets.folder import is_image_file # Not used


def make_dataset(root, dataset_name):
    img_list = []
    
    if dataset_name == 'TSRS_RSNA-Epiphysis_train': # Original training dataset
        image_path = root
        mask_path = root + '_labels'
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: TSRS_RSNA-Epiphysis_train paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return [] # Return empty if paths don't exist
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
        img_list = [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'TSRS_RSNA-Epiphysis_test': # Original testing dataset
        image_path = root
        mask_path = root + '_labels'
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: TSRS_RSNA-Epiphysis_test paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return [] # Return empty if paths don't exist
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
        img_list = [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'JSRT':
        # Assuming downloaded content extracts to jsrt-247-image-lung-segmentation-mask-dataset/images and masks
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'masks')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Did the download complete and extract correctly? Returning empty dataset.")
            return []
        # JSRT images are .png, masks are .png
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.png')]
        img_list = [(os.path.join(image_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'COVID19_Radiography':
        # This dataset is primarily for classification. User is assumed to have provided/generated masks.
        # We'll look for subfolders like COVID, NORMAL, etc., each containing 'images' and 'masks'.
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia'] # Common subfolders
        print(f"Attempting to load COVID19_Radiography from: {root}")
        for sub_name in subfolders:
            sub_image_path = os.path.join(root, sub_name, 'images') # Assuming images are here
            sub_mask_path = os.path.join(root, sub_name, 'masks')   # User is expected to put masks here (e.g., generated masks)
            
            if not os.path.exists(sub_image_path):
                print(f"Warning: Image path {sub_image_path} not found for {sub_name}. Skipping this class.")
                continue
            if not os.path.exists(sub_mask_path):
                print(f"Warning: Mask path {sub_mask_path} not found for {sub_name}. Skipping this class.")
                continue
            
            # Assuming images can be .png or .jpg, and masks are .png
            img_names = [os.path.splitext(f)[0] for f in os.listdir(sub_image_path) if f.lower().endswith(('.png', '.jpg'))]
            for img_name in img_names:
                original_ext = '.png' if os.path.exists(os.path.join(sub_image_path, img_name + '.png')) else '.jpg'
                img_list.append((os.path.join(sub_image_path, img_name + original_ext), 
                                 os.path.join(sub_mask_path, img_name + '.png'))) # Assuming masks are .png
    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'original')
        mask_path = os.path.join(root, 'ground truth')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: {image_path}, {mask_path}. Please ensure manual copy is correct. Returning empty dataset.")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.tif')] # CVC images are often .tif
        img_list = [(os.path.join(image_path, img_name + '.tif'), os.path.join(mask_path, img_name + '.tif')) for img_name in img_names]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    if not img_list:
        print(f"No images found for dataset: {dataset_name} at root: {root}")

    return img_list


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, dataset_name, joint_transform=None, transform=None, target_transform=None, split='train'):
        self.root = root
        self.dataset_name = dataset_name # Store dataset name
        self.imgs = make_dataset(root, dataset_name)
        self.joint_transform = joint_transform # This will be None as per plan
        self.transform = transform # This will be None as per plan
        self.target_transform = target_transform # This will be None as per plan
        self.split = split
        # Only map foreground to 1 and background to 0 for binary segmentation
        self.label_mapping = {val: 1 if val > 0 else 0 for val in range(-1, 31)} 

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # For CVC-ClinicDB and JSRT masks can be multi-channel or single channel,
        # but convert_label expects a PIL Image, which is fine.
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample) 
        else: # split == "test" or "val"
            sample = self.transform_val(sample)
            # For test, we return the original full path to save results with correct names
            sample['name'] = self.imgs[index] 
            return sample
        
    def convert_label(self, label):
        # This function assumes binary segmentation where any non-zero pixel in the mask is foreground (1)
        # and zero is background (0).
        label_array = np.array(label)
        if label_array.ndim == 3: # If mask is RGB or RGBA, take one channel or convert to grayscale first
            label_array = label_array[:,:,0] # Take the first channel
        
        label_index = np.full(label_array.shape, 0, dtype='uint8')
        # Apply the binary mapping based on self.label_mapping
        # Any value in the mask that is > 0 in the original label_mapping (which covers 1-30) will be 1.
        # Otherwise, it's 0.
        # If the original mask values are just 0/255, this will make 255 -> 1.
        label_index[label_array > 0] = 1 # Assuming non-zero pixels are foreground
        
        label_index = Image.fromarray(label_index, mode='P') # 'P' for palettized (single channel)
        return label_index

    def __len__(self):
        return len(self.imgs)
    
    # These transforms directly implement the paper's requirements
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(576,896), # Scale to 576x896
            tr.RandomCrop((576,576)), # Randomly crop to 576x576 during training
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
 
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(576,896), # Scale to 576x896 for validation as well
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)