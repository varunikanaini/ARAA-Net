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


def make_dataset(root, dataset_name):
    img_list = []
    
    if dataset_name == 'TSRS_RSNA-Epiphysis_train' or dataset_name == 'TSRS_RSNA-Epiphysis_test':
        image_path = root
        mask_path = root + '_labels'
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: {dataset_name} paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
        img_list = [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'JSRT':
        # Correct JSRT structure: root/content/jsrt/cxr for images, root/content/jsrt/masks for masks
        image_path = os.path.join(root, 'content', 'jsrt', 'cxr')
        mask_path = os.path.join(root, 'content', 'jsrt', 'masks')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Did the download complete and extract correctly? Returning empty dataset.")
            return []
        # JSRT images are .png, masks are .png
        img_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.lower().endswith('.png')]
        img_list = [(os.path.join(image_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_names]
    elif dataset_name == 'COVID19_Radiography':
        # Correct COVID19_Radiography structure: root/COVID-19_Radiography_Dataset/[CLASS_NAME]/images and masks
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
            
            # Assuming images can be .png or .jpg, and masks are .png
            img_names = [os.path.splitext(f)[0] for f in os.listdir(sub_image_path) if f.lower().endswith(('.png', '.jpg'))]
            for img_name in img_names:
                original_img_path_png = os.path.join(sub_image_path, img_name + '.png')
                original_img_path_jpg = os.path.join(sub_image_path, img_name + '.jpg')
                
                if os.path.exists(original_img_path_png):
                    image_full_path = original_img_path_png
                elif os.path.exists(original_img_path_jpg):
                    image_full_path = original_img_path_jpg
                else:
                    continue # Skip if neither png nor jpg found
                    
                mask_full_path = os.path.join(sub_mask_path, img_name + '.png') # Assuming masks are .png
                
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
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    if not img_list:
        print(f"No images found for dataset: {dataset_name} at root: {root}")

    return img_list


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, joint_transform=None, transform=None, target_transform=None, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.imgs = make_dataset(root, dataset_name)
        # joint_transform, transform, target_transform are not used externally anymore
        # as transforms are handled internally by transform_tr/val
        self.joint_transform = joint_transform 
        self.transform = transform
        self.target_transform = target_transform 
        self.split = split
        self.label_mapping = {val: 1 if val > 0 else 0 for val in range(-1, 31)} 

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample) 
        else: # split == "test" or "val"
            sample = self.transform_val(sample)
            sample['name'] = self.imgs[index] 
            return sample
        
    def convert_label(self, label):
        label_array = np.array(label)
        if label_array.ndim == 3 and label_array.shape[2] >= 1:
            # Assume first channel is the relevant one for segmentation if RGB/RGBA
            label_array = label_array[:, :, 0] 
        elif label_array.ndim == 2:
            pass # Already single channel
        else:
            print(f"Warning: Unexpected label array dimensions: {label_array.shape}. Assuming single channel.")
            # Attempt to flatten or convert to single channel if needed, or raise error.
            # For now, let's assume it's correctly handled or an error will occur downstream.
            
        label_index = np.zeros_like(label_array, dtype='uint8')
        label_index[label_array > 0] = 1 # Any non-zero pixel is foreground
        
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