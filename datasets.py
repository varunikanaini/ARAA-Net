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
import cv2
from PIL import Image
import numpy as np



import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
from torchvision.datasets.folder import is_image_file


def make_dataset(root):
    mask_path = os.path.join(root, 'GT') # Ensure GT path is correctly formed
    image_path = root # Assume images are directly in root
    
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, split='train', scale_h=896, scale_w=576, crop_size=576):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        # Store scale and crop parameters
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.crop_size = crop_size

        self.label_mapping = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6:6, 7:7, 8:8, 9:9, 10:10,
                              11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16:16, 17:17, 18:18, 19:19, 20:20,
                              21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26:26, 27:27, 28:28, 29:29, 30:30}

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample) 
        else: # split == "val" or "test" (using transform_val for both for consistency)
            sample = self.transform_val(sample)
            sample['name'] = self.imgs[index]
            return sample
        
    def convert_label(self, label):
        label_rgb = np.array(label)
        label_index = np.full(label_rgb.shape[:2], 0, dtype='uint8')
        label_index[label_rgb > 0] = 1 
        label_index = Image.fromarray(label_index, mode='P')

        return label_index

    def __len__(self):
        return len(self.imgs)
    
    def transform_tr(self, sample):
        # Apply custom transforms sequentially. Each custom transform expects
        # a dictionary and returns a dictionary.
        sample = tr.RandomHorizontalFlip()(sample)
        sample = tr.FixedResize(self.scale_w, self.scale_h)(sample) # Use passed scale_w, scale_h
        sample = tr.RandomCrop((self.crop_size, self.crop_size))(sample) # Use passed crop_size
        sample = tr.RandomGaussianBlur()(sample)
        sample = tr.RandomGaussianNoise(mean=0., std=0.03, p=0.5)(sample)

        # Handle torchvision.transforms.ColorJitter explicitly:
        # It expects a PIL Image, so apply it directly to sample['image']
        img = sample['image']
        color_jitter_transform = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        img = color_jitter_transform(img)
        sample['image'] = img # Update the image in the dictionary

        # Now apply the final transforms which are designed to handle the dictionary
        composed_final_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        
        return composed_final_transforms(sample)
 
    def transform_val(self, sample):
        # For validation, apply FixedResize and then Normalize/ToTensor
        sample = tr.FixedResize(self.scale_w, self.scale_h)(sample) # Use passed scale_w, scale_h
        
        composed_final_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_final_transforms(sample)