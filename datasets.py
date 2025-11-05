#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torchvision import transforms
import custom_transforms as tr
import config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')

def make_dataset(root, dataset_name, split='train', val_size=0.1, test_size=0.1, random_state=42):
    """
    Creates a list of (image_path, mask_path) tuples for a given dataset and split.
    This function now correctly handles both pre-split and flat datasets.
    """
    dataset_config = config.DATASET_CONFIG[dataset_name]
    structure = dataset_config.get('structure')
    all_pairs = []

    if structure == 'PRE_SPLIT':
        # ... (This logic is unchanged and correct for TSRS_RSNA)
        split_root = os.path.join(root, split)
        mask_dir = os.path.join(root, f"{split}_labels")
        if not (os.path.exists(split_root) and os.path.exists(mask_dir)):
            print(f"Warning: Directory not found for pre-split dataset: {split_root} or {mask_dir}")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(split_root) if f.lower().endswith('.jpg')]
        for name in img_names:
            img_path = os.path.join(split_root, name + '.jpg')
            mask_path = os.path.join(mask_dir, name + '.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))
        return all_pairs

    elif structure == 'FLAT_SPLIT':
        if dataset_name == 'JSRT':
            # ... (This logic is unchanged and correct for JSRT)
            image_dir = os.path.join(root, 'cxr')
            mask_dir = os.path.join(root, 'masks')
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.png')]
                for name in img_names:
                    all_pairs.append((os.path.join(image_dir, name + '.png'), os.path.join(mask_dir, name + '.png')))

        elif dataset_name == 'CVC-ClinicDB':
             # ... (This logic is unchanged and correct for CVC-ClinicDB)
             image_dir = os.path.join(root, 'Original')
             mask_dir = os.path.join(root, 'Ground Truth')
             if os.path.exists(image_dir) and os.path.exists(mask_dir):
                  img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
                  for name in img_names:
                     all_pairs.append((os.path.join(image_dir, name + '.tif'), os.path.join(mask_dir, name + '.tif')))

        # --- NEW LOGIC FOR MONTGOMERY COUNTY ---
        elif dataset_name == 'MontgomeryCounty':
            image_dir = os.path.join(root, 'CXR_png')
            left_mask_dir = os.path.join(root, 'ManualMask', 'leftMask')
            right_mask_dir = os.path.join(root, 'ManualMask', 'rightMask')
            if all(os.path.exists(d) for d in [image_dir, left_mask_dir, right_mask_dir]):
                for img_file in os.listdir(image_dir):
                    if img_file.lower().endswith('.png'):
                        base_name = os.path.splitext(img_file)[0]
                        img_path = os.path.join(image_dir, img_file)
                        # NOTE: The mask filenames in the dataset do NOT have a `_mask` suffix.
                        left_mask_path = os.path.join(left_mask_dir, img_file)
                        right_mask_path = os.path.join(right_mask_dir, img_file)
                        
                        if os.path.exists(left_mask_path) and os.path.exists(right_mask_path):
                            # Store a 3-element tuple for this specific dataset
                            all_pairs.append((img_path, left_mask_path, right_mask_path))
        # ----------------------------------------

        if not all_pairs:
            print(f"Warning: Found 0 image-mask pairs for '{dataset_name}' at root '{root}'. Check paths.")
            return []

        train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=random_state)
        val_proportion = val_size / (1 - test_size)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_proportion, random_state=random_state)

        if split == 'train': return train_pairs
        elif split == 'val': return val_pairs
        elif split == 'test': return test_pairs
        elif split == 'all': return all_pairs
        else: return []
    else:
        raise ValueError(f"Unknown dataset structure '{structure}' for dataset '{dataset_name}'.")

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, split='train', imgs=None):
        self.split = split
        self.dataset_name = dataset_name # Store dataset_name
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = make_dataset(root, dataset_name, split=split)

        if not self.imgs:
            print(f"Warning: The '{self.split}' split for '{dataset_name}' is empty.")

    def __getitem__(self, index):
        try:
            # --- NEW LOGIC TO HANDLE MONTGOMERY'S SPLIT MASKS ---
            if self.dataset_name == 'MontgomeryCounty':
                img_path, left_path, right_path = self.imgs[index]
                left_mask_np = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
                right_mask_np = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
                if left_mask_np is None or right_mask_np is None:
                    raise FileNotFoundError(f"Could not read one or both masks for {img_path}")
                # Combine the two masks into a single mask
                mask_np = cv2.bitwise_or(left_mask_np, right_mask_np)
            else: # Standard logic for all other datasets
                img_path, gt_path = self.imgs[index]
                mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # ----------------------------------------------------

            img_np = cv2.imread(img_path)
            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: {img_path}")
            if mask_np is None: raise FileNotFoundError(f"OpenCV could not read mask for {img_path}")

            if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_np)

            if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
            target = Image.fromarray(mask_np, mode='L')
            label = self.convert_label(target)

        except Exception as e:
            print(f"ERROR processing item at index {index} ({img_path}). Skipping. Error: {e}")
            return None

        sample = {'image': img, 'label': label}

        if self.split == "train":
            return self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)
            # --- UNIVERSAL NAME HANDLING ---
            # Always pass just the image path for consistency
            sample['name'] = img_path
            return sample

    def convert_label(self, label):
        # ... (This function is unchanged and correct)
        label_array = np.array(label)
        if label_array.ndim == 3: label_array = label_array[:, :, 0]
        label_index = np.zeros_like(label_array, dtype='uint8')
        label_index[label_array > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self): return len(self.imgs)

    def transform_tr(self, sample):
        # ... (This function is unchanged)
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(), tr.FixedResize(576,896), tr.RandomCrop((576,576)),
            tr.RandomGaussianBlur(), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        # ... (This function is unchanged)
        composed_transforms = transforms.Compose([
            tr.FixedResize(576,896), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)