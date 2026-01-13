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

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')

CITYSCAPES_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 
    6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 
    13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 
    26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 
    32: 17, 33: 18, -1: 255
}

def encode_cityscapes_label(mask_np):
    label_mask = np.zeros_like(mask_np, dtype=np.uint8)
    for k in CITYSCAPES_MAPPING:
        label_mask[mask_np == k] = CITYSCAPES_MAPPING[k]
    return label_mask

def make_dataset(root, dataset_name, split='train', val_size=0.1, test_size=0.1, random_state=42):
    dataset_config = config.DATASET_CONFIG[dataset_name]
    structure = dataset_config.get('structure')
    all_pairs = []

    if structure == 'PRE_SPLIT':
        split_root = os.path.join(root, split)
        mask_dir = os.path.join(root, f"{split}_labels")
        if not (os.path.exists(split_root) and os.path.exists(mask_dir)):
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
            image_dir = os.path.join(root, 'cxr')
            mask_dir = os.path.join(root, 'masks')
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.png')]
                for name in img_names:
                    all_pairs.append((os.path.join(image_dir, name + '.png'), os.path.join(mask_dir, name + '.png')))
        elif dataset_name == 'CVC-ClinicDB':
             image_dir = os.path.join(root, 'Original')
             mask_dir = os.path.join(root, 'Ground Truth')
             if os.path.exists(image_dir) and os.path.exists(mask_dir):
                  img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
                  for name in img_names:
                      all_pairs.append((os.path.join(image_dir, name + '.tif'), os.path.join(mask_dir, name + '.tif')))
        elif dataset_name == 'MontgomeryCounty':
            image_dir = os.path.join(root, 'CXR_png')
            left_mask_dir = os.path.join(root, 'ManualMask', 'leftMask')
            right_mask_dir = os.path.join(root, 'ManualMask', 'rightMask')
            if all(os.path.exists(d) for d in [image_dir, left_mask_dir, right_mask_dir]):
                for img_file in os.listdir(image_dir):
                    if img_file.lower().endswith('.png'):
                        img_path = os.path.join(image_dir, img_file)
                        left_mask_path = os.path.join(left_mask_dir, img_file)
                        right_mask_path = os.path.join(right_mask_dir, img_file)
                        if os.path.exists(left_mask_path) and os.path.exists(right_mask_path):
                            all_pairs.append((img_path, left_mask_path, right_mask_path))

    elif structure == 'VOC':
        img_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        target_split = 'trainval' if split == 'all' else split
        split_map = {'train': 'train.txt', 'val': 'val.txt', 'trainval': 'trainval.txt', 'test': 'val.txt'}
        txt_name = split_map.get(target_split, 'train.txt')
        txt_path = os.path.join(root, 'ImageSets', 'Segmentation', txt_name)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
            for name in file_names:
                img_path = os.path.join(img_dir, name + '.jpg')
                mask_path = os.path.join(mask_dir, name + '.png')
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    all_pairs.append((img_path, mask_path))
        return all_pairs

    elif structure == 'Cityscapes':
        # 1. Try Standard Split Folders
        if split == 'all': splits_to_check = ['train', 'val']
        elif split == 'test': splits_to_check = ['val']
        else: splits_to_check = [split]
        
        found_standard = False
        img_root = os.path.join(root, 'leftImg8bit')
        mask_root = os.path.join(root, 'gtFine')

        if os.path.exists(img_root) and os.path.exists(mask_root):
            for s in splits_to_check:
                img_dir = os.path.join(img_root, s)
                mask_dir = os.path.join(mask_root, s)
                if not os.path.exists(img_dir): continue
                for city in os.listdir(img_dir):
                    c_img_dir = os.path.join(img_dir, city)
                    c_mask_dir = os.path.join(mask_dir, city)
                    if not os.path.isdir(c_img_dir): continue
                    for f in os.listdir(c_img_dir):
                        if f.endswith('_leftImg8bit.png'):
                            img_path = os.path.join(c_img_dir, f)
                            base = f.replace('_leftImg8bit.png', '')
                            mask_name = base + '_gtFine_labelIds.png' 
                            mask_path = os.path.join(c_mask_dir, mask_name)
                            if os.path.exists(mask_path):
                                all_pairs.append((img_path, mask_path))
                                found_standard = True
                            else:
                                mask_name_alt = base + '_gtFine_labelTrainIds.png'
                                mask_path_alt = os.path.join(c_mask_dir, mask_name_alt)
                                if os.path.exists(mask_path_alt):
                                    all_pairs.append((img_path, mask_path_alt))
                                    found_standard = True
        
        # 2. Robust Fallback: Recursive Search if standard structure fails
        if not found_standard:
            print(f"Standard Cityscapes structure not found in {root}. Attempting recursive search...")
            for root_dir, _, files in os.walk(root):
                for f in files:
                    if f.endswith('_leftImg8bit.png'):
                        img_path = os.path.join(root_dir, f)
                        # Try to find matching mask in parallel directories
                        base_name = f.replace('_leftImg8bit.png', '')
                        
                        # Guess mask path patterns
                        possible_mask_names = [
                            base_name + '_gtFine_labelIds.png',
                            base_name + '_gtFine_labelTrainIds.png'
                        ]
                        
                        # Look in current folder and typical mask folders
                        search_dirs = [
                            root_dir, 
                            root_dir.replace('leftImg8bit', 'gtFine'),
                            root_dir.replace('images', 'masks'),
                            root_dir.replace('imgs', 'labels')
                        ]
                        
                        mask_found = False
                        for d in search_dirs:
                            if not os.path.exists(d): continue
                            for m_name in possible_mask_names:
                                m_path = os.path.join(d, m_name)
                                if os.path.exists(m_path):
                                    all_pairs.append((img_path, m_path))
                                    mask_found = True
                                    break
                            if mask_found: break
                            
        return all_pairs

    elif structure == 'COCO-JSON':
        if COCO is None: raise ImportError("Please install pycocotools")
        img_root = dataset_config.get('img_root', root)
        ann_root = dataset_config.get('ann_root', os.path.join(root, 'annotations'))
        if split == 'all': splits_to_check = ['train', 'val']
        else: splits_to_check = [split]
        year = '2017'
        for s in splits_to_check:
            data_name = s + year
            ann_file = os.path.join(ann_root, f'stuff_{data_name}.json')
            img_dir = os.path.join(img_root, data_name)
            if os.path.exists(ann_file) and os.path.exists(img_dir):
                try:
                    coco_temp = COCO(ann_file)
                    img_ids = coco_temp.getImgIds()
                    for img_id in img_ids:
                        img_info = coco_temp.loadImgs(img_id)[0]
                        file_name = img_info['file_name']
                        full_img_path = os.path.join(img_dir, file_name)
                        all_pairs.append((full_img_path, (img_id, ann_file)))
                except Exception as e:
                    print(f"Error loading COCO {s}: {e}")
        return all_pairs

    if not all_pairs:
        return []

    if split == 'all': return all_pairs

    train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=random_state)
    val_proportion = val_size / (1 - test_size)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_proportion, random_state=random_state)

    if split == 'train': return train_pairs
    elif split == 'val': return val_pairs
    elif split == 'test': return test_pairs
    else: return []

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, split='train', imgs=None):
        self.split = split
        self.dataset_name = dataset_name
        self.coco_objects = {}
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = make_dataset(root, dataset_name, split=split)
        if not self.imgs:
            print(f"Warning: The '{self.split}' split for '{dataset_name}' is empty.")

    def __getitem__(self, index):
        try:
            if self.dataset_name == 'MontgomeryCounty':
                img_path, left_path, right_path = self.imgs[index]
                left_mask_np = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
                right_mask_np = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
                if left_mask_np is None or right_mask_np is None:
                    raise FileNotFoundError(f"Could not read mask for {img_path}")
                mask_np = cv2.bitwise_or(left_mask_np, right_mask_np)

            elif self.dataset_name == 'COCO-Stuff':
                img_path, (img_id, ann_file_path) = self.imgs[index]
                if ann_file_path not in self.coco_objects:
                    self.coco_objects[ann_file_path] = COCO(ann_file_path)
                coco = self.coco_objects[ann_file_path]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                img_np = cv2.imread(img_path)
                if img_np is None: raise FileNotFoundError(f"Image not found: {img_path}")
                h, w = img_np.shape[:2]
                mask_np = np.zeros((h, w), dtype=np.uint8)
                for ann in anns:
                    mask_np[coco.annToMask(ann) > 0] = ann['category_id']

            elif self.dataset_name == 'Cityscapes':
                img_path, gt_path = self.imgs[index]
                mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if mask_np is None: 
                    mask_pil = Image.open(gt_path)
                    mask_np = np.array(mask_pil)
                if 'labelIds' in gt_path and 'labelTrainIds' not in gt_path:
                    mask_np = encode_cityscapes_label(mask_np)

            else:
                img_path, gt_path = self.imgs[index]
                mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                     mask_pil = Image.open(gt_path)
                     mask_np = np.array(mask_pil)

            img_np = cv2.imread(img_path)
            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: {img_path}")
            if mask_np is None: raise FileNotFoundError(f"Mask not found/readable: {self.imgs[index]}")

            if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif img_np.ndim == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            img = Image.fromarray(img_np)
            if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
            
            if self.dataset_name in ['VOC2012', 'COCO-Stuff', 'Cityscapes']:
                target = Image.fromarray(mask_np, mode='P')
            else:
                target = Image.fromarray(mask_np, mode='L')
                
            label = self.convert_label(target)

        except Exception as e:
            print(f"ERROR processing item at index {index}. Error: {e}")
            return None

        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)
            sample['name'] = img_path
            return sample

    def convert_label(self, label):
        label_array = np.array(label)
        if label_array.ndim == 3: label_array = label_array[:, :, 0]
        
        if self.dataset_name in ['VOC2012', 'COCO-Stuff', 'Cityscapes']:
            return Image.fromarray(label_array, mode='P')
        else:
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