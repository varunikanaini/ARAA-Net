#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
from PIL import Image
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
        if not (os.path.exists(split_root) and os.path.exists(mask_dir)): return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(split_root) if f.lower().endswith('.jpg')]
        for name in img_names:
            p1 = os.path.join(split_root, name + '.jpg')
            p2 = os.path.join(mask_dir, name + '.png')
            if os.path.exists(p1) and os.path.exists(p2): all_pairs.append((p1, p2))
        return all_pairs

    elif structure == 'FLAT_SPLIT':
        if dataset_name == 'JSRT':
            image_dir, mask_dir = os.path.join(root, 'cxr'), os.path.join(root, 'masks')
            ext = '.png'
        elif dataset_name == 'CVC-ClinicDB':
            image_dir, mask_dir = os.path.join(root, 'Original'), os.path.join(root, 'Ground Truth')
            ext = '.tif'
        elif dataset_name == 'MontgomeryCounty':
            image_dir = os.path.join(root, 'CXR_png')
            left = os.path.join(root, 'ManualMask', 'leftMask')
            right = os.path.join(root, 'ManualMask', 'rightMask')
            if os.path.exists(image_dir) and os.path.exists(left):
                for f in os.listdir(image_dir):
                    if f.endswith('.png'):
                        all_pairs.append((os.path.join(image_dir, f), os.path.join(left, f), os.path.join(right, f)))
            return all_pairs

        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            for f in os.listdir(image_dir):
                if f.lower().endswith(ext):
                    all_pairs.append((os.path.join(image_dir, f), os.path.join(mask_dir, f)))

    elif structure == 'VOC':
        img_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        target_split = 'trainval' if split == 'all' else split
        split_map = {'train': 'train.txt', 'val': 'val.txt', 'trainval': 'trainval.txt', 'test': 'val.txt'}
        txt_path = os.path.join(root, 'ImageSets', 'Segmentation', split_map.get(target_split, 'train.txt'))
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                names = [x.strip() for x in f.readlines()]
            for name in names:
                p1 = os.path.join(img_dir, name + '.jpg')
                p2 = os.path.join(mask_dir, name + '.png')
                if os.path.exists(p1) and os.path.exists(p2): all_pairs.append((p1, p2))
        return all_pairs

    elif structure == 'Cityscapes':
        # Recursive robust search
        for root_dir, _, files in os.walk(root):
            for f in files:
                if f.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(root_dir, f)
                    base = f.replace('_leftImg8bit.png', '')
                    # Filter by split if needed
                    if split != 'all' and split not in img_path: continue
                    
                    search_dirs = [
                        root_dir.replace('leftImg8bit', 'gtFine'),
                        root_dir.replace('images', 'masks'),
                        root_dir
                    ]
                    for d in search_dirs:
                        if not os.path.exists(d): continue
                        # Try standard labelIds (preferred) then TrainIds
                        m1 = os.path.join(d, base + '_gtFine_labelIds.png')
                        m2 = os.path.join(d, base + '_gtFine_labelTrainIds.png')
                        if os.path.exists(m1):
                            all_pairs.append((img_path, m1)); break
                        elif os.path.exists(m2):
                            all_pairs.append((img_path, m2)); break
        return all_pairs

    elif structure == 'COCO-JSON':
        if COCO is None: raise ImportError("Please install pycocotools")
        img_root = dataset_config.get('img_root', root)
        ann_root = dataset_config.get('ann_root', os.path.join(root, 'annotations'))
        splits = ['train', 'val'] if split == 'all' else [split]
        for s in splits:
            data_name = s + '2017'
            ann_file = os.path.join(ann_root, f'stuff_{data_name}.json')
            img_dir = os.path.join(img_root, data_name)
            if os.path.exists(ann_file) and os.path.exists(img_dir):
                try:
                    coco = COCO(ann_file)
                    for img_id in coco.getImgIds():
                        fname = coco.loadImgs(img_id)[0]['file_name']
                        all_pairs.append((os.path.join(img_dir, fname), (img_id, ann_file)))
                except: pass
        return all_pairs

    if not all_pairs: return []
    if split == 'all': return all_pairs
    
    train_val, test = train_test_split(all_pairs, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
    
    if split == 'train': return train
    elif split == 'val': return val
    return test

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
            print(f"Warning: Empty dataset for {dataset_name} ({split})")

    def __getitem__(self, index):
        try:
            if self.dataset_name == 'MontgomeryCounty':
                img_path, l_path, r_path = self.imgs[index]
                l_mask = cv2.imread(l_path, 0)
                r_mask = cv2.imread(r_path, 0)
                mask_np = cv2.bitwise_or(l_mask, r_mask)
            
            elif self.dataset_name == 'COCO-Stuff':
                img_path, (img_id, ann_file) = self.imgs[index]
                if ann_file not in self.coco_objects: self.coco_objects[ann_file] = COCO(ann_file)
                coco = self.coco_objects[ann_file]
                img_np = cv2.imread(img_path)
                if img_np is None: raise FileNotFoundError
                h, w = img_np.shape[:2]
                mask_np = np.zeros((h, w), dtype=np.uint8)
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                    mask_np[coco.annToMask(ann) > 0] = ann['category_id']
            
            else:
                img_path, mask_path = self.imgs[index]
                img_np = cv2.imread(img_path)
                mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if self.dataset_name == 'Cityscapes':
                    if 'labelIds' in mask_path and 'labelTrainIds' not in mask_path:
                        mask_np = encode_cityscapes_label(mask_np)

            if img_np is None: raise FileNotFoundError(f"Img: {img_path}")
            if mask_np is None: mask_np = np.array(Image.open(mask_path))

            if img_np.ndim == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(img_np)
            mode = 'P' if self.dataset_name in ['VOC2012', 'Cityscapes', 'COCO-Stuff'] else 'L'
            target = Image.fromarray(mask_np, mode=mode)
            label = self.convert_label(target)

        except Exception as e:
            print(f"Error loading {index}: {e}")
            return None

        sample = {'image': img, 'label': label}
        if self.split == 'train': return self.transform_tr(sample)
        sample = self.transform_val(sample)
        sample['name'] = img_path
        return sample

    def convert_label(self, label):
        label_np = np.array(label)
        if label_np.ndim == 3: label_np = label_np[:, :, 0]
        
        if self.dataset_name not in ['VOC2012', 'Cityscapes', 'COCO-Stuff']:
            label_index = np.zeros_like(label_np, dtype='uint8')
            label_index[label_np > 0] = 1
            return Image.fromarray(label_index, mode='P')
        
        return Image.fromarray(label_np.astype('uint8'), mode='P')

    def __len__(self): return len(self.imgs)

    def transform_tr(self, sample):
        return transforms.Compose([
            tr.RandomHorizontalFlip(), tr.FixedResize(576,896), tr.RandomCrop((576,576)),
            tr.RandomGaussianBlur(), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])(sample)

    def transform_val(self, sample):
        return transforms.Compose([
            tr.FixedResize(576,896), tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])(sample)