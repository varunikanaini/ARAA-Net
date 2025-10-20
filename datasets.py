# /kaggle/working/ARAA-Net/datasets.py

import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2 # Using OpenCV for more robust image reading

import custom_transforms as tr # Import your custom transforms module
import config

# --- IMPORT IMAGE_EXTENSIONS and MASK_EXTENSIONS ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(root, dataset_name):
    dataset_items = []

    if 'TSRS_RSNA' in dataset_name:
        image_path = root
        
        if os.path.exists(os.path.join(root, 'GT')):
            mask_path = os.path.join(root, 'GT')
        elif os.path.exists(root + '_labels'):
            mask_path = root + '_labels'
        else:
            raise FileNotFoundError(f"Could not find label directory for {root}. Looked in '{os.path.join(root, 'GT')}' and '{root + '_labels'}'.")

        img_list = []
        for f in os.listdir(image_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_list.append(os.path.splitext(f)[0])

        dataset_items = []
        for img_name in img_list:
            img_full_path = os.path.join(image_path, img_name + '.jpg')
            mask_full_path = os.path.join(mask_path, img_name + '.png')
            
            if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                dataset_items.append((img_full_path, mask_full_path))
            else:
                print(f"Warning: Missing image or mask for {img_name}. Skipping.")

        if not dataset_items:
            raise RuntimeError(f"Found 0 images in {root} with corresponding labels. Please check dataset path and file extensions.")
            
    elif dataset_name == 'JSRT':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'masks')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path): return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))
    
    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Database')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path): continue
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path): return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'DentalPanoramic':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path): return []
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'SixDiseasesChestXRay':
        base_split_folder = root 
        if not os.path.isdir(base_split_folder): return []
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_split_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_split_folder, sub_name, 'masks')
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path): continue
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}.")
    
    if not dataset_items and (dataset_name != 'PLACEHOLDER_FOR_DYNAMIC_SELECTION'):
        print(f"Warning: Found 0 items for dataset '{dataset_name}' at root '{root}'. Please check dataset path, file extensions, and directory structure.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train', kfold_mode=False):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args
        self.kfold_mode = kfold_mode

        self.imgs = make_dataset(self.root, self.dataset_name) 
        
        if not self.imgs:
            print(f"Warning: No images found for dataset '{self.dataset_name}', split '{self.split}' at root '{self.root}'.")
            self.imgs = []
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if dataset_name in ['JSRT', 'COVID19_Radiography']: 
            self.mean = [0.5]
            self.std = [0.5]

        # --- MODIFIED TRANSFORMS FOR TRAINING ---
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=args.min_lesion_area_pixels,
                                       expansion_factor=args.expansion_factor,
                                       min_bbox_size=(args.min_bbox_h, args.min_bbox_w)) if args.min_lesion_area_pixels > 0 else lambda x: x,
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((args.scale_h, args.scale_w)),
                # --- MODIFIED ELASTIC TRANSFORM ---
                tr.ElasticTransform(alpha=50, sigma=7, p=0.7),
                tr.RandomGaussianBlur(),
                # --- MODIFIED COLOR JITTER ---
                tr.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15) if hasattr(tr, 'ColorJitter') else lambda x: x,
                # --- MODIFIED RANDOM AFFINE ---
                tr.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=7, mask_fill_value=0) if hasattr(tr, 'RandomAffine') else lambda x: x,
                tr.RandomCutout(num_holes_range=(1, 4), max_h_size=48, max_w_size=48, fill_value=0, p=0.6) if hasattr(tr, 'RandomCutout') else lambda x: x,
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])
        else: # for val/test splits
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")

            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            elif img_cv.ndim == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

            img = Image.fromarray(img_cv)

            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            target = Image.fromarray(mask_cv, mode='L')
            label = self.convert_label(target)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None

        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)

        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)

        return transformed_sample

    def convert_label(self, label):
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1

        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)