import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
import logging

import custom_transforms as tr
import config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(root, dataset_name):
    dataset_items = []
    logging.debug(f"make_dataset: Processing dataset '{dataset_name}' at root '{root}'")

    if 'TSRS_RSNA' in dataset_name:
        image_path = root
        logging.debug(f"Checking image directory: {image_path}")
        if not os.path.exists(image_path):
            logging.error(f"Image directory does not exist: {image_path}")
            return []

        gt_path = os.path.join(root, 'GT')
        alt_path = root + '_labels'
        if os.path.exists(gt_path):
            mask_path = gt_path
        elif os.path.exists(alt_path):
            mask_path = alt_path
        else:
            logging.error(f"Could not find label directory. Looked in '{gt_path}' and '{alt_path}'")
            return []

        logging.debug(f"Using mask directory: {mask_path}")
        if not os.path.exists(mask_path):
            logging.error(f"Mask directory does not exist: {mask_path}")
            return []

        img_list = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.debug(f"Found {len(img_list)} image files in {image_path}: {img_list[:5] if img_list else []}")

        mask_list = [f for f in os.listdir(mask_path) if f.lower().endswith('.png')]
        logging.debug(f"Found {len(mask_list)} mask files in {mask_path}: {mask_list[:5] if mask_list else []}")

        dataset_items = []
        for img_name in img_list:
            base_name = os.path.splitext(img_name)[0]
            img_full_path = os.path.join(image_path, img_name)
            mask_full_path = os.path.join(mask_path, base_name + '.png')
            
            if os.path.exists(mask_full_path):
                dataset_items.append((img_full_path, mask_full_path))
                logging.debug(f"Paired: {img_full_path} with {mask_full_path}")
            else:
                logging.warning(f"No matching mask for image: {img_name}. Expected: {mask_full_path}")

        if not dataset_items:
            logging.error(f"Found 0 image-label pairs in {root}. Check file extensions and naming consistency.")
            
    elif dataset_name == 'JSRT':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'masks')
        logging.debug(f"JSRT: Checking image_dir={image_path}, mask_dir={mask_path}")
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            logging.error(f"JSRT directories missing: image_dir={image_path}, mask_dir={mask_path}")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                    logging.debug(f"JSRT: Paired {img_full_path} with {mask_full_path}")
                else:
                    logging.warning(f"JSRT: No mask for {img_full_path}")

    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Database')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        logging.debug(f"COVID19_Radiography: Checking base folder {base_dataset_folder}")
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path):
                logging.warning(f"COVID19_Radiography: Skipping subfolder {sub_name}, directories missing")
                continue
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        logging.debug(f"COVID19_Radiography: Paired {img_full_path} with {mask_full_path}")
                    else:
                        logging.warning(f"COVID19_Radiography: No mask for {img_full_path}")

    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        logging.debug(f"CVC-ClinicDB: Checking image_dir={image_path}, mask_dir={mask_path}")
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            logging.error(f"CVC-ClinicDB directories missing: image_dir={image_path}, mask_dir={mask_path}")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                    logging.debug(f"CVC-ClinicDB: Paired {img_full_path} with {mask_full_path}")
                else:
                    logging.warning(f"CVC-ClinicDB: No mask for {img_full_path}")

    elif dataset_name == 'DentalPanoramic':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1')
        logging.debug(f"DentalPanoramic: Checking image_dir={image_path}, mask_dir={mask_path}")
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            logging.error(f"DentalPanoramic directories missing: image_dir={image_path}, mask_dir={mask_path}")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                    logging.debug(f"DentalPanoramic: Paired {img_full_path} with {mask_full_path}")
                else:
                    logging.warning(f"DentalPanoramic: No mask for {img_full_path}")

    elif dataset_name == 'SixDiseasesChestXRay':
        base_split_folder = root 
        logging.debug(f"SixDiseasesChestXRay: Checking base folder {base_split_folder}")
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_split_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_split_folder, sub_name, 'masks')
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path):
                logging.warning(f"SixDiseasesChestXRay: Skipping subfolder {sub_name}, directories missing")
                continue
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        logging.debug(f"SixDiseasesChestXRay: Paired {img_full_path} with {mask_full_path}")
                    else:
                        logging.warning(f"SixDiseasesChestXRay: No mask for {img_full_path}")

    else:
        logging.error(f"Unknown dataset_name: {dataset_name}")
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    logging.info(f"make_dataset: Found {len(dataset_items)} image-label pairs for dataset '{dataset_name}' at root '{root}'")
    if not dataset_items and (dataset_name != 'PLACEHOLDER_FOR_DYNAMIC_SELECTION'):
        logging.warning(f"No items found for dataset '{dataset_name}' at root '{root}'. Check dataset path, file extensions, and directory structure.")
        
    return dataset_items

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train', kfold_mode=False):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args
        self.kfold_mode = kfold_mode

        self.imgs = make_dataset(self.root, self.dataset_name) 
        
        logging.info(f"ImageFolder initialized: dataset={dataset_name}, split={split}, root={root}, found {len(self.imgs)} image-label pairs")
        if not self.imgs:
            logging.warning(f"No images found for dataset '{self.dataset_name}', split '{self.split}' at root '{self.root}'.")
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if dataset_name in ['JSRT', 'COVID19_Radiography']: 
            self.mean = [0.5]
            self.std = [0.5]

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=args.min_lesion_area_pixels,
                                       expansion_factor=args.expansion_factor,
                                       min_bbox_size=(args.min_bbox_h, args.min_bbox_w)) if args.min_lesion_area_pixels > 0 else lambda x: x,
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((args.scale_h, args.scale_w)),
                tr.RandomGaussianBlur(),
                tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if hasattr(tr, 'ColorJitter') else lambda x: x,
                tr.RandomAffine(degrees=7, translate=(0.07, 0.07), scale=(0.95, 1.05), shear=7, mask_fill_value=0) if hasattr(tr, 'RandomAffine') else lambda x: x,
                tr.RandomCutout(num_holes_range=(1, 4), max_h_size=48, max_w_size=48, fill_value=0, p=0.6) if hasattr(tr, 'RandomCutout') else lambda x: x, 
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor() 
            ])
        else:
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
            logging.debug(f"__getitem__[{index}]: img_path={img_path}, label_shape={np.array(label).shape}, label_unique={np.unique(np.array(label))}")

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            logging.error(f"Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None.")
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
        logging.debug(f"convert_label: input_shape={np.array(label).shape}, output_shape={label_index.shape}, unique_values={np.unique(label_index)}")

        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)