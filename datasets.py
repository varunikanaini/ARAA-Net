# datasets.py

import os
import torch
import torch.utils.data as data
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
        self.kfold_mode = kfold_mode # Flag for K-Fold usage

        # Use make_dataset to get image-mask pairs
        self.imgs = make_dataset(self.root, self.dataset_name) 
        
        if not self.imgs:
            print(f"Warning: No images found for dataset '{self.dataset_name}', split '{self.split}' at root '{self.root}'.")
            self.imgs = [] 
        
        # Define mean and std based on dataset, defaulting to ImageNet
        self.mean = (0.485, 0.456, 0.406) # Default ImageNet means
        self.std = (0.229, 0.224, 0.225)  # Default ImageNet stds

        if dataset_name in ['JSRT', 'COVID19_Radiography']: 
            self.mean = [0.5]
            self.std = [0.5]
        # Add more dataset-specific mean/std here if known and different

        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w
        scale_h = args.scale_h
        scale_w = args.scale_w

        # --- Define Transforms ---
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)) if min_lesion_area > 0 else lambda x: x,
                
                # --- BALANCED ENHANCEMENTS & AUGMENTATIONS ---
                # Probabilistic Histogram Equalization (less aggressive than always on)
                tr.HistogramEqualization() if np.random.rand() < 0.5 else lambda x: x, 
                # Probabilistic Wavelet Contrast Enhancement (using params from args)
                tr.WaveletContrastEnhancement(wavelet=args.wavelet_type, level=args.wavelet_level, detail_scale_factor=args.wavelet_detail_scale) if np.random.rand() < 0.2 else lambda x: x, 
                
                # Aggressive Color Jitter (as used in the "99.49 OA" version)
                tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                
                # Conservative Random Affine
                tr.RandomAffine(degrees=7, translate=(0.07, 0.07), scale=(0.95, 1.05), shear=7, mask_fill_value=0),                
                tr.RandomGaussianBlur(radius_range=(0.1, 1.2)),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((scale_h, scale_w)), 
                
                # Moderated Random Cutout (params closer to original, but with slightly higher p)
                tr.RandomCutout(num_holes_range=(1, 4), max_h_size=40, max_w_size=40, fill_value=0, p=0.55), # Slightly higher p=0.55
                
                # Normalization
                tr.Normalize(mean=self.mean, std=self.std), 
                tr.ToTensor()
            ])
        else:  # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        if not self.imgs:
            print(f"Error: Attempted to access index {index} but dataset is empty.")
            return None 
            
        img_path, gt_path = self.imgs[index]
        
        try:
            # Load Image using OpenCV
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")

            # Convert to RGB if needed
            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                print(f"This is color bgr2rgb")
            elif img_cv.ndim == 2: # If grayscale, convert to RGB
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                print(f"This is color gray2rgb")

            img = Image.fromarray(img_cv) # Convert to PIL Image

            # Load Mask using OpenCV in grayscale
            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            target = Image.fromarray(mask_cv, mode='L') # PIL Image in L mode (grayscale)
            
            # Convert to binary mask (0 or 1)
            label = self.convert_label(target)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            print(f"ERROR: Could not open/process image or mask for sample at index {index} ('{img_path}', '{gt_path}'). Error: {e}. Returning None for this sample.")
            return None 

        # Prepare sample for transforms
        sample = {'image': img, 'label': label} # label is PIL Image object in 'P' mode
        
        # Apply the composed transformations
        transformed_sample = self.composed_transforms(sample)
        
        # Add filename for potential debugging or logging
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample

    def convert_label(self, label):
        # Converts the PIL Label Image (mode L) into a binary numpy array (0 or 1)
        label_np = np.array(label, dtype=np.uint8)
        
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)

        # Create a binary mask: 1 if pixel > 0, 0 otherwise.
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 

        # Return as a PIL Image in mode 'P' (Palette)
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)