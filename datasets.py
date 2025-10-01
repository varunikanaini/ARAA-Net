# /kaggle/working/ARAA-Net/datasets.py

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFilter
import numpy as np
from torchvision import transforms
import random 
import cv2 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import pywt # For wavelet enhancement (if needed)

# --- Assume these custom transforms from your previous version exist ---
# If you have re-implemented them for NumPy/Albumentations compatibility, integrate them.
# For now, let's assume they are NOT integrated into the albumentations pipeline directly
# unless specified via A.Lambda.

# Example placeholder for custom transforms that would operate on NumPy arrays:
# def apply_wavelet_enhancement(img_np, wavelet='haar', level=1, detail_scale_factor=1.5): ...
# def apply_histogram_equalization(img_np): ...
# def apply_center_amplification(img_np, mask_np, min_lesion_area_pixels, expansion_factor, min_bbox_size): ...

# --- Helper function for MixUp ---
# This function is typically used *during training*. It should NOT be called for validation or testing.
# Ensure it's only used in the training loop itself, not in the dataset's __getitem__.
def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data.
    Args:
        x: tensor of input
        y: tensor of target
        alpha: mixing coefficient, lambda = np.random.beta(alpha, alpha)
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if batch_size <= 1: # MixUp requires at least 2 samples
        return x, y

    # Choose indices of samples to mix with (shuffling is efficient)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

# --- Data Loading Logic (keep make_dataset as is from your previous corrected version) ---
# ... (your existing corrected make_dataset function is assumed to be here) ...
# Make sure the make_dataset function is the one that correctly handles TSRS_RSNA paths and returns (img_path, mask_path)
# ...

# --- ImageFolder Class ---
class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        # Use the CORRECTED make_dataset function (assuming it's placed above)
        all_imgs = make_dataset(root, dataset_name)
        
        if 'TSRS_RSNA' in dataset_name:
            self.imgs = all_imgs
        else: 
            random.seed(42) 
            random.shuffle(all_imgs)
            
            total_size = len(all_imgs)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size) 
            
            if split == 'train':
                self.imgs = all_imgs[:train_size]
            elif split == 'val':
                self.imgs = all_imgs[train_size : train_size + val_size]
            elif split == 'test':
                self.imgs = all_imgs[train_size + val_size :] 
            else:
                raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. Check dataset path and contents.")

        # --- Albumentations Transforms Setup ---
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # DASEG's fixed preprocessing dimensions
        DASEG_FIXED_RESIZE_W = 576
        DASEG_FIXED_RESIZE_H = 896
        DASEG_TRAIN_CROP_H = 576 # Used for RandomCrop if needed
        DASEG_TRAIN_CROP_W = 576 # Used for RandomCrop if needed

        # --- Define Transforms for Training ---
        if self.split == 'train':
            # IMPORTANT: If `CenterAmplification` was critical, it needs to be applied before
            # the main albumentations pipeline or re-implemented as a custom albumentations transform.
            # For now, assuming it's either omitted or handled externally.
            
            # We need a transform to convert [0,255] uint8 images to [0,1] float images before normalization.
            # Albumentations' `Normalize` expects this range when `max_pixel_value=1.0`.
            
            self.composed_transforms = A.Compose([
                # Resize while maintaining aspect ratio, then pad to fixed size
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR), 
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, border_mode=cv2.BORDER_CONSTANT, value=0),
                
                # Convert image from [0, 255] uint8 to [0.0, 1.0] float
                A.ToFloat(max_value=255.0), 
                
                # --- DASEG-like augmentations (adapted for Albumentations) ---
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ], p=1.0), 
                
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                ], p=1.0),
                
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5), 
                    A.MedianBlur(blur_limit=5, p=0.5),
                ], p=0.5),

                # --- Placeholder for Wavelet and Histogram Equalization ---
                # If you have these as albumentations transforms (e.g., using A.Lambda)
                # integrate them here. Example:
                # A.Lambda(image=custom_wavelet_func, mask=custom_wavelet_func_mask, p=0.5),
                # A.Lambda(image=custom_histo_func, mask=custom_histo_func_mask, p=0.5),

                # --- Random Crop (if DASEG used it for training) ---
                # If RandomCrop was part of the original DASEG training, add it here.
                # A.RandomCrop(height=DASEG_TRAIN_CROP_H, width=DASEG_TRAIN_CROP_W, p=1.0),
                
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), # Normalize [0,1] float image
                ToTensorV2(), # Converts NumPy array to PyTorch tensor
            ])
        else: # Validation/Test Transforms
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ToFloat(max_value=255.0), # Convert to [0,1] float for consistency before normalization
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2(),
            ])

    def __getitem__(self, index):
        if not self.imgs: 
            return None
            
        img_path, gt_path = self.imgs[index]
        try:
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: '{img_path}'.")
            if mask_np is None: raise FileNotFoundError(f"OpenCV could not read mask: '{gt_path}'.")

            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Convert label to binary NumPy array
            # The `convert_label` function should return a NumPy array for albumentations
            label_np = self.convert_label_to_numpy(Image.fromarray(mask_np, mode='L'))

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        # Albumentations expects input as a dictionary with 'image' and 'mask' keys
        sample = {'image': img_np, 'mask': label_np} 
        
        transformed_sample = self.composed_transforms(**sample) # Apply transforms
        
        # Add original filename for testing if needed
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label_to_numpy(self, label_pil):
        """
        Converts a PIL Image label to a binary NumPy array (0 or 1).
        This version is optimized to return a NumPy array directly for albumentations.
        """
        label_np = np.array(label_pil, dtype=np.uint8) # Use uint8 for masks
        
        # Ensure label_np is 2D
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim != 2:
            raise ValueError(f"Unexpected label dimension: {label_np.ndim}")

        # Create a binary mask: 1 for foreground (lesion), 0 for background
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Assumes any non-zero pixel is foreground
        
        return label_index

    def __len__(self):
        return len(self.imgs)