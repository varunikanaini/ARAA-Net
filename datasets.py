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
import pywt 

# --- Data Loading Logic ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(root, dataset_name):
    # --- (Your make_dataset function remains the same as the previous correct version) ---
    # ... Ensure it's correctly defined and returns (img_path, mask_path) tuples ...
    dataset_items = []

    if 'TSRS_RSNA' in dataset_name:
        image_path = root
        mask_dir_candidate_sibling = os.path.join(os.path.dirname(root), os.path.basename(root) + '_labels')
        mask_dir_candidate_in_root = os.path.join(root, 'GT') 

        if os.path.isdir(mask_dir_candidate_sibling): mask_path = mask_dir_candidate_sibling
        elif os.path.isdir(mask_dir_candidate_in_root): mask_path = mask_dir_candidate_in_root
        else:
            print(f"DEBUG: {dataset_name}: Could not find label directory for '{root}'.")
            return []
        
        if not os.path.isdir(image_path):
            print(f"Warning: Image path '{image_path}' is not a directory for dataset '{dataset_name}'.")
            return []
        
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                
                if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
    
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

# --- ImageFolder Class ---
class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        all_imgs = make_dataset(root, dataset_name)
        
        if 'TSRS_RSNA' in dataset_name or dataset_name == 'SixDiseasesChestXRay':
            self.imgs = all_imgs
            if not self.imgs: print(f"Warning: The '{split}' split for {dataset_name} is empty. Expected data at: '{root}'")
        else:
            train_dir = os.path.join(root, 'train')
            val_dir = os.path.join(root, 'val')
            test_dir = os.path.join(root, 'test')

            if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
                print(f"Detected pre-split directories in '{root}'. Using '{split}' split.")
                if split == 'train': self.imgs = make_dataset(train_dir, dataset_name)
                elif split == 'val': self.imgs = make_dataset(val_dir, dataset_name)
                elif split == 'test': self.imgs = make_dataset(test_dir, dataset_name)
                else: raise ValueError(f"Invalid split '{split}'.")
            else:
                if not all_imgs:
                     print(f"Warning: No images found for dataset '{dataset_name}' at root '{root}'.")
                     self.imgs = []
                else:
                    random.seed(42) 
                    random.shuffle(all_imgs)
                    
                    total_size = len(all_imgs)
                    train_ratio = 0.8
                    val_ratio = 0.1
                    
                    train_size = int(train_ratio * total_size)
                    val_size = int(val_ratio * total_size)
                    
                    if total_size < 3: 
                        train_size = 1 if total_size > 0 else 0
                        val_size = 1 if total_size > 1 else 0
                    else:
                        test_size = total_size - train_size - val_size
                        if test_size < 0:
                            if val_size > 1: val_size -= 1
                            test_size = total_size - train_size - val_size
                        test_size = max(0, test_size)

                    if split == 'train': self.imgs = all_imgs[:train_size]
                    elif split == 'val': self.imgs = all_imgs[train_size : train_size + val_size]
                    elif split == 'test': self.imgs = all_imgs[train_size + val_size :] 
                    else: raise ValueError(f"Invalid split '{split}'.")
                    print(f"Performing programmatic split on '{root}'. Total items: {total_size}. Split '{split}': {len(self.imgs)} items.")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. No images loaded. Please check dataset path and contents: '{root}'")

        # --- Albumentations Transforms Setup ---
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        DASEG_FIXED_RESIZE_W = 576
        DASEG_FIXED_RESIZE_H = 896

        if self.split == 'train':
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR), 
                # Corrected PadIfNeeded: 'value' must be a tuple for RGB images.
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, 
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)), # Padding with black for RGB
                
                A.ToFloat(max_value=255.0), 
                
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
                
                # If you have custom transforms, they should be A.Lambda.
                # Ensure custom transforms handle NumPy arrays and return NumPy arrays.
                # Example: A.Lambda(image=custom_wavelet_transform_func, mask=custom_wavelet_transform_func_mask, p=0.5),
                
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), 
                ToTensorV2(), # Converts NumPy array to PyTorch tensor
            ])
        else: # Validation/Test Transforms
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, 
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)), # Padding with black for RGB
                A.ToFloat(max_value=255.0), 
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

            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            
            label_np = self.convert_label_to_numpy(Image.fromarray(mask_np, mode='L'))

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        # Albumentations expects input as a dictionary with 'image' and 'mask' keys
        sample = {'image': img_np, 'mask': label_np} 
        
        transformed_sample = self.composed_transforms(**sample) # Apply transforms
        
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label_to_numpy(self, label_pil):
        label_np = np.array(label_pil, dtype=np.uint8) 
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim != 2:
            raise ValueError(f"Unexpected label dimension: {label_np.ndim}")

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 
        
        return label_index

    def __len__(self):
        return len(self.imgs)

# --- MixUp Helper ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if batch_size <= 1: 
        return x, y

    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y