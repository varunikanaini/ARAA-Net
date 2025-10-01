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
        
        # --- Data Splitting Logic ---
        if 'TSRS_RSNA' in dataset_name or dataset_name == 'SixDiseasesChestXRay':
            # For these datasets, `root` is already a specific split dir (e.g., 'train', 'val', 'test')
            self.imgs = all_imgs
            if not self.imgs: print(f"Warning: The '{split}' split for {dataset_name} is empty. Expected data at: '{root}'")
        else:
            # Handle programmatic splitting if data is not pre-split into subfolders
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
                
                # Adjust split sizes to avoid empty splits
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

        # Define transforms using albumentations
        if self.split == 'train':
            self.composed_transforms = A.Compose([
                # Resize maintaining aspect ratio, then pad to fixed size
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR), 
                # Pad to target size. 'value' MUST be a tuple for RGB images.
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, 
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)), # Padding with black for RGB
                
                A.ToFloat(max_value=255.0), # Convert image to float in [0, 1] range
                
                # --- Augmentations ---
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=1.0), 
                A.OneOf([A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
                         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7)], p=1.0),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5), A.MedianBlur(blur_limit=5, p=0.5)], p=0.5),

                # --- Placeholder for Custom Transforms (e.g., Wavelet, Histogram) ---
                # If you have custom albumentations transforms, they should be A.Lambda
                # Make sure they handle NumPy arrays and return NumPy arrays of correct shape.
                # Example: A.Lambda(image=custom_wavelet_transform_func, mask=custom_wavelet_transform_func_mask, p=0.5),
                
                # --- Random Crop (if applicable and desired) ---
                # If RandomCrop was part of original DASEG training, it should be added here.
                # A.RandomCrop(height=DASEG_TRAIN_CROP_H, width=DASEG_TRAIN_CROP_W, p=1.0),
                
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), # Normalize [0,1] float image
                ToTensorV2(), # Converts NumPy array to PyTorch tensor [C, H, W]
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
            
            # Ensure mask is processed correctly and returns a NumPy array
            label_np = self.convert_label_to_numpy(Image.fromarray(mask_np, mode='L'))

            # --- IMPORTANT: Ensure Mask Shape Consistency ---
            # After all transforms, the mask MUST have the same spatial dimensions as the image.
            # Albumentations transforms should handle this if applied to both image and mask correctly.
            # If an error occurs here, it means the mask's shape is not matching the image's shape after transforms.

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        # Prepare sample for albumentations: both image and mask should be NumPy arrays
        sample = {'image': img_np, 'mask': label_np} 
        
        try:
            transformed_sample = self.composed_transforms(**sample) # Apply transforms
        except Exception as e:
            print(f"ERROR: Albumentations transform failed for sample {index} ({img_path}). Error: {e}. Returning None.")
            return None
        
        # Ensure the mask also has the correct tensor type and shape after transforms
        # ToTensorV2 handles this conversion.
        
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label_to_numpy(self, label_pil):
        label_np = np.array(label_pil, dtype=np.uint8) 
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim != 2:
            # If label_np is already 2D (e.g., from cv2.imread(..., cv2.IMREAD_GRAYSCALE)), this check is fine.
            # If PIL creates it in a way that's not 2D and not reducible to 2D, this will catch it.
            raise ValueError(f"Unexpected label dimension: {label_np.ndim} for {label_pil.size}")

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