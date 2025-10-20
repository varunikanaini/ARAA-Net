# datasets.py
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

    # --- TSRS_RSNA Datasets ---
    if 'TSRS_RSNA' in dataset_name:
        image_path = root
        mask_path = None
        if os.path.exists(os.path.join(root, 'GT')):
            mask_path = os.path.join(root, 'GT')
        elif os.path.exists(os.path.join(root, 'Labels')): # Common alternative folder name
            mask_path = os.path.join(root, 'Labels')
        elif os.path.exists(root + '_labels'): # Another common alternative
            mask_path = root + '_labels'

        if not mask_path:
            raise FileNotFoundError(f"Could not find label directory for {root}. Looked in '{os.path.join(root, 'GT')}', '{os.path.join(root, 'Labels')}', and '{root + '_labels'}'.")

        img_list = []
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_list.append(os.path.splitext(f)[0])

        for img_name in img_list:
            img_full_path = os.path.join(image_path, img_name + '.jpg') # Assuming JPG for images, adjust if needed
            mask_full_path = os.path.join(mask_path, img_name + '.png') # Assuming PNG for masks, adjust if needed
            
            if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                dataset_items.append((img_full_path, mask_full_path))
            else:
                print(f"Warning: Missing image or mask for {img_name}. Found image: {os.path.exists(img_full_path)}, Found mask: {os.path.exists(mask_full_path)}. Skipping.")

    # --- Other Datasets ---
    elif dataset_name == 'JSRT':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'masks')
        if os.path.isdir(image_path) and os.path.isdir(mask_path):
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
            if os.path.isdir(sub_image_path) and os.path.isdir(sub_mask_path):
                for f in os.listdir(sub_image_path):
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(sub_image_path, f)
                        mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                        if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        if os.path.isdir(image_path) and os.path.isdir(mask_path):
            for f in os.listdir(image_path):
                if f.lower().endswith('.tif'):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(image_path, f)
                    mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                    if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'DentalPanoramic':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1') # Assuming 'segmentation_1' is the mask folder
        if os.path.isdir(image_path) and os.path.isdir(mask_path):
            for f in os.listdir(image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(image_path, f)
                    mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Assuming PNG mask
                    if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    elif dataset_name == 'SixDiseasesChestXRay':
        base_split_folder = root 
        if os.path.isdir(base_split_folder):
            subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
            for sub_name in subfolders:
                sub_image_path = os.path.join(base_split_folder, sub_name, 'images')
                sub_mask_path = os.path.join(base_split_folder, sub_name, 'masks')
                if os.path.isdir(sub_image_path) and os.path.isdir(sub_mask_path):
                    for f in os.listdir(sub_image_path):
                        if f.lower().endswith(IMAGE_EXTENSIONS):
                            img_name_base = os.path.splitext(f)[0]
                            img_full_path = os.path.join(sub_image_path, f)
                            mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                            if os.path.exists(mask_full_path): dataset_items.append((img_full_path, mask_full_path))

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}.")
    
    if not dataset_items and dataset_name not in ['PLACEHOLDER_FOR_DYNAMIC_SELECTION']: # Avoid warning for specific placeholders
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
        
        # --- Define Mean/Std based on dataset type ---
        if dataset_name in ['JSRT', 'COVID19_Radiography']: 
            self.mean = [0.5]
            self.std = [0.5]
            # For grayscale images, we'll convert to RGB later in __getitem__ if needed by model
            self.is_grayscale_dataset = True
        else:
            # Default ImageNet means/stds for RGB images
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.is_grayscale_dataset = False

        # --- Define Transforms ---
        if self.split == 'train':
            # Using the enhanced custom transforms
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                # Apply CenterAmplification only if the argument is enabled and meaningful
                tr.CenterAmplification(min_lesion_area_pixels=args.min_lesion_area_pixels,
                                       expansion_factor=args.expansion_factor,
                                       min_bbox_size=(args.min_bbox_h, args.min_bbox_w)) if args.min_lesion_area_pixels > 0 else lambda x: x,
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((args.scale_h, args.scale_w)),
                # Enable ElasticTransform with its current settings
                tr.ElasticTransform(alpha=340, sigma=5, p=0.8),
                tr.RandomGaussianBlur(),
                tr.HistogramEqualization(p=0.4), # Added Histogram Equalization
                tr.WaveletContrastEnhancement(detail_scale_factor=1.8, p=0.4), # Added Wavelet Enhancement
                tr.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                tr.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15, mask_fill_value=0),
                tr.RandomCutout(num_holes_range=(1, 3), max_h_size=48, max_w_size=48, fill_value=0, p=0.5), 
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor() 
            ])
        else: # Validation and Test splits
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                # For validation/test, ensure images are RGB if needed, and normalize
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            # Use OpenCV for robustness, but convert to PIL for transforms
            img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR) # Read as color image
            if img_cv is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")

            # If dataset is grayscale, but model expects RGB (e.g. MobileNetV2 encoder), convert
            if self.is_grayscale_dataset and img_cv.ndim == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            elif self.is_grayscale_dataset and img_cv.ndim == 3 and img_cv.shape[2] == 3:
                # If it was already read as RGB (e.g. .jpg), but is a grayscale dataset, convert to RGB for consistency
                 img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB) # This might be redundant if already RGB, but ensures consistency

            img_pil = Image.fromarray(img_cv)

            # Read mask as grayscale
            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            target_pil = Image.fromarray(mask_cv, mode='L')
            label_pil = self.convert_label(target_pil) # Convert to PIL with correct mode/values

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None

        sample = {'image': img_pil, 'label': label_pil}
        transformed_sample = self.composed_transforms(sample)

        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)

        return transformed_sample

    def convert_label(self, label_pil):
        # This function is intended to ensure the mask has consistent values (e.g., 0 and 1 for binary)
        # and is in a format suitable for PyTorch (like P mode PIL or np array).
        label_np = np.array(label_pil, dtype=np.uint8)
        
        # Handle cases where mask might be RGB or have multiple channels
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim == 3 and label_np.shape[2] == 3: # If mask is accidentally loaded as RGB
             # Assuming the dominant channel or first channel is the one to use
             print(f"Warning: Mask {label_pil} has 3 channels, attempting to convert to single channel.")
             label_np = label_np[:, :, 0] 

        # Binarize the label: Foreground is any non-zero pixel value.
        # This assumes your masks use 0 for background and >0 for foreground.
        # Adjust if your masks use specific values for different classes.
        binary_label_np = (label_np > 0).astype(np.uint8)

        return Image.fromarray(binary_label_np, mode='L') # Return as PIL Image in 'L' mode (8-bit grayscale)

    def __len__(self):
        return len(self.imgs)