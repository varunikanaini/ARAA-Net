# /kaggle/working/ARAA-Net/datasets.py

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
import torch # Needed for potential tensor operations

# --- Placeholder for custom transforms ---
# Assuming 'custom_transforms' module or equivalent is available.
# These are mock implementations if not found.
class CustomTransformPlaceholder:
    def __call__(self, sample):
        return sample

# Mock implementations for missing custom transforms
if not hasattr(CustomTransformPlaceholder, 'FixedResize'): CustomTransformPlaceholder.FixedResize = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'CenterAmplification'): CustomTransformPlaceholder.CenterAmplification = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'RandomAffine'): CustomTransformPlaceholder.RandomAffine = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'RandomGaussianBlur'): CustomTransformPlaceholder.RandomGaussianBlur = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'RandomHorizontalFlip'): CustomTransformPlaceholder.RandomHorizontalFlip = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'RandomCrop'): CustomTransformPlaceholder.RandomCrop = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'ColorJitter'): CustomTransformPlaceholder.ColorJitter = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'WaveletContrastEnhancement'): CustomTransformPlaceholder.WaveletContrastEnhancement = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'RandomCutout'): CustomTransformPlaceholder.RandomCutout = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'Normalize'): CustomTransformPlaceholder.Normalize = lambda **kwargs: lambda x: x
if not hasattr(CustomTransformPlaceholder, 'ToTensor'): CustomTransformPlaceholder.ToTensor = lambda **kwargs: lambda x: x

tr = CustomTransformPlaceholder() # Alias for custom transforms

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(data_info, split):
    """
    Collects image-mask pairs. If specific split directories (train/val/test) are not found,
    it attempts to collect all items from a general location and prepare for programmatic splitting.
    """
    dataset_items = []
    base_path = data_info['path']
    structure = data_info['structure']
    img_ext = data_info.get('image_ext', IMAGE_EXTENSIONS)
    mask_ext = data_info.get('mask_ext', MASK_EXTENSIONS)

    print(f"Looking for dataset '{structure}' split '{split}' in '{base_path}'")

    # --- Data Collection Logic ---
    # This function's primary role is to FIND ALL AVAILABLE image-mask pairs.
    # The actual splitting will be handled in the ImageFolder __init__.

    if structure == 'TSRS_RSNA':
        # TSRS_RSNA is expected to have explicit split folders (train, val, test)
        split_image_dir = os.path.join(base_path, split)
        split_mask_dir = os.path.join(base_path, f"{split}_labels")

        if not os.path.isdir(split_image_dir):
            print(f"Warning: TSRS-like dataset '{structure}' split '{split}': Image directory not found at '{split_image_dir}'.")
            return []
        if not os.path.isdir(split_mask_dir):
            print(f"Warning: TSRS-like dataset '{structure}' split '{split}': Label directory not found at '{split_mask_dir}'.")
            return []

        for f in os.listdir(split_image_dir):
            if f.lower().endswith(img_ext):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(split_image_dir, f)
                mask_found = False
                for ext in mask_ext:
                    mask_full_path = os.path.join(split_mask_dir, img_name_base + ext)
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        mask_found = True
                        break
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in dataset '{structure}', split '{split}'. Skipping.")

    elif structure == 'STANDARD':
        # For STANDARD, we first check if the 'split' directory exists directly.
        # If it does, we assume it contains the data for that split (e.g., JSRT might have 'train' folder with 'images'/'masks' inside).
        # If not, we collect all items from common locations to prepare for programmatic splitting.
        
        explicit_split_path = os.path.join(base_path, split)
        
        if os.path.isdir(explicit_split_path):
            print(f"Found explicit split directory for '{split}' in '{base_path}'. Processing it.")
            # Try to find standard subfolders within the explicit split directory
            image_dir_name_candidates = ['images', 'cxr'] # Common names
            mask_dir_name_candidates = ['masks', 'GT']
            
            img_dir = None
            mask_dir = None

            for img_cand in image_dir_name_candidates:
                potential_img_dir = os.path.join(explicit_split_path, img_cand)
                if os.path.isdir(potential_img_dir):
                    for mask_cand in mask_dir_name_candidates:
                        potential_mask_dir = os.path.join(explicit_split_path, mask_cand)
                        if os.path.isdir(potential_mask_dir):
                            img_dir = potential_img_dir
                            mask_dir = potential_mask_dir
                            break
                    if img_dir: break # Found both image and mask dirs for this split

            if img_dir and mask_dir:
                for f in os.listdir(img_dir):
                    if f.lower().endswith(img_ext):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(img_dir, f)
                        mask_found = False
                        for ext in mask_ext:
                            mask_full_path = os.path.join(mask_dir, img_name_base + ext)
                            if os.path.exists(mask_full_path):
                                dataset_items.append((img_full_path, mask_full_path))
                                mask_found = True
                                break
                        if not mask_found:
                            print(f"Warning: Mask not found for image {f} in explicit split '{split}'. Skipping.")
            else:
                 print(f"Could not find standard image/mask subfolders within '{explicit_split_path}'. Will attempt to collect all items for programmatic split.")
                 # Fall through to collect all items

        # If explicit split dir was not found, or subfolders within it weren't found,
        # collect all items from the base_path to prepare for programmatic splitting.
        if not dataset_items:
            print(f"Collecting all available items from '{base_path}' for programmatic split.")
            
            # Common locations for images and masks if not split
            potential_img_dirs = [
                os.path.join(base_path, 'images'),
                os.path.join(base_path, 'cxr'),
                base_path # If images are directly in base_path
            ]
            potential_mask_dirs = [
                os.path.join(base_path, 'masks'),
                os.path.join(base_path, 'GT'),
                base_path # If masks are directly in base_path
            ]
            
            found_image_dir = None
            found_mask_dir = None

            for p_img in potential_img_dirs:
                if os.path.isdir(p_img):
                    for p_mask in potential_mask_dirs:
                        if os.path.isdir(p_mask):
                            # Check if they seem like corresponding dirs (e.g. not base_path for both)
                            # Simple heuristic: avoid using base_path for both unless it's the only option
                            if p_img == base_path and p_mask == base_path and len(potential_img_dirs) > 1:
                                continue
                            
                            found_image_dir = p_img
                            found_mask_dir = p_mask
                            print(f"Using image dir: '{found_image_dir}', mask dir: '{found_mask_dir}'")
                            break
                    if found_image_dir: break

            if found_image_dir and found_mask_dir:
                for f in os.listdir(found_image_dir):
                    if f.lower().endswith(img_ext):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(found_image_dir, f)
                        mask_found = False
                        for ext in mask_ext:
                            mask_full_path = os.path.join(found_mask_dir, img_name_base + ext)
                            if os.path.exists(mask_full_path):
                                dataset_items.append((img_full_path, mask_full_path))
                                mask_found = True
                                break
                        if not mask_found:
                            print(f"Warning: Mask not found for image {f} in '{base_path}'. Skipping.")
            else:
                print(f"Could not find suitable image/mask directories under '{base_path}' for programmatic collection.")

    elif structure == 'COVID19':
        # COVID19 structure is predefined within the dataset path
        classes = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for cls_name in classes:
            cls_image_path = os.path.join(base_path, 'COVID-19_Radiography_Dataset', cls_name, 'images')
            cls_mask_path = os.path.join(base_path, 'COVID-19_Radiography_Dataset', cls_name, 'masks')
            if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path):
                print(f"Warning: COVID19 class '{cls_name}' image/mask path not found. Skipping class.")
                continue
            for f in os.listdir(cls_image_path):
                if f.lower().endswith(img_ext):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(cls_image_path, f)
                    mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image {f} in class '{cls_name}'. Skipping.")

    elif structure == 'SIX_DISEASES':
        # SIX_DISEASES also has a structure with class folders, potentially under split dirs
        classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        # We assume 'split' here refers to the top-level split folder if it exists
        split_specific_path = os.path.join(base_path, split)
        
        if os.path.isdir(split_specific_path): # Explicit split folder exists
             print(f"Processing SIX_DISEASES split '{split}' from '{split_specific_path}'.")
             for cls_name in classes:
                 cls_image_path = os.path.join(split_specific_path, cls_name, 'images')
                 cls_mask_path = os.path.join(split_specific_path, cls_name, 'masks')
                 if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path):
                     continue # Skip if class folder structure isn't found
                 for f in os.listdir(cls_image_path):
                     if f.lower().endswith(img_ext):
                         img_name_base = os.path.splitext(f)[0]
                         img_full_path = os.path.join(cls_image_path, f)
                         mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                         if os.path.exists(mask_full_path):
                             dataset_items.append((img_full_path, mask_full_path))
                         else:
                             print(f"Warning: Mask not found for SIX_DISEASES image {f} in class '{cls_name}', split '{split}'. Skipping.")
        else:
             print(f"Explicit split directory '{split_specific_path}' not found for SIX_DISEASES. Will attempt to collect all items.")
             # Fall through to collect all items if explicit split dirs not found
             # (This part might need refinement depending on SIX_DISEASES's exact structure if not pre-split)
             # For now, assume it uses 'images' and 'masks' directly if no split dirs.
             potential_img_dirs = [os.path.join(base_path, 'images'), base_path]
             potential_mask_dirs = [os.path.join(base_path, 'masks'), base_path]
             # ... similar collection logic as STANDARD ...
             # This section is simplified; full collection logic as in STANDARD is recommended if needed.
             print("Collecting items directly from base path for SIX_DISEASES (assuming no explicit splits).")
             img_dir_base = os.path.join(base_path, 'images')
             mask_dir_base = os.path.join(base_path, 'masks')
             if os.path.isdir(img_dir_base) and os.path.isdir(mask_dir_base):
                for f in os.listdir(img_dir_base):
                    if f.lower().endswith(img_ext):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(img_dir_base, f)
                        mask_full_path = os.path.join(mask_dir_base, img_name_base + '.png')
                        if os.path.exists(mask_full_path):
                            dataset_items.append((img_full_path, mask_full_path))
                        else:
                            print(f"Warning: Mask not found for SIX_DISEASES image {f} in base path. Skipping.")
             else:
                 print(f"Could not find 'images'/'masks' directories directly under '{base_path}' for SIX_DISEASES.")

    else:
        raise ValueError(f"Unknown dataset structure type: {structure}. Please define handling for this structure.")

    if not dataset_items:
        print(f"Warning: Found 0 image-mask pairs for dataset structure '{structure}' in '{base_path}'. Please check path and dataset structure.")
    
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args

        try:
            dataset_info = config.DATASET_CONFIG[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")

        # --- Data Collection and Splitting ---
        # 1. First, try to get items for the specific 'split' if explicit split folders exist.
        #    make_dataset is designed to do this for TSRS_RSNA and potentially for STANDARD/SIX_DISEASES if they have explicit split dirs.
        found_items_for_split = make_dataset(dataset_info, split)

        # 2. If make_dataset didn't find items for the specific 'split' (e.g., it's a dataset meant for programmatic splitting like JSRT without pre-split folders),
        #    we collect ALL available items and perform the split here.
        if not found_items_for_split and dataset_info['structure'] != 'TSRS_RSNA': # Avoid re-collecting for inherently split datasets like TSRS_RSNA
            print(f"No explicit split data found for '{split}'. Collecting all available items for programmatic 80/10/10 split.")
            # Collect all items, ignoring the 'split' argument for make_dataset to get everything
            all_available_items = make_dataset(dataset_info, 'all') # 'all' is a dummy split, make_dataset should handle it to collect all.

            if not all_available_items:
                raise RuntimeError(f"Failed to collect any data for dataset '{self.dataset_name}' at path '{dataset_info['path']}'. Cannot proceed.")
            
            # Perform programmatic split
            random.seed(42) # Ensure reproducibility
            random.shuffle(all_available_available_items)
            
            total_size = len(all_available_items)
            train_ratio = 0.8
            val_ratio = 0.1
            
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            
            # Adjust for very small datasets to avoid empty splits
            if total_size < 3:
                train_size = 1 if total_size > 0 else 0
                val_size = 1 if total_size > 1 else 0
                test_size = max(0, total_size - train_size - val_size)
            else: # Ensure test_size is non-negative
                test_size = max(0, test_size)

            print(f"Programmatic split: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")

            if split == 'train':
                self.imgs = all_available_items[:train_size]
            elif split == 'val':
                self.imgs = all_available_items[train_size : train_size + val_size]
            elif split == 'test':
                self.imgs = all_available_items[train_size + val_size :]
            else:
                raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")
        
        else:
            # Use the items found directly for the specific split (e.g., from TSRS_RSNA or if explicit splits were found for STANDARD)
            self.imgs = found_items_for_split

        # --- Final Check and Shuffle ---
        if not self.imgs:
            raise RuntimeError(f"No images loaded for dataset '{self.dataset_name}' split '{self.split}'. Check data paths, structure, and split configuration.")
        else:
            print(f"Successfully loaded {len(self.imgs)} samples for {dataset_name} split '{self.split}'.")
        
        # Shuffle training data after it's been assigned
        if self.split == 'train':
            random.shuffle(self.imgs)

        # --- Augmentation Setup ---
        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w
        scale_h = args.scale_h
        scale_w = args.scale_w

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)) if min_lesion_area > 0 else lambda x: x,
                tr.RandomAffine(degrees=7, translate=(0.07, 0.07), scale=(0.95, 1.05), shear=7, mask_fill_value=0),
                tr.RandomGaussianBlur(radius_range=(0.1, 1.2)),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((scale_h, scale_w)),
                tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                tr.WaveletContrastEnhancement(wavelet=args.wavelet_type, level=args.wavelet_level, detail_scale_factor=args.wavelet_detail_scale) if np.random.rand() < 0.2 else lambda x: x,
                tr.RandomCutout(num_holes_range=(1, 4), max_h_size=40, max_w_size=40, fill_value=0, p=0.5),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else:  # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        if not self.imgs: 
            return None
            
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