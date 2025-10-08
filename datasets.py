# /kaggle/working/ARAA-Net/datasets.py

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFilter
import numpy as np
from torchvision import transforms
import random
import cv2
import albumentations as A  # Using Albumentations for transforms
from albumentations.pytorch import ToTensorV2
import pywt # Assuming pywt is needed for WaveletContrastEnhancement if custom_transforms uses it

# --- Placeholder for custom transforms ---
# If your 'custom_transforms' module is not available, these placeholders
# will allow the code to run, but they won't perform any actual transformations.
# Replace with your actual custom transform implementations if needed.
class CustomTransformPlaceholder:
    def __call__(self, sample):
        # In a real scenario, this would apply the transform to the sample
        return sample

# Mock implementations for potentially missing custom transforms
# These ensure the code doesn't break if custom_transforms are not fully defined/imported.
# If you are using torchvision transforms directly with aliases, these might not be strictly necessary.
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

# Alias for custom transforms. If you are using custom_transforms module, import it here.
# from custom_transforms import FixedResize, CenterAmplification, ... etc.
# tr = your_imported_custom_transforms
tr = CustomTransformPlaceholder() # Using placeholder for now

# --- Global definitions for image/mask extensions ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(data_info, split):
    """
    Collects image-mask pairs. It attempts to find explicitly split data first.
    If explicit splits are not found, it collects all available items from common locations
    to prepare for programmatic splitting.
    
    Args:
        data_info (dict): Dataset configuration from config.py.
        split (str): The desired split ('train', 'val', 'test', or 'all' for collection).
                     'all' is used internally for programmatic splitting.

    Returns:
        list: A list of (image_path, mask_path) tuples.
    """
    dataset_items = []
    base_path = data_info['path']
    structure = data_info['structure']
    img_ext = data_info.get('image_ext', IMAGE_EXTENSIONS)
    mask_ext = data_info.get('mask_ext', MASK_EXTENSIONS)

    print(f"make_dataset: Looking for dataset '{structure}' split '{split}' in '{base_path}'")

    # --- Data Collection Logic ---
    
    # 1. Check for explicit split directories (e.g., train/val/test)
    explicit_split_dir_found = False
    potential_split_dirs = ['train', 'val', 'test'] if split != 'all' else [] # Only check for specific splits if 'split' is not 'all'

    if split != 'all' and structure == 'STANDARD' or structure == 'SIX_DISEASES':
        # Check if the directory for the specific split (e.g., 'train') exists directly
        current_split_path = os.path.join(base_path, split)
        if os.path.isdir(current_split_path):
            print(f"make_dataset: Found explicit split directory for '{split}' in '{base_path}'.")
            explicit_split_dir_found = True
            
            # For STANDARD structure, look for 'images'/'masks' (or variants) within the split dir
            if structure == 'STANDARD':
                image_dir_name_candidates = ['images', 'cxr']
                mask_dir_name_candidates = ['masks', 'GT']
                
                img_dir = None
                mask_dir = None

                for img_cand in image_dir_name_candidates:
                    potential_img_dir = os.path.join(current_split_path, img_cand)
                    if os.path.isdir(potential_img_dir):
                        for mask_cand in mask_dir_name_candidates:
                            potential_mask_dir = os.path.join(current_split_path, mask_cand)
                            if os.path.isdir(potential_mask_dir):
                                img_dir = potential_img_dir
                                mask_dir = potential_mask_dir
                                break
                        if img_dir: break
                
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
                    print(f"make_dataset: Could not find standard image/mask subfolders within '{current_split_path}'. Falling back to collection mode.")
                    explicit_split_dir_found = False # Force collection mode

            # For SIX_DISEASES, it might have class folders directly under split dir
            elif structure == 'SIX_DISEASES':
                 classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
                 for cls_name in classes:
                     cls_image_path = os.path.join(current_split_path, cls_name, 'images')
                     cls_mask_path = os.path.join(current_split_path, cls_name, 'masks')
                     if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path):
                         continue
                     for f in os.listdir(cls_image_path):
                         if f.lower().endswith(img_ext):
                             img_name_base = os.path.splitext(f)[0]
                             img_full_path = os.path.join(cls_image_path, f)
                             mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                             if os.path.exists(mask_full_path):
                                 dataset_items.append((img_full_path, mask_full_path))
                             else:
                                 print(f"Warning: Mask not found for SIX_DISEASES image {f} in class '{cls_name}', split '{split}'. Skipping.")
            
            # If we found items using explicit split dirs, return them
            if dataset_items:
                 print(f"make_dataset: Found {len(dataset_items)} items using explicit split '{split}'.")
                 return dataset_items


    # 2. If explicit splits were not used or found, collect all available items.
    #    This happens if 'split' is 'all', or if explicit splits failed/weren't configured for STANDARD/SIX_DISEASES.
    if not dataset_items:
        print(f"make_dataset: No explicit split data found or requested for '{split}'. Collecting all available items from '{base_path}'.")
        
        if structure == 'TSRS_RSNA': # TSRS_RSNA relies on explicit splits, should not reach here for collection mode ideally.
            print("Warning: TSRS_RSNA structure typically requires explicit splits. Attempting collection as fallback.")
            # Fallback collection logic for TSRS_RSNA if needed, similar to STANDARD.
            # For now, we assume it must have explicit splits.
            if not os.path.isdir(base_path): return [] # Ensure base path exists
            for f in os.listdir(base_path): # Look for image files directly in base_path
                 if f.lower().endswith(img_ext):
                     img_name_base = os.path.splitext(f)[0]
                     img_full_path = os.path.join(base_path, f)
                     mask_found = False
                     for ext in mask_ext:
                         mask_full_path = os.path.join(base_path, img_name_base + ext) # Assume mask is also in base_path
                         if os.path.exists(mask_full_path):
                              dataset_items.append((img_full_path, mask_full_path))
                              mask_found = True
                              break
                     if not mask_found:
                          print(f"Warning: Mask not found for TSRS_RSNA image {f} directly in '{base_path}'. Skipping.")


        elif structure == 'STANDARD':
            # Common locations for images and masks if not split
            potential_img_dirs = [
                os.path.join(base_path, 'images'), os.path.join(base_path, 'cxr'), base_path
            ]
            potential_mask_dirs = [
                os.path.join(base_path, 'masks'), os.path.join(base_path, 'GT'), base_path
            ]
            
            found_image_dir = None
            found_mask_dir = None

            for p_img in potential_img_dirs:
                if os.path.isdir(p_img):
                    for p_mask in potential_mask_dirs:
                        if os.path.isdir(p_mask):
                            # Avoid using base_path for both unless it's the only option
                            if p_img == base_path and p_mask == base_path and len(potential_img_dirs) > 1: continue
                            found_image_dir = p_img
                            found_mask_dir = p_mask
                            print(f"make_dataset: Using image dir: '{found_image_dir}', mask dir: '{found_mask_dir}'")
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
                print(f"make_dataset: Could not find suitable image/mask directories under '{base_path}' for collection.")

        elif structure == 'COVID19':
            classes = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
            for cls_name in classes:
                cls_image_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'images')
                cls_mask_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'masks')
                if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path):
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
            # If explicit splits weren't found, try collecting from common locations
            print(f"make_dataset: SIX_DISEASES - explicit split dirs not found. Attempting collection from '{base_path}'.")
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
                print(f"make_dataset: Could not find 'images'/'masks' directories directly under '{base_path}' for SIX_DISEASES collection.")
        
        else:
            raise ValueError(f"Unknown dataset structure type: {structure}. Please define handling for this structure.")

    if not dataset_items:
        print(f"Warning: make_dataset found 0 items for structure '{structure}' in '{base_path}'.")
        
    return dataset_items

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        """
        Initializes the dataset. It handles data collection and splitting.
        If explicit split directories (train/val/test) are not found, it performs
        a programmatic 80/10/10 split on all collected data.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 

        # 1. Attempt to get items for the specific 'split' if explicit split folders exist.
        #    make_dataset is designed to handle this for structures like TSRS_RSNA and
        #    potentially for STANDARD/SIX_DISEASES if they have explicit split dirs.
        found_items_for_split = make_dataset(dataset_name=dataset_name, data_info={'path': root, 'structure': config.DATASET_CONFIG[dataset_name]['structure']}, split=split)

        # 2. If no items were found for the specific 'split' (and it's not a dataset like TSRS_RSNA that MUST have explicit splits),
        #    collect ALL available items and perform the split here.
        if not found_items_for_split and dataset_name != 'TSRS_RSNA-Epiphysis' and dataset_name != 'TSRS_RSNA-Articular-Surface': # Avoid re-collecting for inherently split datasets.
            print(f"ImageFolder: No explicit split data found for '{split}'. Collecting all available items for programmatic 80/10/10 split.")
            
            # Collect all items using make_dataset with a dummy 'all' split trigger
            all_available_items = make_dataset(dataset_name=dataset_name, data_info={'path': root, 'structure': config.DATASET_CONFIG[dataset_name]['structure']}, split='all')

            if not all_available_items:
                raise RuntimeError(f"ImageFolder: Failed to collect any data for dataset '{self.dataset_name}' at path '{root}'. Cannot proceed.")
            
            # Perform programmatic split
            random.seed(42) # Ensure reproducibility
            random.shuffle(all_available_items)
            
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

            print(f"ImageFolder: Performing programmatic split for {dataset_name}. Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")

            if split == 'train': self.imgs = all_available_items[:train_size]
            elif split == 'val': self.imgs = all_available_items[train_size : train_size + val_size]
            elif split == 'test': self.imgs = all_available_items[train_size + val_size :]
            else: raise ValueError(f"ImageFolder: Invalid split '{split}'. Expected 'train', 'val', or 'test'.")
        
        else:
            # Use the items found directly for the specific split
            self.imgs = found_items_for_split

        # --- Final Check and Shuffle ---
        if not self.imgs:
            raise RuntimeError(f"ImageFolder: No images loaded for dataset '{self.dataset_name}' split '{self.split}'. Check data paths, structure, and split configuration. Dataset root: '{root}'.")
        else:
            print(f"ImageFolder: Successfully loaded {len(self.imgs)} samples for {dataset_name} split '{self.split}'.")
        
        # Shuffle training data after it's been assigned
        if self.split == 'train':
            random.shuffle(self.imgs)

        # --- Albumentations Transforms Setup ---
        # These are specific to your project's needs and assumed from previous context.
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Define target sizes, potentially from args or fixed values
        DASEG_FIXED_RESIZE_W = getattr(args, 'scale_w', 576) # Use args.scale_w if available, else default
        DASEG_FIXED_RESIZE_H = getattr(args, 'scale_h', 896) # Use args.scale_h if available, else default

        if self.split == 'train':
            self.composed_transforms = A.Compose([
                # Resize maintaining aspect ratio
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR), 
                # Pad to target size
                A.PadIfNeeded(
                    min_height=DASEG_FIXED_RESIZE_H, 
                    min_width=DASEG_FIXED_RESIZE_W, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    pad_val=(0, 0, 0),  # Black padding for RGB
                ),
                A.ToFloat(max_value=255.0), # Convert to [0.0, 1.0]
                
                # --- Augmentations ---
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=1.0), 
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                ], p=1.0),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5), A.MedianBlur(blur_limit=5, p=0.5)], p=0.5),

                # If using custom transforms with Albumentations (e.g., via A.Lambda)
                # Ensure they return NumPy arrays and are compatible.
                # Example: A.Lambda(image=lambda x, **kwargs: tr.YourCustomTransform()(x), mask=lambda x, **kwargs: tr.YourCustomTransformMask()(x))

                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), 
                ToTensorV2(), 
            ])
        else: # Validation/Test Transforms
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=DASEG_FIXED_RESIZE_H, 
                    min_width=DASEG_FIXED_RESIZE_W, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    pad_val=(0, 0, 0),
                ),
                A.ToFloat(max_value=255.0), 
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2(),
            ])

    def __getitem__(self, index):
        if not self.imgs: 
            return None
            
        img_path, gt_path = self.imgs[index]
        try:
            # Load image and mask using OpenCV
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: '{img_path}'.")
            if mask_np is None: raise FileNotFoundError(f"OpenCV could not read mask: '{gt_path}'.")

            # Convert image from BGR to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) 
            
            # Convert mask to NumPy array (binary: 0 or 1)
            label_np = self.convert_label_to_numpy(Image.fromarray(mask_np, mode='L'))

            # Ensure mask shape is compatible before passing to albumentations
            # Albumentations expects image and mask to have same spatial dimensions initially.
            if label_np.shape[:2] != img_np.shape[:2]:
                 print(f"Warning: Shape mismatch before transform for {img_path}. Image shape: {img_np.shape[:2]}, Mask shape: {label_np.shape}. Resizing mask.")
                 label_np = cv2.resize(label_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        # Create sample dictionary for Albumentations
        sample = {'image': img_np, 'mask': label_np} 
        
        try:
            # Apply transformations
            transformed_sample = self.composed_transforms(**sample) 
        except Exception as e:
            print(f"ERROR: Albumentations transform failed for sample {index} ({img_path}). Error: {e}. Returning None.")
            return None
        
        # Add original filename for testing/validation if not in training mode
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label_to_numpy(self, label_pil):
        """Converts a PIL Image label to a binary NumPy array (0 or 1)."""
        label_np = np.array(label_pil, dtype=np.uint8) 
        # Squeeze channel dimension if present (e.g., shape (H, W, 1))
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim != 2:
            raise ValueError(f"Unexpected label dimension: {label_np.ndim} for {label_pil.size}")

        # Create a binary mask: 1 where label > 0, 0 otherwise
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 
        
        return label_index

    def __len__(self):
        return len(self.imgs)

# --- MixUp Helper ---
# This function is typically used for data augmentation during training.
# It's included here for completeness but might not be used by default.
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

# --- Import config ---
# Assumes config.py is in the same directory or accessible in PYTHONPATH
import config

# --- Global data split dictionaries ---
# These will be populated by ImageFolder during initialization
TRAIN_DATA = []
VAL_DATA = []
TEST_DATA = []

# --- Function to setup dataset loaders ---
def setup_dataloaders(args):
    """
    Sets up DataLoader objects for training, validation, and testing.
    Handles dataset instantiation and potential errors.
    """
    logging.info("Setting up data loaders...")
    
    # Determine base path from config
    try:
        dataset_info = config.DATASET_CONFIG[args.dataset_name]
        base_path = dataset_info['path']
    except KeyError:
        raise ValueError(f"Dataset '{args.dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")
    except Exception as e:
        raise RuntimeError(f"Error accessing dataset configuration for '{args.dataset_name}': {e}")

    train_loader, val_loader, test_loader = None, None, None
    
    # --- Train DataLoader ---
    try:
        train_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='train')
        if train_set and len(train_set) > 0:
            TRAIN_DATA.extend(train_set.imgs) # Store for potential inspection
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate_fn)
            logging.info(f"Train DataLoader created with {len(train_set)} samples.")
        else:
            logging.warning("Train dataset is empty. No training loader created.")
    except RuntimeError as e:
        logging.error(f"Failed to initialize training dataset: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during training dataset setup: {e}")

    # --- Validation DataLoader ---
    try:
        val_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='val')
        if val_set and len(val_set) > 0:
            VAL_DATA.extend(val_set.imgs) # Store for potential inspection
            val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn) # Batch size 1 for validation/testing
            logging.info(f"Validation DataLoader created with {len(val_set)} samples.")
        else:
            logging.warning("Validation dataset is empty. No validation loader created.")
    except RuntimeError as e:
        logging.error(f"Failed to initialize validation dataset: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during validation dataset setup: {e}")

    # --- Test DataLoader ---
    try:
        test_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='test')
        if test_set and len(test_set) > 0:
            TEST_DATA.extend(test_set.imgs) # Store for potential inspection
            test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn) # Batch size 1 for validation/testing
            logging.info(f"Test DataLoader created with {len(test_set)} samples.")
        else:
            logging.warning("Test dataset is empty. No test loader created.")
    except RuntimeError as e:
        logging.error(f"Failed to initialize test dataset: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during test dataset setup: {e}")

    if not train_loader and not val_loader and not test_loader:
        raise RuntimeError("FATAL: No data loaders could be created. Please check dataset configuration and paths.")
        
    return train_loader, val_loader, test_loader