# /kaggle/working/ARAA-Net/datasets.py

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
import torch

# --- Placeholder for custom transforms ---
# Ensure these are correctly imported or defined if 'custom_transforms' is a separate module.
# Using placeholders here for demonstration if not provided.
class CustomTransformPlaceholder:
    def __call__(self, sample): return sample
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
tr = CustomTransformPlaceholder() # Alias

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(data_info, split_requested):
    """
    Collects image-mask pairs. It prioritizes explicit split directories if they exist.
    If not, it attempts to collect all items from common locations to prepare for programmatic splitting.
    This function is primarily for data *collection*. Splitting logic is in ImageFolder.
    """
    dataset_items = []
    base_path = data_info['path']
    structure = data_info['structure']
    img_ext = data_info.get('image_ext', IMAGE_EXTENSIONS)
    mask_ext = data_info.get('mask_ext', MASK_EXTENSIONS)

    print(f"make_dataset: Looking for dataset '{structure}' split '{split_requested}' in '{base_path}'")

    # --- Data Collection Logic ---
    # This function's main goal is to FIND ALL AVAILABLE image-mask pairs.
    # The actual splitting will happen in ImageFolder.__init__.
    
    # Try to find explicit split folders (train, val, test) if 'split_requested' is not 'all'
    explicit_split_found = False
    if split_requested != 'all':
        current_split_dir = os.path.join(base_path, split_requested)
        if os.path.isdir(current_split_dir):
            print(f"make_dataset: Found explicit split directory for '{split_requested}' at '{current_split_dir}'.")
            explicit_split_found = True
            
            # --- Handle STANDARD structure with explicit split folders ---
            if structure == 'STANDARD':
                image_dir_name_candidates = ['images', 'cxr']
                mask_dir_name_candidates = ['masks', 'GT']
                img_dir, mask_dir = None, None

                for img_cand in image_dir_name_candidates:
                    potential_img_dir = os.path.join(current_split_dir, img_cand)
                    if os.path.isdir(potential_img_dir):
                        for mask_cand in mask_dir_name_candidates:
                            potential_mask_dir = os.path.join(current_split_dir, mask_cand)
                            if os.path.isdir(potential_mask_dir):
                                img_dir, mask_dir = potential_img_dir, potential_mask_dir
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
                            if not mask_found: print(f"Warning: Mask not found for image {f} in explicit split '{split_requested}'. Skipping.")
                else:
                    print(f"make_dataset: Could not find standard image/mask subfolders within '{current_split_dir}'.")
                    explicit_split_found = False # Force collection mode

            # --- Handle SIX_DISEASES structure with explicit split folders ---
            elif structure == 'SIX_DISEASES':
                 classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
                 for cls_name in classes:
                     cls_image_path = os.path.join(current_split_dir, cls_name, 'images')
                     cls_mask_path = os.path.join(current_split_dir, cls_name, 'masks')
                     if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path): continue
                     for f in os.listdir(cls_image_path):
                         if f.lower().endswith(img_ext):
                             img_name_base = os.path.splitext(f)[0]
                             img_full_path = os.path.join(cls_image_path, f)
                             mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                             if os.path.exists(mask_full_path):
                                 dataset_items.append((img_full_path, mask_full_path))
                             else:
                                 print(f"Warning: Mask not found for SIX_DISEASES image {f} in class '{cls_name}', split '{split_requested}'. Skipping.")
            
            # --- Handle TSRS_RSNA structure with explicit split folders ---
            elif structure == 'TSRS_RSNA':
                split_image_dir = os.path.join(base_path, split_requested)
                split_mask_dir = os.path.join(base_path, f"{split_requested}_labels")
                if os.path.isdir(split_image_dir) and os.path.isdir(split_mask_dir):
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
                            if not mask_found: print(f"Warning: Mask not found for image {f} in dataset '{structure}', split '{split_requested}'. Skipping.")
                else:
                    print(f"make_dataset: TSRS_RSNA requires '{split_requested}' and '{split_requested}_labels' directories. Not found.")
                    explicit_split_found = False # Force collection mode if these specific dirs aren't present.

            # If we successfully collected items using explicit split dirs, return them.
            if dataset_items:
                 print(f"make_dataset: Found {len(dataset_items)} items using explicit split '{split_requested}'.")
                 return dataset_items

    # --- If explicit split dirs were not found or didn't yield items, collect ALL available items ---
    # This happens for STANDARD/SIX_DISEASES if no explicit split dirs are found, or if split_requested is 'all'.
    if not dataset_items:
        print(f"make_dataset: No explicit split data found or requesting 'all'. Collecting all available items from '{base_path}'.")
        
        # --- Collection logic for STANDARD structure ---
        if structure == 'STANDARD':
            potential_img_dirs = [os.path.join(base_path, 'images'), os.path.join(base_path, 'cxr'), base_path]
            potential_mask_dirs = [os.path.join(base_path, 'masks'), os.path.join(base_path, 'GT'), base_path]
            
            found_image_dir, found_mask_dir = None, None
            for p_img in potential_img_dirs:
                if os.path.isdir(p_img):
                    for p_mask in potential_mask_dirs:
                        if os.path.isdir(p_mask):
                            if p_img == base_path and p_mask == base_path and len(potential_img_dirs) > 1: continue
                            found_image_dir, found_mask_dir = p_img, p_mask
                            print(f"make_dataset: Using image dir: '{found_image_dir}', mask dir: '{found_mask_dir}' for collection.")
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
                        if not mask_found: print(f"Warning: Mask not found for image {f} in collection mode for '{base_path}'. Skipping.")
            else:
                print(f"make_dataset: Could not find suitable image/mask directories under '{base_path}' for STANDARD collection.")

        # --- Collection logic for COVID19 structure ---
        elif structure == 'COVID19':
            classes = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
            for cls_name in classes:
                cls_image_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'images')
                cls_mask_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'masks')
                if not os.path.isdir(cls_image_path) or not os.path.isdir(cls_mask_path): continue
                for f in os.listdir(cls_image_path):
                    if f.lower().endswith(img_ext):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(cls_image_path, f)
                        mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                        if os.path.exists(mask_full_path):
                            dataset_items.append((img_full_path, mask_full_path))
                        else:
                            print(f"Warning: Mask not found for COVID19 image {f} in class '{cls_name}'. Skipping.")
        
        # --- Collection logic for SIX_DISEASES (if explicit splits not found) ---
        elif structure == 'SIX_DISEASES':
            print(f"make_dataset: SIX_DISEASES collection mode. Looking under '{base_path}'.")
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

        # --- Collection logic for TSRS_RSNA (fallback if explicit splits not found) ---
        elif structure == 'TSRS_RSNA':
             print(f"make_dataset: TSRS_RSNA fallback collection mode. Looking under '{base_path}'.")
             if os.path.isdir(base_path):
                for f in os.listdir(base_path):
                     if f.lower().endswith(img_ext):
                         img_name_base = os.path.splitext(f)[0]
                         img_full_path = os.path.join(base_path, f)
                         mask_found = False
                         for ext in mask_ext:
                             mask_full_path = os.path.join(base_path, img_name_base + ext) # Assume mask in same dir
                             if os.path.exists(mask_full_path):
                                  dataset_items.append((img_full_path, mask_full_path))
                                  mask_found = True
                                  break
                         if not mask_found:
                              print(f"Warning: Mask not found for TSRS_RSNA image {f} directly in '{base_path}'. Skipping.")
             else:
                  print(f"make_dataset: Base path '{base_path}' not found for TSRS_RSNA collection.")

    if not dataset_items:
        print(f"Warning: make_dataset found 0 items for structure '{structure}' in '{base_path}'.")
        
    return dataset_items

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        """
        Initializes the dataset. Handles data collection and programmatic splitting (80/10/10)
        if explicit split directories are not found.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 

        # 1. Get dataset info from config
        try:
            dataset_info = config.DATASET_CONFIG[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")
        
        # 2. Collect all available items using make_dataset.
        #    We pass 'all' as the split to ensure make_dataset collects everything.
        #    The actual split will be performed here in __init__.
        all_available_items = make_dataset(data_info=dataset_info, split='all')

        if not all_available_items:
            raise RuntimeError(f"ImageFolder: Failed to collect any data for dataset '{self.dataset_name}' at path '{root}'. Cannot proceed.")
        
        # --- Perform Programmatic Split (80/10/10) ---
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

        # Assign items to self.imgs based on the requested split
        if split == 'train': self.imgs = all_available_items[:train_size]
        elif split == 'val': self.imgs = all_available_items[train_size : train_size + val_size]
        elif split == 'test': self.imgs = all_available_items[train_size + val_size :]
        else: raise ValueError(f"ImageFolder: Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        # --- Final Check and Shuffle ---
        if not self.imgs:
            # This check is crucial: if the requested split resulted in an empty list
            raise RuntimeError(f"ImageFolder: No images loaded for dataset '{self.dataset_name}' split '{self.split}'. Check data paths, structure, and split configuration. Dataset root: '{root}'.")
        else:
            print(f"ImageFolder: Successfully loaded {len(self.imgs)} samples for {dataset_name} split '{self.split}'.")
        
        # Shuffle training data after it's been assigned
        if self.split == 'train':
            random.shuffle(self.imgs)

        # --- Albumentations Transforms Setup ---
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        DASEG_FIXED_RESIZE_W = getattr(args, 'scale_w', 576)
        DASEG_FIXED_RESIZE_H = getattr(args, 'scale_h', 896)

        if self.split == 'train':
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR), 
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, border_mode=cv2.BORDER_CONSTANT, pad_val=(0, 0, 0)),
                A.ToFloat(max_value=255.0), 
                
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=1.0), 
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                ], p=1.0),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5), A.MedianBlur(blur_limit=5, p=0.5)], p=0.5),

                # --- Custom transforms can be integrated here if needed ---
                # Example using A.Lambda for custom transforms:
                # A.Lambda(image=lambda x, **kwargs: tr.YourCustomTransform()(x), mask=lambda x, **kwargs: tr.YourCustomTransformMask()(x))
                
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), 
                ToTensorV2(), 
            ])
        else: # Validation/Test Transforms
            self.composed_transforms = A.Compose([
                A.LongestMaxSize(max_size=max(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=DASEG_FIXED_RESIZE_H, min_width=DASEG_FIXED_RESIZE_W, border_mode=cv2.BORDER_CONSTANT, pad_val=(0, 0, 0)),
                A.ToFloat(max_value=255.0), 
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2(),
            ])

    def __getitem__(self, index):
        if not self.imgs: return None
            
        img_path, gt_path = self.imgs[index]
        try:
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None: raise FileNotFoundError(f"OpenCV could not read image: '{img_path}'.")
            if mask_np is None: raise FileNotFoundError(f"OpenCV could not read mask: '{gt_path}'.")

            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) 
            label_np = self.convert_label_to_numpy(Image.fromarray(mask_np, mode='L'))

            if label_np.shape[:2] != img_np.shape[:2]:
                 print(f"Warning: Shape mismatch before transform for {img_path}. Image shape: {img_np.shape[:2]}, Mask shape: {label_np.shape}. Resizing mask.")
                 label_np = cv2.resize(label_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        sample = {'image': img_np, 'mask': label_np} 
        
        try:
            transformed_sample = self.composed_transforms(**sample) 
        except Exception as e:
            print(f"ERROR: Albumentations transform failed for sample {index} ({img_path}). Error: {e}. Returning None.")
            return None
        
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label_to_numpy(self, label_pil):
        label_np = np.array(label_pil, dtype=np.uint8) 
        if label_np.ndim == 3 and label_np.shape[2] == 1: label_np = label_np.squeeze(2)
        elif label_np.ndim != 2: raise ValueError(f"Unexpected label dimension: {label_np.ndim} for {label_pil.size}")
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 
        return label_index

    def __len__(self):
        return len(self.imgs)

# --- MixUp Helper (optional) ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    if batch_size <= 1: return x, y
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

# --- Import config and setup dataloaders ---
# Assumes config.py is in the same directory or accessible in PYTHONPATH
import config

def setup_dataloaders(args):
    logging.info("Setting up data loaders...")
    
    # Get dataset info from config
    try:
        dataset_info = config.DATASET_CONFIG[args.dataset_name]
        # IMPORTANT: Ensure the 'path' in config.py for JSRT is correct.
        # Based on your error, it should point to the directory containing 'content/jsrt',
        # or directly to 'content/jsrt' if that's where 'cxr' and 'masks' are.
        # If JSRT data is truly at:
        # /kaggle/working/ARAA-Net/data/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/cxr
        # /kaggle/working/ARAA-Net/data/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/masks
        # Then the 'path' in config.py for JSRT should be:
        # os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt')
        base_path = dataset_info['path'] 
    except KeyError:
        raise ValueError(f"Dataset '{args.dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")
    except Exception as e:
        raise RuntimeError(f"Error accessing dataset configuration for '{args.dataset_name}': {e}")

    train_loader, val_loader, test_loader = None, None, None
    
    # --- Train DataLoader ---
    try:
        # Pass the actual root path from config to ImageFolder
        train_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='train')
        if train_set and len(train_set) > 0:
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
            val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)
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
            test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)
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

# --- Placeholder for custom_collate_fn if needed, otherwise use default ---
# This is a dummy implementation; replace if you have a specific collate function.
def custom_collate_fn(batch):
    # Filter out None values which might be returned by __getitem__ on error
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Return None if the batch is empty
    return torch.utils.data.dataloader.default_collate(batch)