# /kaggle/working/ARAA-Net/datasets.py

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
import torch

# --- IMPORT YOUR CUSTOM TRANSFORM CLASSES ---
# Ensure this import path is correct for your project structure.
import custom_transforms as tr

# --- Global definitions for image/mask extensions ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(data_info, split_requested):
    """
    Collects image-mask pairs. Prioritizes explicit split directories if they exist.
    If not, it collects all items from common locations (including JSRT's content/jsrt structure)
    to prepare for programmatic splitting.
    """
    dataset_items = []
    base_path = data_info['path']
    structure = data_info['structure']
    img_ext = data_info.get('image_ext', IMAGE_EXTENSIONS)
    mask_ext = data_info.get('mask_ext', MASK_EXTENSIONS)

    print(f"make_dataset: Looking for dataset '{structure}' split '{split_requested}' in '{base_path}'")

    # Helper function to find items within given image and mask directories
    def find_items_in_dirs(img_dir, mask_dir, img_ext, mask_ext):
        items = []
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            print(f"make_dataset: Image dir '{img_dir}' or Mask dir '{mask_dir}' not found.")
            return items
        
        for f in os.listdir(img_dir):
            if f.lower().endswith(img_ext):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(img_dir, f)
                mask_found = False
                for ext in mask_ext:
                    mask_full_path = os.path.join(mask_dir, img_name_base + ext)
                    if os.path.exists(mask_full_path):
                        items.append((img_full_path, mask_full_path))
                        mask_found = True
                        break
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in '{img_dir}'. Skipping.")
        return items

    # --- Data Collection Logic ---
    # Function aims to FIND ALL AVAILABLE image-mask pairs.
    # The actual splitting logic is in ImageFolder.__init__.

    explicit_split_found = False
    # Check for explicit split directories ONLY if we're not in 'all' collection mode.
    if split_requested != 'all':
        current_split_dir = os.path.join(base_path, split_requested)
        
        if structure == 'STANDARD' and os.path.isdir(current_split_dir):
            print(f"make_dataset: Found explicit split directory for '{split_requested}' at '{current_split_dir}'.")
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
                dataset_items.extend(find_items_in_dirs(img_dir, mask_dir, img_ext, mask_ext))
                explicit_split_found = True
            else:
                print(f"make_dataset: Could not find standard image/mask subfolders within '{current_split_dir}'.")

        elif structure == 'SIX_DISEASES' and os.path.isdir(current_split_dir):
             classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
             for cls_name in classes:
                 cls_image_path = os.path.join(current_split_dir, cls_name, 'images')
                 cls_mask_path = os.path.join(current_split_dir, cls_name, 'masks')
                 dataset_items.extend(find_items_in_dirs(cls_image_path, cls_mask_path, img_ext, mask_ext))
             explicit_split_found = True

        elif structure == 'TSRS_RSNA':
            split_image_dir = os.path.join(base_path, split_requested)
            split_mask_dir = os.path.join(base_path, f"{split_requested}_labels")
            dataset_items.extend(find_items_in_dirs(split_image_dir, split_mask_dir, img_ext, mask_ext))
            explicit_split_found = True

        if dataset_items: # If we found items using explicit split dirs, return them.
             print(f"make_dataset: Found {len(dataset_items)} items using explicit split '{split_requested}'.")
             return dataset_items

    # --- If explicit splits were not found or not applicable, collect ALL available items ---
    if not dataset_items: # This block runs if explicit splits failed or if split_requested was 'all'.
        print(f"make_dataset: No explicit split data found or requesting 'all'. Collecting all available items from '{base_path}'.")
        
        # --- Collection logic for STANDARD structure (JSRT specific adjustment) ---
        if structure == 'STANDARD':
            jsrt_content_path = os.path.join(base_path, 'content', 'jsrt')
            
            candidate_base_dirs = []
            # Add JSRT's specific content path if it exists
            if os.path.isdir(jsrt_content_path): 
                candidate_base_dirs.append(jsrt_content_path)
            
            # Add standard search locations if not JSRT specific content path OR if JSRT content path doesn't exist
            if structure != 'JSRT' or not os.path.isdir(jsrt_content_path):
                candidate_base_dirs.extend([
                    os.path.join(base_path, 'images'), os.path.join(base_path, 'cxr'), base_path
                ])
            
            potential_img_dir_names = ['images', 'cxr'] # Candidates for image subdirectories
            potential_mask_dir_names = ['masks', 'GT']   # Candidates for mask subdirectories

            found_image_dir, found_mask_dir = None, None
            # Iterate through candidate base directories to find the data
            for b_path in candidate_base_dirs:
                if not os.path.isdir(b_path): continue # Skip if this base path doesn't exist
                
                for img_dir_name in potential_img_dir_names:
                    potential_img_dir = os.path.join(b_path, img_dir_name)
                    if os.path.isdir(potential_img_dir):
                        for mask_dir_name in potential_mask_dir_names:
                            potential_mask_dir = os.path.join(b_path, mask_dir_name)
                            if os.path.isdir(potential_mask_dir):
                                found_image_dir, found_mask_dir = potential_img_dir, potential_mask_dir
                                print(f"make_dataset: Using image dir: '{found_image_dir}', mask dir: '{found_mask_dir}' for collection.")
                                break
                        if found_image_dir: break
                if found_image_dir: break # Stop searching if data was found

            if found_image_dir and found_mask_dir:
                dataset_items.extend(find_items_in_dirs(found_image_dir, found_mask_dir, img_ext, mask_ext))
            else:
                print(f"make_dataset: Could not find suitable image/mask directories under '{base_path}' or its JSRT-specific path for STANDARD collection.")

        # --- Collection logic for COVID19 structure ---
        elif structure == 'COVID19':
            classes = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
            for cls_name in classes:
                cls_image_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'images')
                cls_mask_path = os.path.join(base_path, 'COVID-19_Radiography_Database', cls_name, 'masks')
                dataset_items.extend(find_items_in_dirs(cls_image_path, cls_mask_path, img_ext, mask_ext))
        
        # --- Collection logic for SIX_DISEASES structure ---
        elif structure == 'SIX_DISEASES':
            print(f"make_dataset: SIX_DISEASES collection mode. Looking under '{base_path}'.")
            img_dir_base = os.path.join(base_path, 'images')
            mask_dir_base = os.path.join(base_path, 'masks')
            dataset_items.extend(find_items_in_dirs(img_dir_base, mask_dir_base, img_ext, mask_ext))
            if not dataset_items: # Fallback if direct 'images'/'masks' not found
                print(f"make_dataset: Also checking direct subfolders of '{base_path}' for SIX_DISEASES classes.")
                classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
                for cls_name in classes:
                    cls_image_path = os.path.join(base_path, cls_name, 'images')
                    cls_mask_path = os.path.join(base_path, cls_name, 'masks')
                    dataset_items.extend(find_items_in_dirs(cls_image_path, cls_mask_path, img_ext, mask_ext))
        
        # --- Collection logic for TSRS_RSNA ---
        elif structure == 'TSRS_RSNA':
             print(f"make_dataset: TSRS_RSNA fallback collection mode. Looking under '{base_path}'.")
             if os.path.isdir(base_path):
                dataset_items.extend(find_items_in_dirs(base_path, base_path, img_ext, mask_ext)) # Assume mask is in same dir
             else:
                  print(f"make_dataset: Base path '{base_path}' not found for TSRS_RSNA collection.")

    if not dataset_items:
        print(f"Warning: make_dataset found 0 items for structure '{structure}' in '{base_path}'.")
        
    return dataset_items

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        """
        Initializes the dataset. Collects all data and performs programmatic splitting (80/10/10)
        since explicit split directories are assumed to be absent.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 

        try:
            dataset_info = config.DATASET_CONFIG[dataset_name]
            # --- CRITICAL PATH ADJUSTMENT FOR JSRT ---
            # If JSRT data is at: /kaggle/working/ARAA-Net/data/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/
            # THEN config.py's 'path' for JSRT MUST be set to:
            # os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt')
            base_path = dataset_info['path'] 
        except KeyError:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.DATASET_CONFIG. Available: {list(config.DATASET_CONFIG.keys())}")
        except Exception as e:
            raise RuntimeError(f"Error accessing dataset config for '{dataset_name}': {e}")

        # Collect all available items using make_dataset with 'all' split trigger
        all_available_items = make_dataset(data_info=dataset_info, split_requested='all')

        if not all_available_items:
            raise RuntimeError(f"ImageFolder: Failed to collect any data for dataset '{self.dataset_name}' at path '{root}'. Cannot proceed.")
        
        # --- Perform Programmatic Split (80/10/10) ---
        random.seed(42)
        random.shuffle(all_available_items)
        
        total_size = len(all_available_items)
        train_ratio, val_ratio = 0.8, 0.1
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        if total_size < 3:
            train_size, val_size = (1, 1) if total_size > 1 else (1, 0)
            test_size = max(0, total_size - train_size - val_size)
        else:
            test_size = max(0, test_size)

        print(f"ImageFolder: Performing programmatic split for {dataset_name}. Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")

        # Assign items to self.imgs based on the requested split
        if split == 'train': self.imgs = all_available_items[:train_size]
        elif split == 'val': self.imgs = all_available_items[train_size : train_size + val_size]
        elif split == 'test': self.imgs = all_available_items[train_size + val_size :]
        else: raise ValueError(f"ImageFolder: Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        # --- Final Check and Shuffle ---
        if not self.imgs:
            raise RuntimeError(f"ImageFolder: No images loaded for dataset '{self.dataset_name}' split '{self.split}'. Check data paths, structure, and split configuration. Dataset root: '{root}'.")
        else:
            print(f"ImageFolder: Successfully loaded {len(self.imgs)} samples for {dataset_name} split '{self.split}'.")
        
        if self.split == 'train': random.shuffle(self.imgs)

        # --- Transforms Setup (using your custom_transforms) ---
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        DASEG_FIXED_RESIZE_W = getattr(args, 'scale_w', 576)
        DASEG_FIXED_RESIZE_H = getattr(args, 'scale_h', 896)

        # Ensure these `getattr` calls match the arguments your `train.py` script provides.
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H),
                tr.CenterAmplification(min_lesion_area_pixels=getattr(args, 'min_lesion_area_pixels', 576),
                                       expansion_factor=getattr(args, 'expansion_factor', 1.5),
                                       min_bbox_size=(getattr(args, 'min_bbox_h', 32), getattr(args, 'min_bbox_w', 32))) if getattr(args, 'min_lesion_area_pixels', 0) > 0 else lambda x: x,
                tr.RandomAffine(degrees=getattr(args, 'affine_degrees', 7), 
                                translate=(getattr(args, 'affine_translate', 0.07), getattr(args, 'affine_translate', 0.07)), 
                                scale=(getattr(args, 'affine_scale', 0.95), getattr(args, 'affine_scale', 1.05)), 
                                shear=getattr(args, 'affine_shear', 7), 
                                mask_fill_value=0),
                tr.RandomGaussianBlur(radius_range=(getattr(args, 'blur_radius_min', 0.1), getattr(args, 'blur_radius_max', 1.2))),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop(size=(DASEG_FIXED_RESIZE_H, DASEG_FIXED_RESIZE_W)),
                tr.ColorJitter(brightness=getattr(args, 'jitter_brightness', 0.1), 
                               contrast=getattr(args, 'jitter_contrast', 0.1), 
                               saturation=getattr(args, 'jitter_saturation', 0.1), 
                               hue=getattr(args, 'jitter_hue', 0.05)) if getattr(args, 'use_color_jitter', True) else lambda x: x,
                tr.WaveletContrastEnhancement(wavelet=getattr(args, 'wavelet_type', 'haar'), 
                                              level=getattr(args, 'wavelet_level', 1), 
                                              detail_scale_factor=getattr(args, 'wavelet_detail_scale', 1.5)) if random.random() < 0.2 else lambda x: x,
                tr.RandomCutout(num_holes_range=(getattr(args, 'cutout_num_holes_min', 1), getattr(args, 'cutout_num_holes_max', 4)), 
                                max_h_size=getattr(args, 'cutout_max_h', 40), 
                                max_w_size=getattr(args, 'cutout_max_w', 40), 
                                fill_value=0, p=getattr(args, 'cutout_p', 0.5)),
                tr.Normalize(mean=self.mean, std=self.std) # Normalize expects PIL image, converts to tensor internally
            ])
        else: # Validation/Test Transforms
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H),
                tr.Normalize(mean=self.mean, std=self.std) # Normalize expects PIL image, converts to tensor internally
            ])

    def __getitem__(self, index):
        if not self.imgs: return None
            
        img_path, gt_path = self.imgs[index]
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None: raise FileNotFoundError(f"OpenCV could not read image: '{img_path}'.")
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) 
            img_pil = Image.fromarray(img_cv)

            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None: raise FileNotFoundError(f"OpenCV could not read mask: '{gt_path}'.")
            mask_pil = Image.fromarray(mask_cv, mode='L')

            label_np = self.convert_label_to_numpy(mask_pil)
            label_pil_binary = Image.fromarray(label_np, mode='P')

            if label_pil_binary.size != img_pil.size:
                 print(f"Warning: PIL Image size mismatch before transform for {img_path}. Image size: {img_pil.size}, Mask size: {label_pil_binary.size}. Resizing mask.")
                 label_pil_binary = label_pil_binary.resize(img_pil.size, Image.NEAREST)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        sample = {'image': img_pil, 'label': label_pil_binary} 
        
        try:
            transformed_sample = self.composed_transforms(sample) 
        except Exception as e:
            print(f"ERROR: Custom transform pipeline failed for sample {index} ({img_path}). Error: {e}. Returning None.")
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

# --- Import config and setup dataloaders ---
import config # Assumes config.py is accessible

def setup_dataloaders(args):
    logging.info("Setting up data loaders...")
    
    try:
        dataset_info = config.DATASET_CONFIG[args.dataset_name]
        # --- CRITICAL PATH ADJUSTMENT FOR JSRT ---
        # If your JSRT data is at: /kaggle/working/ARAA-Net/data/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/
        # THEN config.py's 'path' for JSRT MUST be set to:
        # os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt')
        # If config.py's path is already correct, this code will use it.
        base_path = dataset_info['path'] 
    except KeyError:
        raise ValueError(f"Dataset '{args.dataset_name}' not found in config.DATASET_CONFIG. Available: {list(config.DATASET_CONFIG.keys())}")
    except Exception as e:
        raise RuntimeError(f"Error accessing dataset config for '{args.dataset_name}': {e}")

    train_loader, val_loader, test_loader = None, None, None
    
    # --- Train DataLoader ---
    try:
        # ImageFolder will now collect all data and split internally based on 'split' argument.
        # The make_dataset function is updated to correctly find JSRT data within 'content/jsrt'.
        train_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='train')
        if train_set and len(train_set) > 0:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate_fn)
            logging.info(f"Train DataLoader created with {len(train_set)} samples.")
        else:
            logging.warning("Train dataset is empty. No training loader created.")
    except RuntimeError as e: logging.error(f"Failed to initialize training dataset: {e}")
    except Exception as e: logging.error(f"Unexpected error during training dataset setup: {e}")

    # --- Validation DataLoader ---
    try:
        val_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='val')
        if val_set and len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)
            logging.info(f"Validation DataLoader created with {len(val_set)} samples.")
        else:
            logging.warning("Validation dataset is empty. No validation loader created.")
    except RuntimeError as e: logging.error(f"Failed to initialize validation dataset: {e}")
    except Exception as e: logging.error(f"Unexpected error during validation dataset setup: {e}")

    # --- Test DataLoader ---
    try:
        test_set = ImageFolder(root=base_path, dataset_name=args.dataset_name, args=args, split='test')
        if test_set and len(test_set) > 0:
            test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)
            logging.info(f"Test DataLoader created with {len(test_set)} samples.")
        else:
            logging.warning("Test dataset is empty. No test loader created.")
    except RuntimeError as e: logging.error(f"Failed to initialize test dataset: {e}")
    except Exception as e: logging.error(f"Unexpected error during test dataset setup: {e}")

    if not train_loader and not val_loader and not test_loader:
        raise RuntimeError("FATAL: No data loaders could be created. Please check dataset configuration and paths.")
        
    return train_loader, val_loader, test_loader

# --- custom_collate_fn ---
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)