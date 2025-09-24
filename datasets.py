# /kaggle/working/ARAA-Net/datasets.py

import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
import logging
from torchvision import transforms
import glob 

# Import custom transforms
import custom_transforms as tr 

# NEW: Import DATA_ROOT and KAGGLE_DATASET_MAPPING from config
# This is a local import to prevent circular dependencies if config.py also imports datasets.
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING 

# --- NEW: Dataset Configuration Dictionary ---
DATASET_CONFIGS = {
    'TSRS_RSNA-Epiphysis': {
        'image_ext': ('.jpg', '.jpeg'), 
        'image_subpath': '', # Images are directly in split root (e.g., train/image.jpg)
        'mask_subpath': 'GT', # Masks are in 'GT' subdirectory relative to split root
        'mask_ext': '.png',
        'has_predefined_splits': True, # Explicitly state it has train/val/test folders
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': '', # Masks have same name as image
        'transform_params': {
            'resize_w': 896, # Target width for FixedResize (PIL expects W,H)
            'resize_h': 576, # Target height for FixedResize
            'crop_size_h': 576, # Height for RandomCrop (PyTorch expects H,W)
            'crop_size_w': 576  # Width for RandomCrop
        },
        'label_map_type': 'binary_0_to_1' # Example: Convert any non-zero pixel to 1
    },
    'JSRT': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'cxr', # Images are in 'cxr' subdirectory relative to root
        'mask_subpath': 'masks', # Masks are in 'masks' subdirectory relative to root
        'mask_ext': '.png',
        'has_predefined_splits': False, # No train/val/test folders, programmatic split needed
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': '', # Masks have same name as image
        'transform_params': {
            'resize_w': 448, # Target width
            'resize_h': 448, # Target height
            'crop_size_h': 448, # Height for RandomCrop
            'crop_size_w': 448  # Width for RandomCrop
        },
        'label_map_type': 'binary_0_to_1'
    },
    # --- THIS ENTRY IS CRUCIAL AND MUST BE PRESENT ---
    'COVID-19_Radiography': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'mask_ext': '.png',
        'has_predefined_splits': False, # Programmatic split
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': True, # This dataset has subfolders like 'COVID', 'Normal' etc.
        'image_subpath_in_category': 'images', # Path relative to 'COVID' folder, e.g., 'COVID/images'
        'mask_subpath_in_category': 'masks',   # Path relative to 'COVID' folder, e.g., 'COVID/masks'
        'mask_suffix': '', # Masks have same name as image
        'transform_params': {
            'resize_w': 448, 
            'resize_h': 448, 
            'crop_size_h': 448, 
            'crop_size_w': 448  
        },
        'label_map_type': 'binary_0_to_1'
    }
    # --- END COVID-19_Radiography Configuration ---
}
# --- END NEW: Dataset Configuration Dictionary ---


def make_dataset(root_path_for_dataset, dataset_name, split_name='all'): 
    """
    Collects all image/mask pairs for a given dataset and split configuration.
    root_path_for_dataset: The base path for the *entire* dataset (e.g., DATA_ROOT/COVID-19_Radiography_Dataset)
                           or for a specific split (e.g., DATA_ROOT/TSRS_RSNA-Epiphysis/train)
    dataset_name: The key from DATASET_CONFIGS
    split_name: 'train', 'val', 'test', or 'all'. For programmatic splits, 'all' collects everything.
    """
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        logging.error(f"Dataset config not found for '{dataset_name}'. Please add it to DATASET_CONFIGS in datasets.py.")
        return []

    image_exts = config['image_ext']
    mask_ext = config['mask_ext']
    mask_suffix = config.get('mask_suffix', '') 
    
    dataset_items = []
    
    logging.info(f"Collecting data for dataset '{dataset_name}' (split: '{split_name}') from '{root_path_for_dataset}' with config: {config}")

    if not os.path.exists(root_path_for_dataset):
        logging.error(f"Dataset base directory not found: '{root_path_for_dataset}'. Please check data path.")
        return []

    search_root = root_path_for_dataset # Base for searching for images/masks


    # --- Case 1: Nested Class Folders (e.g., KOA) ---
    if config.get('has_class_folders_under_split', False):
        logging.info(f"Handling class-folder-nested structure for '{dataset_name}'.")
        class_subdirs = [os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        class_subdirs.sort()
        if not class_subdirs:
            logging.warning(f"No class subdirectories found in '{search_root}'. Expected structure like '{search_root}/0/', '{search_root}/1/', etc.")
            return []
        for class_dir in class_subdirs:
            for ext in image_exts:
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(ext)]
                for img_name in image_files:
                    img_path = os.path.join(class_dir, img_name)
                    img_name_base = os.path.splitext(img_name)[0]
                    mask_path_attempt = os.path.join(class_dir, img_name_base + mask_suffix + mask_ext)
                    if os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else:
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{class_dir}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")

    # --- Case 2: Nested Category Folders (e.g., COVID-19_Radiography) ---
    elif config.get('has_category_folders_under_split', False):
        logging.info(f"Handling category-folder-nested structure for '{dataset_name}'.")
        category_subdirs = [os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        category_subdirs.sort()
        if not category_subdirs:
            logging.warning(f"No category subdirectories found in '{search_root}'. Expected structure like '{search_root}/COVID/', '{search_root}/Normal/', etc.")
            return []
        for category_dir in category_subdirs:
            image_category_path = os.path.join(category_dir, config['image_subpath_in_category'])
            mask_category_path = os.path.join(category_dir, config['mask_subpath_in_category'])
            if not os.path.exists(image_category_path):
                logging.warning(f"Image subpath not found in '{category_dir}': {image_category_path}. Skipping category '{os.path.basename(category_dir)}'.")
                continue
            if not os.path.exists(mask_category_path):
                logging.warning(f"Mask subpath not found in '{category_dir}': {mask_category_path}. Skipping category '{os.path.basename(category_dir)}'.")
                continue
            for ext in image_exts:
                image_files = [f for f in os.listdir(image_category_path) if f.lower().endswith(ext)]
                for img_name in image_files:
                    img_path = os.path.join(image_category_path, img_name)
                    img_name_base = os.path.splitext(img_name)[0]
                    mask_path_attempt = os.path.join(mask_category_path, img_name_base + mask_ext)
                    if os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else:
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{image_category_path}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")

    # --- Case 3: Flat Structure (e.g., RSNA, JSRT) ---
    else: 
        image_subpath_relative = config.get('image_subpath', '')
        mask_subpath_relative = config.get('mask_subpath', '')

        image_dir = os.path.join(search_root, image_subpath_relative)
        mask_dir = os.path.join(search_root, mask_subpath_relative)
        
        logging.info(f"Handling flat/subpath structure for '{dataset_name}'.")
        logging.info(f"Expected images in: '{image_dir}'")
        logging.info(f"Expected masks in: '{mask_dir}'")

        if not os.path.exists(image_dir):
            logging.error(f"Image directory not found for dataset '{dataset_name}': {image_dir}")
            return []
        if not os.path.exists(mask_dir):
            logging.error(f"Mask directory not found for dataset '{dataset_name}': {mask_dir}")
            return []

        logging.info(f"Image directory '{image_dir}' exists.")
        logging.info(f"Mask directory '{mask_dir}' exists.")

        img_list_basenames = []
        for ext in image_exts:
            img_list_basenames.extend([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        for img_name_base in img_list_basenames:
            found_img_path = None
            for ext in image_exts: 
                potential_img_path = os.path.join(image_dir, img_name_base + ext)
                if os.path.exists(potential_img_path):
                    found_img_path = potential_img_path
                    break

            if found_img_path:
                mask_path_attempt = os.path.join(mask_dir, img_name_base + mask_suffix + mask_ext)
                if os.path.exists(mask_path_attempt):
                    dataset_items.append((found_img_path, mask_path_attempt))
                else:
                    logging.warning(f"Skipping: Missing mask for '{img_name_base}'. (Image: {found_img_path}, Mask attempt: {mask_path_attempt})")
            else:
                logging.warning(f"Skipping: Missing image file for '{img_name_base}'.")

    if not dataset_items:
        logging.error(f"No valid image/mask pairs found in '{search_root}' for dataset '{dataset_name}'. "
                      f"Please ensure the data exists and matches the config in datasets.py. Current config: {config}")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, image_mask_paths, args, split='train'): 
        self.imgs = image_mask_paths 
        self.args = args 
        self.split = split
        self.dataset_name = args.dataset_name 
        self.config = DATASET_CONFIGS.get(self.dataset_name)

        if not self.config:
            raise ValueError(f"Dataset configuration for '{self.dataset_name}' not found in DATASET_CONFIGS.")

        transform_params = self.config['transform_params']
        resize_w = transform_params['resize_w']
        resize_h = transform_params['resize_h']
        crop_size_h = transform_params['crop_size_h']
        crop_size_w = transform_params['crop_size_w']

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.FixedResize(resize_w, resize_h), 
                tr.RandomCrop((crop_size_h, crop_size_w)), 
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test splits
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(resize_w, resize_h),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label, 'name': os.path.basename(img_path)}
        transformed_sample = self.composed_transforms(sample)
        
        return transformed_sample
    
    def convert_label(self, label):
        label_rgb = np.array(label)

        if label_rgb.ndim == 3 and label_rgb.shape[2] == 3: 
            label_gray = Image.fromarray(label_rgb).convert('L')
            label_np = np.array(label_gray, dtype=np.uint8)
        elif label_rgb.ndim == 2:
            label_np = label_rgb.astype(np.uint8)
        else:
            logging.warning(f"Unexpected label image dimensions: {label_rgb.shape}. Attempting to convert to 2D.")
            if label_rgb.ndim == 3 and label_rgb.shape[2] == 1:
                 label_np = label_rgb.squeeze(2).astype(np.uint8)
            else:
                 label_np = label_rgb[:,:,0].astype(np.uint8) if label_rgb.ndim == 3 else label_rgb.astype(np.uint8)

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 

        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)