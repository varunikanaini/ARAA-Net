# /kaggle/working/ARAA-Net/datasets.py (MODIFIED for precise structures and robust make_dataset)

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob 
import logging 

import custom_transforms as tr 
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING # NEW IMPORTS: For dynamic dataset paths


# --- NEW: Dataset Configuration Dictionary ---
DATASET_CONFIGS = {
    # Existing RSNA datasets (assuming structure: DATA_ROOT/<dataset_name>/<split>/images/ or /GT/)
    'TSRS_RSNA-Epiphysis': {
        'image_ext': ('.jpg', '.jpeg'), 
        'image_subpath': '', # Images are directly in split root (e.g., train/image.jpg)
        'mask_subpath': 'GT', # Masks are in 'GT' subdirectory relative to split root
        'mask_ext': '.png',
        'has_predefined_splits': True, # <<< NEW: Explicitly state it has train/val/test folders
        'is_nested_under_split_root': False, # Images/masks directly under split_root or simple subpaths
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    'TSRS_RSNA-Articular-Surface': {
        'image_ext': ('.jpg', '.jpeg'),
        'image_subpath': '',
        'mask_subpath': 'GT',
        'mask_ext': '.png',
        'has_predefined_splits': True, # <<< NEW
        'is_nested_under_split_root': False,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    
    # Your KOA dataset (lvv-koa) structure: DATA_ROOT/lvv-koa/train/0/image.png, .../0/image_mask.png
    'KOA': { 
        'image_ext': ('.png', '.jpg', '.jpeg'), 
        'mask_ext': '.png',
        'has_predefined_splits': True, # <<< NEW: train/val/test folders exist under lvv-koa
        'is_nested_under_split_root': True, # Split folder contains further nested structures (class folders)
        'has_class_folders_under_split': True, # Split folder contains class subfolders (0,1,2,3,4)
        'has_category_folders_under_split': False,
        'mask_suffix': '_mask' # CRUCIAL ASSUMPTION: Mask file is 'image_name_mask.png'
    },
    
    # COVID-19 Radiography Database: DATA_ROOT/COVID-19_Radiography_Dataset/COVID/images/image.png, .../COVID/masks/mask.png
    'COVID-19_Radiography': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'mask_ext': '.png',
        'has_predefined_splits': False, # <<< NEW: No train/val/test folders, programmatic split needed
        'is_nested_under_split_root': True, # Contains category folders directly
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': True, # Root folder contains category subfolders (COVID, Normal etc.)
        'image_subpath_in_category': 'images', # Path relative to category folder
        'mask_subpath_in_category': 'masks',   # Path relative to category folder
        'mask_suffix': '' # Masks have same name as image
    },
    
    # JSRT Dataset: DATA_ROOT/jsrt/cxr/image.png, .../masks/mask.png
    # This dataset typically doesn't have predefined train/val/test splits.
    'JSRT': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'cxr', # Images are in 'cxr' subdirectory relative to root
        'mask_subpath': 'masks', # Masks are in 'masks' subdirectory relative to root
        'mask_ext': '.png',
        'has_predefined_splits': False, # <<< NEW: No train/val/test folders, programmatic split needed
        'is_nested_under_split_root': False, # Flat images/masks under cxr/masks
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': '' # Masks have same name as image
    }
}
# --- END NEW: Dataset Configuration Dictionary ---


def make_dataset(root_path_for_dataset, dataset_name, split_name='all'): # <<< MODIFIED: root is now dataset_base_path, added split_name
    """
    Collects all image/mask pairs for a given dataset and split configuration.
    root_path_for_dataset: The base path for the *entire* dataset (e.g., DATA_ROOT/COVID-19_Radiography_Dataset)
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

    # --- Determine the actual root for image/mask searching based on predefined splits ---
    search_root = root_path_for_dataset
    if config['has_predefined_splits'] and split_name != 'all':
        # For datasets with predefined train/val/test folders (e.g., RSNA, KOA)
        search_root = os.path.join(root_path_for_dataset, split_name)
        if not os.path.exists(search_root):
            logging.error(f"Predefined split directory '{split_name}' not found at '{search_root}'.")
            return []
        logging.info(f"Searching within predefined split directory: '{search_root}'")


    # --- Case 1: Nested Class Folders (e.g., KOA: search_root/0/image.png) ---
    if config['has_class_folders_under_split']:
        logging.info(f"Handling class-folder-nested structure for '{dataset_name}'.")
        class_subdirs = [os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        class_subdirs.sort()

        if not class_subdirs:
            logging.warning(f"No class subdirectories found in '{search_root}'. Expected structure like '{search_root}/0/', '{search_root}/1/', etc.")
            return []
        
        for class_dir in class_subdirs:
            for ext in image_exts:
                image_files = glob.glob(os.path.join(class_dir, '*' + ext), recursive=False)
                for img_path in image_files:
                    img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                    
                    mask_path_attempt = None
                    if mask_suffix:
                        mask_path_attempt = os.path.join(class_dir, img_name_base + mask_suffix + mask_ext)
                    else: 
                        mask_path_attempt = os.path.join(class_dir, img_name_base + mask_ext)
                    
                    if mask_path_attempt and os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else:
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{class_dir}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")

    # --- Case 2: Nested Category Folders (e.g., COVID-19 Radiography: search_root/COVID/images/img.png) ---
    elif config['has_category_folders_under_split']:
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
                image_files = glob.glob(os.path.join(image_category_path, '*' + ext), recursive=False)
                for img_path in image_files:
                    img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path_attempt = os.path.join(mask_category_path, img_name_base + mask_ext) # Assume same name, mask_ext in masks folder
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else:
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{image_category_path}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")

    # --- Case 3: Flat Structure (e.g., RSNA, JSRT: search_root/images/img.png or search_root/cxr/img.png) ---
    else: 
        image_subpath_relative = config.get('image_subpath', '')
        mask_subpath_relative = config.get('mask_subpath', '')
        image_dir = os.path.join(search_root, image_subpath_relative)
        mask_dir = os.path.join(search_root, mask_subpath_relative)
        
        logging.info(f"Handling flat structure for '{dataset_name}'.")
        logging.info(f"Expected images in: '{image_dir}'")
        logging.info(f"Expected masks in: '{mask_dir}'")

        if not os.path.exists(image_dir):
            logging.error(f"Image directory not found for flat dataset '{dataset_name}': {image_dir}")
            return []
        if not os.path.exists(mask_dir):
            logging.error(f"Mask directory not found for flat dataset '{dataset_name}': {mask_dir}")
            return []

        logging.info(f"Image directory '{image_dir}' exists.")
        logging.info(f"Mask directory '{mask_dir}' exists.")

        img_list_basenames = []
        for ext in image_exts:
            img_list_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        for img_name_base in img_list_basenames:
            found_img_path = None
            for ext in image_exts:
                potential_img_path = os.path.join(image_dir, img_name_base + ext)
                if os.path.exists(potential_img_path):
                    found_img_path = potential_img_path
                    break

            if found_img_path:
                mask_path_attempt = None
                if mask_suffix:
                    mask_path_attempt = os.path.join(mask_dir, img_name_base + mask_suffix + mask_ext)
                else: 
                    mask_path_attempt = os.path.join(mask_dir, img_name_base + mask_ext)

                if mask_path_attempt and os.path.exists(mask_path_attempt):
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
    def __init__(self, image_mask_paths, args, split='train'): # <<< MODIFIED: Now takes list of (image_path, mask_path)
        self.imgs = image_mask_paths # This is the list from make_dataset
        self.args = args 
        self.split = split
        self.dataset_name = args.dataset_name 

        min_lesion_area = getattr(args, 'min_lesion_area_pixels', 576)
        expansion_factor = getattr(args, 'expansion_factor', 1.5)
        min_bbox_h = getattr(args, 'min_bbox_h', 32)
        min_bbox_w = getattr(args, 'min_bbox_w', 32)

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((args.scale_h, args.scale_w)), 
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
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
        label_np = np.array(label, dtype=np.uint8)
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)