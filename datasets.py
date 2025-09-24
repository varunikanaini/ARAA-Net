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
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING 

# --- NEW: Dataset Configuration Dictionary ---
DATASET_CONFIGS = {
    'TSRS_RSNA-Epiphysis': {
        'image_ext': ('.jpg', '.jpeg'), 
        'image_subpath': '', # Images are directly in split root (e.g., train/image.jpg)
        'mask_subpath': 'GT', # Masks are in 'GT' subdirectory relative to split root
        'mask_ext': '.png',
        'has_predefined_splits': True, # Explicitly state it has train/val/test folders
        'mask_suffix': '', # Masks have same name as image
        'transform_params': {
            'resize_w': 896, # Target width for FixedResize
            'resize_h': 576, # Target height for FixedResize
            'crop_size_h': 576, # Height for RandomCrop (square crop)
            'crop_size_w': 576  # Width for RandomCrop (square crop)
        },
        'label_map_type': 'binary_0_to_1' # Example: Convert any non-zero pixel to 1
    },
    'JSRT': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'cxr', # Images are in 'cxr' subdirectory relative to root
        'mask_subpath': 'masks', # Masks are in 'masks' subdirectory relative to root
        'mask_ext': '.png',
        'has_predefined_splits': False, # No train/val/test folders, programmatic split needed
        'mask_suffix': '', # Masks have same name as image
        'transform_params': {
            'resize_w': 448, # Target width
            'resize_h': 448, # Target height
            'crop_size_h': 448, # Height for RandomCrop
            'crop_size_w': 448  # Width for RandomCrop
        },
        'label_map_type': 'binary_0_to_1'
    }
    # Add other datasets here following a similar structure
}
# --- END NEW: Dataset Configuration Dictionary ---


# REPLACING OLD make_dataset(root) with a more flexible version
def make_dataset(root_path_for_dataset, dataset_name, split_name='all'): 
    """
    Collects all image/mask pairs for a given dataset and split configuration.
    root_path_for_dataset: The base path for the *entire* dataset (e.g., DATA_ROOT/TSRS_RSNA-Epiphysis)
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

    # If predefined splits, the root_path_for_dataset is already pointing to the specific split folder (e.g., /train, /val)
    # If no predefined splits, root_path_for_dataset is the dataset's main root (e.g., /jsrt)
    search_root = root_path_for_dataset 
    
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
        for ext in image_exts: # Iterate through possible extensions
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


class ImageFolder(data.Dataset): # Changed from data.Dataset to Dataset for direct import
    # Now takes `image_mask_paths` (list of tuples) and `args` (Namespace)
    def __init__(self, image_mask_paths, args, split='train'): 
        self.imgs = image_mask_paths 
        self.args = args 
        self.split = split
        self.dataset_name = args.dataset_name 
        self.config = DATASET_CONFIGS.get(self.dataset_name)

        if not self.config:
            raise ValueError(f"Dataset configuration for '{self.dataset_name}' not found in DATASET_CONFIGS.")

        # Dynamically get transform parameters from config
        transform_params = self.config['transform_params']
        resize_w = transform_params['resize_w']
        resize_h = transform_params['resize_h']
        crop_size_h = transform_params['crop_size_h']
        crop_size_w = transform_params['crop_size_w']

        # Define transforms dynamically
        # This replaces the old transform_tr/transform_val methods
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.FixedResize(resize_w, resize_h), # FixedResize expects (w, h)
                tr.RandomCrop((crop_size_h, crop_size_w)), # RandomCrop expects (H, W)
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test splits
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(resize_w, resize_h), # FixedResize expects (w, h)
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
        # This conversion logic (non-zero to 1) is flexible and can be adapted per dataset
        # based on 'label_map_type' in config if different behaviors are needed.
        label_rgb = np.array(label)

        # Ensure it's a 2D array if coming from a grayscale image or has a single channel.
        # If it's RGB, convert to grayscale first or select a channel.
        if label_rgb.ndim == 3 and label_rgb.shape[2] == 3: # If it's an RGB image
            label_gray = Image.fromarray(label_rgb).convert('L') # Convert to grayscale PIL image
            label_np = np.array(label_gray, dtype=np.uint8)
        elif label_rgb.ndim == 2: # Already grayscale or 2D
            label_np = label_rgb.astype(np.uint8)
        else:
            logging.warning(f"Unexpected label image dimensions: {label_rgb.shape}. Attempting to convert to 2D.")
            # Fallback for unexpected dim, e.g., (H,W,1) -> (H,W)
            if label_rgb.ndim == 3 and label_rgb.shape[2] == 1:
                 label_np = label_rgb.squeeze(2).astype(np.uint8)
            else:
                 # Last resort, might lose info if it's genuinely multi-channel meaningful
                 label_np = label_rgb[:,:,0].astype(np.uint8) if label_rgb.ndim == 3 else label_rgb.astype(np.uint8)

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        # Assuming background is 0, foreground is any non-zero value
        label_index[label_np > 0] = 1 

        return Image.fromarray(label_index, mode='P') # 'P' for palettized (single channel)

    def __len__(self):
        return len(self.imgs)