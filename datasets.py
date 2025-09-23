# /kaggle/working/ARAA-Net/datasets.py (MODIFIED for new dataset support and improved logging)

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob 
import logging # <<< ADDED for better warnings/errors

import custom_transforms as tr 

# --- NEW: Dataset Configuration Dictionary ---
# Define specific loading instructions for each dataset.
# Adjust image_subpath/mask_subpath/mask_suffix based on actual dataset structures.
DATASET_CONFIGS = {
    'TSRS_RSNA-Epiphysis': {
        'image_ext': ('.jpg', '.jpeg'), 
        'mask_subpath': 'GT', # Labels are in a 'GT' subdirectory relative to root (e.g., train/GT/)
        'mask_ext': '.png',
        'is_nested': False # Images/labels are directly under train/val/test
    },
    'TSRS_RSNA-Articular-Surface': {
        'image_ext': ('.jpg', '.jpeg'),
        'mask_subpath': 'GT',
        'mask_ext': '.png',
        'is_nested': False
    },
    'KOA': { # Common structure: DATA_ROOT/KOA/train/images/, DATA_ROOT/KOA/train/labels/
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'images', # Images are in an 'images' subdirectory
        'mask_subpath': 'labels',  # Masks are in a 'labels' subdirectory
        'mask_ext': '.png',
        'is_nested': False
    },
    'MURA': { # MURA is often nested. Example: MURA/train/study1/image.png, MURA/train/study1_mask.png
              # This config supports recursive search and a mask_suffix convention.
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'mask_suffix': '_mask', # Example: if mask for 'image.png' is 'image_mask.png'
        'mask_ext': '.png',
        'is_nested': True, # Needs to search subdirectories
        'image_subpath': '', # Images are directly in subfolders (e.g., study1)
        'mask_subpath': '' # Masks are also directly in subfolders, found by suffix
    },
    'Chest-Xray': { # Common structure: DATA_ROOT/Chest-Xray/train/images/, DATA_ROOT/Chest-Xray/train/labels/
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'images',
        'mask_subpath': 'labels',
        'mask_ext': '.png',
        'is_nested': False
    }
}
# --- END NEW: Dataset Configuration Dictionary ---


def make_dataset(root, dataset_name): # <<< MODIFIED: Pass dataset_name
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        logging.error(f"Dataset config not found for '{dataset_name}'. Please add it to DATASET_CONFIGS in datasets.py.")
        return []

    image_exts = config['image_ext']
    mask_ext = config['mask_ext']
    
    dataset_items = []

    if config['is_nested']:
        logging.info(f"Attempting to load nested dataset '{dataset_name}' from '{root}' (recursive search)...")
        # Search for image files recursively in the root (e.g., train/study1/image.png)
        image_paths_raw = []
        for ext in image_exts:
            image_paths_raw.extend(glob.glob(os.path.join(root, '**', '*' + ext), recursive=True))
        
        if not image_paths_raw:
            logging.warning(f"No images found recursively in '{root}' for dataset '{dataset_name}' with extensions {image_exts}. Check path and extensions.")

        for img_path in image_paths_raw:
            img_name_base = os.path.splitext(os.path.basename(img_path))[0]
            img_dir = os.path.dirname(img_path)

            mask_path_attempt = None
            if 'mask_suffix' in config and config['mask_suffix']: 
                # e.g., MURA/train/study1/image1_mask.png for image1.png
                potential_mask_path = os.path.join(img_dir, img_name_base + config['mask_suffix'] + mask_ext)
                if os.path.exists(potential_mask_path):
                    mask_path_attempt = potential_mask_path
            
            # If mask_suffix not used or didn't find, check parallel/same directory based on image_subpath/mask_subpath
            if not mask_path_attempt and 'image_subpath' in config and 'mask_subpath' in config and config['image_subpath'] and config['mask_subpath']:
                # Example: images in /train/images/study1/ and masks in /train/labels/study1/
                if config['image_subpath'] in img_dir: # Check if img_dir contains the image_subpath
                    # Replace image_subpath with mask_subpath in the path
                    mask_dir_for_image = img_dir.replace(os.path.sep + config['image_subpath'], os.path.sep + config['mask_subpath'])
                    potential_mask_path = os.path.join(mask_dir_for_image, img_name_base + mask_ext)
                    if os.path.exists(potential_mask_path):
                        mask_path_attempt = potential_mask_path
            
            # Final fallback: Assume mask is in the exact same directory as image, with same name but mask_ext
            if not mask_path_attempt:
                potential_mask_path = os.path.join(img_dir, img_name_base + mask_ext)
                if os.path.exists(potential_mask_path):
                    mask_path_attempt = potential_mask_path

            if mask_path_attempt and os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                dataset_items.append((img_path, mask_path_attempt))
            else:
                logging.warning(f"Skipping: Missing image or mask for '{img_name_base}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")


    else: # Flat structure (e.g., Epiphysis, Articular-Surface, KOA, Chest-Xray)
        image_subpath = config.get('image_subpath', '')
        mask_subpath = config.get('mask_subpath', '')
        image_dir = os.path.join(root, image_subpath)
        mask_dir = os.path.join(root, mask_subpath)
        
        logging.info(f"Attempting to load flat dataset '{dataset_name}' from '{root}'...")
        logging.info(f"Expected image directory: '{image_dir}'")
        logging.info(f"Expected mask directory: '{mask_dir}'")
        logging.info(f"Expected image extensions: {image_exts}")
        logging.info(f"Expected mask extension: {mask_ext}")

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
                potential_mask_path = os.path.join(mask_dir, img_name_base + mask_ext)
                if os.path.exists(potential_mask_path):
                    dataset_items.append((found_img_path, potential_mask_path))
                else:
                    logging.warning(f"Skipping: Missing mask for '{img_name_base}'. (Image: {found_img_path}, Mask attempt: {potential_mask_path})")
            else:
                logging.warning(f"Skipping: Missing image file for '{img_name_base}'.")

    if not dataset_items:
        logging.error(f"Found 0 images in '{root}' for dataset '{dataset_name}'. "
                      f"Please ensure the dataset exists at '{root}', matches the '{dataset_name}' config in datasets.py, "
                      f"and contains valid image/mask pairs. Current config: {config}")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.args = args 
        self.split = split
        self.dataset_name = args.dataset_name 

        # Pass dataset_name to make_dataset
        self.imgs = make_dataset(root, self.dataset_name) 

        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

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
        else: # Validation/Test - no data augmentation or complex preprocessing
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