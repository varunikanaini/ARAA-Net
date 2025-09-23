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
        'is_nested': False, 
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    'TSRS_RSNA-Articular-Surface': {
        'image_ext': ('.jpg', '.jpeg'),
        'image_subpath': '',
        'mask_subpath': 'GT',
        'mask_ext': '.png',
        'is_nested': False,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    
    # Your KOA dataset (lvv-koa) structure: DATA_ROOT/lvv-koa/train/0/image.png, .../0/image_mask.png
    'KOA': { 
        'image_ext': ('.png', '.jpg', '.jpeg'), 
        'mask_ext': '.png',
        'is_nested': True, # Needs recursive search due to class folders
        'has_class_folders_under_split': True, # Split folder contains class subfolders (0,1,2,3,4)
        'has_category_folders_under_split': False,
        'mask_suffix': '_mask' # CRUCIAL ASSUMPTION: Mask file is 'image_name_mask.png'
    },
    
    # COVID-19 Radiography Database: DATA_ROOT/COVID-19_Radiography_Dataset/COVID/images/image.png, .../COVID/masks/mask.png
    'COVID-19_Radiography': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'mask_ext': '.png',
        'is_nested': True, # Split folder contains further nested structures (category folders)
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': True, # Split folder contains category subfolders (COVID, Normal etc.)
        'image_subpath_in_category': 'images', # Path relative to category folder
        'mask_subpath_in_category': 'masks',   # Path relative to category folder
        'mask_suffix': '' # Masks have same name as image
    },
    
    # JSRT Dataset: DATA_ROOT/jsrt/cxr/image.png, .../masks/mask.png
    # Assuming user will create train/val/test folders *within* the downloaded root for splitting,
    # and then copy 'cxr' and 'masks' into them.
    'JSRT': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'cxr', # Images are in 'cxr' subdirectory relative to split root
        'mask_subpath': 'masks', # Masks are in 'masks' subdirectory relative to split root
        'mask_ext': '.png',
        'is_nested': False, # Not nested structure under split root
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': '' # Masks have same name as image
    }
}
# --- END NEW: Dataset Configuration Dictionary ---


def make_dataset(root, dataset_name): 
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        logging.error(f"Dataset config not found for '{dataset_name}'. Please add it to DATASET_CONFIGS in datasets.py.")
        return []

    image_exts = config['image_ext']
    mask_ext = config['mask_ext']
    mask_suffix = config.get('mask_suffix', '') # Get suffix if defined
    
    dataset_items = []
    
    logging.info(f"Loading dataset '{dataset_name}' from '{root}' with config: {config}")

    if not os.path.exists(root):
        logging.error(f"Root directory for split not found: '{root}'. Please check data path.")
        return []

    # --- Case 1: Nested Class Folders (e.g., KOA: train/0/image.png) ---
    if config['has_class_folders_under_split']:
        logging.info(f"Handling class-folder-nested structure for '{dataset_name}'.")
        class_subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        class_subdirs.sort()

        if not class_subdirs:
            logging.warning(f"No class subdirectories found in '{root}'. Expected structure like '{root}/0/', '{root}/1/', etc.")
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

    # --- Case 2: Nested Category Folders (e.g., COVID-19 Radiography: root/COVID/images/img.png) ---
    elif config['has_category_folders_under_split']:
        logging.info(f"Handling category-folder-nested structure for '{dataset_name}'.")
        category_subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        category_subdirs.sort()

        if not category_subdirs:
            logging.warning(f"No category subdirectories found in '{root}'. Expected structure like '{root}/COVID/', '{root}/Normal/', etc.")
            return []

        for category_dir in category_subdirs:
            image_category_path = os.path.join(category_dir, config['image_subpath_in_category'])
            mask_category_path = os.path.join(category_dir, config['mask_subpath_in_category'])

            if not os.path.exists(image_category_path):
                logging.warning(f"Image subpath not found in '{category_dir}': {image_category_path}. Skipping category.")
                continue
            if not os.path.exists(mask_category_path):
                logging.warning(f"Mask subpath not found in '{category_dir}': {mask_category_path}. Skipping category.")
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

    # --- Case 3: Flat Structure (e.g., RSNA, JSRT, Chest-Xray with images/labels subdirs) ---
    else: 
        image_subpath_relative = config.get('image_subpath', '')
        mask_subpath_relative = config.get('mask_subpath', '')
        image_dir = os.path.join(root, image_subpath_relative)
        mask_dir = os.path.join(root, mask_subpath_relative)
        
        logging.info(f"Handling flat structure for '{dataset_name}'.")
        logging.info(f"Expected images in: '{image_dir}'")
        logging.info(f"Expected masks in: '{mask_dir}'")

        if not os.path.exists(image_dir):
            logging.error(f"Image directory not found for flat dataset '{dataset_name}': {image_dir}")
            return []
        if not os.path.exists(mask_dir):
            logging.error(f"Mask directory not found for flat dataset '{dataset_name}': {mask_dir}")
            return []

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
        logging.error(f"No valid image/mask pairs found in '{root}' for dataset '{dataset_name}'. "
                      f"Please ensure the data exists and matches the config in datasets.py. Current config: {config}")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.args = args 
        self.split = split
        self.dataset_name = args.dataset_name 

        self.imgs = make_dataset(root, self.dataset_name) 

        # Default CenterAmplification args, ensure they exist even if not used by this split
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