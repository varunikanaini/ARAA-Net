import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob 
import logging 

import custom_transforms as tr 
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING


# --- Dataset Configuration Dictionary (Essential for COVID-19_Radiography) ---
DATASET_CONFIGS = {
    'TSRS_RSNA-Epiphysis': {
        'image_ext': ('.jpg', '.jpeg'), 
        'image_subpath': '',
        'mask_subpath': 'GT',
        'mask_ext': '.png',
        'has_predefined_splits': True,
        'is_nested_under_split_root': False,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    'TSRS_RSNA-Articular-Surface': {
        'image_ext': ('.jpg', '.jpeg'),
        'image_subpath': '',
        'mask_subpath': 'GT',
        'mask_ext': '.png',
        'has_predefined_splits': True, 
        'is_nested_under_split_root': False,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    },
    
    'KOA': { 
        'image_ext': ('.png', '.jpg', '.jpeg'), 
        'mask_ext': '.png',
        'is_nested': True,
        'has_predefined_splits': True,
        'has_class_folders_under_split': True,
        'has_category_folders_under_split': False,
        'mask_suffix': '_mask'
    },
    
    'COVID-19_Radiography': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'mask_ext': '.png',
        'has_predefined_splits': False, # Needs programmatic splitting
        'is_nested': True,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': True, # Has COVID, Normal, etc. folders
        'image_subpath_in_category': 'images', # images are in category/images
        'mask_subpath_in_category': 'masks',   # masks are in category/masks
        'mask_suffix': ''
    },
    
    'JSRT': { 
        'image_ext': ('.png', '.jpg', '.jpeg'),
        'image_subpath': 'cxr',
        'mask_subpath': 'masks',
        'mask_ext': '.png',
        'has_predefined_splits': False,
        'is_nested': False,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': False,
        'mask_suffix': ''
    }
}


def make_dataset(root_path_for_dataset, dataset_name, split_name='all'): 
    """
    Collects all image/mask pairs for a given dataset and split configuration.
    This function has been enhanced to handle various dataset structures based on DATASET_CONFIGS.
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

    search_root = root_path_for_dataset
    if config['has_predefined_splits'] and split_name != 'all':
        search_root = os.path.join(root_path_for_dataset, split_name)
        if not os.path.exists(search_root):
            logging.error(f"Predefined split directory '{split_name}' not found at '{search_root}'.")
            return []
        logging.info(f"Searching within predefined split directory: '{search_root}'")


    if config['has_class_folders_under_split']:
        logging.info(f"Handling class-folder-nested structure for '{dataset_name}'.")
        class_subdirs = [os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        class_subdirs.sort()
        if not class_subdirs: 
            logging.warning(f"No class subdirectories found in '{search_root}'.")
        
        for class_dir in class_subdirs:
            for ext in image_exts:
                image_files = glob.glob(os.path.join(class_dir, '*' + ext), recursive=False)
                for img_path in image_files:
                    img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path_attempt = os.path.join(class_dir, img_name_base + mask_suffix + mask_ext) if mask_suffix else os.path.join(class_dir, img_name_base + mask_ext)
                    if mask_path_attempt and os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else: 
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{class_dir}'.")

    elif config['has_category_folders_under_split']:
        logging.info(f"Handling category-folder-nested structure for '{dataset_name}'.")
        category_subdirs = [os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        category_subdirs.sort()
        if not category_subdirs: 
            logging.warning(f"No category subdirectories found in '{search_root}'.")

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
                    mask_path_attempt = os.path.join(mask_category_path, img_name_base + mask_ext)
                    if os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else: 
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{image_category_path}'.")

    else: # Flat Structure
        image_subpath_relative = config.get('image_subpath', '')
        mask_subpath_relative = config.get('mask_subpath', '')
        image_dir = os.path.join(search_root, image_subpath_relative)
        mask_dir = os.path.join(search_root, mask_subpath_relative)
        
        logging.info(f"Handling flat structure for '{dataset_name}'. Expected images in: '{image_dir}', masks in: '{mask_dir}'")

        if not os.path.exists(image_dir): 
            logging.error(f"Image directory not found: {image_dir}")
            return []
        if not os.path.exists(mask_dir): 
            logging.error(f"Mask directory not found: {mask_dir}")
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
                mask_path_attempt = os.path.join(mask_dir, img_name_base + mask_suffix + mask_ext) if mask_suffix else os.path.join(mask_dir, img_name_base + mask_ext)
                if mask_path_attempt and os.path.exists(mask_path_attempt):
                    dataset_items.append((found_img_path, mask_path_attempt))
                else: 
                    logging.warning(f"Skipping: Missing mask for '{img_name_base}'.")
            else: 
                logging.warning(f"Skipping: Missing image file for '{img_name_base}'.")

    if not dataset_items:
        logging.error(f"No valid image/mask pairs found in '{search_root}' for dataset '{dataset_name}'. Please ensure data exists and matches config.")
        
    return dataset_items


# /kaggle/working/ARAA-Net/datasets.py

# ... (rest of the file remains unchanged)

class ImageFolder(data.Dataset):
    """
    A custom dataset class that loads image and mask pairs. 
    It applies transformations based on provided arguments for scaling and cropping.
    """
    def __init__(self, image_mask_paths, args, split='train'): 
        self.imgs = image_mask_paths 
        self.args = args 
        self.split = split

        # Retrieve dimensions from args with defaults if not present
        scale_h = getattr(args, 'scale_h', 576)
        scale_w = getattr(args, 'scale_w', 896)
        # Use crop_size_h/w from args if available, otherwise default to scale_h/w
        crop_h = getattr(args, 'crop_size_h', scale_h)
        crop_w = getattr(args, 'crop_size_w', scale_w)
        
        # Ensure crop dimensions are not larger than scale dimensions if a fixed resize is applied first
        crop_h = min(crop_h, scale_h)
        crop_w = min(crop_w, scale_w)


        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                # CHANGE THIS LINE: Pass h and w as separate arguments in (H, W) order
                tr.FixedResize(scale_h, scale_w), # Fixed: Pass h, w as separate arguments (assuming FixedResize(h,w))
                tr.RandomCrop((crop_h, crop_w)),      # Pass (H, W) tuple to RandomCrop (this remains a tuple)
                tr.RandomGaussianBlur(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                # CHANGE THIS LINE: Pass h and w as separate arguments in (H, W) order
                tr.FixedResize(scale_h, scale_w), # Fixed: Pass h, w as separate arguments (assuming FixedResize(h,w))
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label, 'name': os.path.basename(img_path)}
        transformed_sample = self.composed_transforms(sample)
        
        return transformed_sample
    
    def convert_label(self, label):
        label_gray = label.convert('L')
        label_np = np.array(label_gray, dtype=np.uint8)
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)