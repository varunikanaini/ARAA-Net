import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob 
import logging 

import custom_transforms as tr 
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING


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
        'has_predefined_splits': False,
        'is_nested': True,
        'has_class_folders_under_split': False,
        'has_category_folders_under_split': True,
        'image_subpath_in_category': 'images',
        'mask_subpath_in_category': 'masks',
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
                    mask_path_attempt = os.path.join(mask_category_path, img_name_base + mask_ext)
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path_attempt):
                        dataset_items.append((img_path, mask_path_attempt))
                    else:
                        logging.warning(f"Skipping: Missing image or mask for '{img_name_base}' in '{image_category_path}'. (Image: {img_path}, Mask attempt: {mask_path_attempt})")

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
    def __init__(self, image_mask_paths, args, split='train'): 
        self.imgs = image_mask_paths 
        self.args = args 
        self.split = split

        # Retrieve dimensions from args with defaults
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
                tr.FixedResize(h=scale_h, w=scale_w), # Resize first
                tr.RandomCrop((crop_h, crop_w)), # Then random crop
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(h=scale_h, w=scale_w), # Resize first
                tr.CenterCrop((crop_h, crop_w)), # Then center crop for consistency
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