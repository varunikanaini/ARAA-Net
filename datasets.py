import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
import logging

import custom_transforms as tr
from config import DATA_ROOT, KAGGLE_DATASET_MAPPING

# Dataset Configuration
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

def preprocess_dataset(image_dir, mask_dir, output_mask_dir, image_exts=('.png', '.jpg', '.jpeg'), mask_ext='.png'):
    """Preprocess dataset to ensure masks match image sizes."""
    os.makedirs(output_mask_dir, exist_ok=True)
    for ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f'*{ext}')):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f'{img_name}{mask_ext}')
            if os.path.exists(mask_path):
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path)
                if img.size != mask.size:
                    logging.info(f"Resizing mask {mask_path} from {mask.size} to {img.size}")
                    mask = mask.resize(img.size, Image.NEAREST)
                    mask.save(os.path.join(output_mask_dir, f'{img_name}{mask_ext}'))
                else:
                    mask.save(os.path.join(output_mask_dir, f'{img_name}{mask_ext}'))
            else:
                logging.warning(f"Mask not found for {img_path}")

def make_dataset(root_path_for_dataset, dataset_name, split_name='all'):
    """Collects all image/mask pairs for a given dataset and split configuration."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        logging.error(f"Dataset config not found for '{dataset_name}'.")
        return []

    image_exts = config['image_ext']
    mask_ext = config['mask_ext']
    mask_suffix = config.get('mask_suffix', '')

    dataset_items = []
    logging.info(f"Collecting data for dataset '{dataset_name}' (split: '{split_name}') from '{root_path_for_dataset}'")

    if not os.path.exists(root_path_for_dataset):
        logging.error(f"Dataset base directory not found: '{root_path_for_dataset}'.")
        return []

    search_root = root_path_for_dataset

    if config['has_category_folders_under_split']:
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

    if not dataset_items:
        logging.error(f"No valid image/mask pairs found in '{search_root}' for dataset '{dataset_name}'.")
    return dataset_items

class ImageFolder(data.Dataset):
    """Custom dataset class that loads image and mask pairs with transformations."""
    def __init__(self, image_mask_paths, args, split='train'):
        self.imgs = image_mask_paths
        self.args = args
        self.split = split

        scale_h = getattr(args, 'scale_h', 576)
        scale_w = getattr(args, 'scale_w', 896)
        crop_h = getattr(args, 'crop_size_h', scale_h)
        crop_w = getattr(args, 'crop_size_w', scale_w)
        crop_h = min(crop_h, scale_h)
        crop_w = min(crop_w, scale_w)

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.FixedResize(scale_h, scale_w),
                tr.RandomCrop((crop_h, crop_w)),
                tr.RandomGaussianBlur(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else:
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(scale_h, scale_w),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        logging.info(f"Image path: {img_path}, Size: {img.size}")
        logging.info(f"Mask path: {gt_path}, Size: {target.size}")
        target = self.convert_label(target)
        logging.info(f"Converted mask size: {target.size}")
        sample = {'image': img, 'label': target, 'name': os.path.basename(img_path)}
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