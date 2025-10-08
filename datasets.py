import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
import custom_transforms as tr
import config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(data_info, split):
    dataset_items = []
    base_path = data_info['path']
    structure = data_info['structure']
    img_ext = data_info.get('image_ext', IMAGE_EXTENSIONS)
    mask_ext = data_info.get('mask_ext', MASK_EXTENSIONS)

    print(f"Looking for dataset '{structure}' split '{split}' in '{base_path}'")

    if structure == 'TSRS_RSNA':
        split_image_dir = os.path.join(base_path, split)
        split_mask_dir = os.path.join(base_path, f"{split}_labels")

        if not os.path.exists(split_image_dir):
            print(f"Warning: TSRS-like dataset '{structure}' split '{split}': Image directory not found at '{split_image_dir}'.")
            return []
        if not os.path.exists(split_mask_dir):
            print(f"Warning: TSRS-like dataset '{structure}' split '{split}': Label directory not found at '{split_mask_dir}'.")
            return []

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
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in dataset '{structure}', split '{split}'. Skipping.")

    elif structure == 'STANDARD':
        subfolders = data_info.get('subfolders', {}).get(split)
        if not subfolders:
            print(f"Error: 'subfolders' not defined for STANDARD structure for split '{split}'. Cannot find image/mask dirs.")
            return []

        image_dir_name = subfolders.get('images')
        mask_dir_name = subfolders.get('masks', subfolders.get('GT'))

        if not image_dir_name or not mask_dir_name:
            print(f"Error: Missing 'images' or 'masks'/'GT' dir names in subfolders config for STANDARD structure, split '{split}'.")
            return []

        split_image_path = os.path.join(base_path, split, image_dir_name)
        split_mask_path = os.path.join(base_path, split, mask_dir_name)

        if not os.path.exists(split_image_path):
            print(f"Warning: STANDARD split '{split}' Image path not found: '{split_image_path}'.")
            return []
        if not os.path.exists(split_mask_path):
            print(f"Warning: STANDARD split '{split}' Mask path not found: '{split_mask_path}'.")
            return []

        for f in os.listdir(split_image_path):
            if f.lower().endswith(img_ext):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(split_image_path, f)

                mask_found = False
                for ext in mask_ext:
                    mask_full_path = os.path.join(split_mask_path, img_name_base + ext)
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        mask_found = True
                        break
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in dataset '{structure}', split '{split}'. Skipping.")

    elif structure == 'COVID19':
        classes = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for cls_name in classes:
            cls_image_path = os.path.join(base_path, 'COVID-19_Radiography_Dataset', cls_name, 'images')
            cls_mask_path = os.path.join(base_path, 'COVID-19_Radiography_Dataset', cls_name, 'masks')

            if not os.path.exists(cls_image_path) or not os.path.exists(cls_mask_path):
                print(f"Warning: COVID19 class '{cls_name}' image/mask path not found. Skipping class.")
                continue

            for f in os.listdir(cls_image_path):
                if f.lower().endswith(img_ext):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(cls_image_path, f)
                    mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image {f} in class '{cls_name}'. Skipping.")

    elif structure == 'SIX_DISEASES':
        classes = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        for cls_name in classes:
            split_base_path = os.path.join(base_path, cls_name)
            cls_image_path = os.path.join(split_base_path, 'images')
            cls_mask_path = os.path.join(split_base_path, 'masks')

            if not os.path.exists(cls_image_path) or not os.path.exists(cls_mask_path):
                print(f"Warning: SixDiseasesChestXRay class '{cls_name}' image/mask path not found in split '{split}'. Skipping class.")
                continue

            for f in os.listdir(cls_image_path):
                if f.lower().endswith(img_ext):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(cls_image_path, f)
                    mask_full_path = os.path.join(cls_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for SixDiseasesChestXRay image {f} in class '{cls_name}', split '{split}'. Skipping.")
    else:
        raise ValueError(f"Unknown dataset structure type: {structure}. Please define handling for this structure.")

    if not dataset_items:
        print(f"Warning: Found 0 image-mask pairs for dataset structure '{structure}' split '{split}' in '{base_path}'. Please check path and dataset structure.")

    return dataset_items

class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args

        try:
            dataset_info = config.DATASET_CONFIG[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.DATASET_CONFIG. Available datasets: {list(config.DATASET_CONFIG.keys())}")

        self.imgs = make_dataset(dataset_info, split)
        random.shuffle(self.imgs)

        if not self.imgs:
            raise RuntimeError(f"No images found for dataset '{self.dataset_name}' split '{self.split}'. Check paths, structure in config, and file extensions.")
        else:
            print(f"Found {len(self.imgs)} samples for {dataset_name} split '{self.split}'.")

        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w
        scale_h = args.scale_h
        scale_w = args.scale_w

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)) if min_lesion_area > 0 else lambda x: x,
                tr.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, mask_fill_value=0),
                tr.RandomGaussianBlur(radius_range=(0.1, 1.2)),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((scale_h, scale_w)),
                tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05) if hasattr(tr, 'ColorJitter') else lambda x: x,
                tr.WaveletContrastEnhancement(wavelet=args.wavelet_type, level=args.wavelet_level, detail_scale_factor=args.wavelet_detail_scale) if np.random.rand() < 0.2 else lambda x: x,
                tr.RandomCutout(num_holes_range=(1, 2), max_h_size=32, max_w_size=32, fill_value=0, p=0.3),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else:
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=scale_w, h=scale_h),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")

            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            elif img_cv.ndim == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

            img = Image.fromarray(img_cv)

            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            target = Image.fromarray(mask_cv, mode='L')
            label = self.convert_label(target)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None

        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)

        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)

        return transformed_sample

    def convert_label(self, label):
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1

        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)