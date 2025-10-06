# datasets.py (Corrected make_dataset function for TSRS_RSNA)

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2

import custom_transforms as tr

# Define common image and mask extensions for robustness
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(root_dir, dataset_name, split):
    """
    Creates a list of (image_path, mask_path) tuples for a given dataset and split.
    Handles different dataset structures, including the specific TSRS_RSNA structure
    where images and labels are directly in the split folders.
    """
    dataset_items = []
    
    # --- Logic for TSRS-like datasets ---
    # `root_dir` passed to ImageFolder IS the split directory (e.g., /.../TSRS_RSNA-Epiphysis/train)
    if 'TSRS_RSNA' in dataset_name:
        image_path_actual = root_dir # Images are directly in this root_dir (which is the split folder)
        
        # The label path is formed by taking the parent of root_dir and appending split_labels
        # e.g., if root_dir is /.../TSRS_RSNA-Epiphysis/train
        # then parent is /.../TSRS_RSNA-Epiphysis/
        # and label path becomes /.../TSRS_RSNA-Epiphysis/train_labels/
        
        parent_dir = os.path.dirname(root_dir) # Get the parent directory (e.g., /.../TSRS_RSNA-Epiphysis/)
        mask_path_actual = os.path.join(parent_dir, split + '_labels') # e.g., /.../TSRS_RSNA-Epiphysis/train_labels/
        
        if not os.path.exists(image_path_actual):
            print(f"Warning: TSRS-like dataset '{dataset_name}' split '{split}': Image directory not found at '{image_path_actual}'.")
            return []
        if not os.path.exists(mask_path_actual):
            print(f"Warning: TSRS-like dataset '{dataset_name}' split '{split}': Label directory not found at '{mask_path_actual}'.")
            return []

        # List images and find corresponding masks
        for f in os.listdir(image_path_actual):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path_actual, f)
                
                # Try common mask extensions (your labels are PNG)
                mask_found = False
                for ext in MASK_EXTENSIONS:
                    mask_full_path = os.path.join(mask_path_actual, img_name_base + ext)
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        mask_found = True
                        break
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in {dataset_name}/{split}. Skipping.")

    # --- Logic for JSRT ---
    # Assuming structure: root_dir/cxr/ and .../masks/ where root_dir IS the split folder
    elif dataset_name == 'JSRT':
        image_path = os.path.join(root_dir, 'cxr')
        mask_path = os.path.join(root_dir, 'masks')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith('.png'): 
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f} in split '{split}'. Skipping.")

    # --- Logic for COVID19_Radiography ---
    # root_dir passed to ImageFolder is the dataset base path (e.g., /.../COVID19_Radiography_Dataset)
    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root_dir, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']

        if not os.path.exists(base_dataset_folder):
            print(f"Warning: COVID19_Radiography dataset base folder not found: '{base_dataset_folder}'.")
            return []

        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for COVID19 class '{sub_name}' not found. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image {f} in class '{sub_name}'. Skipping.")

    # --- Logic for CVC-ClinicDB ---
    # root_dir passed to ImageFolder is the specific split folder (e.g., /.../CVC-ClinicDB/train)
    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root_dir, 'Original')
        mask_path = os.path.join(root_dir, 'Ground Truth')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f} in split '{split}'. Skipping.")

    # --- Logic for DentalPanoramic ---
    # root_dir passed to ImageFolder is the specific split folder (e.g., /.../DentalPanoramic/train)
    elif dataset_name == 'DentalPanoramic':
        image_path = os.path.join(root_dir, 'images')
        mask_path = os.path.join(root_dir, 'segmentation_1')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: DentalPanoramic split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f} in split '{split}'. Skipping.")

    # --- Logic for SixDiseasesChestXRay ---
    # root_dir passed to ImageFolder is the specific split folder (e.g., /.../Dataset/train)
    elif dataset_name == 'SixDiseasesChestXRay':
        base_folder_for_split = root_dir
        
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        
        if not os.path.exists(base_folder_for_split):
            print(f"Warning: SixDiseasesChestXRay base path '{base_folder_for_split}' not found. Returning empty dataset.")
            return []

        for sub_name in subfolders:
            sub_image_path = os.path.join(base_folder_for_split, sub_name, 'images')
            sub_mask_path = os.path.join(base_folder_for_split, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for SixDiseasesChestXRay class '{sub_name}' in split '{split}' not found. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for SixDiseasesChestXRay image {f} in class '{sub_name}', split '{split}'. Skipping.")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    # Final check if any items were found
    if not dataset_items:
        print(f"Warning: Found 0 image-mask pairs for dataset '{dataset_name}' split '{split}' in '{root_dir}'. Please check path and dataset structure.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        """
        Args:
            root (str): Path to the dataset split directory (e.g., /kaggle/working/ARAA-Net/data/TSRS_RSNA-Epiphysis/train)
            dataset_name (str): Name of the dataset (e.g., 'TSRS_RSNA-Epiphysis')
            args: Arguments object containing dataset-specific parameters.
            split (str): 'train', 'val', or 'test'.
        """
        self.root = root # This IS the split directory (e.g., .../train)
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        # Call make_dataset. `root` is already the split dir, so make_dataset handles it.
        all_imgs_masks = make_dataset(root, dataset_name, split) 
        self.imgs = all_imgs_masks

        if not self.imgs:
            # This error will now be caught here if make_dataset returns an empty list.
            # The previous error was in DataLoader initialization when num_samples was 0.
            raise RuntimeError(f"No images found for dataset '{self.dataset_name}' split '{self.split}' at root '{self.root}'. Please check directory structure and file extensions.")
        else:
            print(f"Found {len(self.imgs)} samples for {self.dataset_name} split '{self.split}'.")

        # Parameters for transforms from args
        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

        # --- DASEG's fixed preprocessing dimensions ---
        DASEG_FIXED_RESIZE_W = 576
        DASEG_FIXED_RESIZE_H = 896
        DASEG_TRAIN_CROP_H = 576 
        DASEG_TRAIN_CROP_W = 576
        # --- END DASEG dimensions ---

        # Define transforms
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                tr.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, mask_fill_value=0), 
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)),
                tr.WaveletContrastEnhancement(wavelet=args.wavelet_type, level=args.wavelet_level, detail_scale_factor=args.wavelet_detail_scale),
                tr.HistogramEqualization(),
                tr.RandomHorizontalFlip(),
                tr.RandomCrop((DASEG_TRAIN_CROP_H, DASEG_TRAIN_CROP_W)), 
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            img_np = cv2.imread(img_path)
            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif img_np.ndim == 2: # Grayscale, convert to RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            img = Image.fromarray(img_np)
            target = Image.fromarray(mask_np, mode='L')
            
            label = self.convert_label(target)
            
        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None 
        
        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)
        
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label(self, label):
        """
        Converts a PIL Image label to a binary segmentation tensor (0 for background, 1 for foreground).
        Assumes any non-zero pixel value in the original mask is foreground.
        """
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Set foreground pixels to 1
        
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)