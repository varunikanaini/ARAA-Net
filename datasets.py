# /kaggle/working/ARAA-Net/datasets.py
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
    Handles different dataset structures.
    """
    dataset_items = []
    
    # --- Logic for TSRS-like datasets ---
    # Assumes structure: root_dir/<dataset_name>/<split>/images/ and root_dir/<dataset_name>/<split>/GT/
    if 'TSRS_RSNA' in dataset_name:
        image_path = os.path.join(root_dir, dataset_name, split, 'images')
        mask_path = os.path.join(root_dir, dataset_name, split, 'GT') # Common for segmented datasets
        
        # Fallback for masks if 'GT' doesn't exist, e.g., '_labels'
        if not os.path.exists(mask_path):
            mask_path_alt = os.path.join(root_dir, dataset_name, split + '_labels')
            if os.path.exists(mask_path_alt):
                mask_path = mask_path_alt
            else:
                print(f"Warning: TSRS-like dataset '{dataset_name}' split '{split}': Could not find label directory. Looked in '{mask_path}' and '{mask_path_alt}'.")
                return []

        if not os.path.exists(image_path):
            print(f"Warning: TSRS-like dataset '{dataset_name}' split '{split}': Image directory not found at '{image_path}'.")
            return []

        # List images and find corresponding masks
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                
                # Try common mask extensions
                mask_found = False
                for ext in MASK_EXTENSIONS:
                    mask_full_path = os.path.join(mask_path, img_name_base + ext)
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                        mask_found = True
                        break
                if not mask_found:
                    print(f"Warning: Mask not found for image {f} in {dataset_name}/{split}. Skipping.")

    # --- Logic for JSRT ---
    elif dataset_name == 'JSRT':
        # Assumes structure: root_dir/jsrt-247-image-lung-segmentation-mask-dataset/<split>/cxr and .../masks
        # Or if root_dir is already the 'split' folder (e.g. train)
        # Let's assume structure is root_dir/JSRT_dataset_folder/<split>/cxr and .../masks
        # Adjust these paths if your JSRT structure is different.
        base_folder = os.path.join(root_dir, 'content', 'jsrt') # Assuming data is extracted into 'content/jsrt'
        image_path = os.path.join(base_folder, split, 'cxr')
        mask_path = os.path.join(base_folder, split, 'masks')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith('.png'): # JSRT images are typically PNG
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # JSRT masks are PNG
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f} in split '{split}'. Skipping.")

    # --- Logic for COVID19_Radiography ---
    elif dataset_name == 'COVID19_Radiography':
        # Assumes structure: root_dir/COVID-19_Radiography_Dataset/<Class>/images/ and .../masks/
        base_dataset_folder = os.path.join(root_dir, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia'] # Classes

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
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # Masks are PNG
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image {f} in class '{sub_name}'. Skipping.")

    # --- Logic for CVC-ClinicDB ---
    elif dataset_name == 'CVC-ClinicDB':
        # Assumes structure: root_dir/CVC-ClinicDB/<split>/Original/ and .../Ground Truth/
        image_path = os.path.join(root_dir, dataset_name, split, 'Original')
        mask_path = os.path.join(root_dir, dataset_name, split, 'Ground Truth')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'): # Images are TIFF
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif') # Masks are TIFF
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f} in split '{split}'. Skipping.")

    # --- Logic for DentalPanoramic ---
    elif dataset_name == 'DentalPanoramic':
        # Assumes structure: root_dir/dental_panoramic_xrays/<split>/images/ and .../segmentation_1/
        image_path = os.path.join(root_dir, dataset_name, split, 'images')
        mask_path = os.path.join(root_dir, dataset_name, split, 'segmentation_1')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: DentalPanoramic split '{split}' paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []

        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Masks are PNG
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f} in split '{split}'. Skipping.")

    # --- Logic for SixDiseasesChestXRay ---
    elif dataset_name == 'SixDiseasesChestXRay':
        # Assumes structure: root_dir/Dataset/<split>/<Class>/images/ and .../<Class>/masks/
        base_dataset_folder = os.path.join(root_dir, 'Dataset') # This seems to be the root based on the path in config
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia'] # Classes
        
        if not os.path.exists(base_dataset_folder):
            print(f"Warning: SixDiseasesChestXRay base path '{base_dataset_folder}' not found. Returning empty dataset.")
            return []

        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, split, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, split, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for SixDiseasesChestXRay class '{sub_name}' in split '{split}' not found. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # Masks are PNG
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
            root (str): Path to the dataset's root directory (e.g., /kaggle/working/ARAA-Net/data)
            dataset_name (str): Name of the dataset (e.g., 'TSRS_RSNA-Epiphysis')
            args: Arguments object containing dataset-specific parameters.
            split (str): 'train', 'val', or 'test'.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        all_imgs_masks = make_dataset(root, dataset_name, split) # Pass root to make_dataset
        self.imgs = all_imgs_masks # Directly use the items from make_dataset

        if not self.imgs:
            raise RuntimeError(f"No images found for dataset '{self.dataset_name}' split '{self.split}' at root '{self.root}'. Please check directory structure and file extensions.")
        else:
            print(f"Found {len(self.imgs)} samples for {self.dataset_name} split '{self.split}'.")

        # Parameters for transforms from args
        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

        # --- DASEG's fixed preprocessing dimensions (hardcoded for faithful comparison) ---
        # These values override any --scale-h/--scale-w passed via command line
        # to ensure the LASA-Unet model processes images at the exact same resolution
        # as the DASEG model's internal hardcoded transformations.
        DASEG_FIXED_RESIZE_W = 576
        DASEG_FIXED_RESIZE_H = 896 # DASEG FixedResize(w=576, h=896)
        DASEG_TRAIN_CROP_H = 576 # DASEG RandomCrop((576,576))
        DASEG_TRAIN_CROP_W = 576
        # --- END DASEG dimensions ---

        # Define transforms
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Added ColorJitter
                # RandomAffine parameters: degrees, translate (fraction of width/height), scale, shear, fill colors
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
                tr.ToTensor() # Converts PIL images to Tensors and labels to LongTensors
            ])
        else: # Validation/Test - no random augmentation, only resizing and normalization
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            # Read image in RGB format
            img_np = cv2.imread(img_path)
            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")
            if img_np.ndim == 3 and img_np.shape[2] == 3: # Already RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif img_np.ndim == 2: # Grayscale, convert to RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            # Read mask in grayscale
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            img = Image.fromarray(img_np)
            target = Image.fromarray(mask_np, mode='L') # Ensure mask is loaded as 'L' mode
            
            # Convert label: Binary segmentation (0: background, 1: foreground)
            # Assuming masks have pixel values > 0 for foreground.
            label = self.convert_label(target)
            
        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None # Return None for corrupted samples to be handled by collate_fn
        
        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)
        
        # Add image name for potential debugging/saving predictions
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label(self, label):
        """
        Converts a PIL Image label to a binary segmentation tensor (0 for background, 1 for foreground).
        Assumes any non-zero pixel value in the original mask is foreground.
        """
        label_np = np.array(label, dtype=np.uint8)
        # Ensure it's a 2D array
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Set foreground pixels to 1
        
        # Convert to PIL Image with 'P' mode for palette-based representation
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)
