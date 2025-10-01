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

def make_dataset(root, dataset_name):
    """
    Creates a list of (image_path, mask_path) tuples.
    Args:
        root (str): The root directory to search for data. For TSRS_RSNA, this is expected to be the split directory (e.g., '/path/to/test').
                    For other datasets, it might be a parent directory if pre-split subfolders exist, or the split directory itself.
        dataset_name (str): The name of the dataset (e.g., 'TSRS_RSNA-Epiphysis').
    Returns:
        list: A list of (image_path, mask_path) tuples.
    """
    dataset_items = []

    # Handling TSRS_RSNA-Epiphysis and TSRS_RSNA-Articular-Surface (assuming similar structure)
    if 'TSRS_RSNA' in dataset_name:
        # 'root' is expected to be the directory containing images for the specific split (e.g., '/kaggle/working/ARAA-Net/data/TSRS_RSNA-Epiphysis/test')
        image_path = root 
        
        # Determine mask path based on the provided 'root' (image_path)
        # If root is like '.../test', mask_path should be '.../test_labels'
        # If root is like '.../train', mask_path should be '.../train_labels'
        mask_dir_candidate_sibling = os.path.join(os.path.dirname(root), os.path.basename(root) + '_labels') # e.g., /path/to/dataset/test_labels
        
        # Fallback to looking for 'GT' inside the image directory if the sibling directory isn't found
        mask_dir_candidate_in_root = os.path.join(root, 'GT') 

        if os.path.isdir(mask_dir_candidate_sibling):
            mask_path = mask_dir_candidate_sibling
        elif os.path.isdir(mask_dir_candidate_in_root):
            mask_path = mask_dir_candidate_in_root
        else:
            print(f"DEBUG: {dataset_name}: Could not find label directory for '{root}'. Looked in '{mask_dir_candidate_sibling}' and '{mask_dir_candidate_in_root}'.")
            return []
        
        # Ensure image_path is a directory
        if not os.path.isdir(image_path):
            print(f"Warning: Image path '{image_path}' is not a directory for dataset '{dataset_name}'. Returning empty dataset.")
            return []
        
        # Iterate through files in the image_path directory
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                # Assuming masks are named the same as images and have .png extension
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                
                if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    # print(f"Warning: Missing image or mask for {img_name_base} in {dataset_name}. Skipping. Image: {img_full_path}, Mask: {mask_full_path}")
                    pass # Suppress warnings for missing pairs to avoid clutter if some exist

    elif dataset_name == 'JSRT':
        # Assumes JSRT data is directly under root, with 'images' and 'masks' subfolders
        image_path = os.path.join(root, 'images') # Adjusted to expect images/masks subfolders
        mask_path = os.path.join(root, 'masks')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            print(f"Warning: JSRT paths not found: '{image_path}', '{mask_path}'. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image '{f}'. Skipping.")

    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path):
                print(f"Warning: Image or mask path for '{sub_name}' not found in '{base_dataset_folder}'. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image '{f}' in '{sub_name}'. Skipping.")

    elif dataset_name == 'CVC-ClinicDB':
        # Assumes CVC-ClinicDB data is directly under root, with 'Original' and 'Ground Truth' subfolders
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: '{image_path}', '{mask_path}'. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image '{f}'. Skipping.")

    elif dataset_name == 'DentalPanoramic':
        # Assumes DentalPanoramic data is directly under root, with 'images' and 'segmentation_1' subfolders
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1')
        if not os.path.isdir(image_path) or not os.path.isdir(mask_path):
            print(f"Warning: DentalPanoramic paths not found: '{image_path}', '{mask_path}'. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Assuming masks are png
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image '{f}'. Skipping.")

    elif dataset_name == 'SixDiseasesChestXRay':
        # Assumes SixDiseasesChestXRay data has 'train', 'val', 'test' splits and within each, subfolders by disease
        # The 'root' passed here will be the specific split folder (e.g., DATA_ROOT/Dataset/train)
        base_split_folder = root # e.g., DATA_ROOT/Dataset/train
        if not os.path.isdir(base_split_folder):
            print(f"Warning: SixDiseasesChestXRay split path not found: '{base_split_folder}'. Returning empty dataset.")
            return []

        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_split_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_split_folder, sub_name, 'masks')
            
            if not os.path.isdir(sub_image_path) or not os.path.isdir(sub_mask_path):
                print(f"Warning: Image or mask path for '{sub_name}' not found in '{base_split_folder}'. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # Assuming masks are png
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for {sub_name} image '{f}'. Skipping.")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    # Final check to ensure data was found
    if not dataset_items and (dataset_name != 'PLACEHOLDER_FOR_DYNAMIC_SELECTION'): # Avoid error if placeholder is used incorrectly
        print(f"Warning: Found 0 items for dataset '{dataset_name}' at root '{root}'. Please check dataset path, file extensions, and directory structure.")
        # Consider raising an error here if a valid dataset is expected to have data.
        # raise RuntimeError(f"Found 0 images in {root} for dataset {dataset_name} with corresponding labels. Please check dataset path and file extensions.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        # --- Data Loading and Splitting Logic ---
        # make_dataset expects `root` to be the directory containing the split (e.g., 'train', 'val', 'test')
        # for datasets like TSRS_RSNA and SixDiseasesChestXRay where the caller (main function) provides the specific split path.
        # For other datasets, make_dataset might expect a higher-level root, and we perform splitting here.
        
        all_imgs = make_dataset(root, dataset_name)
        
        # If make_dataset already returned items for a specific split (like TSRS_RSNA, or SixDiseasesChestXRay if `root` is already a split folder),
        # we use them directly. Otherwise, we perform programmatic splitting if the `root` is a general dataset directory.
        
        if 'TSRS_RSNA' in dataset_name or dataset_name == 'SixDiseasesChestXRay':
            # For these datasets, the `root` passed to __init__ is already expected to be a specific split folder (e.g., 'train', 'val', 'test').
            # `make_dataset` will find items within that specific folder. No further splitting is needed here.
            self.imgs = all_imgs
            if not self.imgs: # If the found list is empty, log a warning.
                print(f"Warning: The '{split}' split for {dataset_name} is empty. Expected data at: '{root}'")
        else:
            # For other datasets, we check if they are pre-split into train/val/test subdirectories within the provided 'root'.
            train_dir = os.path.join(root, 'train')
            val_dir = os.path.join(root, 'val')
            test_dir = os.path.join(root, 'test')

            if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
                # Data is already pre-split into subdirectories
                print(f"Detected pre-split directories in '{root}'. Using '{split}' split.")
                if split == 'train':
                    self.imgs = make_dataset(train_dir, dataset_name)
                elif split == 'val':
                    self.imgs = make_dataset(val_dir, dataset_name)
                elif split == 'test':
                    self.imgs = make_dataset(test_dir, dataset_name)
                else:
                    raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
            else:
                # Perform programmatic splitting if data is not pre-split into subfolders.
                # This branch assumes 'root' is a single folder containing all images/masks for this dataset,
                # and we need to divide them into train/val/test.
                if not all_imgs:
                     print(f"Warning: No images found for dataset '{dataset_name}' at root '{root}'.")
                     self.imgs = []
                else:
                    random.seed(42) # For reproducibility
                    random.shuffle(all_imgs)
                    
                    total_size = len(all_imgs)
                    # Define split sizes, ensuring at least one item for each split if possible and avoiding index errors
                    train_ratio = 0.8
                    val_ratio = 0.1
                    # test_ratio is implicitly the remainder

                    train_size = int(train_ratio * total_size)
                    val_size = int(val_ratio * total_size)
                    
                    # Adjust sizes to ensure valid splits and prevent empty splits if possible for small datasets
                    if total_size < 3: # Very small dataset
                        train_size = 1 if total_size > 0 else 0
                        val_size = 1 if total_size > 1 else 0
                        test_size = total_size - train_size - val_size # Will be 0 or 1
                    else:
                        # Ensure test_size gets the remainder and is at least 1 if there are enough items
                        test_size = total_size - train_size - val_size
                        # Distribute if test_size ends up negative due to rounding or if train/val took all
                        if test_size < 0:
                            if val_size > 1: val_size -= 1
                            test_size = total_size - train_size - val_size # Recalculate
                        test_size = max(0, test_size) # Ensure not negative

                    if split == 'train':
                        self.imgs = all_imgs[:train_size]
                    elif split == 'val':
                        self.imgs = all_imgs[train_size : train_size + val_size]
                    elif split == 'test':
                        self.imgs = all_imgs[train_size + val_size :] # Remaining for test
                    else:
                        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
                    print(f"Performing programmatic split on '{root}'. Total items: {total_size}. Split '{split}': {len(self.imgs)} items (Train: {train_size}, Val: {val_size}, Test remainder).")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. No images loaded. Please check dataset path and contents: '{root}'")

        # --- Transforms ---
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

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
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
        else: # Validation/Test - no data augmentation or complex preprocessing
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        if not self.imgs: # Handle case where self.imgs is empty
            return None
            
        img_path, gt_path = self.imgs[index]
        try:
            # Using cv2.imread for potentially better performance and handling of various formats
            img_np = cv2.imread(img_path)
            # Ensure mask is read in grayscale
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: '{img_path}'. File might be corrupted or path incorrect.")
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: '{gt_path}'. File might be corrupted or path incorrect.")

            # Convert BGR (OpenCV default) to RGB if it's a color image
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(img_np)
            target = Image.fromarray(mask_np, mode='L') # mode='L' for grayscale
            
            label = self.convert_label(target)
        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: '{img_path}', '{gt_path}'. Error: {e}. Returning None for this sample.")
            return None 
        
        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)
        
        if self.split != 'train':
            # Add original filename for potential logging/saving of results
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label(self, label):
        """
        Converts a PIL Image label to a binary NumPy array (0 or 1) and returns it as a PIL Image.
        """
        label_np = np.array(label, dtype=np.uint8) # Use uint8 for masks
        
        # Ensure label_np is 2D
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        elif label_np.ndim != 2:
            raise ValueError(f"Unexpected label dimension: {label_np.ndim}")

        # Create a binary mask: 1 for foreground (lesion), 0 for background
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Assumes any non-zero pixel is foreground
        
        return Image.fromarray(label_index, mode='P') # Return as PIL Image in 'P' mode for consistency

    def __len__(self):
        return len(self.imgs)