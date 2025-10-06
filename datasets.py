# datasets.py
import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random 
import cv2 

# Import custom transforms
import custom_transforms as tr

# Define common image and mask extensions for robustness
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp') # Masks are typically single-channel PNGs or TIFs

def make_dataset(root, dataset_name, split='train'): # Added 'split' argument
    dataset_items = []

    # --- Handling TSRS_RSNA Datasets ---
    if 'TSRS_RSNA' in dataset_name:
        # Assumes structure like: DATA_ROOT/TSRS_RSNA-Epiphysis/train/images/
        # and DATA_ROOT/TSRS_RSNA-Epiphysis/train/GT/ (for masks)
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'GT') # Assuming masks are in a 'GT' subdirectory

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"DEBUG: TSRS_RSNA dataset ({split}) path not found. Expected: {image_dir}, {mask_dir}. Please ensure data is organized correctly.")
            return []

        for f in os.listdir(image_dir):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_dir, f)
                
                # Mask filename might be the same base name, or follow a convention
                # Assuming masks are PNGs and named like image_name.png
                mask_full_path = os.path.join(mask_dir, img_name_base + '.png') 
                
                if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Missing image or mask for {img_name_base} in {dataset_name} ({split}). Skipping.")

    # --- Handling JSRT Dataset ---
    elif dataset_name == 'JSRT':
        # Assumes structure like: DATA_ROOT/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/cxr/
        # and masks in content/jsrt/masks/
        image_path = os.path.join(root, 'content', 'jsrt', 'cxr')
        mask_path = os.path.join(root, 'content', 'jsrt', 'masks')
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
            
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'): # JSRT uses PNG
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f}. Skipping.")

    # --- Handling COVID19_Radiography Dataset ---
    elif dataset_name == 'COVID19_Radiography':
        # Assumes structure: COVID-19_Radiography_Dataset/COVID/images/, COVID-19_Radiography_Dataset/COVID/masks/ etc.
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        
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
                        print(f"Warning: Mask not found for COVID19 image {f}. Skipping.")

    # --- Handling CVC-ClinicDB Dataset ---
    elif dataset_name == 'CVC-ClinicDB':
        # Assumes structure: DATA_ROOT/Original/ (images), DATA_ROOT/Ground Truth/ (masks)
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
            
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'): # Images are TIF
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif') # Masks are TIF
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f}. Skipping.")

    # --- Handling DentalPanoramic Dataset ---
    elif dataset_name == 'DentalPanoramic':
        # Assumes structure: DATA_ROOT/images/ (images), DATA_ROOT/segmentation_1/ (masks)
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1') # Or 'segmentation_2', etc. if multiple mask types exist
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: DentalPanoramic paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
            
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Masks are PNG
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f}. Skipping.")

    # --- Handling SixDiseasesChestXRay Dataset ---
    elif dataset_name == 'SixDiseasesChestXRay':
        # Assumes structure: DATA_ROOT/train/Covid/images/, DATA_ROOT/train/Covid/masks/ etc.
        # The 'root' passed here will be the dataset's base path like DATA_ROOT/Dataset/train
        base_image_path = os.path.join(root, 'images') # Assuming common images dir if not per-class
        base_mask_path = os.path.join(root, 'masks') # Assuming common masks dir if not per-class
        
        if not os.path.exists(base_image_path) or not os.path.exists(base_mask_path):
            print(f"Warning: SixDiseasesChestXRay base image/mask paths not found at {base_image_path}, {base_mask_path}. Trying per-class paths.")
            # Fallback to per-class structure if common directories don't exist
            subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
            for sub_name in subfolders:
                sub_image_path = os.path.join(root, sub_name, 'images')
                sub_mask_path = os.path.join(root, sub_name, 'masks')
                
                if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                    print(f"Warning: Image or mask path for SixDiseases class '{sub_name}' not found. Skipping class.")
                    continue
                
                for f in os.listdir(sub_image_path):
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        img_name_base = os.path.splitext(f)[0]
                        img_full_path = os.path.join(sub_image_path, f)
                        mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # Masks are PNG
                        if os.path.exists(mask_full_path):
                            dataset_items.append((img_full_path, mask_full_path))
                        else:
                            print(f"Warning: Mask not found for SixDiseases class '{sub_name}' image {f}. Skipping.")
        else: # If common image/mask dirs found (This structure might be less common for this dataset)
            print("Using common image/mask directories for SixDiseasesChestXRay.")
            for f in os.listdir(base_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(base_image_path, f)
                    mask_full_path = os.path.join(base_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for SixDiseases image {f} in common directory. Skipping.")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    if not dataset_items:
        print(f"ERROR: Found 0 images in {root} for dataset '{dataset_name}' with split '{split}'. Please check dataset path and file extensions.")
        # Raising an error here is critical for training/testing to stop if data is missing
        raise RuntimeError(f"No data found for dataset {dataset_name} split {split}.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        # make_dataset will now handle the split directory logic
        all_imgs = make_dataset(root, dataset_name, split=split)
        
        self.imgs = all_imgs # The 'root' argument to ImageFolder is now the specific split directory
        
        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. Please check dataset path, structure, and file extensions.")
            # No need to raise error here, __getitem__ and __len__ will handle it.

        # Preprocessing parameters
        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

        # --- DASEG's fixed preprocessing dimensions (hardcoded for faithful comparison) ---
        # These values override any --scale-h/--scale-w passed via command line
        # to ensure the model processes images at the exact same resolution.
        DASEG_FIXED_RESIZE_W = 576
        DASEG_FIXED_RESIZE_H = 896 
        DASEG_TRAIN_CROP_H = 576
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
        else: # Validation/Test - no data augmentation
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=DASEG_FIXED_RESIZE_W, h=DASEG_FIXED_RESIZE_H), 
                # DASEG's evaluation doesn't seem to have other preprocessing steps like Wavelet/HE
                # If they are part of the evaluation protocol, they should be added here.
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        if not self.imgs: # Handle case where dataset is empty
            return None

        img_path, gt_path = self.imgs[index]
        try:
            # Use OpenCV for reading, then convert to PIL Image
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            # Convert BGR to RGB if it's a color image
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(img_np)
            # Target is a mask, ensure it's loaded as grayscale and then converted to PIL Image
            target = Image.fromarray(mask_np, mode='L')
            
            label = self.convert_label(target) # Convert PIL label mask to indexed format
        
        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, IndexError) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None # Return None to signal a problem with this sample
        
        # Sample dictionary containing PIL Images
        sample = {'image': img, 'label': label}
        
        # Apply transformations
        transformed_sample = self.composed_transforms(sample)
        
        # Add filename for potential debugging/saving results, especially for validation/test splits
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label(self, label):
        """Converts a PIL Image mask to a binary indexed mask (0 or 1)."""
        label_np = np.array(label, dtype=np.uint8)
        
        # Ensure label_np is 2D (remove channel dimension if present)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        
        # Create a binary mask: 1 for foreground (lesions/regions of interest), 0 for background
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Assuming any non-zero pixel in the mask is the foreground class
        
        # Return as a PIL Image with mode 'P' (palette/indexed) for compatibility with some transforms,
        # although it's just 0s and 1s. The ToTensor transform will convert this to a tensor.
        return Image.fromarray(label_index, mode='P') 

    def __len__(self):
        return len(self.imgs)