# /kaggle/working/ARAA-Net/datasets.py (FINAL VERSION WITH NEW PREPROCESSING AND MULTI-DATASET SUPPORT)

import os
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random # Added for programmatic splitting
import cv2 # Added for robust image loading (especially for TIFFs and problematic JPEGs)

import custom_transforms as tr 

# Define common image and mask extensions for robustness (from reference datasets.py)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp') # Masks are often png or tiff


# Modified make_dataset to accept dataset_name
def make_dataset(root, dataset_name):
    dataset_items = []

    # Original TSRS_RSNA-Epiphysis dataset logic
    if dataset_name == 'TSRS_RSNA-Epiphysis':
        # The 'root' argument here will already be pointing to DATA_ROOT/TSRS_RSNA-Epiphysis/train or DATA_ROOT/TSRS_RSNA-Epiphysis/val
        image_base_path = root
        
        # Check for original 'GT' or '_labels' convention
        if os.path.exists(os.path.join(root, 'GT')):
            mask_base_path = os.path.join(root, 'GT')
        elif os.path.exists(root + '_labels'): # This is how the original config.py generates the mask path for train/val for TSRS
            mask_base_path = root + '_labels'
        else:
            print(f"Warning: Could not find label directory for {dataset_name} at {root}. Looked in '{os.path.join(root, 'GT')}' and '{root + '_labels'}'. Returning empty dataset.")
            return []

        # List images and pair with masks
        img_names = []
        for f in os.listdir(image_base_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')): # Original uses these extensions
                img_names.append(os.path.splitext(f)[0])

        for img_name in img_names:
            img_full_path = os.path.join(image_base_path, img_name + '.jpg') # Original primarily uses .jpg
            mask_full_path = os.path.join(mask_base_path, img_name + '.png') # Original primarily uses .png
            
            if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                dataset_items.append((img_full_path, mask_full_path))
            else:
                print(f"Warning: Missing image or mask for {img_name} in {dataset_name}. Skipping.")

    # New datasets logic (adapted from the provided reference datasets.py)
    elif dataset_name == 'jsrt-247-image-lung-segmentation-mask-dataset':
        # 'root' here will be DATA_ROOT/jsrt-247-image-lung-segmentation-mask-dataset
        image_path = os.path.join(root, 'content', 'jsrt', 'cxr')
        mask_path = os.path.join(root, 'content', 'jsrt', 'masks')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'): # JSRT images are .png
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # JSRT masks are .png
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f}. Skipping.")
    
    elif dataset_name == 'covid19-radiography-database':
        # 'root' here will be DATA_ROOT/covid19-radiography-database
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} (COVID19_Radiography) not found. Skipping this class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # COVID masks are typically .png
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID-19 image {f}. Skipping.")

    elif dataset_name == 'CVC-ClinicDB':
        # 'root' here will be DATA_ROOT/CVC-ClinicDB
        image_path = os.path.join(root, 'Original') 
        mask_path = os.path.join(root, 'Ground Truth') 
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'): # CVC images are often .tif
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif') # CVC masks are often .tif
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f}. Skipping.")
    
    elif dataset_name == 'dental_panoramic_xrays':
        # 'root' here will be DATA_ROOT/dental_panoramic_xrays
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1') # Using segmentation_1 by default as per reference
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: DentalPanoramic paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Assuming masks are .png
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f}. Skipping.")

    elif dataset_name == 'Dataset': # SixDiseasesChestXRay uses 'Dataset' as its base folder name
        # 'root' here will be DATA_ROOT/Dataset
        base_dataset_folder = os.path.join(root, 'train') # As per reference, data is in 'train' subfolder
        # Corrected subfolder names to match observed Title Case in the reference 'train.py'
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia'] 
        
        if not os.path.exists(base_dataset_folder):
            print(f"Warning: SixDiseasesChestXRay base path not found: {base_dataset_folder}. Returning empty dataset.")
            return []

        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} (SixDiseasesChestXRay) not found. Skipping this class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png') # Assuming masks are .png
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for {sub_name} image {f}. Skipping.")
    else:
        # Fallback for unexpected dataset_name or error in config.py's DATASET_NAME
        raise ValueError(f"Unknown dataset_name: '{dataset_name}'. Please check config.py and ensure it matches a known dataset base folder name.")
    
    if not dataset_items:
        raise RuntimeError(f"Found 0 images in {root} for dataset {dataset_name} with corresponding labels. Please check dataset path and file extensions.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.split = split
        self.args = args
        self.dataset_name = args.dataset_name # Get dataset_name from args

        # This part handles the initial loading and potential programmatic split
        # For 'TSRS_RSNA-Epiphysis', root already points to DATA_ROOT/TSRS_RSNA-Epiphysis/train or DATA_ROOT/TSRS_RSNA-Epiphysis/val
        # For other datasets, root will be DATA_ROOT/<dataset_base_folder_name>
        if self.dataset_name == 'TSRS_RSNA-Epiphysis':
            # TSRS uses predefined train/val folders, so no programmatic splitting here
            self.imgs = make_dataset(root, self.dataset_name)
        else:
            # For new datasets, perform programmatic 80/10/10 split
            all_imgs = make_dataset(root, self.dataset_name)
            
            random.seed(42) # For reproducibility
            random.shuffle(all_imgs)
            
            total_size = len(all_imgs)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            
            if split == 'train':
                self.imgs = all_imgs[:train_size]
            elif split == 'val':
                self.imgs = all_imgs[train_size : train_size + val_size]
            elif split == 'test':
                self.imgs = all_imgs[train_size + val_size :]
            else:
                raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test' for programmatic splitting datasets.")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. Check dataset path and contents.")

        # --- Original preprocessing arguments from the initial datasets.py ---
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
                # <<< NEW PREPROCESSING STEPS >>>
                tr.WaveletContrastEnhancement(wavelet=args.wavelet_type, level=args.wavelet_level, detail_scale_factor=args.wavelet_detail_scale),
                tr.HistogramEqualization(),
                # <<< END NEW PREPROCESSING >>>
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
        try:
            # Use OpenCV for more robust image and mask reading
            # Read image as is (color or grayscale)
            img_np = cv2.imread(img_path)
            # Read mask as grayscale to ensure it's 2D for convert_label
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            # Convert OpenCV's BGR to RGB if it's a color image
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            # If img_np is grayscale (ndim=2), PIL.Image.fromarray will handle it correctly.
            
            img = Image.fromarray(img_np)
            # Ensure mask_np is 2D before converting to PIL Image 'L' mode
            if mask_np.ndim == 3: # If mask was read as 3 channels (e.g., a color mask), take one.
                mask_np = mask_np[:, :, 0]
            target = Image.fromarray(mask_np, mode='L') # 'L' mode for grayscale

            label = self.convert_label(target)
        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError) as e: 
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Skipping this sample.")
            return None # Return None for corrupted samples
        
        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)
        
        if self.split != 'train':
            # Store original image name (just the filename) for visualization/tracking
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    # Updated convert_label for robustness (from reference datasets.py)
    def convert_label(self, label):
        label_np = np.array(label, dtype=np.uint8)
        
        # Ensure label_np is 2D. If it was loaded as a 3-channel image but contains
        # only grayscale info, take the first channel. This makes it robust to
        # how different image libraries might load "grayscale" masks (e.g., as RGB with R=G=B).
        if label_np.ndim == 3 and label_np.shape[2] >= 1:
            label_np = label_np[:, :, 0]
        elif label_np.ndim == 2:
            pass # Already 2D
        else:
            print(f"Warning: Unexpected label array dimensions in convert_label: {label_np.shape}. Attempting to assume single channel.")
            if label_np.ndim > 2: # Attempt to reshape if higher dim
                label_np = label_np.reshape(label_np.shape[0], label_np.shape[1])
            else: # If 1D or other unexpected, fall back to original
                print(f"Warning: Could not reshape label_np. Using as is, may cause issues.")

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 # Convert any non-zero pixel to 1 (foreground)
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)