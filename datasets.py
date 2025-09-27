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
    dataset_items = []

    # Handling TSRS_RSNA-Epiphysis and TSRS_RSNA-Articular-Surface (assuming similar structure)
    if 'TSRS_RSNA' in dataset_name:
        image_path = root # 'root' here will be e.g., DATA_ROOT/TSRS_RSNA-Epiphysis/train
        if os.path.exists(os.path.join(root, 'GT')):
            mask_path = os.path.join(root, 'GT')
        elif os.path.exists(root + '_labels'): # Matches original DASEG structure e.g., val_labels
            mask_path = root + '_labels'
        else:
            print(f"DEBUG: {dataset_name}: Could not find label directory for {root}. Looked in '{os.path.join(root, 'GT')}' and '{root + '_labels'}'.")
            return []
        
        for f in os.listdir(image_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png') # Assuming masks are png
                
                if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Missing image or mask for {img_name_base} in {dataset_name}. Skipping.")

    elif dataset_name == 'JSRT':
        image_path = os.path.join(root, 'content', 'jsrt', 'cxr')
        mask_path = os.path.join(root, 'content', 'jsrt', 'masks')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: JSRT paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.png'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f}. Skipping.")

    elif dataset_name == 'COVID19_Radiography':
        base_dataset_folder = os.path.join(root, 'COVID-19_Radiography_Dataset')
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} not found. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for COVID19 image {f}. Skipping.")

    elif dataset_name == 'CVC-ClinicDB':
        image_path = os.path.join(root, 'Original')
        mask_path = os.path.join(root, 'Ground Truth')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: CVC-ClinicDB paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.tif')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f}. Skipping.")

    elif dataset_name == 'DentalPanoramic':
        image_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'segmentation_1')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Warning: DentalPanoramic paths not found: {image_path}, {mask_path}. Returning empty dataset.")
            return []
        for f in os.listdir(image_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_path, f)
                mask_full_path = os.path.join(mask_path, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f}. Skipping.")

    elif dataset_name == 'SixDiseasesChestXRay':
        base_dataset_folder = os.path.join(root, 'train')
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        
        if not os.path.exists(base_dataset_folder):
            print(f"Warning: SixDiseasesChestXRay base path not found: {base_dataset_folder}. Returning empty dataset.")
            return []

        for sub_name in subfolders:
            sub_image_path = os.path.join(base_dataset_folder, sub_name, 'images')
            sub_mask_path = os.path.join(base_dataset_folder, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} not found. Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for {sub_name} image {f}. Skipping.")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    if not dataset_items:
        raise RuntimeError(f"Found 0 images in {root} for dataset {dataset_name} with corresponding labels. Please check dataset path and file extensions.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        all_imgs = make_dataset(root, dataset_name)
        
        if 'TSRS_RSNA' in dataset_name:
            self.imgs = all_imgs # For TSRS, 'root' is already specific to train/val
        else: # For other datasets, perform programmatic splitting
            random.seed(42) # For reproducibility
            random.shuffle(all_imgs)
            
            total_size = len(all_imgs)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size) # 10% for validation
            
            if split == 'train':
                self.imgs = all_imgs[:train_size]
            elif split == 'val':
                self.imgs = all_imgs[train_size : train_size + val_size]
            elif split == 'test':
                self.imgs = all_imgs[train_size + val_size :] # Remaining 10% for test
            else:
                raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        if not self.imgs:
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. Check dataset path and contents.")

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
        img_path, gt_path = self.imgs[index]
        try:
            img_np = cv2.imread(img_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

            if img_np is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")
            if mask_np is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
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
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)
        
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1 
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)