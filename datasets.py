# datasets.py
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

def make_dataset(root, dataset_name, split):
    """
    Creates a list of (image_path, mask_path) tuples for the specified dataset and split.
    """
    dataset_items = []
    
    # --- Construct specific path based on dataset and split ---
    current_split_path = os.path.join(root, split)
    
    # TSRS_RSNA Datasets (Epiphysis and Articular-Surface)
    if 'TSRS_RSNA' in dataset_name:
        image_dir = current_split_path # e.g., DATA_ROOT/TSRS_RSNA-Epiphysis/train
        mask_dir = os.path.join(root, f"{split}_labels") # e.g., DATA_ROOT/TSRS_RSNA-Epiphysis/train_labels
        
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory not found for TSRS_RSNA dataset ({dataset_name}, split '{split}'): {image_dir}. Skipping.")
            return []
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory not found for TSRS_RSNA dataset ({dataset_name}, split '{split}'): {mask_dir}. Skipping.")
            return []

        for f in os.listdir(image_dir):
            if f.lower().endswith(('.jpg', '.jpeg')): # Assuming JPG for images in TSRS
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_dir, f)
                mask_full_path = os.path.join(mask_dir, img_name_base + '.png') # Assuming PNG for masks in TSRS

                if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Missing image or mask for {img_name_base} in {dataset_name} ({split}). Skipping.")

    # JSRT Dataset
    elif dataset_name == 'JSRT':
        image_dir = os.path.join(root, 'cxr') # JSRT structure under root
        mask_dir = os.path.join(root, 'masks')
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Warning: JSRT paths not found for split '{split}' at {root}. Expected dirs 'cxr', 'masks'. Returning empty.")
            return []
        
        # JSRT has images and masks in root directories, no explicit train/val/test folders for the dataset itself usually
        # We'll use the provided split logic in the dataset class for JSRT if it's small enough, or assume all data if root points to full dataset
        # For consistency, we'll assume root points to the directory *containing* cxr/masks for JSRT
        # If a specific split is requested and JSRT doesn't have them, this will be handled by the ImageFolder class split logic.
        for f in os.listdir(image_dir):
            if f.lower().endswith('.png'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_dir, f)
                mask_full_path = os.path.join(mask_dir, img_name_base + '.png')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for JSRT image {f}. Skipping.")

    # COVID19_Radiography Dataset
    elif dataset_name == 'COVID19_Radiography':
        # This dataset has 'train' and 'test' folders within its main directory
        # The subfolders are 'COVID', 'NORMAL', etc.
        base_data_dir = os.path.join(root, 'COVID-19_Radiography_Dataset')
        
        if not os.path.exists(base_data_dir):
            print(f"Warning: COVID19_Radiography base path not found: {base_data_dir}. Returning empty.")
            return []

        # We expect the split to be handled by the class's internal splitting logic if the dataset doesn't have explicit subfolders for train/val/test
        # However, if the path provided IS already split (e.g., root/split/COVID/images), we need to adapt.
        # Let's assume for now that the main logic handles it by selecting from *all* available images and splitting.
        # If COVID19_Radiography has its own train/test structure, that needs to be handled here.
        # For now, we will collect all images and rely on the ImageFolder's split.
        
        subfolders = ['COVID', 'NORMAL', 'Lung_Opacity', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_data_dir, sub_name, 'images')
            sub_mask_path = os.path.join(base_data_dir, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} not found in COVID19_Radiography. Skipping class.")
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

    # CVC-ClinicDB Dataset
    elif dataset_name == 'CVC-ClinicDB':
        image_dir = os.path.join(root, 'Original') # CVC-ClinicDB structure under root
        mask_dir = os.path.join(root, 'Ground Truth')
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Warning: CVC-ClinicDB paths not found for split '{split}' at {root}. Expected dirs 'Original', 'Ground Truth'. Returning empty.")
            return []
        for f in os.listdir(image_dir):
            if f.lower().endswith('.tif'):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_dir, f)
                mask_full_path = os.path.join(mask_dir, img_name_base + '.tif')
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for CVC image {f}. Skipping.")

    # DentalPanoramic Dataset
    elif dataset_name == 'DentalPanoramic':
        image_dir = os.path.join(root, 'images') # DentalPanoramic structure under root
        mask_dir = os.path.join(root, 'segmentation_1') # Assuming this is the mask folder
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Warning: DentalPanoramic paths not found for split '{split}' at {root}. Expected dirs 'images', 'segmentation_1'. Returning empty.")
            return []
        for f in os.listdir(image_dir):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                img_name_base = os.path.splitext(f)[0]
                img_full_path = os.path.join(image_dir, f)
                mask_full_path = os.path.join(mask_dir, img_name_base + '.png') # Assuming PNG masks
                if os.path.exists(mask_full_path):
                    dataset_items.append((img_full_path, mask_full_path))
                else:
                    print(f"Warning: Mask not found for DentalPanoramic image {f}. Skipping.")

    # SixDiseasesChestXRay Dataset
    elif dataset_name == 'SixDiseasesChestXRay':
        # This dataset has train/val/test folders, and then disease subfolders
        base_data_dir = os.path.join(root, split) # e.g., DATA_ROOT/Dataset/train
        
        if not os.path.exists(base_data_dir):
            print(f"Warning: SixDiseasesChestXRay split path not found: {base_data_dir}. Returning empty.")
            return []
        
        subfolders = ['Covid', 'Normal', 'Tuberculosis', 'Bacterial Pneumonia', 'Pneumothorax', 'Viral Pneumonia']
        for sub_name in subfolders:
            sub_image_path = os.path.join(base_data_dir, sub_name, 'images')
            sub_mask_path = os.path.join(base_data_dir, sub_name, 'masks')
            
            if not os.path.exists(sub_image_path) or not os.path.exists(sub_mask_path):
                print(f"Warning: Image or mask path for {sub_name} not found in SixDiseasesChestXRay (split '{split}'). Skipping class.")
                continue
            
            for f in os.listdir(sub_image_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    img_name_base = os.path.splitext(f)[0]
                    img_full_path = os.path.join(sub_image_path, f)
                    mask_full_path = os.path.join(sub_mask_path, img_name_base + '.png')
                    if os.path.exists(mask_full_path):
                        dataset_items.append((img_full_path, mask_full_path))
                    else:
                        print(f"Warning: Mask not found for {sub_name} image {f} in SixDiseasesChestXRay. Skipping.")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. This dataset does not have a defined data loading mechanism.")
    
    if not dataset_items:
        print(f"Warning: Found 0 images for dataset '{dataset_name}' with split '{split}'. Please check dataset paths and file extensions.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train'):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args 
        
        # Call make_dataset with the correct split and root
        all_imgs = make_dataset(root, dataset_name, split)
        
        # Perform programmatic splitting ONLY IF the dataset itself doesn't have explicit train/val/test folders
        # And if the provided 'root' directory is not already the specific split directory.
        # For TSRS_RSNA, root is already the specific split dir (e.g., .../train)
        # For JSRT, COVID19_Radiography, CVC-ClinicDB, DentalPanoramic, SixDiseasesChestXRay, we assume root is the dataset's top level directory.
        
        perform_internal_split = True
        if 'TSRS_RSNA' in dataset_name:
            # For TSRS_RSNA, the 'root' passed is already the split directory (e.g., train, val)
            # so make_dataset returns items for that specific split. No further splitting needed.
            self.imgs = all_imgs
            perform_internal_split = False
        
        if perform_internal_split:
            # For other datasets, we assume 'root' points to the main dataset folder
            # and we need to split the collected images.
            random.seed(42) # For reproducibility
            random.shuffle(all_imgs)
            
            total_size = len(all_imgs)
            # The split sizes here (0.8, 0.1, 0.1) are for datasets where we *don't* already have pre-defined splits.
            # If the dataset IS pre-split (like TSRS_RSNA), make_dataset will handle it.
            # If a dataset like JSRT is small, this split might lead to empty splits.
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
            print(f"Warning: {self.split} split for {self.dataset_name} is empty. Check dataset path and contents, or adjust splitting ratio if dataset is small.")
            # Optionally raise an error if this is critical for the dataset
            # raise RuntimeError(f"Empty split '{split}' for dataset {dataset_name}. Check dataset configuration.")


        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

        # --- DASEG's fixed preprocessing dimensions (hardcoded for faithful comparison) ---
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
        # Assuming label 0 is background and any non-zero value is foreground (class 1)
        label_index[label_np > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)