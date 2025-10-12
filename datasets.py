# datasets.py

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms
from glob import glob

# Define default extensions if not provided by config
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

class ImageFolder(data.Dataset):
    """
    Custom Dataset class for loading image-mask pairs from specified directory structures.
    Supports different dataset structures (STANDARD, TSRS_RSNA, COVID19, SIX_DISEASES).
    Handles data augmentation transformations.
    """
    def __init__(self, root, dataset_name, args, split='train', kfold_mode=False):
        """
        Args:
            root (str): Path to the dataset directory.
            dataset_name (str): Name of the dataset (used to retrieve config).
            args (argparse.Namespace): Parsed command-line arguments containing dataset-specific info and augmentation parameters.
            split (str): 'train', 'val', or 'test'.
            kfold_mode (bool): If True, assumes 'index' refers to an item within a subset
                                and modifies __getitem__ behavior slightly. Used for K-Fold.
        """
        self.root = root
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        self.kfold_mode = kfold_mode # Flag for K-Fold usage

        self.dataset_config = args.DATASET_CONFIG[dataset_name]
        self.image_ext = self.dataset_config.get('image_ext', IMAGE_EXTENSIONS)
        self.mask_ext = self.dataset_config.get('mask_ext', MASK_EXTENSIONS)
        self.num_classes = self.dataset_config['num_classes']

        self.image_dir, self.mask_dir = self._get_data_paths()
        
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Image or mask directory not found for split '{split}' in dataset '{dataset_name}'. Expected image dir: {self.image_dir}, mask dir: {self.mask_dir}")

        self.image_files = sorted(self._get_image_files())
        self.mask_files = sorted(self._get_mask_files())

        if len(self.image_files) == 0 or len(self.mask_files) == 0:
            raise RuntimeError(f"No images or masks found for split '{split}' in dataset '{dataset_name}'. Check extensions and directory structure.")
        if len(self.image_files) != len(self.mask_files):
            print(f"Warning: Number of images ({len(self.image_files)}) does not match number of masks ({len(self.mask_files)}) for split '{split}' in dataset '{dataset_name}'. This might cause issues.")
            # Potentially truncate to the shorter list if desired, or raise error
            min_len = min(len(self.image_files), len(self.mask_files))
            self.image_files = self.image_files[:min_len]
            self.mask_files = self.mask_files[:min_len]


        self.transform = self._get_transform()

    def _get_data_paths(self):
        """Determines the image and mask directory paths based on dataset structure."""
        structure = self.dataset_config.get('structure', 'STANDARD')
        subfolders = self.dataset_config.get('subfolders', None)

        if structure == 'STANDARD':
            if subfolders and self.split in subfolders:
                img_folder = subfolders[self.split].get('images', '')
                mask_folder = subfolders[self.split].get('masks', '')
            else: # Assume default 'images' and 'masks' folders if not specified per split
                img_folder = 'images'
                mask_folder = 'masks'
            img_path = os.path.join(self.root, img_folder)
            mask_path = os.path.join(self.root, mask_folder)
            
        elif structure == 'TSRS_RSNA':
            # TSRS_RSNA datasets often have train/val/test directly under root,
            # with images and masks inside each split folder.
            img_path = os.path.join(self.root, self.split)
            mask_path = os.path.join(self.root, self.split) # Assuming masks are in the same dir or a subfolder
            # If masks are in a specific subfolder like 'Masks', adjust accordingly:
            # mask_path = os.path.join(self.root, self.split, 'Masks')
            
        elif structure == 'COVID19':
            # COVID19 dataset has 'images' and 'masks' folders at the root,
            # and inside those, subfolders for train/test.
            if self.split == 'train': # For COVID19, train/val/test splits are not explicit folders, but based on file naming.
                                      # We assume all data is loaded, and filtering happens elsewhere or implicitly.
                                      # Here, we load from the main image/mask dirs.
                img_path = os.path.join(self.root, 'images')
                mask_path = os.path.join(self.root, 'masks')
            elif self.split == 'test': # Assuming test data might be separate
                img_path = os.path.join(self.root, 'images')
                mask_path = os.path.join(self.root, 'masks')
            else: # Fallback or handle validation if it exists
                img_path = os.path.join(self.root, 'images')
                mask_path = os.path.join(self.root, 'masks')
            
        elif structure == 'SIX_DISEASES':
            # Assuming the structure is: Dataset/split/images and Dataset/split/masks
            img_path = os.path.join(self.root, self.split, 'images')
            mask_path = os.path.join(self.root, self.split, 'masks')
            
        else:
            raise ValueError(f"Unsupported dataset structure: {structure}")
            
        return img_path, mask_path

    def _get_image_files(self):
        """Finds all image files matching the specified extensions."""
        files = []
        for ext in self.image_ext:
            files.extend(glob(os.path.join(self.image_dir, f'*{ext}')))
        return files

    def _get_mask_files(self):
        """Finds all mask files matching the specified extensions."""
        files = []
        for ext in self.mask_ext:
            # Handle cases where mask files might have different naming conventions (e.g., '_mask.png')
            # This glob pattern is a basic attempt; may need refinement based on dataset specifics.
            files.extend(glob(os.path.join(self.mask_dir, f'*{ext}')))
            # Example for TSRS_RSNA where mask name might match image name exactly:
            # files.extend(glob(os.path.join(self.mask_dir, f'*{ext}')))
        
        # Basic filtering to ensure mask files are likely associated with image files
        # This is heuristic and might need adjustment. A robust approach links files by name.
        filtered_files = []
        image_basenames = {os.path.splitext(os.path.basename(f))[0] for f in self.image_files}
        for mask_file in files:
            mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
            # Try to match basename directly or with common suffixes like '_mask'
            if mask_basename in image_basenames or \
               mask_basename.replace('_mask', '') in image_basenames or \
               mask_basename.replace('_gt', '') in image_basenames: # Add other common suffixes
                filtered_files.append(mask_file)
                
        if len(filtered_files) != len(self.image_files):
             print(f"Warning: Mask file count ({len(filtered_files)}) after matching differs from image count ({len(self.image_files)}). Check naming conventions. Proceeding with {len(filtered_files)} masks.")
             # Re-aligning files based on basename matching
             aligned_masks = []
             image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.image_files}
             
             for img_file in self.image_files:
                 img_basename = os.path.splitext(os.path.basename(img_file))[0]
                 found_mask = None
                 
                 # Try exact match first
                 potential_mask_path = os.path.join(self.mask_dir, f"{img_basename}{self.mask_ext[0]}") # Use first mask ext
                 if os.path.exists(potential_mask_path):
                      found_mask = potential_mask_path
                 else:
                      # Try common suffixes
                      for ext in self.mask_ext:
                           for suffix in ['_mask', '_gt']:
                                potential_mask_path = os.path.join(self.mask_dir, f"{img_basename}{suffix}{ext}")
                                if os.path.exists(potential_mask_path):
                                     found_mask = potential_mask_path
                                     break
                           if found_mask: break
                 
                 if found_mask:
                      aligned_masks.append(found_mask)
                 else:
                      print(f"Warning: Could not find a matching mask for image: {img_file}. Skipping this image.")
                      # Remove corresponding image file to keep lists aligned
                      self.image_files.remove(img_file) 

             self.mask_files = aligned_masks
             # Ensure image_files is updated based on successful mask finding
             self.image_files = [f for f in self.image_files if os.path.join(self.mask_dir, f"{os.path.splitext(os.path.basename(f))[0]}{self.mask_ext[0]}") in self.mask_files or \
                                 any(os.path.exists(os.path.join(self.mask_dir, f"{os.path.splitext(os.path.basename(f))[0]}{suffix}{ext}")) for suffix in ['_mask', '_gt'] for ext in self.mask_ext)]


        return self.mask_files # Return the potentially re-aligned mask files


    def _get_transform(self):
        """Defines the data augmentation pipeline based on arguments."""
        transform_list = []

        # Fixed resize based on backbone input size is often applied first
        if self.args.scale_h and self.args.scale_w:
             transform_list.append(transforms.Resize((self.args.scale_h, self.args.scale_w), interpolation=transforms.InterpolationMode.BILINEAR))
             # PIL.Image.BILINEAR might be used if not using torchvision.transforms directly for resize
             # transform_list.append(FixedResize(self.args.scale_w, self.args.scale_h)) 

        # Center Amplification - should be applied before random crops
        if self.args.min_lesion_area_pixels > 0:
            transform_list.append(custom_transforms.CenterAmplification(
                min_lesion_area_pixels=self.args.min_lesion_area_pixels,
                expansion_factor=self.args.expansion_factor,
                min_bbox_size=(self.args.min_bbox_h, self.args.min_bbox_w)
            ))

        # Random horizontal flip
        transform_list.append(custom_transforms.RandomHorizontalFlip())
        
        # Random affine transformations
        transform_list.append(custom_transforms.RandomAffine(
            degrees=10, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1), 
            shear=10
        ))
        
        # Random crop - ensure it uses the size defined by scale_h/scale_w if provided, otherwise aspect ratio
        # For segmentation, usually you want crops of a fixed size or aspect ratio.
        # If scale_h/scale_w are set, RandomCrop might not be needed if FixedResize already handles it.
        # If you need random crops AFTER resizing, define a size here.
        # Example: transform_list.append(custom_transforms.RandomCrop((self.args.scale_h, self.args.scale_w)))

        # Random Gaussian Blur
        transform_list.append(custom_transforms.RandomGaussianBlur(radius_range=(0.1, 1.0)))
        
        # Contrast enhancement methods
        if self.args.wavelet_type: # Wavelet contrast enhancement
            transform_list.append(custom_transforms.WaveletContrastEnhancement(
                wavelet=self.args.wavelet_type,
                level=self.args.wavelet_level,
                detail_scale_factor=self.args.wavelet_detail_scale
            ))
        else: # Fallback to histogram equalization if no wavelet specified
            transform_list.append(custom_transforms.HistogramEqualization())

        # Color jitter (applies to image only)
        transform_list.append(custom_transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ))

        # Random Cutout
        transform_list.append(custom_transforms.RandomCutout(p=0.3))

        # Convert PIL Image to PyTorch Tensor and Normalize
        # Mean and Std are typically standard for ImageNet models, but might need dataset-specific values
        # For medical images, normalization often uses dataset mean/std.
        # If not specified, use common values.
        mean = [0.485, 0.456, 0.406] if self.args.dataset_name not in ['JSRT', 'COVID19_Radiography'] else [0.5, 0.5, 0.5] # Example for non-RGB or different datasets
        std = [0.229, 0.224, 0.225] if self.args.dataset_name not in ['JSRT', 'COVID19_Radiography'] else [0.5, 0.5, 0.5]
        
        # Adjust normalization based on number of channels
        if self.args.backbone in ['inception_v3']: # Inception expects 3 channels
             mean = [0.5, 0.5, 0.5]
             std = [0.5, 0.5, 0.5]
             
        if self.dataset_config.get('structure') in ['JSRT', 'COVID19_Radiography']: # Grayscale datasets
            mean = [0.5]
            std = [0.5]
            transform_list.append(transforms.Lambda(lambda img: transforms.functional.to_tensor(img.convert('L')))) # Ensure grayscale conversion and tensor
        else: # RGB datasets
            transform_list.append(transforms.Lambda(lambda img: transforms.functional.to_tensor(img)))

        transform_list.append(transforms.Normalize(mean=mean, std=std))

        # Convert PIL label to PyTorch LongTensor (do this AFTER transformations that modify PIL images)
        # This is done within the __getitem__ after image transformations are applied.
        
        return transforms.Compose(transform_list)

    def _get_image_files(self):
        """Finds all image files matching the specified extensions."""
        files = []
        for ext in self.image_ext:
            files.extend(glob(os.path.join(self.image_dir, f'*{ext}')))
        # Filter out mask files if they accidentally got included due to similar extensions
        files = [f for f in files if not any(f.endswith(mask_ext) for mask_ext in self.mask_ext)]
        return files

    def _get_mask_files(self):
        """Finds all mask files matching the specified extensions and attempts to link them to images."""
        files = []
        for ext in self.mask_ext:
            files.extend(glob(os.path.join(self.mask_dir, f'*{ext}')))
        
        # --- File Alignment Logic ---
        # Create a mapping from image basename to its full path
        image_basename_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in self.image_files}
        
        aligned_mask_files = []
        unmatched_images = []

        for mask_file in files:
            mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
            matched = False
            
            # Try direct basename match
            if mask_basename in image_basename_to_path:
                aligned_mask_files.append(mask_file)
                matched = True
            else:
                # Try matching with common suffixes like '_mask', '_gt'
                for suffix in ['_mask', '_gt']:
                    potential_basename = mask_basename.replace(suffix, '')
                    if potential_basename in image_basename_to_path:
                        aligned_mask_files.append(mask_file)
                        matched = True
                        break # Found a match with a suffix
            
            if not matched:
                print(f"Warning: Mask file '{os.path.basename(mask_file)}' not directly matched to an image basename. Checking if it's an exact image file.")
                # If it's not matched and doesn't seem like a mask file (e.g., if mask ext is also in image ext)
                if mask_file in self.image_files: # Check if this mask file is actually an image file
                     print(f"  '{os.path.basename(mask_file)}' seems to be an image file, not a mask. Removing from mask list.")
                else:
                    print(f"  '{os.path.basename(mask_file)}' is an unmatched mask file. It might be ignored if it doesn't correspond to an image.")

        # Now, ensure we only keep image files that have a corresponding mask file
        final_image_files = []
        final_mask_files = []

        for mask_file in aligned_mask_files:
             mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
             found_image_for_mask = False
             
             # Search for the corresponding image file
             for img_file in self.image_files:
                  img_basename = os.path.splitext(os.path.basename(img_file))[0]
                  if img_basename == mask_basename or \
                     img_basename == mask_basename.replace('_mask', '') or \
                     img_basename == mask_basename.replace('_gt', ''):
                         final_image_files.append(img_file)
                         final_mask_files.append(mask_file)
                         found_image_for_mask = True
                         break # Found the image for this mask
             
             if not found_image_for_mask:
                  print(f"Warning: Image file not found for mask '{os.path.basename(mask_file)}'. This mask will be ignored.")

        if len(final_image_files) != len(final_mask_files):
            print(f"Error during file alignment: Mismatch after final check. Img count: {len(final_image_files)}, Mask count: {len(final_mask_files)}. Check dataset integrity and naming.")
            # Decide how to handle: raise error, or proceed with mismatch (not recommended)
            # For now, we will proceed but warn
            
        self.image_files = final_image_files
        self.mask_files = final_mask_files
        
        if not self.image_files or not self.mask_files:
             raise RuntimeError("After alignment, no valid image-mask pairs were found.")

        return self.mask_files


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Loads an image and its corresponding mask.
        Applies transformations.
        Returns a dictionary containing 'image' and 'label'.
        Returns None if sample loading or processing fails.
        """
        if self.kfold_mode:
            # In kfold_mode, index is already an index into the subset.
            # No need to adjust index, directly use it.
            pass
        
        # Ensure index is within bounds (though sampler should handle this)
        if index >= len(self.image_files):
            print(f"Error: Index {index} out of bounds for dataset size {len(self.image_files)}.")
            return None

        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        try:
            # Load Image
            img = Image.open(image_path).convert('RGB') # Always convert to RGB for consistency, handle grayscale internally in transforms if needed
            
            # Load Mask
            mask = Image.open(mask_path)
            
            # Ensure mask is in a format suitable for segmentation (e.g., 'L' for grayscale labels)
            # If masks are multi-channel, they might need specific handling or conversion.
            if mask.mode != 'L':
                mask = mask.convert('L') # Convert mask to grayscale

            # Apply transformations
            if self.transform:
                # Apply image transforms
                img_transformed = self.transform(img)
                
                # Apply mask transform separately (e.g., resizing, converting to tensor)
                # Note: Some transforms like RandomAffine, CenterAmplification modify PIL images.
                # Ensure these are compatible with both image and mask.
                # Here, we assume transform handles image, and we'll apply specific mask processing.
                
                # Re-apply resize if not handled by main transform and needed for mask
                if self.args.scale_h and self.args.scale_w:
                     mask = mask.resize((self.args.scale_w, self.args.scale_h), Image.NEAREST)

                # Convert mask to tensor
                mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

                return {'image': img_transformed, 'label': mask_tensor}
            else:
                # If no transform, just convert to tensors
                img_tensor = transforms.functional.to_tensor(img)
                mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
                return {'image': img_tensor, 'label': mask_tensor}

        except Exception as e:
            print(f"Error loading or processing sample at index {index} (Image: {image_path}, Mask: {mask_path}): {e}")
            return None # Return None to indicate failure

# --- Helper functions for dataset creation ---
# These might be used if you need to create specific datasets or splits manually.
# For now, ImageFolder directly loads from pre-defined paths.

def make_dataset(image_dir, mask_dir, image_ext=IMAGE_EXTENSIONS, mask_ext=MASK_EXTENSIONS):
    """
    Creates a list of (image, mask) file paths.
    Assumes a 1-to-1 correspondence based on filenames (after stripping extensions).
    """
    image_files = []
    for ext in image_ext:
        image_files.extend(glob(os.path.join(image_dir, f'*{ext}')))
    
    mask_files = []
    for ext in mask_ext:
        mask_files.extend(glob(os.path.join(mask_dir, f'*{ext}')))

    image_files = sorted(image_files)
    mask_files = sorted(mask_files)

    # Basic alignment based on filename (stripping extension)
    data = []
    image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    
    for mask_file in mask_files:
        mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
        matched = False
        for img_basename, img_path in image_map.items():
            if img_basename == mask_basename or \
               img_basename == mask_basename.replace('_mask', '') or \
               img_basename == mask_basename.replace('_gt', ''):
                data.append((img_path, mask_file))
                matched = True
                break
        if not matched:
            print(f"Warning: Mask file '{os.path.basename(mask_file)}' has no matching image. Skipping.")
            
    if not data:
        raise RuntimeError("No image-mask pairs found. Check dataset structure and file naming.")

    return data

# --- Import custom_transforms module ---
# Ensure custom_transforms.py is in the same directory or accessible via PYTHONPATH
import custom_transforms