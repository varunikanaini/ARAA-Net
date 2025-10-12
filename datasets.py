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

# --- Import custom_transforms module ---
# Ensure custom_transforms.py is in the same directory or accessible via PYTHONPATH
import custom_transforms

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
        self.mask_files = sorted(self._get_mask_files()) # This will also align the files

        if len(self.image_files) == 0 or len(self.mask_files) == 0:
            raise RuntimeError(f"No images or masks found or aligned for split '{split}' in dataset '{dataset_name}'. Check extensions and directory structure.")
        
        # Final check after alignment to ensure counts match and are non-zero
        if len(self.image_files) != len(self.mask_files):
            print(f"Critical Error: Image/Mask file count mismatch after alignment ({len(self.image_files)} images, {len(self.mask_files)} masks). Dataset is likely corrupted or naming conventions are not handled.")
            # Raise an error to halt execution if alignment failed severely
            raise RuntimeError("Alignment of image and mask files failed.")

        self.transform = self._get_transform()

    def _get_data_paths(self):
        """Determines the image and mask directory paths based on dataset structure."""
        structure = self.dataset_config.get('structure', 'STANDARD')
        subfolders = self.dataset_config.get('subfolders', None)

        img_path, mask_path = None, None

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
            mask_path = os.path.join(self.root, self.split) 
            # If masks are in a specific subfolder like 'Masks', adjust accordingly:
            # mask_path = os.path.join(self.root, self.split, 'Masks')
            
        elif structure == 'COVID19':
            # COVID19 dataset has 'images' and 'masks' folders at the root,
            # and inside those, subfolders for train/test.
            if self.split == 'train': 
                img_path = os.path.join(self.root, 'images')
                mask_path = os.path.join(self.root, 'masks')
            elif self.split == 'test': 
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
        # Filter out mask files if they accidentally got included due to similar extensions
        files = [f for f in files if not any(f.endswith(mask_ext) for mask_ext in self.mask_ext)]
        return files

    def _get_mask_files(self):
        """Finds all mask files matching the specified extensions and attempts to link them to images."""
        all_mask_files_found = []
        for ext in self.mask_ext:
            all_mask_files_found.extend(glob(os.path.join(self.mask_dir, f'*{ext}')))
        
        # --- File Alignment Logic ---
        # Create a mapping from image basename to its full path
        image_basename_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in self.image_files}
        
        aligned_mask_files = []
        
        for mask_file in all_mask_files_found:
            mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
            matched_image_path = None
            
            # Try direct basename match
            if mask_basename in image_basename_to_path:
                matched_image_path = image_basename_to_path[mask_basename]
            else:
                # Try matching with common suffixes like '_mask', '_gt'
                for suffix in ['_mask', '_gt']:
                    potential_basename = mask_basename.replace(suffix, '')
                    if potential_basename in image_basename_to_path:
                        matched_image_path = image_basename_to_path[potential_basename]
                        break # Found a match with a suffix
            
            if matched_image_path:
                aligned_mask_files.append(mask_file)
            else:
                print(f"Warning: Mask file '{os.path.basename(mask_file)}' not matched to any image basename. It might be ignored.")

        # Now, filter self.image_files to only include those that have a corresponding mask
        final_image_files = []
        final_mask_files = []
        
        processed_image_basenames = set() # To avoid duplicates if multiple masks map to the same image

        for mask_file in aligned_mask_files:
             mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
             
             for img_file in self.image_files:
                  img_basename = os.path.splitext(os.path.basename(img_file))[0]
                  
                  # Check for direct or suffixed match
                  if img_basename == mask_basename or \
                     img_basename == mask_basename.replace('_mask', '') or \
                     img_basename == mask_basename.replace('_gt', ''):
                      
                      if img_basename not in processed_image_basenames: # Ensure unique image
                           final_image_files.append(img_file)
                           final_mask_files.append(mask_file)
                           processed_image_basenames.add(img_basename)
                           break # Found the image for this mask, move to next mask
                           
        # Update the instance's file lists with the aligned and filtered ones
        self.image_files = final_image_files
        self.mask_files = final_mask_files
        
        if not self.image_files or not self.mask_files:
             raise RuntimeError("After alignment, no valid image-mask pairs were found. Please check dataset integrity and naming conventions.")

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
            img = Image.open(image_path).convert('RGB') # Always convert to RGB for consistency
            
            # Load Mask
            mask = Image.open(mask_path)
            
            # Ensure mask is in a format suitable for segmentation (e.g., 'L' for grayscale labels)
            if mask.mode != 'L':
                mask = mask.convert('L') # Convert mask to grayscale

            # Apply transformations
            if self.transform:
                # Apply image transforms
                img_transformed = self.transform(img)
                
                # Apply mask transform separately (e.g., resizing, converting to tensor)
                # Certain PIL-based transforms like resize might be applied here.
                # Ensure transforms that modify PIL objects are applied correctly.
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
    This function is not directly used by ImageFolder but can be helpful for dataset management.
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