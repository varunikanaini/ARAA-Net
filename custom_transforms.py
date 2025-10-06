# /kaggle/working/ARAA-Net/custom_transforms.py (Corrected)

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
import pywt
import random

# --- ColorJitter Augmentation (Added as per error message) ---
class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        """
        Randomly change the brightness, contrast, saturation, and hue of an image.
        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
            hue (float or tuple of float (min, max)): How much to jitter hue.
        """
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Apply ColorJitter to the image
        img = self.jitter(img)
        return {'image': img, 'label': label}

# --- Cutout Augmentation ---
class Cutout(object):
    """
    Randomly masks out one or more square regions in the image.
    """
    def __init__(self, num_holes=1, max_h_size=64, max_w_size=64, fill_value=0, always_apply=False):
        self.num_holes = num_holes
        self.max_h_size = max_h_size # Max height of the cutout square
        self.max_w_size = max_w_size # Max width of the cutout square
        self.fill_value = fill_value # Fill value for the cutout region (0 for black)
        self.always_apply = always_apply # Whether to apply augmentation always or randomly

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        
        # Apply augmentation randomly (50% chance) unless always_apply is True
        if not self.always_apply and np.random.rand() < 0.5:
            return {'image': img, 'label': label}

        img_np = np.array(img) # Convert PIL Image to numpy array for manipulation
        h, w, c = img_np.shape # Get image dimensions (h, w, channels)

        for _ in range(self.num_holes):
            # Randomly choose size for the hole (ensure it's not larger than image dimensions)
            hole_h = random.randint(1, min(self.max_h_size, h))
            hole_w = random.randint(1, min(self.max_w_size, w))

            # Randomly choose top-left corner for the hole
            # Ensure the entire hole fits within the image bounds
            y1 = random.randint(0, h - hole_h)
            x1 = random.randint(0, w - hole_w)
            y2 = y1 + hole_h
            x2 = x1 + hole_w

            # Fill the cutout region in the image
            # Use self.fill_value for all channels
            img_np[y1:y2, x1:x2, :] = self.fill_value
            
            # Note: Cutout is applied only to the image here. If masks need it,
            # you would need to apply it to the label_np array as well, likely with `self.mask_fill_value`.
            # For segmentation masks, fill_value should typically be 0.

        img_out = Image.fromarray(img_np)
        return {'image': img_out, 'label': label} # Return original label

# --- RandomHorizontalFlip ---
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': label}

# --- FixedResize ---
class FixedResize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img.resize((self.w, self.h), Image.BILINEAR)
        label = label.resize((self.w, self.h), Image.NEAREST) # Use NEAREST for masks
        return {'image': img, 'label': label}

# --- RandomCrop ---
class RandomCrop(object):
    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        w, h = img.size
        th, tw = self.size
        
        # If image is already the target size, return as is
        if w == tw and h == th:
            return {'image': img, 'label': label}
        
        # If image is smaller than target size, resize it to target size (with padding implicitly handled by Pillow's resize)
        if h < th or w < tw:
            img = img.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)
            return {'image': img, 'label': label}

        # If image is larger, perform random crop
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        img = img.crop((j, i, j + tw, i + th))
        label = label.crop((j, i, j + tw, i + th))
        return {'image': img, 'label': label}

# --- RandomGaussianBlur ---
class RandomGaussianBlur(object):
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5: # Apply Gaussian blur with 50% probability
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return {'image': img, 'label': label}

# --- Normalize ---
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Convert PIL Image to Tensor and then normalize
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.mean, self.std)
        return {'image': img, 'label': label}

# --- ToTensor ---
class ToTensor(object):
    """ Converts the PIL label to a PyTorch LongTensor. Image is assumed to be already a Tensor. """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Convert PIL Image label to numpy array, then to LongTensor
        label_np = np.array(label, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_np).long()
        return {'image': img, 'label': label_tensor}

# --- CenterAmplification ---
class CenterAmplification(object):
    """
    Applies center-based amplification to small lesions, as described in STS-Net Method 4.
    Operates on PIL Images. This transform should ideally be applied AFTER FixedResize
    and BEFORE RandomCrop.
    """
    def __init__(self, min_lesion_area_pixels=576, expansion_factor=1.5, min_bbox_size=(32, 32)):
        self.min_lesion_area_pixels = min_lesion_area_pixels
        self.expansion_factor = expansion_factor
        self.min_bbox_size = min_bbox_size # (h, w) for minimum expanded bbox size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)
        
        lesion_coords = np.argwhere(label_np > 0) # Find coordinates of non-zero pixels (lesions)
        lesion_area = len(lesion_coords)

        # Apply amplification only if the lesion area is within the specified range (0 < area < threshold)
        if 0 < lesion_area < self.min_lesion_area_pixels:
            # Find bounding box of the lesion
            y_min, x_min = lesion_coords.min(axis=0)
            y_max, x_max = lesion_coords.max(axis=0)

            h_lesion, w_lesion = y_max - y_min + 1, x_max - x_min + 1
            center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

            # Calculate expanded dimensions, ensuring minimum size is met
            expanded_h = max(self.min_bbox_size[0], int(h_lesion * self.expansion_factor))
            expanded_w = max(self.min_bbox_size[1], int(w_lesion * self.expansion_factor))
            
            img_w, img_h = img.size # Original image dimensions
            
            # Calculate raw expanded crop coordinates around the center
            cx1_raw = center_x - expanded_w // 2
            cy1_raw = center_y - expanded_h // 2
            
            cx2_raw = center_x + (expanded_w // 2) + (expanded_w % 2)
            cy2_raw = center_y + (expanded_h // 2) + (expanded_h % 2)

            # Clamp coordinates to image boundaries
            crop_x_min = max(0, cx1_raw)
            crop_y_min = max(0, cy1_raw)
            crop_x_max = min(img_w, cx2_raw)
            crop_y_max = min(img_h, cy2_raw)

            current_crop_w = crop_x_max - crop_x_min
            current_crop_h = crop_y_max - crop_y_min

            # Adjust if clamped crop is smaller than desired expanded size
            if current_crop_w < expanded_w:
                if crop_x_min == 0: # If left edge was clamped
                    crop_x_max = min(img_w, crop_x_min + expanded_w)
                else: # If right edge was clamped or it was centered, adjust left edge
                    crop_x_min = max(0, crop_x_max - expanded_w)
            
            if current_crop_h < expanded_h:
                if crop_y_min == 0: # If top edge was clamped
                    crop_y_max = min(img_h, crop_y_min + expanded_h)
                else: # If bottom edge was clamped or it was centered, adjust top edge
                    crop_y_min = max(0, crop_y_max - expanded_h)
            
            # Final clamping to ensure minimum dimensions after adjustments
            crop_x_max = max(crop_x_min + self.min_bbox_size[1], crop_x_max)
            crop_y_max = max(crop_y_min + self.min_bbox_size[0], crop_y_max)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)
            crop_x_min = max(0, crop_x_max - self.min_bbox_size[1])
            crop_y_min = max(0, crop_y_max - self.min_bbox_size[0])

            # Crop the image and label
            img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            label_cropped = label.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            
            # Resize the cropped image and label back to the original image size
            # This is crucial to maintain consistency for subsequent transformations like FixedResize
            target_img_size = img.size # Use original image size for resizing back
            img = img_cropped.resize(target_img_size, Image.BILINEAR)
            label = label_cropped.resize(target_img_size, Image.NEAREST)

        return {'image': img, 'label': label}


# --- RandomAffine ---
class RandomAffine(object):
    def __init__(self, degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10, fill_color=(0,0,0), mask_fill_value=0):
        """
        Applies random affine transformations to the image and mask.
        Args:
            degrees (float or tuple): Range of degrees to select from.
            translate (tuple): Range of translation (x, y).
            scale (tuple): Range of scale (x, y).
            shear (float or tuple): Range of shear.
            fill_color (int or tuple): Pixel value to fill for transformed pixels (for image).
            mask_fill_value (int): Pixel value to fill for transformed pixels (for mask).
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill_color = fill_color
        self.mask_fill_value = mask_fill_value

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        
        # Generate random parameters for affine transformation
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        tx = 0.0
        ty = 0.0
        if self.translate:
            tx = np.random.uniform(-self.translate[0], self.translate[0])
            ty = np.random.uniform(-self.translate[1], self.translate[1])
        
        # Handle scale parameter: can be a scalar or a tuple (min, max)
        s = 1.0
        if isinstance(self.scale, (tuple, list)):
            s = np.random.uniform(self.scale[0], self.scale[1])
        elif self.scale is not None: # Assume scalar
            s = np.random.uniform(1.0 - self.scale, 1.0 + self.scale)
        
        sh = 0.0
        if self.shear:
            sh = np.random.uniform(-self.shear, self.shear)

        # Apply affine transformation to both image and label
        img = transforms.functional.affine(img, angle=angle, 
                                           translate=(tx, ty),
                                           scale=s,
                                           shear=sh,
                                           fill=self.fill_color) # Use 'fill' for image
        
        label = transforms.functional.affine(label, angle=angle, 
                                             translate=(tx, ty),
                                             scale=s,
                                             shear=sh,
                                             fill=self.mask_fill_value) # Use 'fill' for mask

        return {'image': img, 'label': label}