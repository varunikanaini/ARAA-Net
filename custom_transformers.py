# /kaggle/working/ARAA-Net/custom_transforms.py
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': label}

class FixedResize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Use Image.LANCZOS for higher quality resizing for images, but BILINEAR is often fine too.
        img = img.resize((self.w, self.h), Image.BILINEAR) 
        label = label.resize((self.w, self.h), Image.NEAREST) # For labels, use NEAREST
        return {'image': img, 'label': label}

class RandomCrop(object):
    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return {'image': img, 'label': label}
        
        # Simple random crop, not lesion-aware to keep it generic
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        img = img.crop((j, i, j + tw, i + th))
        label = label.crop((j, i, j + tw, i + th))
        return {'image': img, 'label': label}

class RandomGaussianBlur(object):
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return {'image': img, 'label': label}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = transforms.functional.to_tensor(img) # Convert to tensor first
        img = transforms.functional.normalize(img, self.mean, self.std)
        return {'image': img, 'label': label}

class ToTensor(object):
    """ Converts the PIL label to a PyTorch LongTensor. Image is assumed to be already a Tensor. """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        return {'image': img, 'label': label}

class CenterAmplification(object):
    """
    Applies center-based amplification to small lesions, as described in STS-Net Method 4.
    Operates on PIL Images. This transform should be applied AFTER FixedResize
    but BEFORE RandomCrop if RandomCrop is used to ensure consistent image dimensions.
    """
    def __init__(self, min_lesion_area_pixels=576, expansion_factor=1.5, min_bbox_size=(32, 32)):
        self.min_lesion_area_pixels = min_lesion_area_pixels
        self.expansion_factor = expansion_factor
        self.min_bbox_size = min_bbox_size # (h, w) for minimum expanded bbox size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)
        
        lesion_coords = np.argwhere(label_np > 0)
        lesion_area = len(lesion_coords)

        # Only apply if a lesion exists and is considered "small"
        if 0 < lesion_area < self.min_lesion_area_pixels:
            # Find lesion bounding box
            y_min, x_min = lesion_coords.min(axis=0)
            y_max, x_max = lesion_coords.max(axis=0)

            h_lesion, w_lesion = y_max - y_min + 1, x_max - x_min + 1
            center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

            # Expand bounding box dimensions
            expanded_h = max(self.min_bbox_size[0], int(h_lesion * self.expansion_factor))
            expanded_w = max(self.min_bbox_size[1], int(w_lesion * self.expansion_factor))
            
            img_w, img_h = img.size # Current image size (after FixedResize)
            
            # Calculate new crop coordinates, ensuring they stay within image bounds
            cx1 = max(0, center_x - expanded_w // 2)
            cy1 = max(0, center_y - expanded_h // 2)
            cx2 = min(img_w, center_x + expanded_w // 2 + (expanded_w % 2))
            cy2 = min(img_h, center_y + expanded_h // 2 + (expanded_h % 2))

            # Adjust if the calculated box is smaller than desired due to image boundaries
            if cx2 - cx1 < expanded_w:
                if cx1 == 0: cx2 = min(img_w, cx1 + expanded_w)
                else: cx1 = max(0, cx2 - expanded_w)
            
            if cy2 - cy1 < expanded_h:
                if cy1 == 0: cy2 = min(img_h, cy1 + expanded_h)
                else: cy1 = max(0, cy2 - expanded_h)
            
            # Final crop coordinates
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = cx1, cy1, cx2, cy2

            # Perform crop
            img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            label_cropped = label.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            
            # Resize cropped image and label back to the original image dimensions (after FixedResize)
            target_img_size = img.size # (W, H)
            img = img_cropped.resize(target_img_size, Image.BILINEAR)
            label = label_cropped.resize(target_img_size, Image.NEAREST)

        return {'image': img, 'label': label}