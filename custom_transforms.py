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
        img = img.resize((self.w, self.h), Image.BILINEAR) 
        label = label.resize((self.w, self.h), Image.NEAREST)
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
        
        if h < th or w < tw:
            img = img.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)
            return {'image': img, 'label': label}

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
        img = transforms.functional.to_tensor(img)
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
        
        lesion_coords = np.argwhere(label_np > 0)
        lesion_area = len(lesion_coords)

        if 0 < lesion_area < self.min_lesion_area_pixels:
            y_min, x_min = lesion_coords.min(axis=0)
            y_max, x_max = lesion_coords.max(axis=0)

            h_lesion, w_lesion = y_max - y_min + 1, x_max - x_min + 1
            center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

            expanded_h = max(self.min_bbox_size[0], int(h_lesion * self.expansion_factor))
            expanded_w = max(self.min_bbox_size[1], int(w_lesion * self.expansion_factor))
            
            img_w, img_h = img.size
            
            cx1_raw = center_x - expanded_w // 2
            cy1_raw = center_y - expanded_h // 2
            
            cx2_raw = center_x + (expanded_w // 2) + (expanded_w % 2)
            cy2_raw = center_y + (expanded_h // 2) + (expanded_h % 2)

            crop_x_min = max(0, cx1_raw)
            crop_y_min = max(0, cy1_raw)
            crop_x_max = min(img_w, cx2_raw)
            crop_y_max = min(img_h, cy2_raw)

            current_crop_w = crop_x_max - crop_x_min
            current_crop_h = crop_y_max - crop_y_min

            if current_crop_w < expanded_w:
                if crop_x_min == 0:
                    crop_x_max = min(img_w, crop_x_min + expanded_w)
                else:
                    crop_x_min = max(0, crop_x_max - expanded_w)
            
            if current_crop_h < expanded_h:
                if crop_y_min == 0:
                    crop_y_max = min(img_h, crop_y_min + expanded_h)
                else:
                    crop_y_min = max(0, crop_y_max - expanded_h)
            
            crop_x_max = max(crop_x_min + self.min_bbox_size[1], crop_x_max)
            crop_y_max = max(crop_y_min + self.min_bbox_size[0], crop_y_max)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)
            crop_x_min = max(0, crop_x_max - self.min_bbox_size[1])
            crop_y_min = max(0, crop_y_max - self.min_bbox_size[0])

            img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            label_cropped = label.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            
            target_img_size = img.size
            img = img_cropped.resize(target_img_size, Image.BILINEAR)
            label = label_cropped.resize(target_img_size, Image.NEAREST)

        return {'image': img, 'label': label}