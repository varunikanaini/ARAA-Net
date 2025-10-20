import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torchvision import transforms
import pywt
import random
from torchvision.transforms import functional as TF
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

# --- Elastic Transform ---
class ElasticTransform(object):
    """
    Apply elastic deformation on a PIL image and its corresponding mask.
    Increased alpha and sigma for stronger deformation, and p for higher probability.
    """
    def __init__(self, alpha=340, sigma=5, p=0.8): # Increased alpha, sigma, and p
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        img, label = sample['image'], sample['label']
        
        img_np = np.array(img)
        label_np = np.array(label)

        shape = img_np.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        transformed_label_np = map_coordinates(label_np, indices, order=0, mode='reflect').reshape(shape)

        if len(img_np.shape) == 3 and img_np.shape[2] > 1:
            channels = [map_coordinates(img_np[..., i], indices, order=1, mode='reflect').reshape(shape) for i in range(img_np.shape[2])]
            transformed_img_np = np.stack(channels, axis=-1)
        else:
            transformed_img_np = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
        
        img = Image.fromarray(transformed_img_np.astype(np.uint8))
        label = Image.fromarray(transformed_label_np.astype(np.uint8))
        
        return {'image': img, 'label': label}

# --- Random Horizontal Flip ---
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5): # Default p is usually fine, but you can increase if dataset is highly symmetric
        self.p = p
    def __call__(self, sample):
        if np.random.rand() < self.p:
            img = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            label = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)
            return {'image': img, 'label': label}
        return sample

# --- Fixed Resize ---
class FixedResize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, sample):
        img = sample['image'].resize((self.w, self.h), Image.BILINEAR)
        label = sample['label'].resize((self.w, self.h), Image.NEAREST)
        return {'image': img, 'label': label}

# --- Random Crop ---
class RandomCrop(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return sample

        if h < th or w < tw:
            # Pad if smaller than desired crop size
            padding_h = max(0, th - h)
            padding_w = max(0, tw - w)
            img = TF.pad(img, (0, 0, padding_w, padding_h), fill=0) # Pad with black
            label = TF.pad(label, (0, 0, padding_w, padding_h), fill=0) # Pad label with background
            w, h = img.size

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        img_cropped = img.crop((j, i, j + tw, i + th))
        label_cropped = label.crop((j, i, j + tw, i + th))
        return {'image': img_cropped, 'label': label_cropped}

# --- Random Gaussian Blur ---
class RandomGaussianBlur(object):
    def __init__(self, radius_range=(0.1, 2.0), p=0.5): # Increased p
        self.radius_range = radius_range
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            img = sample['image'].filter(ImageFilter.GaussianBlur(radius=radius))
            return {'image': img, 'label': sample['label']}
        return sample

# --- Normalize ---
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.mean, self.std)
        return {'image': img, 'label': label}

# --- ToTensor ---
class ToTensor(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)
        if label_np.max() > 1: # Assuming binary segmentation where non-zero is foreground
            label_np = (label_np > 0).astype(np.int64) # More robust binarization
        else:
            label_np = label_np.astype(np.int64)
        label = torch.from_numpy(label_np).long()
        return {'image': img, 'label': label}

# --- Center Amplification ---
class CenterAmplification(object):
    """
    Operates on PIL Images. Applied AFTER FixedResize and BEFORE RandomCrop.
    """
    def __init__(self, min_lesion_area_pixels=576, expansion_factor=1.5, min_bbox_size=(32, 32)):
        self.min_lesion_area_pixels = min_lesion_area_pixels
        self.expansion_factor = expansion_factor
        self.min_bbox_size = min_bbox_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)

        # Ensure label_np is binary for accurate area calculation
        binary_label_np = (label_np > 0).astype(np.uint8)
        lesion_coords = np.argwhere(binary_label_np > 0)
        
        if lesion_coords.size == 0: # No lesions found
            return sample

        lesion_area = len(lesion_coords)

        if 0 < lesion_area < self.min_lesion_area_pixels:
            y_min, x_min = lesion_coords.min(axis=0)
            y_max, x_max = lesion_coords.max(axis=0)

            h_lesion, w_lesion = y_max - y_min + 1, x_max - x_min + 1
            center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

            expanded_h = max(self.min_bbox_size[0], int(h_lesion * self.expansion_factor))
            expanded_w = max(self.min_bbox_size[1], int(w_lesion * self.expansion_factor))

            img_w, img_h = img.size

            # Calculate desired crop box centered on lesion center
            cx1_raw = center_x - expanded_w // 2
            cy1_raw = center_y - expanded_h // 2
            cx2_raw = center_x + (expanded_w + 1) // 2 # Use ceiling division to ensure expanded_w is covered
            cy2_raw = center_y + (expanded_h + 1) // 2

            # Clamp the crop box to image boundaries
            crop_x_min = max(0, cx1_raw)
            crop_y_min = max(0, cy1_raw)
            crop_x_max = min(img_w, cx2_raw)
            crop_y_max = min(img_h, cy2_raw)

            # Adjust if clamping made the box too small or inverted
            if crop_x_max <= crop_x_min: crop_x_max = crop_x_min + 1
            if crop_y_max <= crop_y_min: crop_y_max = crop_y_min + 1

            # Ensure the cropped box can still accommodate the minimum size if possible
            # This part is tricky and might need adjustment based on how you want to handle edge cases
            # The goal is to have a crop that's AT LEAST min_bbox_size and attempts to cover expanded region
            current_crop_w = crop_x_max - crop_x_min
            current_crop_h = crop_y_max - crop_y_min

            # If after clamping, the crop is smaller than expanded dimensions, shift it
            if current_crop_w < expanded_w:
                shift = expanded_w - current_crop_w
                if crop_x_min == 0: # Shift to the right
                    crop_x_max = min(img_w, crop_x_max + shift)
                else: # Shift to the left (towards origin)
                    crop_x_min = max(0, crop_x_min - shift)
            if current_crop_h < expanded_h:
                shift = expanded_h - current_crop_h
                if crop_y_min == 0: # Shift down
                    crop_y_max = min(img_h, crop_y_max + shift)
                else: # Shift up
                    crop_y_min = max(0, crop_y_max - shift)
            
            # Final clamp to ensure it's within bounds after shifts
            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)

            # Ensure minimum crop size is respected IF possible within image bounds
            crop_w = crop_x_max - crop_x_min
            crop_h = crop_y_max - crop_y_min
            if crop_w < self.min_bbox_size[1]:
                target_w = min(img_w, self.min_bbox_size[1])
                if crop_x_min == 0:
                    crop_x_max = min(img_w, crop_x_min + target_w)
                else:
                    crop_x_min = max(0, crop_x_max - target_w)
            if crop_h < self.min_bbox_size[0]:
                target_h = min(img_h, self.min_bbox_size[0])
                if crop_y_min == 0:
                    crop_y_max = min(img_h, crop_y_min + target_h)
                else:
                    crop_y_min = max(0, crop_y_max - target_h)
            
            # Final check on dimensions
            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)
            
            if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min: # If somehow invalid, return original
                return sample

            img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            label_cropped = label.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

            # Resize the cropped region back to the original image size
            # This effectively "zooms in" on the amplified region
            img = img_cropped.resize(img.size, Image.BILINEAR)
            label = label_cropped.resize(label.size, Image.NEAREST)

        return {'image': img, 'label': label}

# --- Histogram Equalization ---
class HistogramEqualization(object):
    """
    Applies histogram equalization. Increased p value.
    """
    def __init__(self, p=0.4): # Increased p
        self.p = p
    def __call__(self, sample):
        if np.random.rand() < self.p:
            img = sample['image']
            if img.mode == 'RGB':
                img_ycbcr = img.convert('YCbCr')
                Y, Cb, Cr = img_ycbcr.split()
                Y_eq = ImageOps.equalize(Y)
                img_eq = Image.merge('YCbCr', (Y_eq, Cb, Cr)).convert('RGB')
            else:
                img_eq = ImageOps.equalize(img)
            sample['image'] = img_eq
        return sample

# --- Wavelet Contrast Enhancement ---
class WaveletContrastEnhancement(object):
    """
    Applies DWT based contrast enhancement. Increased p value and detail_scale_factor.
    """
    def __init__(self, wavelet='haar', level=1, detail_scale_factor=1.8, p=0.4): # Increased detail_scale_factor and p
        self.wavelet = wavelet
        self.level = level
        self.detail_scale_factor = detail_scale_factor
        self.p = p

        if self.wavelet not in pywt.wavelist(kind='discrete'):
            print(f"Warning: Wavelet '{self.wavelet}' not found. Falling back to 'haar'.")
            self.wavelet = 'haar'

    def __call__(self, sample):
        if np.random.rand() < self.p:
            img_pil = sample['image']

            if img_pil.mode == 'RGB':
                img_gray = img_pil.convert('L')
                original_mode = 'RGB'
            else:
                img_gray = img_pil
                original_mode = 'L'

            img_np = np.array(img_gray, dtype=np.float32) / 255.0

            coeffs_list = pywt.wavedec2(img_np, self.wavelet, mode='periodization', level=self.level)

            modified_coeffs_list = [coeffs_list[0]]
            for d_level in coeffs_list[1:]:
                cH, cV, cD = d_level
                cH_e = cH * self.detail_scale_factor
                cV_e = cV * self.detail_scale_factor
                cD_e = cD * self.detail_scale_factor
                modified_coeffs_list.append((cH_e, cV_e, cD_e))

            img_reconstructed = pywt.waverec2(modified_coeffs_list, self.wavelet, mode='periodization')

            img_reconstructed = np.clip(img_reconstructed, 0, 1)
            img_enhanced_pil = Image.fromarray((img_reconstructed * 255).astype(np.uint8))

            if original_mode == 'RGB':
                img_enhanced_pil = img_enhanced_pil.convert('RGB')

            sample['image'] = img_enhanced_pil
        return sample

# --- Random Cutout ---
class RandomCutout(object):
    """
    Increased num_holes, max_h_size, max_w_size, and p.
    """
    def __init__(self, num_holes_range=(1, 3), max_h_size=48, max_w_size=48, fill_value=0, p=0.5): # Increased parameters
        self.num_holes_range = num_holes_range
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img, label = sample['image'], sample['label']
        w, h = img.size
        num_holes = random.randint(self.num_holes_range[0], self.num_holes_range[1])

        for _ in range(num_holes):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            # Ensure cutout has at least 1x1 size
            hole_w = random.randint(1, self.max_w_size)
            hole_h = random.randint(1, self.max_h_size)
            x2 = min(w, x1 + hole_w)
            y2 = min(h, y1 + hole_h)
            
            # Draw on image
            draw_img = ImageDraw.Draw(img)
            draw_img.rectangle([x1, y1, x2, y2], fill=self.fill_value)

            # Draw on label (fill with background value 0)
            draw_label = ImageDraw.Draw(label)
            draw_label.rectangle([x1, y1, x2, y2], fill=0)

        return {'image': img, 'label': label}

# --- Random Affine ---
class RandomAffine(object):
    """
    Slightly increased parameters for more variety.
    """
    def __init__(self, degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15, mask_fill_value=0): # Increased parameters
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.mask_fill_value = mask_fill_value

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        angle = random.uniform(-self.degrees, self.degrees)
        translate_x = random.uniform(-self.translate[0], self.translate[0]) * img.size[0]
        translate_y = random.uniform(-self.translate[1], self.translate[1]) * img.size[1]
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear_angle = random.uniform(-self.shear, self.shear)

        # torchvision.transforms.functional.affine expects translate as (tx, ty)
        # and scale as single factor, shear as angle in degrees.
        # The existing implementation is correct.

        img = TF.affine(img, angle=angle, translate=(translate_x, translate_y), 
                        scale=scale_factor, shear=shear_angle, 
                        interpolation=TF.InterpolationMode.BILINEAR)

        label = TF.affine(label, angle=angle, translate=(translate_x, translate_y), 
                          scale=scale_factor, shear=shear_angle, 
                          interpolation=TF.InterpolationMode.NEAREST, fill=self.mask_fill_value)

        return {'image': img, 'label': label}

# --- Color Jitter ---
class ColorJitter(object):
    """
    Increased parameters for more aggressive color augmentation.
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1): # Increased parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        
        # Apply color jitter only to the image, not the label
        img = TF.adjust_brightness(img, random.uniform(max(0, 1 - self.brightness), 1 + self.brightness))
        img = TF.adjust_contrast(img, random.uniform(max(0, 1 - self.contrast), 1 + self.contrast))
        img = TF.adjust_saturation(img, random.uniform(max(0, 1 - self.saturation), 1 + self.saturation))
        img = TF.adjust_hue(img, random.uniform(-self.hue, self.hue))
        
        return {'image': img, 'label': label}