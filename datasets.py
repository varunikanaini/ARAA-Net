# /kaggle/working/araa/ARAA-Net/datasets.py (FINAL VERSION)

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms

# We will use custom_transforms for Normalize and ToTensor which work with dicts
import custom_transforms as tr 

def make_dataset(root):
    # This function is fine as-is.
    mask_path = os.path.join(root, 'GT')
    image_path = root
    mask_path = root + '_labels'
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

# In datasets.py, replace the entire ImageFolder class with this:

class ImageFolder(data.Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.imgs = make_dataset(root)
        self.split = split
        
        # --- THIS IS THE FIX ---
        # The transforms now correctly use the arguments passed from the training script
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                # Use the scale_w (or a dedicated crop_size arg) for the random crop
                tr.RandomCrop((args.scale_h, args.scale_w)), 
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)
        
        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)
        
        return transformed_sample
    
    def convert_label(self, label):
        label_rgb = np.array(label)
        label_index = np.full(label_rgb.shape[:2], 0, dtype='uint8')
        for k in range(1, 30):
            label_index[label_rgb == k] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)