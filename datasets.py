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

class ImageFolder(data.Dataset):
    # It now accepts args to use command-line parameters
    def __init__(self, root, args, split='train'):
        self.root = root
        self.imgs = make_dataset(root)
        self.split = split
        
        if self.split == 'train':
            # This pipeline uses your command-line args and adds better augmentation
            self.composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                # Use scale from args, then crop to a fixed size for training stability
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                # RandomResizedCrop is excellent for handling objects of different sizes
                tr.RandomCrop((576, 576)), 
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                # For validation, we just resize deterministically based on your args
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
        label_index = Image.fromarray(label_index, mode='P')
        return label_index

    def __len__(self):
        return len(self.imgs)