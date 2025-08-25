# /kaggle/working/ARAA-Net/datasets.py (MODIFIED)

import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
# We will use the custom transforms for Normalize and ToTensor, but torchvision for others
import custom_transforms as tr 

def make_dataset(root):
    # This function is fine as-is.
    mask_path = os.path.join(root, 'GT')
    image_path = root
    mask_path = root + '_labels'
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, split='train'):
        self.root = root
        self.imgs = make_dataset(root)
        self.split = split
        
        # Let's define the transforms directly here for clarity
        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                # NEW Augmentations
                transforms.RandomResizedCrop(size=(576, 576), scale=(0.5, 2.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
                tr.RandomGaussianBlur(),
                
                # Keep your custom Normalize and ToTensor
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else: # Validation/Test
            self.composed_transforms = transforms.Compose([
                # For validation, we want deterministic resizing, not random.
                tr.FixedResize(w=576, h=896), # Maintain aspect ratio before normalizing
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
            transformed_sample['name'] = self.imgs[index]
        
        return transformed_sample
    
    def convert_label(self, label):
        # This function is fine as-is.
        label_rgb = np.array(label)
        label_index = np.full(label_rgb.shape[:2], 0, dtype='uint8')
        for k in range(1,30):
            label_index[label_rgb == k] = 1
        label_index = Image.fromarray(label_index, mode='P')
        return label_index

    def __len__(self):
        return len(self.imgs)