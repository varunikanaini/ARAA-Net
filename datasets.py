# datasets.py

import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np

def make_dataset(root):
    # Logic to find image and label pairs
    mask_path = root + '_labels'
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def convert_label(self, label_img):
        """
        Converts a multi-class label image into a binary (0/1) mask.
        This is the critical missing step.
        """
        label_array = np.array(label_img, dtype=np.uint8)
        # Create a new array initialized to 0 (background)
        binary_mask = np.zeros_like(label_array)
        # Set all pixels with a value > 0 to 1 (foreground)
        binary_mask[label_array > 0] = 1
        # Convert back to a PIL Image
        return Image.fromarray(binary_mask, mode='L')

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        
        # Load the original ground truth image
        original_target = Image.open(gt_path)
        
        # Convert it to a binary mask using the essential function
        target = self.convert_label(original_target)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # The target is now a (1, H, W) tensor of 0s and 1s. Squeeze it to (H, W).
        return {'image': img, 'label': target.squeeze(0)}

    def __len__(self):
        return len(self.imgs)