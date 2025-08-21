# datasets.py

import os
import os.path
import torch.utils.data as data
from PIL import Image

def make_dataset(root):
    # Your original logic for finding image and label pairs
    mask_path = root + '_labels'
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    # This is your original, correct ImageFolder class
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L') # Use 'L' for single-channel labels

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # The output of target_transform will be (1, H, W), so we squeeze it to (H, W)
        return {'image': img, 'label': target.squeeze(0)}

    def __len__(self):
        return len(self.imgs)