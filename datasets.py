# /kaggle/working/ARAA-Net/datasets.py (FINAL VERSION WITH CENTERAMPLIFICATION)

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms

import custom_transforms as tr 

def make_dataset(root):
    image_path = root
    
    if os.path.exists(os.path.join(root, 'GT')):
        mask_path = os.path.join(root, 'GT')
    elif os.path.exists(root + '_labels'):
        mask_path = root + '_labels'
    else:
        raise FileNotFoundError(f"Could not find label directory for {root}. Looked in '{os.path.join(root, 'GT')}' and '{root + '_labels'}'.")

    img_list = []
    for f in os.listdir(image_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_list.append(os.path.splitext(f)[0])

    dataset_items = []
    for img_name in img_list:
        img_full_path = os.path.join(image_path, img_name + '.jpg')
        mask_full_path = os.path.join(mask_path, img_name + '.png')
        
        if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
            dataset_items.append((img_full_path, mask_full_path))
        else:
            print(f"Warning: Missing image or mask for {img_name}. Skipping.")

    if not dataset_items:
        raise RuntimeError(f"Found 0 images in {root} with corresponding labels. Please check dataset path and file extensions.")
        
    return dataset_items


class ImageFolder(data.Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.imgs = make_dataset(root)
        self.split = split
        self.args = args

        min_lesion_area = args.min_lesion_area_pixels
        expansion_factor = args.expansion_factor
        min_bbox_h = args.min_bbox_h
        min_bbox_w = args.min_bbox_w

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.FixedResize(w=args.scale_w, h=args.scale_h),
                tr.CenterAmplification(min_lesion_area_pixels=min_lesion_area,
                                       expansion_factor=expansion_factor,
                                       min_bbox_size=(min_bbox_h, min_bbox_w)),
                tr.RandomHorizontalFlip(),
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
        label_np = np.array(label, dtype=np.uint8)
        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1
        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)