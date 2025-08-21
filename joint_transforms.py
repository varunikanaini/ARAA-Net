# joint_transforms.py

import random
from PIL import Image
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class Resize(object):
    def __init__(self, size):
        # size: (h, w)
        self.size = tuple(reversed(size))  # PIL expects (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, lbl_fill=None):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        assert img.size == lbl.size, 'size of img and lbl should be the same.'
        
        # Pad if needed
        if self.pad_if_needed:
            w, h = img.size
            pad_h = max(self.size[0] - h, 0)
            pad_w = max(self.size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, 0, pad_w, pad_h), self.fill)
                lbl = F.pad(lbl, (0, 0, pad_w, pad_h), self.lbl_fill)

        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)