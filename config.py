#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

tsrs_epiphysis_path = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
tsrs_articular_path = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')
jsrt_path = os.path.join(data_root, 'jsrt')
cvc_path = os.path.join(data_root, 'CVC-ClinicDB')
montgomery_path = os.path.join(data_root, 'MontgomerySet')
voc_path = os.path.join(data_root, 'VOCdevkit', 'VOC2012')
cityscapes_path = os.path.join(data_root, 'Cityscapes')
coco_img_path = '/kaggle/input/coco-2017-dataset/coco2017'
coco_ann_path = os.path.join(data_root, 'coco_stuff', 'annotations')

DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': tsrs_epiphysis_path,
        'structure': 'PRE_SPLIT',
        'num_classes': 2
    },
    'TSRS_RSNA-Articular-Surface': {
        'path': tsrs_articular_path,
        'structure': 'PRE_SPLIT',
        'num_classes': 2
    },
    'JSRT': {
        'path': jsrt_path,
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'CVC-ClinicDB': {
        'path': cvc_path,
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'MontgomeryCounty': {
        'path': montgomery_path,
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'VOC2012': {
        'path': voc_path,
        'structure': 'VOC',
        'num_classes': 21,
        'ignore_index': 255
    },
    'Cityscapes': {
        'path': cityscapes_path,
        'structure': 'Cityscapes',
        'num_classes': 19,
        'ignore_index': 255
    },
    'COCO-Stuff': {
        'path': coco_img_path,
        'structure': 'COCO-JSON',
        'num_classes': 92,
        'ignore_index': 255,
        'img_root': coco_img_path,
        'ann_root': coco_ann_path
    }
}

print("--- Resolved Dataset Paths ---")
for name, cfg in DATASET_CONFIG.items():
    print(f"{name} Path is pointing to: {cfg['path']}")
print("------------------------------")