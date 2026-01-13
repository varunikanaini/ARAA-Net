#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

WORK_DIR = '/kaggle/working/ARAA-Net'
DATA_ROOT = os.path.join(WORK_DIR, 'data')
os.makedirs(DATA_ROOT, exist_ok=True)

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
        'structure': 'PRE_SPLIT',
        'num_classes': 2
    },
    'TSRS_RSNA-Articular-Surface': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Articular-Surface'),
        'structure': 'PRE_SPLIT',
        'num_classes': 2
    },
    'JSRT': {
        'path': os.path.join(DATA_ROOT, 'jsrt'),
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'CVC-ClinicDB': {
        'path': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'MontgomeryCounty': {
        'path': os.path.join(DATA_ROOT, 'MontgomerySet'),
        'structure': 'FLAT_SPLIT',
        'num_classes': 2
    },
    'VOC2012': {
        'path': os.path.join(DATA_ROOT, 'VOCdevkit', 'VOC2012'),
        'structure': 'VOC',
        'num_classes': 21,
        'ignore_index': 255
    },
    'Cityscapes': {
        'path': os.path.join(DATA_ROOT, 'cityscapes'),
        'structure': 'Cityscapes',
        'num_classes': 19,
        'ignore_index': 255
    },
    'COCO-Stuff': {
        'path': '/kaggle/input/coco-2017-dataset/coco2017',
        'structure': 'COCO-JSON',
        'num_classes': 92,
        'ignore_index': 255,
        'img_root': '/kaggle/input/coco-2017-dataset/coco2017',
        'ann_root': os.path.join(DATA_ROOT, 'coco_stuff', 'annotations')
    }
}

print("--- Resolved Dataset Paths ---")
for name, cfg in DATASET_CONFIG.items():
    print(f"{name:<25} : {cfg['path']}")
print("------------------------------")