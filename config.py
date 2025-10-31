#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# --- Base Directories ---
data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': os.path.join(data_root, 'TSRS_RSNA-Epiphysis'),
        'structure': 'PRE_SPLIT',
    },
    'TSRS_RSNA-Articular-Surface': {
        'path': os.path.join(data_root, 'TSRS_RSNA-Articular-Surface'),
        'structure': 'PRE_SPLIT',
    },
    'JSRT': {
        'path': os.path.join(data_root, 'jsrt-247-image-lung-segmentation-mask-dataset'),
        'structure': 'FLAT_SPLIT',
    },
    'CVC-ClinicDB': {
        'path': os.path.join(data_root, 'CVC-ClinicDB'),
        'structure': 'FLAT_SPLIT',
    },
}