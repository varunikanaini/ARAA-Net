#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang
"""
import os

# --- Base Directories ---
data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# =========================================================================
# === NEW: Centralized Dataset Configuration (The Correct Approach) ===
# This dictionary is now the single source of truth for dataset paths and types.
# 'PRE_SPLIT':  The script will look for 'train' and 'test' subfolders inside the path.
# 'FLAT_SPLIT': The script will load all images from the root and split them 80/10/10.
# =========================================================================
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
    # Add any other datasets here following the same pattern
}