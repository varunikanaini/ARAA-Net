#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# --- Base Directories ---
data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# =========================================================================
# === FINAL FIX: Point directly to the correct local paths ===
# All complex pathing logic is removed for simplicity and correctness.
# =========================================================================

# --- TSRS Datasets ---
tsrs_epiphysis_path = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
tsrs_articular_path = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')

# --- JSRT Dataset ---
# This now points to the exact directory you specified.
jsrt_path = os.path.join(data_root, 'jsrt')

# --- CVC-ClinicDB Dataset ---
cvc_path = os.path.join(data_root, 'CVC-ClinicDB')


# --- Centralized Dataset Configuration ---
DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': tsrs_epiphysis_path,
        'structure': 'PRE_SPLIT',
    },
    'TSRS_RSNA-Articular-Surface': {
        'path': tsrs_articular_path,
        'structure': 'PRE_SPLIT',
    },
    'JSRT': {
        'path': jsrt_path,  # Use the final, correct local path
        'structure': 'FLAT_SPLIT',
    },
    'CVC-ClinicDB': {
        'path': cvc_path,
        'structure': 'FLAT_SPLIT',
    },
}

