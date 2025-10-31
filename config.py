#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# --- Base Directories ---
data_root = './data' # Local data root
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# =========================================================================
# === KAGGLE-AWARE PATHING (THE FINAL FIX) ===
# This logic checks if the standard Kaggle input directory exists.
# If it does, it uses that path. Otherwise, it falls back to the local path.
# This makes the script work both locally and on Kaggle without changes.
# =========================================================================

# --- TSRS Datasets (Assumed to be in the local ./data directory) ---
tsrs_epiphysis_path = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
tsrs_articular_path = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')

# --- JSRT Dataset ---
kaggle_jsrt_path = '/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset'
local_jsrt_path = os.path.join(data_root, 'jsrt-247-image-lung-segmentation-mask-dataset')
# Use the Kaggle path if it exists, otherwise use the local path
jsrt_path = kaggle_jsrt_path if os.path.exists(kaggle_jsrt_path) else local_jsrt_path

# --- CVC-ClinicDB Dataset ---
kaggle_cvc_path = '/kaggle/input/cvcclinicdb' # Example Kaggle path, adjust if needed
local_cvc_path = os.path.join(data_root, 'CVC-ClinicDB')
cvc_path = kaggle_cvc_path if os.path.exists(kaggle_cvc_path) else local_cvc_path


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
        'path': jsrt_path,  # Use the dynamically determined path
        'structure': 'FLAT_SPLIT',
    },
    'CVC-ClinicDB': {
        'path': cvc_path, # Use the dynamically determined path
        'structure': 'FLAT_SPLIT',
    },
    # Add any other datasets here following the same pattern
}

print("--- Resolved Dataset Paths ---")
print(f"JSRT Path: {DATASET_CONFIG['JSRT']['path']}")
print("------------------------------")