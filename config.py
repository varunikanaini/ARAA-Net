#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# --- Base Directories ---
data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# =========================================================================
# === KAGGLE-AWARE PATHING (FINAL, CORRECTED VERSION) ===
# This logic now correctly points to the subdirectory where the data actually resides.
# =========================================================================

# --- TSRS Datasets (Assumed to be in the local ./data directory) ---
tsrs_epiphysis_path = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
tsrs_articular_path = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')

# --- JSRT Dataset ---
# 1. Define the top-level Kaggle and local directories
kaggle_jsrt_toplevel_path = '/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset'
local_jsrt_toplevel_path = os.path.join(data_root, 'jsrt-247-image-lung-segmentation-mask-dataset')

# 2. Determine which top-level directory exists
base_jsrt_path = kaggle_jsrt_toplevel_path if os.path.exists(kaggle_jsrt_toplevel_path) else local_jsrt_toplevel_path

# 3. CRITICAL FIX: Point to the actual data subfolder, which is often named 'jsrt'
#    The `datasets.py` script will now look for 'cxr' inside this final path.
jsrt_path = os.path.join(base_jsrt_path, 'jsrt')

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
        'path': jsrt_path,  # Use the final, corrected path
        'structure': 'FLAT_SPLIT',
    },
    'CVC-ClinicDB': {
        'path': cvc_path,
        'structure': 'FLAT_SPLIT',
    },
}

print("--- Resolved Dataset Paths ---")
print(f"JSRT Path is now pointing to: {DATASET_CONFIG['JSRT']['path']}")
print("------------------------------")