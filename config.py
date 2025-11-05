#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# --- Base Directories ---
data_root = './data'
os.makedirs(data_root, exist_ok=True)
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# =========================================================================
# === Centralized Dataset Paths ===
# =========================================================================

# --- TSRS Datasets ---
tsrs_epiphysis_path = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
tsrs_articular_path = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')

# --- JSRT Dataset ---
jsrt_path = os.path.join(data_root, 'jsrt')

# --- CVC-ClinicDB Dataset ---
cvc_path = os.path.join(data_root, 'CVC-ClinicDB')

# --- Montgomery County Dataset ---
montgomery_path = os.path.join(data_root, 'MontgomerySet') # Using your specified path

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
        'path': jsrt_path,
        'structure': 'FLAT_SPLIT',
    },
    'CVC-ClinicDB': {
        'path': cvc_path,
        'structure': 'FLAT_SPLIT',
    },
    # --- NEW DATASET ENTRY ---
    'MontgomeryCounty': {
        'path': montgomery_path,
        'structure': 'FLAT_SPLIT', # It needs splitting, so this is correct
    },
    # --------------------------
}

print("--- Resolved Dataset Paths ---")
for name, cfg in DATASET_CONFIG.items():
    print(f"{name} Path is pointing to: {cfg['path']}")
print("------------------------------")