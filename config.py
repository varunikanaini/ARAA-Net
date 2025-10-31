#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import os

# Base directory for all datasets
data_root = './data'
os.makedirs(data_root, exist_ok=True)

# --- Original dataset (TSRS_RSNA-Epiphysis) ---
tsrs_rsna_epiphysis_base = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
epiphysis_train_path = os.path.join(tsrs_rsna_epiphysis_base, 'train')
epiphysis_test_path = os.path.join(tsrs_rsna_epiphysis_base, 'test')

# =========================================================================
# === FIX: Added paths for the Articular-Surface dataset ===
# This mirrors the structure of the Epiphysis dataset as requested.
# =========================================================================
tsrs_rsna_articular_base = os.path.join(data_root, 'TSRS_RSNA-Articular-Surface')
articular_train_path = os.path.join(tsrs_rsna_articular_base, 'train')
articular_test_path = os.path.join(tsrs_rsna_articular_base, 'test')
# =========================================================================

# Backbone path - keep as is
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

# --- Paths for other potential datasets (downloads are disabled) ---
jsrt_dataset_base = os.path.join(data_root, 'jsrt-247-image-lung-segmentation-mask-dataset')
covid_dataset_base = os.path.join(data_root, 'covid19-radiography-database')
cvc_clinicdb_base = os.path.join(data_root, 'CVC-ClinicDB')
dental_panoramic_base = os.path.join(data_root, 'dental_panoramic_xrays')
six_diseases_base = os.path.join(data_root, 'Dataset')


# Dictionary to map dataset names to their root paths
DATASET_PATHS = {
    # Epiphysis Paths
    'TSRS_RSNA-Epiphysis_train': epiphysis_train_path,
    'TSRS_RSNA-Epiphysis_test': epiphysis_test_path,

    # =========================================================================
    # === FIX: Added dictionary keys for the Articular-Surface dataset ===
    # This resolves the KeyError.
    # =========================================================================
    'TSRS_RSNA-Articular-Surface_train': articular_train_path,
    'TSRS_RSNA-Articular-Surface_test': articular_test_path,
    # =========================================================================

    # Other Dataset Paths
    'JSRT': jsrt_dataset_base,
    'COVID19_Radiography': covid_dataset_base,
    'CVC-ClinicDB': cvc_clinicdb_base,
    'DentalPanoramic': dental_panoramic_base,
    'SixDiseasesChestXRay': six_diseases_base,
}