#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import os
import kagglehub

# Base directory for all datasets
data_root = './data'
os.makedirs(data_root, exist_ok=True)

# Original dataset (TSRS_RSNA-Epiphysis)
tsrs_rsna_epiphysis_base = os.path.join(data_root, 'TSRS_RSNA-Epiphysis')
cod_training_root = os.path.join(tsrs_rsna_epiphysis_base, 'train')
chameleon_path = os.path.join(tsrs_rsna_epiphysis_base, 'test')

# Backbone path - keep as is
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'


# JSRT dataset
jsrt_dataset_base = os.path.join(data_root, 'jsrt-247-image-lung-segmentation-mask-dataset')
# Download JSRT dataset if not already present or if the target directory is empty
if not os.path.exists(jsrt_dataset_base) or not os.listdir(jsrt_dataset_base):
    print(f"Downloading JSRT dataset to {data_root}...")
    kagglehub.dataset_download("abduzzami/jsrt-247-image-lung-segmentation-mask-dataset", path=data_root)
    print("JSRT dataset downloaded.")

# COVID-19 Radiography Database dataset
covid_dataset_base = os.path.join(data_root, 'covid19-radiography-database')
# Download COVID-19 Radiography Database dataset if not already present or if the target directory is empty
if not os.path.exists(covid_dataset_base) or not os.listdir(covid_dataset_base):
    print(f"Downloading COVID-19 Radiography Database dataset to {data_root}...")
    kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database", path=data_root)
    print("COVID-19 Radiography Database dataset downloaded.")

# CVC-ClinicDB dataset (user manually copies)
# User is expected to manually copy 'original' and 'ground truth' folders into cvc_clinicdb_base.
cvc_clinicdb_base = os.path.join(data_root, 'CVC-ClinicDB')
print(f"For CVC-ClinicDB: Please ensure your 'CVC-ClinicDB' folder with 'original' and 'ground truth' subfolders is placed at: {cvc_clinicdb_base}")


# Dictionary to map dataset names to their root paths
DATASET_PATHS = {
    'TSRS_RSNA-Epiphysis_train': cod_training_root,
    'TSRS_RSNA-Epiphysis_test': chameleon_path,
    'JSRT': jsrt_dataset_base,
    'COVID19_Radiography': covid_dataset_base,
    'CVC-ClinicDB': cvc_clinicdb_base,
}