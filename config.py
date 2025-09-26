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
jsrt_dataset_name_kaggle = "abduzzami/jsrt-247-image-lung-segmentation-mask-dataset"
jsrt_dataset_base = os.path.join(data_root, os.path.basename(jsrt_dataset_name_kaggle))

# Only attempt Kagglehub download if the directory is truly empty/missing
if not os.path.exists(jsrt_dataset_base) or not os.listdir(jsrt_dataset_base):
    print(f"Downloading JSRT dataset to {data_root}...")
    kagglehub.dataset_download(jsrt_dataset_name_kaggle, path=data_root)
    print("JSRT dataset downloaded.")

# COVID-19 Radiography Database dataset
covid_dataset_name_kaggle = "tawsifurrahman/covid19-radiography-database"
covid_dataset_base = os.path.join(data_root, os.path.basename(covid_dataset_name_kaggle))

# Only attempt Kagglehub download if the directory is truly empty/missing
if not os.path.exists(covid_dataset_base) or not os.listdir(covid_dataset_base):
    print(f"Downloading COVID-19 Radiography Database dataset to {data_root}...")
    kagglehub.dataset_download(covid_dataset_name_kaggle, path=data_root)
    print("COVID-19 Radiography Database dataset downloaded.")

# CVC-ClinicDB dataset (user manually copies)
cvc_clinicdb_base = os.path.join(data_root, 'CVC-ClinicDB')
print(f"For CVC-ClinicDB: Please ensure your 'CVC-ClinicDB' folder with 'original' and 'ground truth' subfolders is placed at: {cvc_clinicdb_base}")


# New: Panoramic Dental X-rays With Segmented Mandibles
# As per user's clarification, this dataset is expected at 'data/dental_panoramic_xrays'
dental_panoramic_base = os.path.join(data_root, 'dental_panoramic_xrays')
print(f"For DentalPanoramic: Please ensure the dataset content is placed at: {dental_panoramic_base}")
print(f"  If running on Kaggle, you might need to copy from /kaggle/input/panoramic-dental-x-rays-with-segmented-mandibles to {dental_panoramic_base}")

# New: 6-Diseases Chest X-Ray Dataset with Masks
# As per user's clarification, this dataset is expected at 'data/Dataset'
six_diseases_base = os.path.join(data_root, 'Dataset')
print(f"For SixDiseasesChestXRay: Please ensure the dataset content is placed at: {six_diseases_base}")
print(f"  If running on Kaggle, you might need to copy from /kaggle/input/6-diseases-chest-x-ray-dataset-with-masks to {six_diseases_base}")


# Dictionary to map dataset names to their root paths
DATASET_PATHS = {
    'TSRS_RSNA-Epiphysis_train': cod_training_root,
    'TSRS_RSNA-Epiphysis_test': chameleon_path,
    'JSRT': jsrt_dataset_base,
    'COVID19_Radiography': covid_dataset_base,
    'CVC-ClinicDB': cvc_clinicdb_base,
    'DentalPanoramic': dental_panoramic_base,
    'SixDiseasesChestXRay': six_diseases_base,
}