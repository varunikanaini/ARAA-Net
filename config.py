# /kaggle/working/ARAA-Net/config.py

import os
import kagglehub
import shutil
import logging

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data' 

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Function to download and prepare KaggleHub datasets (unchanged) ---
def download_and_extract_kaggle_dataset(dataset_id, target_dir):
    # ... (function body remains unchanged) ...
    pass


# --- Specific KaggleHub Dataset IDs and their target names in DATA_ROOT ---
KAGGLE_DATASET_MAPPING = {
    'TSRS_RSNA-Epiphysis': {'id': None, 'local_dir_name': 'TSRS_RSNA-Epiphysis'}, 
    'TSRS_RSNA-Articular-Surface': {'id': None, 'local_dir_name': 'TSRS_RSNA-Articular-Surface'}, 
    'KOA': { 
        'id': None, 
        'local_dir_name': 'lvv-koa' 
    },
    'MURA': {
        'id': 'murrphyh/mura-v11', 
        'local_dir_name': 'MURA-v11' 
    },
    'COVID-19_Radiography': { 
        'id': 'tawsifurrahman/covid19-radiography-database',
        'local_dir_name': 'COVID-19_Radiography_Dataset'
    },
    'JSRT': { 
        'id': 'abduzzami/jsrt-247-image-lung-segmentation-mask-dataset',
        'local_dir_name': 'jsrt'
    },
    # --- ADD THIS NEW ENTRY ---
    'CVC-ClinicDB': {
        'id': None, # Not downloaded from KaggleHub
        'local_dir_name': 'CVC-ClinicDB' # Expects folder name 'CVC-ClinicDB' under DATA_ROOT
    }
}