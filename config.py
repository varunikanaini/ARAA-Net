# /kaggle/working/ARAA-Net/config.py
import os

# READ data from the original, read-only INPUT directory
# Ensure this path exists and contains your datasets.
# For Kaggle, you'd typically upload datasets to this directory.
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# This DATASET_NAME variable is now a placeholder; the actual dataset
# will be chosen via command-line arguments in train/test/visualize scripts.
DATASET_NAME = 'PLACEHOLDER_FOR_DYNAMIC_SELECTION'

# --- Define paths for all datasets ---
# These paths assume the datasets are organized under DATA_ROOT
# For Kaggle environments, you might need to manually copy input datasets
# to these DATA_ROOT subdirectories.

# The keys here MUST match the --dataset-name argument used in scripts.
DATASET_PATHS = {
    # Original datasets (assuming Articular-Surface is also structured like Epiphysis)
    'TSRS_RSNA-Epiphysis': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
    'TSRS_RSNA-Articular-Surface': os.path.join(DATA_ROOT, 'TSRS_RSNA-Articular-Surface'),
    # New datasets
    'JSRT': os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset'),
    'COVID19_Radiography': os.path.join(DATA_ROOT, 'covid19-radiography-database'),
    'CVC-ClinicDB': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
    'DentalPanoramic': os.path.join(DATA_ROOT, 'dental_panoramic_xrays'),
    'SixDiseasesChestXRay': os.path.join(DATA_ROOT, 'Dataset'), # The root folder in the user's reference code
}

# Ensure DATA_ROOT and CKPT_ROOT exist
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
