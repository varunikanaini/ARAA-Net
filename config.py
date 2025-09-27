import os

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# This DATASET_NAME variable is now a placeholder; the actual dataset
# will be chosen via command-line arguments in train/test/visualize scripts.
DATASET_NAME = 'PLACEHOLDER_FOR_DYNAMIC_SELECTION'

# --- Define paths for all datasets ---
# These paths assume the datasets are organized under DATA_ROOT
# For Kaggle environments, you might need to manually copy input datasets
# to these DATA_ROOT subdirectories as the script does not handle downloads.

DATASET_PATHS = {
    # Original dataset
    'TSRS_RSNA-Epiphysis': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
    # New datasets
    'JSRT': os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset'),
    'COVID19_Radiography': os.path.join(DATA_ROOT, 'covid19-radiography-database'),
    'CVC-ClinicDB': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
    'DentalPanoramic': os.path.join(DATA_ROOT, 'dental_panoramic_xrays'),
    'SixDiseasesChestXRay': os.path.join(DATA_ROOT, 'Dataset'), # The root folder in the user's reference code
}

# Ensure DATA_ROOT and CKPT_ROOT exist (if not handled by Kaggle or environment)
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)

# Placeholder for backbone path, if used
# backbone_path = './backbone/resnet/resnet50-19c8e357.pth'