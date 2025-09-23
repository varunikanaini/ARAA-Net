# /kaggle/working/ARAA-Net/config.py
import os

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# Removed: DATASET_NAME, cod_training_root, test_path as they are now dynamic.