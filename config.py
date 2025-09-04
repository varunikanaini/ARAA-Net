import os

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# DATASET_NAME is now passed via command-line arguments to train.py, test.py, etc.
# These paths will be constructed dynamically in the scripts.
# For example:
# dataset_path = os.path.join(DATA_ROOT, <your_dataset_name>)
# cod_training_root = os.path.join(dataset_path, 'train')
# test_path = os.path.join(dataset_path, 'val')