import os

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# The specific dataset to use
DATASET_NAME = 'TSRS_RSNA-Epiphysis'

# --- Construct the final, absolute paths for train and validation ---
dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
cod_training_root = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'val')

# --- NEW: Teacher-Student Specific Hyperparameters ---
PSEUDO_LABEL_CONF_THRESHOLD = 0.9  # Confidence threshold for teacher's predictions to be considered pseudo-labels
EMA_DECAY_RATE = 0.999            # Exponential Moving Average decay rate for teacher model