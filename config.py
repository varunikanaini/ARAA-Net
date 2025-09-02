import os

# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'

# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# The specific dataset to use
DATASET_NAME = 'TSRS_RSNA-Articular-Surface'

# --- Construct the final, absolute paths for train and validation ---
dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
cod_training_root = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'val')


# Original was 0.9. Trying 0.75 to make the teacher slightly less conservative,
# which might help it mine more "missing" lesions at lower resolutions.
PSEUDO_LABEL_CONF_THRESHOLD = 0.75 
EMA_DECAY_RATE = 0.999            