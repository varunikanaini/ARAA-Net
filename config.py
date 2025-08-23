# config.py

import os

DATA_ROOT = 'kaggle/working/ARAA-Net/data'

CKPT_ROOT = '/kaggle/working/araa/ckpt'

DATASET_NAME = 'TSRS_RSNA-Epiphysis'
# DATASET_NAME = 'TSRS_RSNA-Articular-Surface'



dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
cod_training_root = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')