# config.py

import os

DATA_ROOT = './data'

CKPT_ROOT = '/content/drive/MyDrive/araa/ARAA-Net/ckpt'

DATASET_NAME = 'TSRS_RSNA-Epiphysis'
# DATASET_NAME = 'TSRS_RSNA-Articular-Surface'



dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
cod_training_root = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')