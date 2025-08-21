# config.py

import os

DATA_ROOT = './data'

CKPT_ROOT = './ckpt'

DATASET_NAME = 'TSRS_RSNA-Epiphysis'

dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
cod_training_root = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')