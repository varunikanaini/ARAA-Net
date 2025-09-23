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


# --- NEW: Function to download and prepare KaggleHub datasets ---
def download_and_extract_kaggle_dataset(dataset_id, target_dir):
    """
    Downloads a KaggleHub dataset and extracts it to the target directory.
    If the target directory already contains the dataset (or part of it), skips download.
    Returns the final path where the dataset is ready to be used.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # KaggleHub's download function returns the path where it puts the dataset.
    # This path is usually /kaggle/input/<owner>/<dataset-slug>/<version-number>/
    logging.info(f"Attempting to download KaggleHub dataset '{dataset_id}' to a temporary location...")
    try:
        downloaded_path_input = kagglehub.dataset_download(dataset_id)
        logging.info(f"Downloaded temporarily to: {downloaded_path_input}")
    except Exception as e:
        logging.error(f"Error downloading KaggleHub dataset '{dataset_id}': {e}")
        return None

    # Determine the actual top-level folder name inside the downloaded content
    # e.g., for "tawsifurrahman/covid19-radiography-database", it's "COVID-19_Radiography_Dataset"
    # for "abduzzami/jsrt-247-image-lung-segmentation-mask-dataset", it's "jsrt-247-image-lung-segmentation-mask-dataset"
    actual_dataset_folder_name = os.listdir(downloaded_path_input)[0] if os.listdir(downloaded_path_input) else ""
    source_dataset_root = os.path.join(downloaded_path_input, actual_dataset_folder_name)

    # Define the final destination path within DATA_ROOT
    final_destination_path = os.path.join(target_dir, os.path.basename(source_dataset_root))

    if os.path.exists(final_destination_path) and os.listdir(final_destination_path):
        logging.info(f"Dataset already appears to be in '{final_destination_path}'. Skipping copy.")
        return final_destination_path
    elif os.path.exists(final_destination_path) and not os.listdir(final_destination_path):
        logging.warning(f"Target directory '{final_destination_path}' exists but is empty. Proceeding with copy.")
        shutil.rmtree(final_destination_path) # Clean up empty dir before copy
        
    logging.info(f"Copying contents from '{source_dataset_root}' to '{final_destination_path}'...")
    try:
        shutil.copytree(source_dataset_root, final_destination_path)
        logging.info(f"✅ Successfully copied dataset to: '{final_destination_path}'")
        return final_destination_path
    except Exception as e:
        logging.error(f"Error copying dataset: {e}. Source: '{source_dataset_root}', Dest: '{final_destination_path}'")
        return None

# --- Specific KaggleHub Dataset IDs and their target names in DATA_ROOT ---
KAGGLE_DATASET_MAPPING = {
    'KOA': { # This is your lvv-koa, which you should place under DATA_ROOT/KOA manually or adjust script
        'id': None, # Assuming this is local for now, not KaggleHub ID
        'local_dir_name': 'lvv-koa' # The folder name under DATA_ROOT
    },
    'MURA': {
        'id': 'murrphyh/mura-v11', # Example MURA dataset ID, replace if yours is different
        'local_dir_name': 'MURA-v11' # The folder name under DATA_ROOT for MURA
    },
    'Chest-Xray': { # This refers to the COVID-19 Radiography Database you linked
        'id': 'tawsifurrahman/covid19-radiography-database',
        'local_dir_name': 'COVID-19_Radiography_Dataset' # The folder name under DATA_ROOT for this
    },
    'JSRT': { # This refers to the JSRT-247 dataset you linked
        'id': 'abduzzami/jsrt-247-image-lung-segmentation-mask-dataset',
        'local_dir_name': 'jsrt-247-image-lung-segmentation-mask-dataset' # The folder name under DATA_ROOT for this
    }
}