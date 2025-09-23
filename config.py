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
    
    logging.info(f"Attempting to download KaggleHub dataset '{dataset_id}' to a temporary location...")
    try:
        downloaded_path_input = kagglehub.dataset_download(dataset_id)
        logging.info(f"Downloaded temporarily to: {downloaded_path_input}")
    except Exception as e:
        logging.error(f"Error downloading KaggleHub dataset '{dataset_id}': {e}")
        return None

    # Determine the actual top-level folder name inside the downloaded content
    # For "tawsifurrahman/covid19-radiography-database", it's usually "COVID-19_Radiography_Dataset"
    # For "abduzzami/jsrt-247-image-lung-segmentation-mask-dataset", it's "jsrt" or "jsrt-247..."
    
    # Let's try to be more robust by looking for the common base folder
    top_level_items = os.listdir(downloaded_path_input)
    if not top_level_items:
        logging.error(f"Downloaded path '{downloaded_path_input}' is empty.")
        return None
    
    # Try to find a folder that likely contains the actual dataset content
    # Often, KaggleHub downloads to /kaggle/input/dataset_id/version/DATA_FOLDER/
    source_dataset_root = None
    if len(top_level_items) == 1 and os.path.isdir(os.path.join(downloaded_path_input, top_level_items[0])):
        source_dataset_root = os.path.join(downloaded_path_input, top_level_items[0])
    else:
        # If multiple items or files, assume the downloaded_path_input itself is the root to copy
        source_dataset_root = downloaded_path_input
        logging.warning(f"Multiple items or no single root folder found in downloaded path. Assuming '{downloaded_path_input}' is the source root.")

    if not source_dataset_root:
        logging.error("Could not determine source dataset root from downloaded path.")
        return None

    # Use the local_dir_name from KAGGLE_DATASET_MAPPING for the final destination folder name
    # This ensures consistency with DATASET_CONFIGS in datasets.py
    final_destination_folder_name = [v['local_dir_name'] for k, v in KAGGLE_DATASET_MAPPING.items() if v and v.get('id') == dataset_id][0]
    final_destination_path = os.path.join(target_dir, final_destination_folder_name)

    if os.path.exists(final_destination_path) and os.listdir(final_destination_path):
        logging.info(f"Dataset already appears to be in '{final_destination_path}'. Skipping copy.")
        return final_destination_path
    elif os.path.exists(final_destination_path) and not os.listdir(final_destination_path):
        logging.warning(f"Target directory '{final_destination_path}' exists but is empty. Cleaning up before copy.")
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
    'TSRS_RSNA-Epiphysis': {'id': None, 'local_dir_name': 'TSRS_RSNA-Epiphysis'}, # Not KaggleHub ID, use default path
    'TSRS_RSNA-Articular-Surface': {'id': None, 'local_dir_name': 'TSRS_RSNA-Articular-Surface'}, # Not KaggleHub ID, use default path
    'KOA': { 
        'id': None, 
        'local_dir_name': 'lvv-koa' # The folder name under DATA_ROOT, per your structure
    },
    'MURA': {
        'id': 'murrphyh/mura-v11', # Example MURA dataset ID, replace if yours is different
        'local_dir_name': 'MURA-v11' # The folder name under DATA_ROOT for MURA
    },
    'COVID-19_Radiography': { # <<< MODIFIED: 'local_dir_name' to match extracted folder name
        'id': 'tawsifurrahman/covid19-radiography-database',
        'local_dir_name': 'COVID-19_Radiography_Dataset' # Actual folder name after extraction
    },
    'JSRT': { 
        'id': 'abduzzami/jsrt-247-image-lung-segmentation-mask-dataset',
        'local_dir_name': 'jsrt' # Often extracts to 'jsrt' subfolder
    }
}