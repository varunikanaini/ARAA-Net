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

    top_level_items = os.listdir(downloaded_path_input)
    if not top_level_items:
        logging.error(f"Downloaded path '{downloaded_path_input}' is empty.")
        return None
    
    source_dataset_root = None
    # Prioritize single directory if it exists, otherwise assume input_path is root
    if len(top_level_items) == 1 and os.path.isdir(os.path.join(downloaded_path_input, top_level_items[0])):
        source_dataset_root = os.path.join(downloaded_path_input, top_level_items[0])
    else:
        source_dataset_root = downloaded_path_input
        logging.warning(f"Multiple items or no single root folder found in downloaded path. Assuming '{downloaded_path_input}' is the source root to copy.")

    if not source_dataset_root:
        logging.error("Could not determine source dataset root from downloaded path.")
        return None

    # Determine the final destination path within DATA_ROOT based on KAGGLE_DATASET_MAPPING
    # This requires dataset_id to match a key in KAGGLE_DATASET_MAPPING
    final_destination_folder_name = None
    for ds_name, ds_info in KAGGLE_DATASET_MAPPING.items():
        if ds_info and ds_info.get('id') == dataset_id:
            final_destination_folder_name = ds_info['local_dir_name']
            break
    
    if not final_destination_folder_name:
        logging.error(f"Kaggle Dataset ID '{dataset_id}' not found in KAGGLE_DATASET_MAPPING. Exiting.")
        return None

    # --- NEW ADDITION START ---
    # Heuristic: If the determined source_dataset_root contains a single subdirectory
    # that matches the expected final_destination_folder_name,
    # then the actual content is likely nested one level deeper.
    if source_dataset_root:
        try:
            nested_items = [d for d in os.listdir(source_dataset_root) if os.path.isdir(os.path.join(source_dataset_root, d))]
            if len(nested_items) == 1 and nested_items[0] == final_destination_folder_name:
                logging.info(f"Detected nested dataset structure: '{os.path.basename(source_dataset_root)}/{nested_items[0]}'. Adjusting source root for copy.")
                source_dataset_root = os.path.join(source_dataset_root, nested_items[0])
        except FileNotFoundError:
            logging.warning(f"Source dataset root '{source_dataset_root}' not found during nested check. Skipping adjustment.")
        except Exception as e:
            logging.warning(f"Error during nested directory check in '{source_dataset_root}': {e}. Skipping adjustment.")
    # --- NEW ADDITION END ---

    final_destination_path = os.path.join(target_dir, final_destination_folder_name)

    if os.path.exists(final_destination_path) and os.listdir(final_destination_path):
        logging.info(f"Dataset already appears to be in '{final_destination_path}'. Skipping copy.")
        return final_destination_path
    elif os.path.exists(final_destination_path) and not os.listdir(final_destination_path):
        logging.warning(f"Target directory '{final_destination_path}' exists but is empty. Cleaning up before copy.")
        shutil.rmtree(final_destination_path) 
    
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
    'TSRS_RSNA-Epiphysis': {'id': None, 'local_dir_name': 'TSRS_RSNA-Epiphysis'}, 
    'TSRS_RSNA-Articular-Surface': {'id': None, 'local_dir_name': 'TSRS_RSNA-Articular-Surface'}, 
    'KOA': { 
        'id': None, 
        'local_dir_name': 'lvv-koa' 
    },
    'MURA': {
        'id': 'murrphyh/mura-v11', 
        'local_dir_name': 'MURA-v11' 
    },
    'COVID-19_Radiography': { 
        'id': 'tawsifurrahman/covid19-radiography-database',
        'local_dir_name': 'COVID-19_Radiography_Dataset' # Actual extracted folder name
    },
    'JSRT': { 
        'id': 'abduzzami/jsrt-247-image-lung-segmentation-mask-dataset',
        'local_dir_name': 'jsrt' # Common extracted folder name
    }
}