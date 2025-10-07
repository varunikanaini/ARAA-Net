# config.py
import os

# --- Base Directories ---
# READ data from the original, read-only INPUT directory
DATA_ROOT = '/kaggle/working/ARAA-Net/data'
# WRITE checkpoints to the new, writable WORKING directory
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

# Ensure these directories exist
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)

# --- Dataset Configurations ---
# Define a dictionary where keys are dataset names and values describe how to load them.
# This is more flexible than just paths, as it can include structure details.
# For simplicity here, we'll include paths and structure hints.

# Dataset structures:
# - 'TSRS_RSNA': Assumes split folders (train, val, test) directly under the dataset path,
#                with images in `split_name` and labels in `split_name_labels`.
# - 'STANDARD': Assumes split folders (train, val, test) directly under the dataset path,
#               with images and labels in subfolders named 'images' and 'masks' (or 'GT').
#               Example: /dataset_root/split_name/images, /dataset_root/split_name/masks

DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
        'structure': 'TSRS_RSNA', # Special handling for TSRS
        'num_classes': 2, # Example: Binary segmentation
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png', '.tif', '.tiff', '.bmp'),
    },
    'TSRS_RSNA-Articular-Surface': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Articular-Surface'),
        'structure': 'TSRS_RSNA',
        'num_classes': 2,
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png', '.tif', '.tiff', '.bmp'),
    },
    'JSRT': {
        'path': os.path.join(DATA_ROOT, 'jsrt-247-image-lung-segmentation-mask-dataset'),
        'structure': 'STANDARD',
        'subfolders': { # For STANDARD structure, define subfolder names per split
            'train': {'images': 'cxr', 'masks': 'masks'},
            'val': {'images': 'cxr', 'masks': 'masks'},
            'test': {'images': 'cxr', 'masks': 'masks'},
        },
        'num_classes': 2, # Assuming binary for lung segmentation
        'image_ext': ('.png',),
        'mask_ext': ('.png',),
    },
    'COVID19_Radiography': {
        'path': os.path.join(DATA_ROOT, 'covid19-radiography-database'),
        'structure': 'COVID19', # Custom structure for COVID dataset
        'num_classes': 4, # COVID, Normal, Lung_Opacity, Viral Pneumonia
        'image_ext': ('.jpg',),
        'mask_ext': ('.png',),
    },
    'CVC-ClinicDB': {
        'path': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
        'structure': 'STANDARD',
        'subfolders': {
            'train': {'images': 'Original', 'masks': 'Ground Truth'}, # Assuming 'train' split exists within CVC-ClinicDB
            'val': {'images': 'Original', 'masks': 'Ground Truth'},   # You might need to adjust these based on actual split dirs
            'test': {'images': 'Original', 'masks': 'Ground Truth'},
        },
        'num_classes': 2, # Polyps
        'image_ext': ('.tif',),
        'mask_ext': ('.tif',),
    },
    'DentalPanoramic': {
        'path': os.path.join(DATA_ROOT, 'dental_panoramic_xrays'),
        'structure': 'STANDARD',
        'subfolders': {
            'train': {'images': 'images', 'masks': 'segmentation_1'},
            'val': {'images': 'images', 'masks': 'segmentation_1'},
            'test': {'images': 'images', 'masks': 'segmentation_1'},
        },
        'num_classes': 2, # Example: Tooth vs Background
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png',),
    },
    'SixDiseasesChestXRay': {
        'path': os.path.join(DATA_ROOT, 'Dataset'), # Assuming this is the root for the split folders
        'structure': 'SIX_DISEASES', # Custom structure for this dataset
        'num_classes': 6, # Covid, Normal, TB, Bacterial Pneumonia, Pneumothorax, Viral Pneumonia
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png',),
    },
}

# --- Backbone Definitions ---
# List of supported backbones and their typical channel outputs at different stages
# This helps in configuring the U-Net decoder layers.
BACKBONE_CHANNELS = {
    'vgg16': {'e1': 64, 'e2': 128, 'e3': 256, 'e4': 512, 'bottleneck': 512},
    'resnet50': {'e1': 64, 'e2': 256, 'e3': 512, 'e4': 1024, 'bottleneck': 2048},
    'inception_v3': {'e1': 64, 'e2': 256, 'e3': 768, 'e4': 1280, 'bottleneck': 2048}, # Approximate channels
    'efficientnet_b0': {'e1': 32, 'e2': 48, 'e3': 136, 'e4': 384, 'bottleneck': 1280}, # From torchvision output (may need verification)
    'efficientnet_b3': {'e1': 40, 'e2': 56, 'e3': 160, 'e4': 448, 'bottleneck': 1536}, # From torchvision output (may need verification)
}

# --- Default Training Parameters (can be overridden by args) ---
DEFAULT_ARGS = {
    # Dataset specific defaults
    'dataset_name': 'TSRS_RSNA-Epiphysis',
    'num_workers': 4,

    # Model specifics
    'backbone': 'vgg16',
    'lasa_kernels': [1, 3, 5, 7],

    # Training parameters
    'epochs': 100,
    'batch_size': 4,
    'lr': 0.0005,
    'weight_decay': 0.0001,
    'patience': 15,

    # Image preprocessing parameters
    'scale_h': 896,
    'scale_w': 576,

    # Deep Supervision weights
    'deep_supervision_weights': [0.2, 0.4, 0.6, 0.8, 1.0],

    # Loss function parameters
    'focal_alpha': 0.5,
    'focal_gamma': 2.0,
    'focal_loss_weight': 1.0,
    'dice_loss_weight': 1.0,

    # Data augmentation parameters
    'min_lesion_area_pixels': 576,
    'expansion_factor': 1.5,
    'min_bbox_h': 32,
    'min_bbox_w': 32,
    'wavelet_type': 'haar',
    'wavelet_level': 1,
    'wavelet_detail_scale': 1.5,

    # Scheduler parameters
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
}

# --- Helper function to get dataset config ---
def get_dataset_info(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]