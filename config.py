# config.py (Modified for better backbone resolution handling)
import os

# --- Base Directories ---
DATA_ROOT = '/kaggle/working/ARAA-Net/data'
CKPT_ROOT = '/kaggle/working/ARAA-Net/ckpt'

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)

# --- Dataset Configurations ---
# ... (Your existing DATASET_CONFIG remains the same) ...
DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
        'structure': 'TSRS_RSNA',
        'num_classes': 2,
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
        'subfolders': {
            'train': {'images': 'cxr', 'masks': 'masks'},
            'val': {'images': 'cxr', 'masks': 'masks'},
            'test': {'images': 'cxr', 'masks': 'masks'},
        },
        'num_classes': 2,
        'image_ext': ('.png',),
        'mask_ext': ('.png',),
    },
    'COVID19_Radiography': {
        'path': os.path.join(DATA_ROOT, 'covid19-radiography-database'),
        'structure': 'COVID19',
        'num_classes': 4,
        'image_ext': ('.jpg',),
        'mask_ext': ('.png',),
    },
    'CVC-ClinicDB': {
        'path': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
        'structure': 'STANDARD',
        'subfolders': {
            'train': {'images': 'Original', 'masks': 'Ground Truth'},
            'val': {'images': 'Original', 'masks': 'Ground Truth'},
            'test': {'images': 'Original', 'masks': 'Ground Truth'},
        },
        'num_classes': 2,
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
        'num_classes': 2,
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png',),
    },
    'SixDiseasesChestXRay': {
        'path': os.path.join(DATA_ROOT, 'Dataset'),
        'structure': 'SIX_DISEASES',
        'num_classes': 6,
        'image_ext': ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'),
        'mask_ext': ('.png',),
    },
}

# --- Backbone Definitions ---
# Define typical input resolutions for each backbone
BACKBONE_INPUT_RESOLUTIONS = {
    'vgg16': (224, 224),
    'resnet50': (224, 224),
    'inception_v3': (299, 299),
    'efficientnet_b0': (224, 224),
    'efficientnet_b3': (300, 300), # Common size for B3
}

# Define channel info (keep this)
BACKBONE_CHANNELS = {
    'vgg16': {'e1': 64, 'e2': 128, 'e3': 256, 'e4': 512, 'bottleneck': 512},
    'resnet50': {'e1': 64, 'e2': 256, 'e3': 512, 'e4': 1024, 'bottleneck': 2048},
    'inception_v3': {'e1': 64, 'e2': 256, 'e3': 768, 'e4': 1280, 'bottleneck': 2048},
    'efficientnet_b0': {'e1': 32, 'e2': 48, 'e3': 136, 'e4': 384, 'bottleneck': 1280},
    'efficientnet_b3': {'e1': 40, 'e2': 56, 'e3': 160, 'e4': 448, 'bottleneck': 1536},
}

# --- Default Training Parameters ---
DEFAULT_ARGS = {
    # Dataset specific defaults
    'dataset_name': 'TSRS_RSNA-Epiphysis',
    'num_workers': 4,

    # Model specifics
    'backbone': 'vgg16',
    'lasa_kernels': [1, 3, 5, 7],

    # Training parameters
    'epochs': 50,
    'batch_size': 4,
    'lr': 0.0005,
    'weight_decay': 0.0001,
    'patience': 15,

    # Image preprocessing parameters - MAKE THESE BACKBONE-DEPENDENT
    # We'll handle this lookup in get_args() now
    # 'scale_h': 896, # OLD GLOBAL DEFAULT
    # 'scale_w': 576, # OLD GLOBAL DEFAULT

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
    
    # LR Scheduler type
    'lr_scheduler_type': 'ReduceLROnPlateau', # Default scheduler
    'scheduler_T0': 10, # For CosineAnnealingWarmRestarts
    'scheduler_T_mult': 2, # For CosineAnnealingWarmRestarts
}

# --- Helper function to get dataset config ---
def get_dataset_info(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]

# --- Helper function to get backbone input resolution ---
def get_backbone_resolution(backbone_name):
    if backbone_name not in BACKBONE_INPUT_RESOLUTIONS:
        print(f"Warning: Resolution for backbone '{backbone_name}' not found in BACKBONE_INPUT_RESOLUTIONS. Using default 224x224.")
        return (224, 224) # Fallback resolution
    return BACKBONE_INPUT_RESOLUTIONS[backbone_name]