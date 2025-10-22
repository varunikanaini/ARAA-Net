import torch
import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from custom_transforms import ProportionalResizePad, Normalize, ToTensor
import config
from torchvision import transforms

def visualize_predictions(model, data_loader, device, num_images=10, output_dir='visualization_outputs'):
    model.eval()
    
    # --- ADDED: Create the output directory if it doesn't exist ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualization images to: {output_dir}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            if i >= num_images:
                break
            
            image = sample['image'].to(device)
            gt_mask = sample['label'].squeeze(0).cpu().numpy()
            
            # Get the original image name to use in the filename
            image_name = sample.get('name', [f'image_{i+1}'])[0]
            
            outputs = model(image)
            pred = torch.argmax(outputs[-1], dim=1).squeeze(0).cpu().numpy()
            
            image_to_show = image.squeeze(0).cpu() * std + mean
            image_to_show = np.clip(image_to_show.permute(1, 2, 0).numpy(), 0, 1)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(image_to_show)
            axes[0].set_title(f"Original: {image_name}")
            axes[0].axis('off')
            
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title("Prediction")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # --- MODIFIED: Save the figure instead of trying to show it ---
            save_path = os.path.join(output_dir, f"comparison_{image_name.replace('.jpg', '.png')}")
            plt.savefig(save_path)
            plt.close(fig) # Close the figure to free up memory

    print("Visualization complete.")

# --- (The main function remains exactly the same) ---
def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--num-images', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = f"{args.backbone}_FreezeTune_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    model_path = os.path.join(config.CKPT_ROOT, exp_name, 'best_checkpoint.pth')
    
    model = Light_LASA_Unet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    dataset_path = dataset_info['path']
    
    class DummyArgs:
        scale_h = 224
        scale_w = 224

    test_transforms = transforms.Compose([
        ProportionalResizePad(output_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])

    test_ds = ImageFolder(root=os.path.join(dataset_path, 'test'), dataset_name=args.dataset_name, args=DummyArgs(), split='test')
    test_ds.composed_transforms = test_transforms
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    visualize_predictions(model, test_loader, device, args.num_images)

if __name__ == '__main__':
    main()
# # /kaggle/working/ARAA-Net/visualize_results.py

# import torch
# import torchvision.transforms as transforms # <-- IMPORTANT: Ensure this line is present!
# from torchvision.transforms import functional as F
# import torchvision.transforms # <-- Also ensure this is imported if you use 'transforms.Compose' directly later
# import numpy as np
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import os
# import sys
# import cv2
# import os
# import sys

# # --- Setup Project Path ---
# project_path = '/kaggle/working/ARAA-Net'
# if project_path not in sys.path:
#     sys.path.insert(0, project_path)

# # --- Import your custom modules ---
# # Ensure these paths are correct relative to your project structure
# import custom_transforms as tr # <-- IMPORTANT: Ensure this line is present!
# from light_lasa_unet import Light_LASA_Unet
# from datasets import ImageFolder
# from seg_utils import ConfusionMatrix
# from misc import AvgMeter
# from torch.utils.data import DataLoader, Dataset
# # ... (rest of your visualize_results.py script) ...

# # --- Configuration ---
# # You might need to adjust these based on your setup
# EXPERIMENT_DIR = '/kaggle/working/ARAA-Net/ckpt/mobilenet_v2_FreezeTune_LASA_epiphysis' # Path to your experiment results
# CHECKPOINT_NAME = 'best_checkpoint.pth'
# TEST_DATA_PATH = '/kaggle/working/ARAA-Net/data/TSRS_RSNA-Epiphysis/test' # Path to your TEST dataset folder
# DATASET_NAME = 'TSRS_RSNA-Epiphysis' # The name of your dataset as used in datasets.py
# NUM_CLASSES = 2 # Adjust if you have more than 2 classes
# BATCH_SIZE = 1 # Set to 1 for individual image analysis
# NUM_WORKERS = 2 # Adjust based on your system

# # Define colors for visualization (RGB tuples)
# # Must match the number of classes.
# # Index 0 is background, Index 1 is class 1, etc.
# COLOR_MAP = np.array([
#     [0, 0, 0],      # Class 0 (Background) - Black
#     [255, 0, 0]     # Class 1 (Epiphysis) - Red
#     # Add more colors if you have more classes
#     # e.g., [0, 255, 0] for Class 2, [0, 0, 255] for Class 3, etc.
# ])

# # --- Helper function to convert mask to RGB with specific colors ---
# def apply_color_map(mask_np, color_map):
#     """Converts a class index mask to an RGB image using the provided color map."""
#     if mask_np.ndim != 2:
#         raise ValueError("Mask must be 2D (class indices).")
#     if color_map.shape[0] != np.max(mask_np) + 1:
#         print(f"Warning: Color map size ({color_map.shape[0]}) does not match max class index + 1 ({np.max(mask_np) + 1}). May lead to incorrect colors.")
        
#     height, width = mask_np.shape
#     colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for class_idx, color in enumerate(color_map):
#         # Ensure class_idx is within the bounds of the mask values
#         if class_idx < color_map.shape[0]:
#             colored_mask[mask_np == class_idx] = color
            
#     return colored_mask

# # --- Main Visualization Function ---
# def visualize_segmentation_results(model, test_loader, device, num_classes, color_map, device_name):
#     model.eval()
    
#     # Get default args for dataset loading (needed by ImageFolder)
#     # We only need scale_h and scale_w for the transforms
#     # For testing, FixedResize uses these. Ensure they match your training.
#     # If you don't have a config.py with DEFAULT_ARGS, you might need to hardcode these
#     # or pass them differently. For now, let's assume a basic setup.
    
#     # --- Attempt to load default args from config.py ---
#     try:
#         import config
#         # Make sure these match what your model expects during training
#         # If your model was trained with different scales, adjust here.
#         scale_h, scale_w = config.DEFAULT_ARGS.get('scale_h', 224), config.DEFAULT_ARGS.get('scale_w', 224)
#         # Fetch standard mean/std if available, otherwise use default for ImageFolder
#         img_mean = config.DEFAULT_ARGS.get('mean', [0.485, 0.456, 0.406])
#         img_std = config.DEFAULT_ARGS.get('std', [0.229, 0.224, 0.225])
        
#     except ImportError:
#         print("config.py not found. Using default image dimensions and normalization.")
#         scale_h, scale_w = 224, 224 # Fallback values
#         img_mean = [0.485, 0.456, 0.406]
#         img_std = [0.229, 0.224, 0.225]
#     except AttributeError:
#         print("config.DEFAULT_ARGS not found. Using default image dimensions and normalization.")
#         scale_h, scale_w = 224, 224 # Fallback values
#         img_mean = [0.485, 0.456, 0.406]
#         img_std = [0.229, 0.224, 0.225]

#     print(f"Using image scale: {scale_h}x{scale_w}")
    
#     # --- Re-initialize ImageFolder with correct transforms for testing ---
#     # We only need FixedResize, Normalize, ToTensor for evaluation
#     # IMPORTANT: Ensure the transforms used here match the final stage of your training transforms
#     test_transforms = transforms.Compose([
#         tr.FixedResize(w=scale_w, h=scale_h),
#         tr.Normalize(mean=img_mean, std=img_std),
#         tr.ToTensor()
#     ])
    
#     # Create a dummy args object if needed, or pass specific values
#     # We need args.scale_h and args.scale_w for the ImageFolder init if it uses them directly
#     class DummyArgs:
#         def __init__(self, scale_h, scale_w, min_lesion_area_pixels=0, expansion_factor=1.0, min_bbox_h=32, min_bbox_w=32):
#             self.scale_h = scale_h
#             self.scale_w = scale_w
#             self.min_lesion_area_pixels = min_lesion_area_pixels # For CenterAmplification, set to 0 for testing
#             self.expansion_factor = expansion_factor
#             self.min_bbox_h = min_bbox_h
#             self.min_bbox_w = min_bbox_w
#             # Add other attributes if ImageFolder requires them
#             self.mean = img_mean
#             self.std = img_std
            
#     dummy_args = DummyArgs(scale_h=scale_h, scale_w=scale_w)

#     # Re-create test_dataset with the specific transforms if your ImageFolder constructor is complex
#     # Alternatively, if ImageFolder uses a split argument, pass 'test'
#     test_dataset = ImageFolder(root=TEST_DATA_PATH, dataset_name=DATASET_NAME, args=dummy_args, split='test')
    
#     # If the loader passed to this function already uses the correct test transforms,
#     # you can skip re-initializing test_dataset and test_loader.
#     # For this example, we assume we are creating them here.
#     # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn) # If you have custom_collate_fn

#     print(f"Starting visualization on {len(test_dataset)} images from {TEST_DATA_PATH}")

#     # Use tqdm for progress bar
#     from tqdm import tqdm

#     # To store results for later analysis if needed
#     misclassified_samples = []

#     confmat = ConfusionMatrix(num_classes) if 'ConfusionMatrix' in sys.modules else None

#     with torch.no_grad():
#         for i, sample in enumerate(tqdm(test_loader, desc="Visualizing Results")):
#             if sample is None:
#                 print(f"Skipping None sample at index {i}.")
#                 continue
                
#             images = sample['image'].to(device)
#             labels = sample['label'].to(device) # Ground truth mask indices

#             # Get model outputs (this returns a tuple of aux outputs and final output)
#             outputs = model(images)
            
#             # The last output in the tuple is the prediction from the final layer
#             # Deep supervision outputs are earlier in the tuple
#             predicted_mask_logits = outputs[-1] 
#             predicted_mask_indices = torch.argmax(predicted_mask_logits, dim=1).squeeze(0).cpu().numpy() # Shape: (H, W)

#             # Convert PIL/Tensor ground truth mask to numpy indices
#             # Ensure labels are in the correct format (e.g., LongTensor of class indices)
#             ground_truth_indices = labels.squeeze(0).cpu().numpy() # Shape: (H, W)

#             # Original image tensor to numpy for visualization
#             # Assuming input is CHW, convert to HWC and then to PIL Image if needed
#             original_image_pil = F.to_pil_image(images.squeeze(0).cpu())

#             # --- Generate Difference Mask ---
#             # This mask highlights areas where prediction differs from ground truth
#             difference_mask = np.zeros_like(predicted_mask_indices, dtype=np.uint8)
#             misclassified_pixels = (predicted_mask_indices != ground_truth_indices)
#             difference_mask[misclassified_pixels] = 1 # Mark misclassified pixels as class 1 for visualization

#             # --- Apply color maps ---
#             gt_colored = apply_color_map(ground_truth_indices, color_map)
#             pred_colored = apply_color_map(predicted_mask_indices, color_map)
#             # Highlight differences with a bright color (e.g., Green)
#             diff_colored = np.zeros_like(gt_colored, dtype=np.uint8)
#             diff_colored[difference_mask == 1] = [0, 255, 0] # Green for differences

#             # --- Calculate Confusion Matrix Update (optional) ---
#             if confmat:
#                 confmat.update(ground_truth_indices.flatten(), predicted_mask_indices.flatten())

#             # --- Display results using Matplotlib ---
#             fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # 4 columns: Img, GT, Pred, Diff

#             # Original Image
#             axes[0].imshow(original_image_pil)
#             axes[0].set_title(f"Original {sample.get('name', f'#{i}')}")
#             axes[0].axis('off')

#             # Ground Truth Mask
#             axes[1].imshow(gt_colored)
#             axes[1].set_title("Ground Truth")
#             axes[1].axis('off')

#             # Predicted Mask
#             axes[2].imshow(pred_colored)
#             axes[2].set_title("Prediction")
#             axes[2].axis('off')

#             # Difference Mask
#             axes[3].imshow(diff_colored)
#             axes[3].set_title("Misclassified Pixels (Green)")
#             axes[3].axis('off')
            
#             plt.tight_layout()
#             # Save the figure
#             output_dir = os.path.join(EXPERIMENT_DIR, 'visualization_outputs')
#             os.makedirs(output_dir, exist_ok=True)
#             plt.savefig(os.path.join(output_dir, f'sample_{i:04d}_viz.png'), dpi=300)
#             # plt.show() # Uncomment if you want to display plots interactively
#             plt.close(fig) # Close the figure to free memory

#             # Store misclassified samples if needed for further analysis
#             if np.any(misclassified_pixels):
#                 misclassified_samples.append({
#                     'index': i,
#                     'filename': sample.get('name', f'sample_{i:04d}'),
#                     'original_image': original_image_pil,
#                     'ground_truth': ground_truth_indices,
#                     'prediction': predicted_mask_indices,
#                     'difference_mask': difference_mask # Binary mask of misclassified pixels
#                 })
                
#     print("\n--- Visualization Complete ---")
#     print(f"Generated {i+1} visualizations. Saved to: {output_dir}")
#     if confmat:
#         acc, _, iou, fwiou, dice = confmat.compute()
#         print(f"Overall Accuracy on visualized set: {acc.item():.4f}")
#         print(f"Mean IoU on visualized set: {iou.mean().item():.4f}")
#         print(f"Confusion Matrix:\n{confmat.mat}")

#     return misclassified_samples

# # --- Main Execution Block ---
# if __name__ == '__main__':
#     # --- Load Model ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Instantiate the model
#     # IMPORTANT: Ensure num_classes matches your dataset and model
#     model = Light_LASA_Unet(num_classes=NUM_CLASSES, lasa_kernels=[1, 3, 5, 7]).to(device)
    
#     # Load the checkpoint
#     checkpoint_path = os.path.join(EXPERIMENT_DIR, CHECKPOINT_NAME)
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Checkpoint not found at {checkpoint_path}")
#         sys.exit(1)

#     try:
#         # Load state dict, ensuring it matches the model architecture
#         state_dict = torch.load(checkpoint_path, map_location=device)
        
#         # If the checkpoint contains more than just the model state dict (e.g., optimizer, epoch)
#         if 'model_state_dict' in state_dict:
#             model.load_state_dict(state_dict['model_state_dict'])
#             print(f"Loaded model state dict from '{checkpoint_path}' (found 'model_state_dict' key).")
#         else:
#             model.load_state_dict(state_dict)
#             print(f"Loaded model state dict directly from '{checkpoint_path}'.")

#     except Exception as e:
#         print(f"Error loading checkpoint {checkpoint_path}: {e}")
#         print("Please ensure the checkpoint file is valid and matches the model architecture.")
#         sys.exit(1)

#     # --- Load Dataset and DataLoader ---
#     # Use the same dataset configuration as your training/testing
#     # IMPORTANT: Adjust TEST_DATA_PATH, DATASET_NAME, and NUM_CLASSES accordingly.
#     # The `args` object passed to ImageFolder is simplified for visualization.
#     try:
#         import config
#         # Fetch default args from config if available
#         scale_h, scale_w = config.DEFAULT_ARGS.get('scale_h', 224), config.DEFAULT_ARGS.get('scale_w', 224)
#         img_mean = config.DEFAULT_ARGS.get('mean', [0.485, 0.456, 0.406])
#         img_std = config.DEFAULT_ARGS.get('std', [0.229, 0.224, 0.225])
#     except (ImportError, AttributeError):
#         scale_h, scale_w = 224, 224
#         img_mean = [0.485, 0.456, 0.406]
#         img_std = [0.229, 0.224, 0.225]
#         print("config.py or DEFAULT_ARGS not found. Using fallback values for image scale and normalization.")

#     class DummyArgs:
#         def __init__(self, scale_h, scale_w, mean, std):
#             self.scale_h = scale_h
#             self.scale_w = scale_w
#             self.mean = mean
#             self.std = std
#             # Add any other attributes that ImageFolder might need (e.g., for transforms)
#             self.min_lesion_area_pixels = 0 # Disable for visualization
#             self.expansion_factor = 1.0
#             self.min_bbox_h = 32
#             self.min_bbox_w = 32

#     dummy_args = DummyArgs(scale_h=scale_h, scale_w=scale_w, mean=img_mean, std=img_std)
    
#     # Ensure your custom_transforms.py is in the sys.path
#     # If it's not in the root, you might need to adjust sys.path.insert()
#     try:
#         from custom_transforms import FixedResize, Normalize, ToTensor
#     except ImportError:
#         print("Error: Could not import custom transforms. Make sure custom_transforms.py is accessible.")
#         sys.exit(1)

#     # Re-create the test dataset and loader using the specified path and dataset name
#     # Ensure your ImageFolder handles the 'test' split correctly.
#     try:
#         test_dataset = ImageFolder(root=TEST_DATA_PATH, dataset_name=DATASET_NAME, args=dummy_args, split='test')
#         if not test_dataset or len(test_dataset) == 0:
#              raise ValueError("Test dataset is empty. Check TEST_DATA_PATH and DATASET_NAME.")
             
#         # Use custom_collate_fn if your dataset returns None for invalid samples
#         # For this script, we'll handle None within the visualization loop directly
#         def custom_collate_fn(batch):
#             batch = [item for item in batch if item is not None]
#             return torch.utils.data.dataloader.default_collate(batch) if batch else None

#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=BATCH_SIZE,
#             shuffle=False, # Important for consistent visualization order
#             num_workers=NUM_WORKERS,
#             collate_fn=custom_collate_fn # Use if your dataset can return None
#         )
#     except Exception as e:
#         print(f"Error setting up DataLoader: {e}")
#         print("Please check TEST_DATA_PATH, DATASET_NAME, and ensure your ImageFolder class is correctly configured.")
#         sys.exit(1)

#     # --- Run Visualization ---
#     misclassified_info = visualize_segmentation_results(
#         model,
#         test_loader,
#         device,
#         num_classes=NUM_CLASSES,
#         color_map=COLOR_MAP,
#         device_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
#     )

#     # You can further analyze `misclassified_info` here if needed
#     # For example, count how many samples had misclassifications, or analyze them by filename.
#     print(f"\nAnalysis of misclassified samples (if any):")
#     if misclassified_info:
#         print(f"Found {len(misclassified_info)} samples with misclassifications out of {len(test_loader)} total.")
#         # You can further inspect 'misclassified_info' here
#         # For example:
#         # for item in misclassified_info:
#         #     print(f"- Filename: {item['filename']}, Number of misclassified pixels: {np.sum(item['difference_mask'])}")
#     else:
#         print("No misclassifications found in the visualized subset (or all pixels were correctly classified).")