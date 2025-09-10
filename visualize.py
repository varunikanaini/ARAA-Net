# /kaggle/working/ARAA-Net/visualize.py (FINAL 'lasa' BRANCH VERSION)
import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from models.daseg import daseg
from datasets import ImageFolder
from config import DATA_ROOT, CKPT_ROOT
from misc import check_mkdir

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis')
    parser.add_argument('--backbone', type=str, default='vgg16')
    parser.add_argument('--image-index', type=int, default=10)
    parser.add_argument('--scale-h', type=int, default=576)
    parser.add_argument('--scale-w', type=int, default=576)

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CHANGE 2: USE NEW DIRECTORY FORMAT ---
    exp_name = f"LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}_{args.backbone}"
    
    model_path = os.path.join(CKPT_ROOT, exp_name, 'best_checkpoint.pth')
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Checkpoint not found at {model_path}")
        return

    net = daseg(backbone_name=args.backbone).to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    net.eval()
    print(f"✅ Model loaded from {model_path}")

    # --- CHANGE 3: USE 'test' SPLIT FOR DATA ---
    test_data_path = os.path.join(DATA_ROOT, args.dataset_name, 'test')
    test_set = ImageFolder(test_data_path, args, split='test')

    if args.image_index >= len(test_set):
        print(f"❌ ERROR: Image index {args.image_index} is out of bounds.")
        return

    sample = test_set[args.image_index]
    image_tensor = sample['image'].unsqueeze(0).to(device)
    label_tensor = sample['label']
    image_name = os.path.basename(sample['name'][0])
    print(f"✅ Visualizing image: {image_name} (index {args.image_index})")

    with torch.no_grad():
        _, _, _, _, pred_logits = net(image_tensor)
    prediction_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    ground_truth_mask = label_tensor.numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np); axes[0].set_title('Original Image'); axes[0].axis('off')
    axes[1].imshow(ground_truth_mask, cmap='gray'); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(prediction_mask, cmap='gray'); axes[2].set_title('Prediction'); axes[2].axis('off')
    
    output_dir = os.path.join(CKPT_ROOT, 'visual_results', exp_name)
    check_mkdir(output_dir)
    save_path = os.path.join(output_dir, f"result_index_{args.image_index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()