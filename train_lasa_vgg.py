# /kaggle/working/ARAA-Net/train_enhanced_vgg.py
import os
import sys
import logging
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import the NEW Enhanced Model ---
from lasa_vgg_model import Enhanced_LASA_VGG_UNet
from config import CKPT_ROOT, DATASET_PATHS
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

# --- LOSS FUNCTION CLASSES (Unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.reduction, self.ignore_index = alpha, gamma, reduction, ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            mask = (targets != self.ignore_index).float()
            return (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth, self.ignore_index = smooth, ignore_index

    def forward(self, inputs, targets):
        pred_probs = F.softmax(inputs, dim=1)[:, 1]
        true_flat = (targets == 1).float().view(-1)
        pred_flat = pred_probs.view(-1)
        mask = (targets.view(-1) != self.ignore_index).float()
        intersection = (pred_flat * true_flat * mask).sum()
        union = (pred_flat * mask).sum() + (true_flat * mask).sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)

# --- SCRIPT ARGUMENTS ---
def get_args():
    parser = argparse.ArgumentParser(description='Train Enhanced LASA-VGG-UNet')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Corrected weights for 5 outputs (4 auxiliary + 1 final)
    parser.add_argument('--deep-supervision-weights', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--focal-loss-weight', type=float, default=1.0)
    parser.add_argument('--dice-loss-weight', type=float, default=1.2) # Increased emphasis on Dice
    parser.add_argument('--patience', type=int, default=20)
    # (Other dummy args for ImageFolder remain the same)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--scheduler-patience', type=int, default=7)
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576)
    parser.add_argument('--expansion-factor', type=float, default=1.5)
    parser.add_argument('--min-bbox-h', type=int, default=32)
    parser.add_argument('--min-bbox-w', type=int, default=32)
    parser.add_argument('--wavelet-type', type=str, default='haar')
    parser.add_argument('--wavelet-level', type=int, default=1)
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5)
    
    try: args = parser.parse_args()
    except SystemExit: args = parser.parse_args([])
    if len(args.deep_supervision_weights) != 5:
        parser.error("deep-supervision-weights must have 5 values for the 5 outputs.")
    return args

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file, logging.StreamHandler(sys.stdout))])

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def evaluate_model(net, loader, device, loss_fns, loss_weights, mode="Validating"):
    net.eval()
    confmat = ConfusionMatrix(num_classes=2)
    val_loss = AvgMeter()
    focal_loss, dice_loss = loss_fns
    f_w, d_w, ds_w = loss_weights
    with torch.no_grad():
        for data in tqdm(loader, desc=mode, leave=False):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(inputs)
            total_loss = sum(ds_w[i] * (f_w * focal_loss(out, labels.long()) + d_w * dice_loss(out, labels.long())) for i, out in enumerate(outputs))
            val_loss.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), outputs[-1].argmax(1).flatten())
    _, _, class_iou, _, _ = confmat.compute()
    mIoU = class_iou.mean().item()
    logging.info(f"--- {mode} mIoU: {mIoU:.4f} | Loss: {val_loss.avg:.4f} ---")
    if mode == "Validating": net.train()
    return mIoU

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = f"ENHANCED_VGG16_LASA_UNet_{args.dataset_name}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path)
    setup_logging(exp_path)
    logging.info(f"Starting experiment: {exp_name}\nArgs: {args}")

    base_path = DATASET_PATHS[args.dataset_name]
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    train_set = ImageFolder(train_path, args.dataset_name, args, split='train')
    val_set = ImageFolder(val_path, args.dataset_name, args, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate_fn)

    net = Enhanced_LASA_VGG_UNet(num_classes=2).to(device)
    loss_fns = (FocalLoss().to(device), DiceLoss().to(device))
    loss_weights = (args.focal_loss_weight, args.dice_loss_weight, args.deep_supervision_weights)
    
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)

    best_mIoU, patience = 0.0, 0
    for epoch in range(args.epochs):
        net.train()
        epoch_loss = AvgMeter()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data in progress_bar:
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            total_loss = sum(loss_weights[2][i] * (loss_weights[0] * loss_fns[0](out, labels.long()) + loss_weights[1] * loss_fns[1](out, labels.long())) for i, out in enumerate(outputs))
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss.update(total_loss.item(), inputs.size(0))
            progress_bar.set_postfix(loss=epoch_loss.avg, lr=optimizer.param_groups[0]['lr'])
        
        current_mIoU = evaluate_model(net, val_loader, device, loss_fns, loss_weights)
        scheduler.step(current_mIoU)

        if current_mIoU > best_mIoU:
            best_mIoU, patience = current_mIoU, 0
            torch.save(net.state_dict(), os.path.join(exp_path, 'best_checkpoint.pth'))
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Model saved.")
        else:
            patience += 1
            logging.info(f"⚠️ No improvement for {patience} epochs. Best mIoU: {best_mIoU:.4f}.")
        
        if patience >= args.patience:
            logging.info("--- Early stopping triggered ---")
            break

if __name__ == '__main__':
    main()```

---

### 3. The Corresponding Test Script: `test_enhanced_vgg.py`

To evaluate your newly trained model, use this corresponding test script.

```python
# /kaggle/working/ARAA-Net/test_enhanced_vgg.py
import os
import sys
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path: sys.path.insert(0, project_path)

from enhanced_lasa_vgg_unet import Enhanced_LASA_VGG_UNet
from config import CKPT_ROOT, DATASET_PATHS
from datasets import ImageFolder
from seg_utils import ConfusionMatrix
from misc import check_mkdir
from train_enhanced_vgg import FocalLoss, DiceLoss, custom_collate_fn # Reuse from train script

def get_test_args():
    parser = argparse.ArgumentParser(description='Test Enhanced LASA-VGG-UNet')
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', help='Dataset to test on')
    # (Dummy args for ImageFolder)
    parser.add_argument('--scale-h', type=int, default=896)
    parser.add_argument('--scale-w', type=int, default=576)
    parser.add_argument('--min-lesion-area-pixels', type=int, default=576)
    parser.add_argument('--expansion-factor', type=float, default=1.5)
    parser.add_argument('--min-bbox-h', type=int, default=32)
    parser.add_argument('--min-bbox-w', type=int, default=32)
    parser.add_argument('--wavelet-type', type=str, default='haar')
    parser.add_argument('--wavelet-level', type=int, default=1)
    parser.add_argument('--wavelet-detail-scale', type=float, default=1.5)
    try: return parser.parse_args()
    except SystemExit: return parser.parse_args([])

def main():
    args = get_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"ENHANCED_VGG16_LASA_UNet_{args.dataset_name}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    
    log_file = os.path.join(exp_path, 'final_test_results.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"--- Starting Final Test for {exp_name} ---")

    # Load Model
    checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    net = Enhanced_LASA_VGG_UNet(num_classes=2).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    logging.info("Model loaded from best checkpoint.")

    # Load Data
    base_path = DATASET_PATHS[args.dataset_name]
    test_path = os.path.join(base_path, 'val') # Use validation set for testing as per original setup
    test_set = ImageFolder(test_path, args.dataset_name, args, split='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False, collate_fn=custom_collate_fn)

    # Evaluation
    confmat = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(inputs)
            confmat.update(labels.flatten(), outputs[-1].argmax(1).flatten())

    # Log Results
    global_acc, class_acc, class_iou, fwiou, mDice = confmat.compute()
    mIoU = class_iou.mean().item()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"\n--- Final Test Results ({timestamp}) ---\n"
                 f"Global Accuracy = {global_acc.item():.4f}\n"
                 f"Mean IoU (mIoU) = {mIoU:.4f}\n"
                 f"Mean Dice       = {mDice:.4f}\n"
                 f"FWIoU           = {fwiou.item():.4f}\n"
                 f"Class IoU       = {class_iou.cpu().numpy()}\n")

if __name__ == '__main__':
    main()```

### How to Run the Enhanced VGG16 Version

1.  **Save the new files:**
    *   `enhanced_lasa_vgg_unet.py`
    *   `train_enhanced_vgg.py`
    *   `test_enhanced_vgg.py`

2.  **Execute the new training script from your terminal or notebook:**

    ```bash
    python /kaggle/working/ARAA-Net/train_enhanced_vgg.py \
        --dataset-name 'TSRS_RSNA-Epiphysis' \
        --epochs 100 \
        --batch-size 6 \
        --lr 0.001 
    ```

3.  **After training, run the test script to get final performance metrics:**

    ```bash
    python /kaggle/working/ARAA-Net/test_enhanced_vgg.py --dataset-name 'TSRS_RSNA-Epiphysis'
    ```

This revised approach provides a clear path to improving your model's accuracy and mIoU while staying with the VGG16 backbone, and the architectural changes are designed to be as efficient as possible.