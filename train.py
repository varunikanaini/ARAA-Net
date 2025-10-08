import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from datasets import ImageFolder
import config
from lasa_unet_model import LASAUNet
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Lovasz-Softmax implementation
def lovasz_softmax(probs, labels, classes='present', ignore_index=None):
    if probs.dim() != 4 or labels.dim() != 3:
        raise ValueError("Probs must be [B, C, H, W], labels must be [B, H, W]")
    
    batch_size, num_classes, h, w = probs.size()
    losses = []
    
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        target_c = (labels == c).float()
        prob_c = probs[:, c, :, :]
        intersection = (prob_c * target_c).sum(dim=(1, 2))
        union = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection
        jaccard_loss = 1 - (intersection + 1e-10) / (union + 1e-10)
        losses.append(jaccard_loss.mean())
    
    return torch.stack(losses).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if inputs.shape[2:] != targets.shape[1:]:
            raise ValueError(f"Input shape {inputs.shape} and target shape {targets.shape} spatial dimensions do not match")
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        if inputs.shape[2:] != targets.shape[1:]:
            raise ValueError(f"Input shape {inputs.shape} and target shape {targets.shape} spatial dimensions do not match")
        inputs = torch.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

def evaluate_model(model, data_loader, device, num_classes, logger):
    model.eval()
    total_loss = 0.0
    iou = torch.zeros((num_classes, num_classes), device=device)
    total_samples = 0
    
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice_loss_fn = DiceLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None or batch['image'].size(0) == 0:
                continue
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]  # Use final output for evaluation
            
            try:
                focal_loss = focal_loss_fn(outputs, targets) * args.focal_loss_weight
                dice_loss = dice_loss_fn(outputs, targets) * args.dice_loss_weight
                lovasz_loss = lovasz_softmax(torch.softmax(outputs, dim=1), targets) * 0.5
                loss = focal_loss + dice_loss + lovasz_loss
            except ValueError as e:
                logger.error(f"Loss computation failed: {e}. Skipping batch.")
                continue
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            for c in range(num_classes):
                pred_c = (preds == c).float()
                target_c = (targets == c).float()
                intersection = (pred_c * target_c).sum().item()
                union = pred_c.sum().item() + target_c.sum().item() - intersection
                iou[c, c] += intersection
                iou[c, :] += pred_c.sum().item()
                iou[:, c] += target_c.sum().item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    miou = (iou.diag() / (iou.sum(0) + iou.sum(1) - iou.diag() + 1e-10)).mean().item()
    fwiou = ((iou.diag() / (iou.sum(0) + iou.sum(1) - iou.diag() + 1e-10)) * (iou.sum(1) / iou.sum())).sum().item()
    dice = (2 * iou.diag() / (iou.sum(0) + iou.sum(1) + 1e-10)).mean().item()
    oa = (iou.diag().sum() / iou.sum()).item()
    
    # Per-class IoU and Dice
    class_metrics = []
    for i in range(num_classes):
        class_iou = iou[i, i] / (iou[i, :].sum() + iou[:, i].sum() - iou[i, i] + 1e-10)
        class_dice = (2 * class_iou) / (1 + class_iou)
        class_metrics.append((class_iou.item(), class_dice.item()))
        logger.info(f"Class {i} IoU: {class_iou:.4f}, Dice: {class_dice:.4f}")
    
    return avg_loss, oa, miou, fwiou, dice, class_metrics

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger()
    
    class Args:
        dataset_name = 'TSRS_RSNA-Epiphysis'
        backbone = 'vgg16'
        epochs = 100
        batch_size = 4
        lr = 0.0005
        weight_decay = 0.0005
        patience = 0
        scale_h = 224
        scale_w = 224
        lasa_kernels = [1, 3, 5, 7]
        deep_supervision_weights = [0.2, 0.4, 0.6, 0.8, 1.0]
        focal_alpha = 0.5
        focal_gamma = 2.0
        focal_loss_weight = 0.5
        dice_loss_weight = 1.5
        min_lesion_area_pixels = 400
        expansion_factor = 1.5
        min_bbox_h = 32
        min_bbox_w = 32
        wavelet_type = 'haar'
        wavelet_level = 1
        wavelet_detail_scale = 1.5
        scheduler_type = 'CosineAnnealingWarmRestarts'
        scheduler_patience = 5
        scheduler_factor = 0.5
        scheduler_min_lr = 1e-06
        scheduler_T0 = 10
        scheduler_T_mult = 2
        test_only = False
        resume = False
        fine_tune_epochs = 20
        num_workers = 2
        dataset_path = '/kaggle/working/ARAA-Net/data/TSRS_RSNA-Epiphysis'
        dataset_structure = 'TSRS_RSNA'
        num_classes = 2
    
    global args
    args = Args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = LASAUNet(backbone=args.backbone, num_classes=args.num_classes, lasa_kernels=args.lasa_kernels).to(device)
    
    if args.fine_tune_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Backbone '{args.backbone}' frozen for Phase 1 training.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=args.scheduler_T_mult, eta_min=args.scheduler_min_lr)
    
    train_set = ImageFolder(args.dataset_path, args.dataset_name, args, split='train')
    val_set = ImageFolder(args.dataset_path, args.dataset_name, args, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice_loss_fn = DiceLoss()
    
    best_miou = 0.0
    start_epoch = 0
    
    checkpoint_dir = f"/kaggle/working/ARAA-Net/ckpt/vgg16_LASA1_3_5_7_DSW0.2_0.4_0.6_0.8_1.0_FLW0.5_DLW1.5_epiphysis"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    
    if args.resume and os.path.exists(best_checkpoint_path):
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_miou = checkpoint.get('best_miou', 0.0)
            logger.info(f"Resumed training from epoch {start_epoch}, best mIoU: {best_miou}")
        except Exception as e:
            logger.error(f"Could not load checkpoint: {e}. Starting from scratch.")
    
    accum_steps = 2  # Gradient accumulation for effective batch size of 8
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if epoch == args.fine_tune_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
            logger.info(f"Backbone '{args.backbone}' unfrozen for Phase 2 training.")
        
        running_loss = 0.0
        total_samples = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            if batch is None or batch['image'].size(0) == 0:
                continue
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(images)
            if isinstance(outputs, list):
                if len(outputs) != len(args.deep_supervision_weights):
                    logger.warning(f"Mismatch: {len(outputs)} outputs, {len(args.deep_supervision_weights)} weights. Using last output.")
                    outputs = [outputs[-1]]
                    try:
                        logger.debug(f"Output shape: {outputs[0].shape}, Target shape: {targets.shape}")
                        focal_loss = focal_loss_fn(outputs[0], targets) * args.focal_loss_weight
                        dice_loss = dice_loss_fn(outputs[0], targets) * args.dice_loss_weight
                        lovasz_loss = lovasz_softmax(torch.softmax(outputs[0], dim=1), targets) * 0.5
                        loss = focal_loss + dice_loss + lovasz_loss
                    except ValueError as e:
                        logger.error(f"Loss computation failed: {e}. Skipping batch.")
                        continue
                else:
                    loss = 0
                    for j, out in enumerate(outputs):
                        try:
                            logger.debug(f"Output {j} shape: {out.shape}, Target shape: {targets.shape}")
                            focal_loss = focal_loss_fn(out, targets) * args.focal_loss_weight
                            dice_loss = dice_loss_fn(out, targets) * args.dice_loss_weight
                            lovasz_loss = lovasz_softmax(torch.softmax(out, dim=1), targets) * 0.5
                            loss += args.deep_supervision_weights[j] * (focal_loss + dice_loss + lovasz_loss)
                        except ValueError as e:
                            logger.error(f"Loss computation failed for output {j}: {e}. Skipping output.")
                            continue
            else:
                try:
                    logger.debug(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                    focal_loss = focal_loss_fn(outputs, targets) * args.focal_loss_weight
                    dice_loss = dice_loss_fn(outputs, targets) * args.dice_loss_weight
                    lovasz_loss = lovasz_softmax(torch.softmax(outputs, dim=1), targets) * 0.5
                    loss = focal_loss + dice_loss + lovasz_loss
                except ValueError as e:
                    logger.error(f"Loss computation failed: {e}. Skipping batch.")
                    continue
            
            loss = loss / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * images.size(0) * accum_steps
            total_samples += images.size(0)
        
        if total_samples > 0:
            epoch_loss = running_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {epoch_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1}/{args.epochs}, No valid samples processed.")
        
        scheduler.step()
        
        logger.info("--- Validating Summary ---")
        avg_loss, oa, miou, fwiou, dice, class_metrics = evaluate_model(model, val_loader, device, args.num_classes, logger)
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  OA (Overall Accuracy): {oa:.4f}")
        logger.info(f"  mIoU (Mean IoU): {miou:.4f}")
        logger.info(f"  FWIoU (Frequency Weighted IoU): {fwiou:.4f}")
        logger.info(f"  Dice (Mean Dice Coefficient): {dice:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou
            }, best_checkpoint_path)
            logger.info(f"✅ New best mIoU: {best_miou:.4f}. Saving best model to '{best_checkpoint_path}'.")
        else:
            logger.info(f"⚠️ Validation mIoU did not improve for {epoch - start_epoch + 1} epoch(s). Best mIoU: {best_miou:.4f}.")

if __name__ == "__main__":
    main()