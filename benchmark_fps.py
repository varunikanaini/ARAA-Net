# --- FINAL VERSION: FPS + Segmentation Performance Benchmarking ---

import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from lasa_unet_model import LASA_Unet
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from misc import check_mkdir
from seg_utils import ConfusionMatrix, get_boundary_mask
from boundary_loss import BoundaryLoss
from train_light import FocalLoss, DiceLoss, CenterLoss
import config

def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up ({num_warmup} inferences)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    logging.info(f"Starting FPS benchmark ({num_inference} inferences)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_inference):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_inference / total_time
    return fps

def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, boundary_loss_fn, center_loss_fn, args):
    net.eval()
    confmat = ConfusionMatrix(args.num_classes)
    loss_recorder = torch.tensor(0.0, device=device)
    num_samples = 0

    num_decoder_stages = 4

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Testing", leave=False):
            if data is None:
                continue

            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs_tuple = net(inputs)

            total_loss_batch = 0
            seg_output_idx = 0

            for i in range(num_decoder_stages):
                seg_pred = outputs_tuple[i * 3]
                boundary_pred = outputs_tuple[i * 3 + 1]
                center_pred = outputs_tuple[i * 3 + 2]

                seg_labels = labels.long()
                f_loss = focal_loss_fn(seg_pred, seg_labels)
                d_loss = dice_loss_fn(seg_pred, seg_labels)
                b_loss = boundary_loss_fn(boundary_pred, seg_labels)
                c_loss = center_loss_fn(center_pred, seg_labels)

                weight_index = seg_output_idx
                seg_weight = args.deep_supervision_weights[weight_index] if weight_index < len(args.deep_supervision_weights) else 1.0

                total_loss_batch += seg_weight * (
                    (args.focal_loss_weight * f_loss) +
                    (args.dice_loss_weight * d_loss) +
                    (args.boundary_loss_weight * b_loss) +
                    (args.center_loss_weight * c_loss)
                )
                seg_output_idx += 1

            final_seg_pred = outputs_tuple[-2]
            final_center_pred = outputs_tuple[-1]
            f_loss_final = focal_loss_fn(final_seg_pred, seg_labels)
            d_loss_final = dice_loss_fn(final_seg_pred, seg_labels)
            c_loss_final = center_loss_fn(final_center_pred, seg_labels)

            final_seg_weight = args.deep_supervision_weights[-1] if args.deep_supervision_weights else 1.0
            total_loss_batch += final_seg_weight * (
                (args.focal_loss_weight * f_loss_final) +
                (args.dice_loss_weight * d_loss_final) +
                (args.center_loss_weight * c_loss_final)
            )

            # Update confusion matrix with boundary masks
            target_boundary = get_boundary_mask(labels)
            pred_boundary = get_boundary_mask(final_seg_pred.argmax(1))
            confmat.update(labels.flatten(), final_seg_pred.argmax(1).flatten(),
                         target_boundary.flatten(), pred_boundary.flatten())

            loss_recorder += total_loss_batch * inputs.size(0)
            num_samples += inputs.size(0)

    acc_global, acc, iu, FWIoU, mDice, boundary_iu = confmat.compute()
    avg_loss = loss_recorder / num_samples if num_samples > 0 else 0.0

    logging.info(f"--- Test Summary --- Loss: {avg_loss:.4f}, OA: {acc_global.item():.4f}, "
                 f"mIoU: {iu.mean().item():.4f}, FWIoU: {FWIoU.item():.4f}, "
                 f"mDice: {mDice:.4f}, Boundary IoU: {boundary_iu:.4f}")
    return {'mIoU': iu.mean().item(), 'OA': acc_global.item(), 'boundary_iou': boundary_iu}

def main():
    parser = argparse.ArgumentParser(description='Benchmark Model FPS, Parameters, FLOPs, and Segmentation Performance')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--dataset-name', type=str, default='TSRS_RSNA-Epiphysis', choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--input-h', type=int, default=224)
    parser.add_argument('--input-w', type=int, default=224)
    parser.add_argument('--num-warmup', type=int, default=20)
    parser.add_argument('--num-inference', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.error("A GPU is required for meaningful FPS benchmarking.")
        return

    benchmark_log_dir = os.path.join(config.CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f'benchmark_{args.backbone}_input{args.input_h}x{args.input_w}.log'
    setup_logging_benchmark(benchmark_log_dir, filename=log_filename)

    logging.info(f"--- Benchmarking model with {args.backbone} backbone on {args.dataset_name} ---")
    logging.info(f"Input size for benchmark: {args.input_h}x{args.input_w}")

    # Instantiate model
    model_name = ""
    if args.backbone == 'mobilenet_v2':
        model = Light_LASA_Unet(num_classes=2, lasa_kernels=config.DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))
        model_name = "Light_LASA_Unet"
    else:
        model = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=config.DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))
        model_name = "LASA_Unet"
    
    model.to(device).eval()
    logging.info(f"Instantiated model: {model_name}")

    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        try:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            logging.info(f"Model loaded from {args.checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading checkpoint {args.checkpoint_path}: {e}")
            return

    # Calculate Parameters and FLOPs
    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info(f"\n--- Model Complexity ---")
    logging.info(f"Total Trainable Parameters: {params_formatted}")
    logging.info(f"FLOPs (MACs): {macs_formatted}")

    # Benchmark FPS
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    logging.info(f"\n--- Performance on {torch.cuda.get_device_name(0)} ---")
    logging.info(f"Achieved FPS: {fps:.2f}")
    logging.info(f"Average Inference Time per frame: {1000/fps:.2f} ms")

    # Evaluate segmentation performance
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    args.deep_supervision_weights = [0.1, 0.3, 0.5, 0.7, 1.0]
    args.focal_loss_weight = 0.5
    args.dice_loss_weight = 1.5
    args.boundary_loss_weight = 0.8
    args.center_loss_weight = 0.3
    args.focal_alpha = 0.5
    args.focal_gamma = 2

    test_ds = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, 'test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                             collate_fn=lambda batch: torch.utils.data.dataloader.default_collate([item for item in batch if item is not None]))

    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    boundary_loss_fn = BoundaryLoss(device=device, hausdorff_weight=0.3).to(device)
    center_loss_fn = CenterLoss().to(device)

    metrics = evaluate_model(model, test_loader, device, focal_loss_fn, dice_loss_fn, boundary_loss_fn, center_loss_fn, args)
    logging.info(f"\n--- Segmentation Performance ---")
    logging.info(f"mIoU: {metrics['mIoU']:.4f}")
    logging.info(f"OA: {metrics['OA']:.4f}")
    logging.info(f"Boundary IoU: {metrics['boundary_iou']:.4f}")
    logging.info(f"---------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()