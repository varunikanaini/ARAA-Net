# /kaggle/working/ARAA-Net/benchmark_fps.py (FINAL & CORRECTED - Includes Parameter Counting)
import torch
import argparse
import time
import os
import sys
import logging

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from lasa_vgg_model import LASA_Unet # Use generalized LASA_Unet
from misc import check_mkdir
from config import CKPT_ROOT # <<< FIXED: Import CKPT_ROOT from config


def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def count_parameters(model):
    """Counts total trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Benchmarks model FPS on GPU."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up ({num_warmup} inferences)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    logging.info(f"Starting benchmark ({num_inference} inferences)...")
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

def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-Unet FPS and Parameters')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to benchmark')
    parser.add_argument('--input-h', type=int, default=896, help='Height of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--input-w', type=int, default=576, help='Width of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if not torch.cuda.is_available():
        logging.error("❌ ERROR: A GPU is required for FPS benchmarking. Running on CPU is not representative.")
        sys.exit(1)

    device = torch.device("cuda")

    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_results')
    check_mkdir(benchmark_log_dir)
    setup_logging_benchmark(benchmark_log_dir, filename=f'{args.backbone}_benchmark.log')

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone and input size {args.input_h}x{args.input_w} ---")
    logging.info(f"Arguments: {args}")

    model = LASA_Unet(num_classes=2, backbone_name=args.backbone) 

    total_trainable_params = count_parameters(model)
    logging.info(f"Total Trainable Parameters in LASA-Unet ({args.backbone}): {total_trainable_params}")
    logging.info(f"Total Trainable Parameters (Millions): {total_trainable_params / 1_000_000:.2f} M")

    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info("\n--- FPS Benchmark Results ---")
    logging.info(f"Model: LASA-Unet with {args.backbone} backbone")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Achieved: {fps:.2f} FPS ({1000/fps:.2f} ms/frame)")
    logging.info("---------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()