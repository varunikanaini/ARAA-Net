# /kaggle/working/ARAA-Net/benchmark_fps.py
# --- FINAL VERSION ---

import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Model, Config, and Utilities ---
from lasa_unet_model import LASA_Unet 
from misc import check_mkdir
import config
from config import CKPT_ROOT, DEFAULT_ARGS, BACKBONE_CHANNELS

# --- Logging Setup ---
def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Parameter Counting (Handled by thop, but can be kept) ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- FPS Benchmarking ---
def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
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

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-Unet FPS, Parameters, and FLOPs')
    parser.add_argument('--backbone', type=str, default='vgg19', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--input-h', type=int, default=224)
    parser.add_argument('--input-w', type=int, default=224)
    parser.add_argument('--num-warmup', type=int, default=20)
    parser.add_argument('--num-inference', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.error("A GPU is required for meaningful FPS benchmarking.")
        return

    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f'benchmark_{args.backbone}_input{args.input_h}x{args.input_w}.log'
    setup_logging_benchmark(benchmark_log_dir, filename=log_filename)

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone ---")
    logging.info(f"Input size for benchmark: {args.input_h}x{args.input_w}")

    model = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))
    model.to(device).eval()

    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)

    # Calculate Parameters and FLOPs using thop
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info(f"\n--- Model Complexity ---")
    logging.info(f"Total Trainable Parameters: {params_formatted}")
    logging.info(f"FLOPs (MACs): {macs_formatted}")
    logging.info(f"--------------------------")

    # Benchmark FPS
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info(f"\n--- Performance on {torch.cuda.get_device_name(0)} ---")
    logging.info(f"Achieved FPS: {fps:.2f}")
    logging.info(f"Average Inference Time per frame: {1000/fps:.2f} ms")
    logging.info(f"---------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()