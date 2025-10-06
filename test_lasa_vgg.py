# benchmark_fps.py (CORRECTED CONTENT)

import torch
import argparse
import time
import os
import sys
import logging
import thop # Import thop for FLOPS calculation

# Ensure project_path is in sys.path
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import LASA_Unet and other utilities
from lasa_vgg_model import LASA_Unet
from misc import check_mkdir
from config import CKPT_ROOT

def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    """Configures logging for the benchmarking script."""
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to prevent duplicate logs if run multiple times
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def count_parameters(model):
    """Counts total trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def calculate_flops(model, input_h, input_w, device):
    """Calculates FLOPS for the model using thop."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)
    
    try:
        # Using thop to calculate MACs (Multiply-Accumulate Operations)
        # MACs are approximately 2 * FLOPS
        macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2 # Approximate FLOPS
        return flops, params
    except Exception as e:
        logging.error(f"Error during FLOPS calculation with thop: {e}")
        return 0, 0 # Return 0 if calculation fails

def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Benchmarks model FPS on GPU."""
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up ({num_warmup} inferences)...")
    with torch.no_grad(): # No need to calculate gradients for benchmarking
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize GPU to ensure all previous operations are complete before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logging.info(f"Starting benchmark ({num_inference} inferences)...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_inference):
            _ = model(dummy_input)

    # Synchronize GPU again after all inference runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_inference / total_time
    return fps

def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-Unet FPS, FLOPS, and Parameters')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to benchmark')
    parser.add_argument('--input-h', type=int, default=896, help='Height of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--input-w', type=int, default=576, help='Width of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    parser.add_argument('--full-model', action='store_true', help='Benchmark the full model (including decoder). Default is to benchmark encoder + LASA only to avoid decoder overhead specific to input size.')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if not torch.cuda.is_available():
        logging.error("❌ ERROR: A GPU is required for FPS and FLOPS benchmarking. Running on CPU is not representative.")
        sys.exit(1)

    device = torch.device("cuda")

    # Set up logging for benchmark results
    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_results')
    check_mkdir(benchmark_log_dir)
    setup_logging_benchmark(benchmark_log_dir, filename=f'{args.backbone}_benchmark.log')

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone and input size {args.input_h}x{args.input_w} ---")
    logging.info(f"Arguments: {args}")

    # Instantiate the model
    model = LASA_Unet(num_classes=2, backbone_name=args.backbone)

    # --- Calculate Parameters ---
    total_trainable_params = count_parameters(model)
    logging.info(f"Total Trainable Parameters (LASA-Unet, {args.backbone}): {total_trainable_params:,}")
    logging.info(f"Total Trainable Parameters (Millions): {total_trainable_params / 1_000_000:.2f} M")

    # --- Calculate FLOPS ---
    flops, params_thop = calculate_flops(model, args.input_h, args.input_w, device)
    if flops > 0:
        logging.info(f"Total FLOPS (approximate): {flops:,}")
        logging.info(f"Total FLOPS (Giga): {flops / 1e9:.2f} G")
        # Note: thop's params count might differ slightly from our count_parameters if it includes non-trainable params,
        # or if there are modules it doesn't fully process. We'll log both for reference.
        logging.info(f"Total Parameters (thop calculation): {params_thop:,}")
    else:
        logging.warning("FLOPS calculation failed. Skipping FLOPS reporting.")


    # --- Benchmark FPS ---
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)

    logging.info("\n--- Performance Benchmark Summary ---")
    logging.info(f"Model: LASA-Unet with {args.backbone} backbone")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Achieved FPS: {fps:.2f}")
    logging.info(f"Inference Time per Frame: {1000/fps:.2f} ms")
    logging.info(f"Total Trainable Parameters: {total_trainable_params:,}")
    if flops > 0:
        logging.info(f"Approximate FLOPS: {flops / 1e9:.2f} G")
    logging.info("-------------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()