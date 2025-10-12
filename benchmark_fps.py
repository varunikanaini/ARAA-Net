# /kaggle/working/ARAA-Net/benchmark_fps.py
import torch
import argparse
import time
import os
import sys
import logging
import copy # Needed for potentially copying models if you were to modify them

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Model, Config, and Utilities ---
# MODIFIED: Import the generalized LASA_Unet model
from lasa_unet_model import LASA_Unet 
from misc import check_mkdir
# FIXED: Import CKPT_ROOT, DEFAULT_ARGS, BACKBONE_CHANNELS from config
from config import CKPT_ROOT, DEFAULT_ARGS, BACKBONE_CHANNELS 

# --- Logging Setup ---
def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to avoid duplicate log entries
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- Parameter Counting ---
def count_parameters(model):
    """Counts total trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# --- FPS Benchmarking ---
def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Benchmarks model FPS on GPU."""
    model.to(device)
    model.eval()
    # Create dummy input with appropriate dimensions and data type
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up ({num_warmup} inferences)...")
    # Warm-up run
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    logging.info(f"Starting benchmark ({num_inference} inferences)...")
    torch.cuda.synchronize() # Ensure all previous operations are complete
    start_time = time.time()
    
    # Actual benchmark loop
    with torch.no_grad():
        for _ in range(num_inference):
            _ = model(dummy_input)
    torch.cuda.synchronize() # Wait for all inference operations to complete
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_inference / total_time
    return fps

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-Unet FPS and Parameters')
    
    # --- Dynamically add arguments based on config defaults ---
    # Backbone selection
    parser.add_argument('--backbone', type=str, default=DEFAULT_ARGS.get('backbone', 'vgg16'), # Use default from config
                        choices=BACKBONE_CHANNELS.keys(), help='Backbone architecture to benchmark')
    
    # Input dimensions for benchmarking (use common defaults or train defaults)
    # Suggest using dimensions that match common training scales.
    parser.add_argument('--input-h', type=int, default=None, help='Height of dummy input image')
    parser.add_argument('--input-w', type=int, default=None, help='Width of dummy input image')
    
    # Benchmarking parameters
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # Fallback for environments like notebooks that might exit prematurely
        args = parser.parse_args([])

    # --- Set Input Dimensions based on Backbone if not provided ---
    if args.input_h is None or args.input_w is None:
        try:
            # Get default input resolution from config for the selected backbone
            input_h, input_w = config.get_backbone_resolution(args.backbone)
            args.input_h = input_h
            args.input_w = input_w
            logging.info(f"Using default input resolution for backbone '{args.backbone}': {args.input_h}x{args.input_w}")
        except Exception as e:
            logging.error(f"Could not determine default input resolution for backbone '{args.backbone}': {e}")
            logging.error("Please specify --input-h and --input-w manually.")
            sys.exit(1)

    # --- Device Check ---
    if not torch.cuda.is_available():
        logging.error("❌ ERROR: A GPU is required for FPS benchmarking. Ensure CUDA is available and properly configured.")
        sys.exit(1)
    device = torch.device("cuda")

    # --- Logging Setup ---
    # Use a generic directory for benchmarks, separate from dataset-specific checkpoints
    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    # Create a specific log file for this benchmark run
    log_filename = f'benchmark_{args.backbone}_input{args.input_h}x{args.input_w}.log'
    setup_logging_benchmark(benchmark_log_dir, filename=log_filename)

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone ---")
    logging.info(f"Input size for benchmark: {args.input_h}x{args.input_w}")
    logging.info(f"Arguments used: {vars(args)}")

    # --- Instantiate the Model ---
    # Instantiate the LASA_Unet model with the specified backbone
    # num_classes can be set to a default (e.g., 2).
    # lasa_kernels can be benchmarked with defaults or specified if relevant to FPS.
    model = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))

    # --- Calculate Parameters ---
    total_trainable_params = count_parameters(model)
    logging.info(f"Total Trainable Parameters in LASA-Unet ({args.backbone}): {total_trainable_params:,}") # Format with comma for readability
    logging.info(f"Total Trainable Parameters (Millions): {total_trainable_params / 1_000_000:.2f} M")

    # --- Benchmark FPS ---
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info("\n--- FPS Benchmark Results ---")
    logging.info(f"Model: LASA-Unet ({args.backbone} backbone)")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Warmup Inferences: {args.num_warmup}")
    logging.info(f"Benchmark Inferences: {args.num_inference}")
    logging.info(f"Achieved FPS: {fps:.2f}")
    logging.info(f"Average Inference Time per frame: {1000/fps:.2f} ms") # Convert seconds to milliseconds
    logging.info("---------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()