# benchmark_fps.py
import torch
import argparse
import time
import os
import sys
import logging
from thop import profile # For FLOPS calculation

# Ensure project_path is in sys.path
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import LASA_Unet and other utilities
from lasa_unet_model import LASA_Unet # Assumes model file is renamed
from misc import check_mkdir
from config import CKPT_ROOT 

# Helper function to setup logging
def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    """Configures logging for the benchmarking script."""
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to prevent duplicate logs if run multiple times
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Function to count trainable parameters
def count_parameters(model):
    """Counts total trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# Function to calculate FLOPS
def count_flops(model, input_shape, device):
    """Calculates FLOPS for a model given an input shape."""
    model.to(device)
    model.eval()
    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape, dtype=torch.float32).to(device)
    
    # Use thop for FLOPS calculation
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params # params here is total params, similar to count_parameters

# Function to benchmark FPS
def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Benchmarks model FPS on GPU."""
    model.to(device)
    model.eval() 
    
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
    parser.add_argument('--backbone', type=str, default='vgg16', 
                        choices=['vgg16', 'resnet50', 'inception_v3', 'efficientnet_b0', 'efficientnet_b3'], 
                        help='Backbone architecture to benchmark')
    parser.add_argument('--input-h', type=int, default=896, 
                        help='Height of dummy input image (aligned with DASEG evaluation resize)') 
    parser.add_argument('--input-w', type=int, default=576, 
                        help='Width of dummy input image (aligned with DASEG evaluation resize)') 
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    
    # Argument for LASA kernels, to reflect model configuration
    parser.add_argument('--lasa-kernels', nargs='+', type=int, default=[1, 3, 5, 7], 
                        help='List of kernel sizes for LASA module. Example: --lasa-kernels 1 3 5 7')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([]) # Fallback for interactive environments

    if not torch.cuda.is_available():
        logging.error("❌ ERROR: A GPU is required for accurate FPS and FLOPS benchmarking. Running on CPU is not representative.")
        sys.exit(1)

    device = torch.device("cuda")

    # Set up logging for benchmark results
    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_results')
    check_mkdir(benchmark_log_dir)
    # Dynamic log filename based on backbone and kernels
    lasa_kernels_str = "_".join(map(str, args.lasa_kernels))
    log_filename = f'{args.backbone}_LASA_Kernels{lasa_kernels_str}_benchmark.log'
    setup_logging_benchmark(benchmark_log_dir, filename=log_filename)

    logging.info(f"--- Benchmarking LASA-Unet ---")
    logging.info(f"Backbone: {args.backbone}")
    logging.info(f"LASA Kernels: {args.lasa_kernels}")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Arguments: {vars(args)}")

    # Instantiate the model with the specified backbone and kernels
    # num_classes=2 is a placeholder, not relevant for parameter/FLOP count
    model = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=args.lasa_kernels) 

    # --- Calculate Parameters ---
    total_trainable_params = count_parameters(model)
    logging.info(f"Total Trainable Parameters: {total_trainable_params:,}")
    logging.info(f"Total Trainable Parameters (Millions): {total_trainable_params / 1_000_000:.2f} M")

    # --- Calculate FLOPS ---
    # Use a representative input shape for FLOPS calculation
    input_shape = (1, 3, args.input_h, args.input_w) 
    try:
        flops, _ = count_flops(model, input_shape, device)
        logging.info(f"Total FLOPS: {flops / 1e9:.2f} G") # Report in Giga FLOPS
    except Exception as e:
        logging.warning(f"Could not calculate FLOPS: {e}. Ensure 'thop' is installed correctly.")
        flops = 0 # Set to 0 if calculation fails

    # --- Benchmark FPS ---
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info("\n--- Benchmark Summary ---")
    logging.info(f"Model: LASA-Unet ({args.backbone}, LASA Kernels={args.lasa_kernels})")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Achieved FPS: {fps:.2f}")
    logging.info(f"Inference Time per frame (ms): {1000 / fps:.2f}")
    logging.info(f"Total Trainable Params: {total_trainable_params:,}")
    logging.info(f"Total FLOPS: {flops / 1e9:.2f} G")
    logging.info("---------------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()