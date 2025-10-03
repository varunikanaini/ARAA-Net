# /kaggle/working/ARAA-Net/benchmark_fps.py (CORRECTED CONTENT)

import torch
import argparse
import time
import os
import sys
import logging

# Ensure project_path is in sys.path
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import LASA_Unet and other utilities
from lasa_vgg_model import LASA_Unet 
from misc import check_mkdir
from config import CKPT_ROOT 

# Attempt to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    logging.warning("thop library not found. FLOPs calculation will be skipped. Install with: pip install thop")


def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    """Configures logging for the benchmarking script."""
    log_file = os.path.join(log_dir, filename)
    # Clear existing handlers to prevent duplicate logs if run multiple times
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])


def count_parameters(model):
    """Counts total trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Benchmarks model FPS on GPU. Returns FPS and inference time per frame."""
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    # Create a dummy input tensor with the specified input size
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up ({num_warmup} inferences)...")
    with torch.no_grad(): # No need to calculate gradients for benchmarking
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU to ensure all previous operations are complete before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize() 

    logging.info(f"Starting benchmark ({num_inference} inferences)...")
    start_time = time.time() # Record start time
    with torch.no_grad():
        for _ in range(num_inference):
            _ = model(dummy_input)
    
    # Synchronize GPU again after all inference runs
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    end_time = time.time() # Record end time

    total_time = end_time - start_time # Total time for inference
    if total_time == 0: total_time = 1e-6 # Prevent division by zero if too fast
    fps = num_inference / total_time # Calculate FPS
    avg_inference_time_ms = (total_time / num_inference) * 1000 # Average time per frame in ms
    
    return fps, avg_inference_time_ms

def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-Unet FPS, FLOPs, and Parameters')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Backbone architecture to benchmark')
    parser.add_argument('--input-h', type=int, default=896, help='Height of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--input-w', type=int, default=576, help='Width of dummy input image (aligned with DASEG evaluation resize)') # Aligned default
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of segmentation classes')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # Useful for notebooks or if script is run without args to avoid exiting
        args = parser.parse_args([])

    if not torch.cuda.is_available():
        logging.error("❌ ERROR: A GPU is required for FPS and FLOPs benchmarking. Running on CPU is not representative.")
        sys.exit(1) # Exit if CUDA is not available

    device = torch.device("cuda")

    # Set up logging directory and file
    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_results')
    check_mkdir(benchmark_log_dir)
    setup_logging_benchmark(benchmark_log_dir, filename=f'{args.backbone}_benchmark.log')

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone ---")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}, Num Classes: {args.num_classes}")
    logging.info(f"Arguments: {args}")

    # Instantiate the model with the specified backbone
    model = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone) 

    # --- Calculate Parameters ---
    total_trainable_params = count_parameters(model)
    logging.info(f"Total Trainable Parameters in LASA-Unet ({args.backbone}): {total_trainable_params}")
    logging.info(f"Total Trainable Parameters (Millions): {total_trainable_params / 1_000_000:.2f} M")

    # --- Calculate FLOPs (if thop is available) ---
    if THOP_AVAILABLE:
        try:
            # Create a dummy input tensor on the correct device
            dummy_input_flops = torch.randn(1, 3, args.input_h, args.input_w, dtype=torch.float32).to(device)
            
            # Use profile to get FLOPs and MACs
            # Ensure model is on the correct device and in eval mode for profiling
            model.to(device)
            model.eval()
            macs, params_thop = profile(model, inputs=(dummy_input_flops,), verbose=False)
            
            # Convert MACs to GFLOPs
            gflops = macs / 1e9 
            
            logging.info(f"Total MACs (Multiply-Accumulate operations): {macs:.2f}")
            logging.info(f"Total GFLOPs: {gflops:.2f}")
            # Optionally log params again if thop's params differ from sum(p.numel())
            # logging.info(f"Total Parameters (from thop): {params_thop / 1e6:.2f} M") 
            
        except Exception as e:
            logging.error(f"Error calculating FLOPs: {e}")
    else:
        logging.warning("Skipping FLOPs calculation as thop library is not installed.")

    # --- Benchmark FPS ---
    logging.info(f"Benchmarking FPS with {args.num_warmup} warmup inferences and {args.num_inference} benchmark inferences...")
    fps, avg_inference_time_ms = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info("\n--- Benchmark Summary ---")
    logging.info(f"Model: LASA-Unet with {args.backbone} backbone")
    logging.info(f"Input Size: {args.input_h}x{args.input_w}")
    logging.info(f"Achieved: {fps:.2f} FPS")
    logging.info(f"Inference Time per Frame: {avg_inference_time_ms:.2f} ms")
    logging.info(f"Total Parameters: {total_trainable_params / 1_000_000:.2f} M")
    if THOP_AVAILABLE:
        logging.info(f"Total GFLOPs: {gflops:.2f}")
    logging.info("-------------------------")
    logging.info("✅ Benchmark completed.")

if __name__ == '__main__':
    main()