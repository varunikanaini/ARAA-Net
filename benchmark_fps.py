# /kaggle/working/ARAA-Net/benchmark_fps.py (Updated with FLOPs)
import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format # <-- IMPORT THOP

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import Model, Config, and Utilities ---
from lasa_unet_model import LASA_Unet 
from misc import check_mkdir
# FIXED: Import config correctly
import config 
from config import CKPT_ROOT, DEFAULT_ARGS, BACKBONE_CHANNELS 

# --- Logging Setup ---
def setup_logging_benchmark(log_dir, filename='benchmark_results.log'):
    log_file = os.path.join(log_dir, filename)
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
    
    parser.add_argument('--backbone', type=str, default=DEFAULT_ARGS.get('backbone', 'vgg16'),
                        choices=BACKBONE_CHANNELS.keys(), help='Backbone architecture to benchmark')
    parser.add_argument('--input-h', type=int, default=None, help='Height of dummy input image')
    parser.add_argument('--input-w', type=int, default=None, help='Width of dummy input image')
    parser.add_argument('--num-warmup', type=int, default=20, help='Number of warmup inferences')
    parser.add_argument('--num-inference', type=int, default=100, help='Number of inferences for actual benchmark')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if args.input_h is None or args.input_w is None:
        try:
            input_h, input_w = config.get_backbone_resolution(args.backbone)
            args.input_h = input_h
            args.input_w = input_w
        except Exception as e:
            # Fallback to a default if get_backbone_resolution is not in config
            logging.warning(f"Could not get resolution from config: {e}. Falling back to 224x224.")
            args.input_h, args.input_w = 224, 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f'benchmark_{args.backbone}_input{args.input_h}x{args.input_w}.log'
    setup_logging_benchmark(benchmark_log_dir, filename=log_filename)

    logging.info(f"--- Benchmarking LASA-Unet with {args.backbone} backbone ---")
    logging.info(f"Input size for benchmark: {args.input_h}x{args.input_w}")
    logging.info(f"Arguments used: {vars(args)}")

    model = LASA_Unet(num_classes=2, backbone_name=args.backbone, lasa_kernels=DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))
    model.to(device)
    model.eval()

    # --- Create a single dummy input for all benchmarks ---
    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)

    # --- Calculate Parameters, MACs (FLOPs), and FPS ---
    total_trainable_params = count_parameters(model)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Use clever_format to easily convert to M (Millions) and G (Giga)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info(f"\n--- Model Complexity ---")
    logging.info(f"Total Trainable Parameters: {params_formatted}")
    logging.info(f"FLOPs (MACs): {macs_formatted}") # thop calculates MACs, which is ~FLOPs/2. Often used interchangeably.
    logging.info(f"--------------------------")

    if device.type == 'cuda':
        fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
        logging.info(f"\n--- Performance on {torch.cuda.get_device_name(0)} ---")
        logging.info(f"Achieved FPS: {fps:.2f}")
        logging.info(f"Average Inference Time per frame: {1000/fps:.2f} ms")
        logging.info(f"---------------------------------")
    else:
        logging.warning("GPU not available, skipping FPS benchmark.")

    logging.info("✅ Benchmark completed.")


if __name__ == '__main__':
    main()