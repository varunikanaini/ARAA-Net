

import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format
from collections import OrderedDict

# --- Setup Project Path and Imports ---
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from config import backbone_path
from daseg import daseg
from misc import check_mkdir

# --- Logging Setup ---
def setup_logging(log_dir, filename='benchmark.log'):
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

# --- Model Wrapper for FLOPs Calculation ---
class ModelWrapperForThop(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The daseg model returns 5 outputs, we only need the last one for FLOPs
        *_, final_output = self.model(x)
        return final_output

# --- Main Benchmark Function ---
def main():
    parser = argparse.ArgumentParser(description='Benchmark DANet Model')
    parser.add_argument('--input-h', type=int, default=576)
    parser.add_argument('--input-w', type=int, default=576)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: A GPU is required for FPS benchmarking.")
        return

    benchmark_log_dir = os.path.join('./ckpt', 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f'benchmark_DANet_{args.input_h}x{args.input_w}.log'
    setup_logging(benchmark_log_dir, filename=log_filename)

    logging.info("=" * 50)
    logging.info("Benchmarking DANet (daseg model)")
    logging.info(f"Input Resolution: {args.input_h}x{args.input_w}")
    logging.info("=" * 50)

    # --- Instantiate Model ---
    model = daseg(backbone_path).to(device).eval()
    
    # --- Calculate Parameters and FLOPs ---
    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    model_for_thop = ModelWrapperForThop(model)
    
    logging.info("Calculating model parameters and FLOPs...")
    macs, params = profile(model_for_thop, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info("\n--- Model Complexity ---")
    logging.info(f"Total Parameters: {params_formatted}")
    logging.info(f"FLOPs (GFLOPs): {macs_formatted}")
    logging.info("------------------------\n")

    # --- Benchmark FPS ---
    with torch.no_grad():
        for _ in range(50): _ = model(dummy_input) # Warm-up
        
    torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(200): _ = model(dummy_input)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    fps = 200 / (end_time - start_time)
    
    logging.info(f"--- Performance on {torch.cuda.get_device_name(0)} ---")
    logging.info(f"Frames Per Second (FPS): {fps:.2f}")
    logging.info(f"Avg. Inference Time:     {(1000/fps):.2f} ms")
    logging.info("----------------------------------------")
    logging.info("✅ Benchmark completed successfully.")

if __name__ == '__main__':
    main()