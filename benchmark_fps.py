# /kaggle/working/ARAA-Net/benchmark_fps_lasa_vgg.py
import torch
import argparse
import time
import os
import sys

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from lasa_vgg_model import LASA_VGG_Unet

def benchmark(model, device, input_h, input_w):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    print("Performing GPU warm-up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    print("Starting benchmark (100 inferences)...")
    torch.cuda.synchronize() 
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    fps = 100 / total_time
    return fps

def main():
    parser = argparse.ArgumentParser(description='Benchmark LASA-VGG-Unet FPS')
    parser.add_argument('--input-h', type=int, default=448)
    parser.add_argument('--input-w', type=int, default=448)
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if not torch.cuda.is_available():
        print("❌ ERROR: A GPU is required for FPS benchmarking.")
        return

    device = torch.device("cuda")
    
    # --- Benchmark the LASA-VGG-Unet model ---
    print(f"--- Benchmarking LASA-VGG-Unet with input size {args.input_h}x{args.input_w} ---")
    model = LASA_VGG_Unet(num_classes=2)
    fps = benchmark(model, device, args.input_h, args.input_w)
    
    print("\n\n--- FPS Benchmark Results ---")
    print(f"Model: LASA-VGG-Unet")
    print(f"Input Size: {args.input_h}x{args.input_w}")
    print(f"Achieved: {fps:.2f} FPS ({1000/fps:.2f} ms/frame)")
    print("---------------------------------")

if __name__ == '__main__':
    main()