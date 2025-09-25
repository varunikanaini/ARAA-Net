import torch
import time
import os
import argparse
from collections import OrderedDict

# Ensure these imports match your project structure
from daseg import daseg
from config import backbone_path

def main():
    parser = argparse.ArgumentParser(description='Model Benchmarking (FPS and Trainable Parameters)')
    parser.add_argument('--input_size_w', type=int, default=896, help='Input image width for benchmarking')
    parser.add_argument('--input_size_h', type=int, default=576, help='Input image height for benchmarking')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for FPS calculation')
    parser.add_argument('--num_warmup', type=int, default=10, help='Number of warm-up runs for FPS')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of actual runs for FPS calculation')
    parser.add_argument('--snapshot', type=str, default='',
                        help='Optional: Path to a trained model snapshot (e.g., ckpt/DANet_JSRT/best.pth) to load weights.')
    
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate the model
    model = daseg(backbone_path=backbone_path)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Load snapshot if provided
    if args.snapshot:
        if not os.path.exists(args.snapshot):
            print(f"Warning: Model snapshot not found at {args.snapshot}. Skipping loading weights.")
        else:
            print(f"Loading model weights from: {args.snapshot}")
            state_dict = torch.load(args.snapshot, map_location=device)
            # Remove 'module.' prefix if the model was saved from DataParallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("Model weights loaded successfully.")
    else:
        print("No snapshot provided, using randomly initialized model for benchmarking.")

    # 2. Calculate Total Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")

    # 3. Calculate FPS
    print(f"\n--- FPS Calculation ({args.batch_size}x3x{args.input_size_h}x{args.input_size_w}) ---")
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.input_size_h, args.input_size_w).to(device)

    # Warm-up runs
    print(f"Performing {args.num_warmup} warm-up runs...")
    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure actual inference time
    print(f"Performing {args.num_runs} actual runs for measurement...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.num_runs):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Calculate FPS
    total_time = end_time - start_time
    avg_inference_time = total_time / args.num_runs
    fps = args.batch_size / avg_inference_time if avg_inference_time > 0 else float('inf')

    print(f"Total inference time for {args.num_runs} runs: {total_time:.4f} seconds")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print(f"Frames Per Second (FPS) with batch size {args.batch_size}: {fps:.2f}\n")

if __name__ == '__main__':
    main()