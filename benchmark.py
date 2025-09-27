# /kaggle/working/ARAA-Net/benchmark.py (FOR DANET MODEL)
import torch
import time
import os
import argparse
from collections import OrderedDict
import logging 
import sys 

from daseg import daseg
from config import backbone_path, CKPT_ROOT 
from misc import check_mkdir 

def main():
    parser = argparse.ArgumentParser(description='Model Benchmarking (FPS and Trainable Parameters)')
    parser.add_argument('--input_size_w', type=int, default=896, help='Input image width for benchmarking') # DEFAULT IS KEPT AT USER'S 896
    parser.add_argument('--input_size_h', type=int, default=576, help='Input image height for benchmarking') # DEFAULT IS KEPT AT USER'S 576
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for FPS calculation') 
    parser.add_argument('--num_warmup', type=int, default=10, help='Number of warm-up runs for FPS')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of actual runs for FPS calculation')
    parser.add_argument('--snapshot', type=str, default='',
                        help='Optional: Path to a trained model snapshot (e.g., ckpt/DANet_JSRT/best.pth) to load weights.')
    parser.add_argument('--dataset', type=str, default='TSRS_RSNA-Epiphysis', # ADDED for logging and distinct folders
                        choices=[
                            'TSRS_RSNA-Epiphysis', 
                            'JSRT', 
                            'COVID19_Radiography', 
                            'CVC-ClinicDB', 
                            'DentalPanoramic', 
                            'SixDiseasesChestXRay'
                        ], help='Dataset name for logging purposes, relevant if --snapshot is used (e.g., JSRT, COVID19_Radiography)')
    
    args = parser.parse_args()

    benchmark_log_dir = os.path.join(CKPT_ROOT, 'benchmark_results_DANet') 
    check_mkdir(benchmark_log_dir)
    log_file = os.path.join(benchmark_log_dir, f'DANet_{args.dataset}_benchmark.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]) 
    logger = logging.getLogger()

    logger.info(f"Using device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"Benchmarking Arguments: {args}")

    model = daseg(backbone_path=backbone_path)
    model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.eval() 

    if args.snapshot:
        if not os.path.exists(args.snapshot):
            logger.warning(f"Model snapshot not found at {args.snapshot}. Skipping loading weights.")
        else:
            logger.info(f"Loading model weights from: {args.snapshot}")
            state_dict = torch.load(args.snapshot, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            logger.info("Model weights loaded successfully.")
    else:
        logger.info("No snapshot provided, using randomly initialized model for benchmarking.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"\nTotal Trainable Parameters: {total_params:,}") 
    sys.stdout.flush() 

    logger.info(f"\n--- FPS Calculation (Batch Size: {args.batch_size}, Input: 3x{args.input_size_h}x{args.input_size_w}) ---")
    print(f"\n--- FPS Calculation (Batch Size: {args.batch_size}, Input: 3x{args.input_size_h}x{args.input_size_w}) ---") 
    sys.stdout.flush() 
    
    dummy_input = torch.randn(args.batch_size, 3, args.input_size_h, args.input_size_w).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    logger.info(f"Performing {args.num_warmup} warm-up runs...")
    print(f"Performing {args.num_warmup} warm-up runs...") 
    sys.stdout.flush() 
    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Performing {args.num_runs} actual runs for measurement...")
    print(f"Performing {args.num_runs} actual runs for measurement...") 
    sys.stdout.flush() 
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.num_runs):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_inference_time = total_time / args.num_runs
    fps = args.batch_size / avg_inference_time if avg_inference_time > 0 else float('inf')

    final_benchmark_str = (
        f"Total inference time for {args.num_runs} runs: {total_time:.4f} seconds\n"
        f"Average inference time per batch: {avg_inference_time:.4f} seconds\n"
        f"Frames Per Second (FPS) with batch size {args.batch_size}: {fps:.2f}\n"
    )
    logger.info(final_benchmark_str)
    print(final_benchmark_str) 
    sys.stdout.flush() 

if __name__ == '__main__':
    main()
