# /kaggle/working/ARAA-Net/benchmark_fps.py
# Example usage: !python benchmark_fps.py --backbone resnet50 --dataset-name TSRS_RSNA-Epiphysis --input-h 896 --input-w 576

import torch
import argparse
import time
import os

# --- Add project path to run script from anywhere ---
import sys
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg
import torch.nn.functional as F 

def benchmark(model, device, backbone_name, input_h, input_w):
    """Runs a benchmark for a given model."""
    model.to(device)
    model.eval()

    # Adjust input size based on backbone (and potentially command-line args)
    if backbone_name == 'inception_v3':
        # InceptionV3 typically uses 299x299, override if specified input is not
        if input_h != 299 or input_w != 299:
            print(f"⚠️ Warning: InceptionV3 usually expects 299x299. Benchmarking with {input_h}x{input_w}.")
        dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)
    else:
        dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)


    # --- WARM-UP ---
    print("Performing GPU warm-up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # --- TIMING ---
    print("Starting benchmark (100 inferences)...")
    torch.cuda.synchronize() 
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_frame = total_time / 100
    fps = 1 / avg_time_per_frame
    
    return fps, avg_time_per_frame

def main():
    parser = argparse.ArgumentParser(description='Benchmark model FPS')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'vgg16', 'inception_v3'], help='Backbone to benchmark')
    parser.add_argument('--dataset-name', type=str, default='N/A', help='Name of the dataset being used (for logging purposes, not directly used in benchmark logic)') # Added dataset-name for logging
    parser.add_argument('--input-h', type=int, default=896, help='Height of the dummy input image for benchmarking')
    parser.add_argument('--input-w', type=int, default=576, help='Width of the dummy input image for benchmarking')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ ERROR: A GPU is required for FPS benchmarking. No CUDA device found.")
        return

    device = torch.device("cuda")
    
    # Determine input dimensions based on backbone (and potentially command-line args)
    # The benchmark function will handle the inception_v3 specific override if needed
    input_h_bench = args.input_h
    input_w_bench = args.input_w

    # --- 1. Benchmark Your Model WITH MFR ---
    print(f"--- Benchmarking Model with MultiscaleFeatureRefinement (MFR) module ({args.backbone}) on dataset '{args.dataset_name}' ---")
    model_with_mfr = daseg(backbone_name=args.backbone)
    fps_mfr, time_mfr = benchmark(model_with_mfr, device, args.backbone, input_h_bench, input_w_bench)
    
    # --- 2. Benchmark the Baseline Model WITHOUT MFR ---
    original_daseg_forward = daseg.forward
    
    def forward_no_mfr(self, x):
        original_size = x.size()[2:]
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pos_feat, pred4 = self.positioning(l4)
        
        f3_feat, pred3 = self.focus3(l3, pos_feat, pred4)
        f2_feat, pred2 = self.focus2(l2, f3_feat, pred3) 
        f1_feat, pred1 = self.focus1(l1, f2_feat, pred2) 
        _, pred0 = self.focus0(l0, f1_feat, pred1)
        
        pred4 = F.interpolate(pred4, size=original_size, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=original_size, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=original_size, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, size=original_size, mode='bilinear', align_corners=True)
        pred0 = F.interpolate(pred0, size=original_size, mode='bilinear', align_corners=True)
        return pred4, pred3, pred2, pred1, pred0

    print(f"\n--- Benchmarking Baseline Model WITHOUT MultiscaleFeatureRefinement (MFR) module ({args.backbone}) on dataset '{args.dataset_name}' ---")
    daseg.forward = forward_no_mfr
    
    model_no_mfr = daseg(backbone_name=args.backbone)
    fps_no_mfr, time_no_mfr = benchmark(model_no_mfr, device, args.backbone, input_h_bench, input_w_bench)
    
    daseg.forward = original_daseg_forward

    # --- 3. Print Results ---
    print("\n\n--- FPS Benchmark Results ---")
    print(f"Backbone: {args.backbone}")
    print(f"Dataset (for context): {args.dataset_name}")
    print(f"Input Size for Benchmark: {input_h_bench}x{input_w_bench}")
    print(f"Baseline (No MFR):   {fps_no_mfr:.2f} FPS ({time_no_mfr*1000:.2f} ms/frame)")
    print(f"Your Model (With MFR): {fps_mfr:.2f} FPS ({time_mfr*1000:.2f} ms/frame)")
    print("---------------------------------")
    print(f"Original ARAA-Net paper's reported FPS (for 'ARAA-Net' on ResNet50): 2.33 FPS (Table I in paper)")
    
    if fps_mfr > 2.33: 
        print(f"\n✅ Success! Your model's FPS ({fps_mfr:.2f}) is higher than the original paper's reported FPS (2.33).")
    else:
        print(f"\n⚠️ Note: Your model's FPS ({fps_mfr:.2f}) is lower than or comparable to the paper's reported FPS (2.33). Further optimization might be needed if speed is critical.")
        
    if time_no_mfr > 0: 
        overhead = ((time_mfr - time_no_mfr) / time_no_mfr) * 100
        print(f"\nThe MultiscaleFeatureRefinement (MFR) modules add approximately {overhead:.2f}% time overhead compared to the baseline.")


if __name__ == '__main__':
    main()