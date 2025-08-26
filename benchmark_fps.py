# /kaggle/working/ARAA-Net/benchmark_fps.py
# !python benchmark_fps.py
import torch
import argparse
import time

# --- Add project path to run script from anywhere ---
import sys
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from daseg import daseg

def benchmark(model, device):
    """Runs a benchmark for a given model."""
    model.to(device)
    model.eval()

    # Create a dummy input tensor
    # Batch size 1, 3 channels (RGB), and the validation image size
    dummy_input = torch.randn(1, 3, 896, 576, dtype=torch.float32).to(device)

    # --- WARM-UP ---
    # The first few inferences are always slower, so we run them once
    # without timing to warm up the GPU.
    print("Performing GPU warm-up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # --- TIMING ---
    # We use torch.cuda.synchronize() to ensure we are measuring
    # the true time the GPU takes to finish, not just the time it
    # takes to launch the kernel. This is critical for accurate timing.
    print("Starting benchmark...")
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
    if not torch.cuda.is_available():
        print("❌ ERROR: A GPU is required for FPS benchmarking.")
        return

    device = torch.device("cuda")
    
    # --- 1. Benchmark Your Model WITH LASA ---
    print("--- Benchmarking Model with LASA module ---")
    model_with_lasa = daseg(backbone_name='resnet50')
    fps_lasa, time_lasa = benchmark(model_with_lasa, device)
    
    # --- 2. Benchmark the Baseline Model WITHOUT LASA ---
    # We will temporarily modify the daseg forward pass in memory to disable LASA
    
    # Keep the original forward method
    original_forward = daseg.forward
    
    # Define a new, simpler forward method that bypasses LASA
    def forward_no_lasa(self, x):
        original_size = x.size()[2:]
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        # BYPASS LASA: Use l1 and l2 directly
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pos_feat, pred4 = self.positioning(l4)
        f3_feat, pred3 = self.focus3(l3, pos_feat, pred4)
        f2_feat, pred2 = self.focus2(l2, f3_feat, pred3) # Use l2
        f1_feat, pred1 = self.focus1(l1, f2_feat, pred2) # Use l1
        _, pred0 = self.focus0(l0, f1_feat, pred1)
        # ... resize predictions
        pred4 = F.interpolate(pred4, size=original_size, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=original_size, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=original_size, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, size=original_size, mode='bilinear', align_corners=True)
        pred0 = F.interpolate(pred0, size=original_size, mode='bilinear', align_corners=True)
        return pred4, pred3, pred2, pred1, pred0

    print("\n--- Benchmarking Baseline Model WITHOUT LASA module ---")
    # Temporarily replace the forward method for benchmarking
    daseg.forward = forward_no_lasa
    import torch.nn.functional as F
    model_no_lasa = daseg(backbone_name='resnet50')
    fps_no_lasa, time_no_lasa = benchmark(model_no_lasa, device)
    
    # Restore the original forward method
    daseg.forward = original_forward

    # --- 3. Print Results ---
    print("\n\n--- FPS Benchmark Results ---")
    print(f"Baseline (No LASA):   {fps_no_lasa:.2f} FPS ({time_no_lasa*1000:.2f} ms/frame)")
    print(f"Your Model (With LASA): {fps_lasa:.2f} FPS ({time_lasa*1000:.2f} ms/frame)")
    print("---------------------------------")
    print(f"Original ARAA-Net paper's reported FPS: 2.33 FPS")
    
    if fps_lasa > 30:
        print("\n✅ Success! Your model's FPS is significantly higher than the paper's claim.")
    else:
        print("\n⚠️ Note: Your model's FPS is lower than expected. Check GPU utilization.")
        
    overhead = ((time_lasa - time_no_lasa) / time_no_lasa) * 100
    print(f"The LASA modules add approximately {overhead:.2f}% time overhead.")


if __name__ == '__main__':
    main()