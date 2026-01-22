#!/usr/bin/env python3
"""
Test proper FP16 inference implementation
"""

import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

def test_fp16_inference():
    """Test proper FP16 inference with autocast"""
    print("🔬 Testing Proper FP16 Inference Implementation")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping FP16 test")
        return
    
    device = torch.device('cuda')
    print(f"✅ Using device: {device}")
    
    # Create a simple test model (same architecture as deepfake detector)
    print("📦 Creating test ResNet-50 model...")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    # Move to GPU and set to eval mode (inference mode)
    model = model.to(device)
    model.eval()  # CRITICAL: Set to inference mode
    print("✅ Model in eval mode:", not model.training)
    
    # Test data
    test_input = torch.randn(1, 3, 224, 224, device=device)
    
    print("\\n🧪 Testing Inference Modes:")
    
    # Test 1: Regular FP32 inference
    print("\\n1️⃣ FP32 Inference:")
    model.float()  # Ensure FP32
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output_fp32 = model(test_input)
        end_time.record()
        torch.cuda.synchronize()
        
        fp32_time = start_time.elapsed_time(end_time)
        print(f"✅ FP32 inference time: {fp32_time:.2f}ms")
        print(f"✅ Output shape: {output_fp32.shape}")
        print(f"✅ Output dtype: {output_fp32.dtype}")
    
    # Test 2: Proper FP16 inference with autocast
    print("\\n2️⃣ FP16 Autocast Inference (RECOMMENDED):")
    model.float()  # Keep model in FP32
    with torch.no_grad():  # Disable gradients for inference
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.cuda.amp.autocast():  # Use autocast for FP16
            output_autocast = model(test_input)
        end_time.record()
        torch.cuda.synchronize()
        
        autocast_time = start_time.elapsed_time(end_time)
        print(f"✅ Autocast inference time: {autocast_time:.2f}ms")
        print(f"✅ Output shape: {output_autocast.shape}")
        print(f"✅ Output dtype: {output_autocast.dtype}")
    
    # Test 3: Full model FP16 (NOT recommended for inference)
    print("\\n3️⃣ Full Model FP16 (NOT RECOMMENDED):")
    try:
        model_half = model.half()  # Convert entire model to FP16
        test_input_half = test_input.half()
        
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output_half = model_half(test_input_half)
            end_time.record()
            torch.cuda.synchronize()
            
            half_time = start_time.elapsed_time(end_time)
            print(f"⚠️ Full FP16 inference time: {half_time:.2f}ms")
            print(f"⚠️ Output shape: {output_half.shape}")
            print(f"⚠️ Output dtype: {output_half.dtype}")
            print("❌ This method can cause numerical instability")
    except Exception as e:
        print(f"❌ Full FP16 failed: {e}")
    
    # Performance comparison
    print("\\n🏆 PERFORMANCE COMPARISON:")
    if 'autocast_time' in locals() and 'fp32_time' in locals():
        speedup = fp32_time / autocast_time
        print(f"📊 FP32 time: {fp32_time:.2f}ms")
        print(f"📊 Autocast time: {autocast_time:.2f}ms")
        print(f"🚀 Speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("✅ Autocast provides significant speedup!")
        else:
            print("ℹ️ Speedup may vary with larger models")
    
    # Test numerical consistency
    print("\\n🧮 NUMERICAL CONSISTENCY:")
    if 'output_fp32' in locals() and 'output_autocast' in locals():
        diff = torch.abs(output_fp32 - output_autocast).max().item()
        print(f"📏 Max difference between FP32 and autocast: {diff:.6f}")
        if diff < 1e-3:
            print("✅ Outputs are numerically consistent")
        else:
            print("⚠️ Significant numerical differences detected")
    
    print("\\n🎯 BEST PRACTICES SUMMARY:")
    print("✅ Use model.eval() for inference")
    print("✅ Use torch.no_grad() to disable gradients") 
    print("✅ Use torch.cuda.amp.autocast() for FP16")
    print("❌ Avoid model.half() for inference")
    print("✅ Keep model in FP32, let autocast handle precision")

if __name__ == "__main__":
    test_fp16_inference()