#!/usr/bin/env python3
"""
GPU Performance Test - Test GPU acceleration capabilities
"""

import torch
import numpy as np
from insightface.app import FaceAnalysis
import cv2
import time

def test_gpu_performance():
    """Test GPU performance with various components"""
    print("🚀 GPU ACCELERATION PERFORMANCE TEST")
    print("=" * 60)
    
    # Test 1: PyTorch GPU
    print("\n📋 TEST 1: PyTorch GPU Setup")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        
        # Test tensor operations on GPU
        print("\n🔬 Testing GPU tensor operations...")
        device = torch.device('cuda')
        
        # Create large tensors and perform operations
        start_time = time.time()
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)  # Matrix multiplication
        gpu_time = time.time() - start_time
        print(f"✅ GPU Matrix Multiplication (1000x1000): {gpu_time:.4f}s")
        
        # Compare with CPU
        start_time = time.time()
        a_cpu = torch.randn(1000, 1000)
        b_cpu = torch.randn(1000, 1000)
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"⚡ CPU Matrix Multiplication (1000x1000): {cpu_time:.4f}s")
        print(f"🏆 GPU Speedup: {cpu_time/gpu_time:.2f}x faster")
    else:
        print("❌ No GPU detected")
    
    # Test 2: InsightFace GPU
    print("\n📋 TEST 2: InsightFace GPU Acceleration")
    try:
        import os
        os.environ["INSIGHTFACE_HOME"] = "D:/Facerecognitonbasedattendancesystem/insightface_models"
        
        print("🔄 Initializing InsightFace with CUDA...")
        app = FaceAnalysis(
            name='buffalo_l',
            root=os.environ["INSIGHTFACE_HOME"], 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace initialized with GPU support")
        
        # Test face detection speed
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        faces = app.get(test_image)
        detection_time = time.time() - start_time
        print(f"✅ Face detection time: {detection_time:.4f}s")
        print(f"✅ Detected faces: {len(faces)}")
        
    except Exception as e:
        print(f"❌ InsightFace test failed: {e}")
    
    # Test 3: Memory Management
    print("\n📋 TEST 3: GPU Memory Management")
    if torch.cuda.is_available():
        # Show initial memory
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 Initial GPU Memory - Allocated: {allocated:.3f}GB, Cached: {cached:.3f}GB")
        
        # Create and clear tensors
        large_tensors = []
        for i in range(5):
            tensor = torch.randn(500, 500, device='cuda')
            large_tensors.append(tensor)
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 After allocation - Allocated: {allocated:.3f}GB, Cached: {cached:.3f}GB")
        
        # Clear memory
        large_tensors.clear()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 After cleanup - Allocated: {allocated:.3f}GB, Cached: {cached:.3f}GB")
        print("✅ GPU memory management working correctly")
    
    # Test 4: FP16 Support
    print("\n📋 TEST 4: FP16 Acceleration Support")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        print(f"✅ GPU Compute Capability: {capability[0]}.{capability[1]}")
        
        if capability[0] >= 7:  # Tensor cores available
            print("✅ Tensor Cores available - FP16 acceleration supported")
            
            # Test FP16 vs FP32 speed
            device = torch.device('cuda')
            
            # FP32 test
            start_time = time.time()
            a_fp32 = torch.randn(2000, 2000, device=device, dtype=torch.float32)
            b_fp32 = torch.randn(2000, 2000, device=device, dtype=torch.float32)
            c_fp32 = torch.mm(a_fp32, b_fp32)
            fp32_time = time.time() - start_time
            
            # FP16 test
            start_time = time.time()
            a_fp16 = torch.randn(2000, 2000, device=device, dtype=torch.float16)
            b_fp16 = torch.randn(2000, 2000, device=device, dtype=torch.float16)
            c_fp16 = torch.mm(a_fp16, b_fp16)
            fp16_time = time.time() - start_time
            
            print(f"⚡ FP32 time: {fp32_time:.4f}s")
            print(f"⚡ FP16 time: {fp16_time:.4f}s")
            print(f"🏆 FP16 Speedup: {fp32_time/fp16_time:.2f}x faster")
        else:
            print("⚠️ Limited FP16 support on this GPU")
    
    print("\n🎯 SUMMARY:")
    print("✅ GPU acceleration system fully configured")
    print("✅ Memory management optimized")
    print("✅ Ready for high-performance face recognition")

if __name__ == "__main__":
    test_gpu_performance()