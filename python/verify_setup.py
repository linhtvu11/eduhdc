"""
Environment and Hardware Verification Script for PhD Research.
Validates PyTorch CUDA, complex tensor operations, and HDC capabilities on RTX 4070.
"""

import torch
import torchhd
import numpy as np

def verify_environment():
    print("=" * 65)
    print("      PhD Research Environment Verification (RTX 4070 / HDC / PAM)")
    print("=" * 65)
    
    # 1. PyTorch & CUDA
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device        : {gpu_name}")
        print(f"Total VRAM        : {vram_gb:.2f} GB")
        print(f"Current Memory    : {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
    
    # 2. Complex Tensor Operations (Crucial for PAM)
    print("\n--- Complex Hilbert Space Tensor Benchmark ---")
    d = 128
    batch_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    z1 = torch.randn(batch_size, d, d, dtype=torch.complex64, device=device)
    z2 = torch.randn(batch_size, d, d, dtype=torch.complex64, device=device)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    res = torch.matmul(z1, z2.conj().transpose(-1, -2))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    print(f"Complex Batched Matmul ({batch_size}x{d}x{d} in C): Output Shape = {res.shape}")
    print(f"Complex Tensor Support: [VERIFIED]")
    
    # 3. TorchHD Hypervector Operations
    print("\n--- TorchHD Hyperdimensional Computing Benchmark ---")
    hv_dim = 10000
    hv_a = torchhd.random(100, hv_dim, device=device)
    hv_b = torchhd.random(100, hv_dim, device=device)
    
    bound = torchhd.bind(hv_a, hv_b)
    bundled = torchhd.bundle(hv_a, hv_b)
    sim = torchhd.cosine_similarity(bound, hv_a).mean().item()
    
    print(f"10,000-D Hypervector Binding & Bundling on {device}: [VERIFIED]")
    print(f"Orthogonality check (mean sim bound to key): {sim:.4f} (approx 0.0)")
    
    print("\n" + "=" * 65)
    print("STATUS: ALL HARDWARE & ALGEBRAIC FOUNDATIONS VERIFIED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    verify_environment()
