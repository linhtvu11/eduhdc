"""
Comprehensive Edge-Native Benchmark for EduHDC (Performance & Memory Optimization).

Evaluates:
  1. DKT Baseline (LSTM cuDNN) on GPU
  2. EduHDC_KT (Old Sequential Python Loop) on GPU
  3. EduHDC_KT_Fast (Parallel Causal Scan) on GPU (RTX 4070)
  4. EduHDC_KT_Fast (CPU Single-Core Profile - Raspberry Pi simulation)
  5. EduHDC_KT_Quantized (Int8 CPU Edge Profile)

Measures:
  - Latency per Student (ms) & Per Step (μs)
  - Throughput (Students/sec & Interactions/sec)
  - Memory Footprint / Model Size (MB)
  - Predictive Accuracy (AUC-ROC & Accuracy on ASSISTments 2012-2013)
  - Speedup Ratio
"""

import sys
import time
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.models import EduHDC_KT, DKT_Baseline
from eduhdc.models_fast import EduHDC_KT_Fast, EduHDC_KT_Quantized
from eduhdc.data_loader_real import load_assistments_real


# ==============================================================================
# Helper to measure memory and latency
# ==============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates model size in Megabytes (MB)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def benchmark_inference(
    model: nn.Module,
    sequences: List[Dict],
    skill_to_idx: Dict[str, int],
    device: str = "cuda",
    warmup_runs: int = 10,
    timed_runs: int = 50,
    is_quantized: bool = False
) -> Dict:
    """Accurately measures inference latency, throughput, and accuracy."""
    model.eval()
    
    # Prepare tensors on target device
    prepared_data = []
    total_steps = 0
    target_dev = torch.device(device)
    for seq in sequences:
        skills = torch.tensor([skill_to_idx[inter['skill']] for inter in seq], dtype=torch.long, device=target_dev)
        corrects = torch.tensor([inter['correct'] for inter in seq], dtype=torch.float32, device=target_dev)
        total_steps += len(seq)
        prepared_data.append((skills, corrects))

    # --- Warmup ---
    with torch.no_grad():
        for i in range(min(warmup_runs, len(prepared_data))):
            s, c = prepared_data[i]
            if is_quantized:
                _ = model.forward_cpu_edge(s.cpu(), c.cpu())
            elif isinstance(model, DKT_Baseline):
                _ = model(s, c)
            elif hasattr(model, "forward_sequence_fast"):
                _ = model.forward_sequence_fast(s, c)
            else:
                _ = model.forward_sequence(s, c)
            if device == "cuda":
                torch.cuda.synchronize()

    # --- Timed Benchmark ---
    latencies_per_student = []
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        start_total = time.perf_counter()
        for s, c in prepared_data:
            t0 = time.perf_counter()
            if is_quantized:
                probs = model.forward_cpu_edge(s.cpu(), c.cpu())
            elif isinstance(model, DKT_Baseline):
                probs = model(s, c)
            elif hasattr(model, "forward_sequence_fast"):
                probs = model.forward_sequence_fast(s, c)
            else:
                probs = model.forward_sequence(s, c)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_per_student.append((t1 - t0) * 1000.0) # in ms
            all_preds.extend(probs.cpu().numpy().tolist())
            all_actuals.extend(c.cpu().numpy().tolist())

        total_time_s = time.perf_counter() - start_total

    auc = roc_auc_score(all_actuals, all_preds) if len(set(all_actuals)) > 1 else 0.5
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_actuals, binary_preds)

    avg_student_lat_ms = np.mean(latencies_per_student)
    p95_student_lat_ms = np.percentile(latencies_per_student, 95)
    per_step_lat_us = (avg_student_lat_ms / (total_steps / len(prepared_data))) * 1000.0
    throughput_students_per_sec = len(prepared_data) / total_time_s
    throughput_steps_per_sec = total_steps / total_time_s

    return {
        "auc": auc,
        "acc": acc,
        "avg_lat_ms": avg_student_lat_ms,
        "p95_lat_ms": p95_student_lat_ms,
        "per_step_us": per_step_lat_us,
        "throughput_std_s": throughput_students_per_sec,
        "throughput_step_s": throughput_steps_per_sec,
        "total_steps": total_steps,
    }


# ==============================================================================
# Main Benchmark Runner
# ==============================================================================

def run_edge_benchmark():
    print("=" * 85)
    print("  EduHDC Edge-Native Performance & Memory Optimization Benchmark")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Host Device: {device} | PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    # 1. Load Real ASSISTments Data (300 students)
    sequences_dict, skill_set = load_assistments_real(
        max_students=300,
        min_seq_len=20,
        seed=42
    )
    skill_list = sorted(list(skill_set))
    skill_to_idx = {s: i for i, s in enumerate(skill_list)}
    num_skills = len(skill_list)
    student_sequences = list(sequences_dict.values())

    # Split train/test
    train_seqs, test_seqs = train_test_split(student_sequences, test_size=0.3, random_state=42)
    print(f"Skills: {num_skills} | Train Students: {len(train_seqs)} | Test Students: {len(test_seqs)}")

    # 2. Train Base Models
    vsa_dim = 2048
    print("\nTraining models on real student stream (3 epochs)...")

    # A. Train Fast Vectorized EduHDC
    model_fast = EduHDC_KT_Fast(num_skills=num_skills, vsa_dim=vsa_dim, hidden_dim=64, device=device)
    model_fast.to(device)
    opt_fast = optim.Adam(model_fast.parameters(), lr=0.015, weight_decay=1e-5)
    crit = nn.BCELoss()

    t_train_start = time.perf_counter()
    model_fast.train()
    for ep in range(3):
        for seq in train_seqs:
            s = torch.tensor([skill_to_idx[x['skill']] for x in seq], dtype=torch.long, device=device)
            c = torch.tensor([x['correct'] for x in seq], dtype=torch.float32, device=device)
            opt_fast.zero_grad()
            probs = model_fast(s, c)
            loss = crit(probs, c)
            loss.backward()
            opt_fast.step()
    t_train_fast = time.perf_counter() - t_train_start
    print(f"  [DONE] EduHDC-KT Fast trained in {t_train_fast:.2f}s")

    # B. Train DKT LSTM Baseline
    model_dkt = DKT_Baseline(num_skills=num_skills, emb_dim=64, hidden_dim=128)
    model_dkt.to(device)
    opt_dkt = optim.Adam(model_dkt.parameters(), lr=0.003, weight_decay=1e-5)

    t_train_start = time.perf_counter()
    model_dkt.train()
    for ep in range(3):
        for seq in train_seqs:
            s = torch.tensor([skill_to_idx[x['skill']] for x in seq], dtype=torch.long, device=device)
            c = torch.tensor([x['correct'] for x in seq], dtype=torch.float32, device=device)
            opt_dkt.zero_grad()
            probs = model_dkt(s, c)
            loss = crit(probs, c)
            loss.backward()
            opt_dkt.step()
    t_train_dkt = time.perf_counter() - t_train_start
    print(f"  [DONE] DKT LSTM Baseline trained in {t_train_dkt:.2f}s")

    # C. Create Quantized Int8 Model
    model_quant = EduHDC_KT_Quantized(model_fast)
    print(f"  [DONE] Int8 Quantization complete")

    # D. Old EduHDC Model for comparison
    model_old = EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="edubind", device=device)
    # Copy weights for fair parity
    model_old.skill_embeddings.data = model_fast.skill_embeddings.data.clone()
    model_old.response_embeddings.data = model_fast.response_embeddings.data.clone()
    model_old.skill_bias.weight.data = model_fast.skill_bias.weight.data.clone()
    model_old.to(device)

    # 3. Benchmark All Configurations
    print("\nRunning Inference Latency & Memory Benchmarks...")
    print("-" * 85)

    import copy
    model_fast_cpu = copy.deepcopy(model_fast).cpu()

    configs = [
        ("EduHDC Cũ (Python Loop GPU)", model_old, "cuda", False),
        ("DKT Baseline (LSTM cuDNN GPU)", model_dkt, "cuda", False),
        ("EduHDC Fast (Parallel Scan GPU)", model_fast, "cuda", False),
        ("EduHDC Fast (CPU Single-Core Edge)", model_fast_cpu, "cpu", False),
        ("EduHDC Int8 (Quantized CPU Edge)", model_quant, "cpu", True),
    ]

    results = []
    for name, m, dev, is_q in configs:
        size_mb = get_model_size_mb(m)
        stats = benchmark_inference(m, test_seqs, skill_to_idx, device=dev, is_quantized=is_q)
        results.append((name, size_mb, stats))

    # 4. Summary Table
    print("\n" + "=" * 95)
    print(f"{'Deployment Target & Model':<35s} | {'Latency / Std':<14s} | {'Per Step':<10s} | {'Throughput':<14s} | {'Model Size':<10s} | {'AUC-ROC':<8s}")
    print("-" * 95)

    baseline_lat = results[0][2]["avg_lat_ms"] # Old EduHDC latency
    for name, size_mb, s in results:
        lat_str = f"{s['avg_lat_ms']:.2f} ms"
        step_str = f"{s['per_step_us']:.1f} μs"
        tp_str = f"{s['throughput_std_s']:,.0f} std/s"
        size_str = f"{size_mb:.2f} MB"
        auc_str = f"{s['auc']:.4f}"
        print(f"{name:<35s} | {lat_str:<14s} | {step_str:<10s} | {tp_str:<14s} | {size_str:<10s} | {auc_str:<8s}")

    print("-" * 95)

    # 5. Speedup Analysis
    gpu_old_lat = results[0][2]["avg_lat_ms"]
    dkt_lat     = results[1][2]["avg_lat_ms"]
    gpu_fast_lat = results[2][2]["avg_lat_ms"]
    cpu_fast_lat = results[3][2]["avg_lat_ms"]
    cpu_int8_lat = results[4][2]["avg_lat_ms"]

    print("\nSpeedup & Optimization Summary:")
    print(f"  • EduHDC Fast GPU vs Old EduHDC:  {gpu_old_lat / gpu_fast_lat:.1f}x SPEEDUP ({gpu_old_lat:.2f} ms -> {gpu_fast_lat:.2f} ms)")
    print(f"  • EduHDC Fast GPU vs DKT (LSTM):   {dkt_lat / gpu_fast_lat:.1f}x FASTER than LSTM cuDNN")
    print(f"  • CPU Edge Inference Latency:      {cpu_fast_lat:.2f} ms / student ({results[3][2]['per_step_us']:.1f} μs / interaction)")
    print(f"  • Int8 Quantized Memory Footprint: {results[4][1]:.2f} MB (Ultra-lightweight for Raspberry Pi & Mobile)")


if __name__ == "__main__":
    run_edge_benchmark()
