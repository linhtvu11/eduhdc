"""
Rigorous Neuro-Symbolic Knowledge Tracing Benchmark on Real ASSISTments 2012-2013.
Compares:
  1. Standard DKT (LSTM Baseline)
  2. Commutative MAP-KT (Bipolar MAP + Readout)
  3. Commutative HRR-KT (Real HRR + Readout)
  4. Non-Commutative EduHDC-KT (EduBind + Readout)

Metrics:
  - AUC-ROC & Accuracy (5-Fold Cross Validation)
  - Trainable Parameter Count
  - Inference Latency (ms / student)
  - Statistical Significance (Paired Student's t-test vs Baselines)
"""

import sys
import os
import json
import time
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from scipy import stats as scipy_stats

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.models import EduHDC_KT, DKT_Baseline, SAKT_Baseline, SimpleKT_Baseline, AKT_Baseline
from eduhdc.data_loader_real import load_assistments_real

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "assistments" / \
            "2012-2013-data-with-predictions-4-final.csv"

RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")


# ==============================================================================
# Training & Evaluation Loop for KT
# ==============================================================================

def _make_batches(sequences, skill_to_idx, device, batch_size=64, min_len=5):
    """Collate variable-length student sequences into padded (skills, correct, mask)
    batches on device. Returns a list of (skills, corrects, mask) tuples."""
    seqs = [s for s in sequences if len(s) >= min_len]
    # bucket by length to reduce padding waste
    seqs.sort(key=len)
    batches = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        T = max(len(s) for s in chunk)
        B = len(chunk)
        sk = torch.zeros(B, T, dtype=torch.long, device=device)
        co = torch.zeros(B, T, dtype=torch.long, device=device)
        ma = torch.zeros(B, T, dtype=torch.float32, device=device)
        for b, s in enumerate(chunk):
            L = len(s)
            sk[b, :L] = torch.tensor([skill_to_idx[it['skill']] for it in s],
                                     dtype=torch.long, device=device)
            co[b, :L] = torch.tensor([it['correct'] for it in s],
                                     dtype=torch.long, device=device)
            ma[b, :L] = 1.0
        batches.append((sk, co, ma))
    return batches


def train_and_eval_kt_model(
    model: nn.Module,
    train_sequences: List[Dict],
    test_sequences: List[Dict],
    skill_to_idx: Dict[str, int],
    epochs: int = 15,
    lr: float = 0.005,
    device: str = "cuda",
    val_frac: float = 0.15,
    patience: int = 3,
    batch_size: int = 64,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """
    Trains a KT model on train_sequences, evaluates on test_sequences.
    VECTORIZED: students are processed in padded batches (forward_batch), which is
    ~50-200x faster than per-student forward_sequence.
    D3 FIX: internal validation split + early-stopping on val-AUC (fair to all models).
    Returns: (test_auc, test_acc, latency_ms_per_student)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction="none")

    # --- Internal train/val split for early-stopping ---
    train_sequences = [s for s in train_sequences if len(s) >= 5]
    n_val = max(1, int(len(train_sequences) * val_frac))
    rng = np.random.RandomState(0)
    perm0 = rng.permutation(len(train_sequences))
    val_ids = set(perm0[:n_val].tolist())
    tr_seqs = [s for i, s in enumerate(train_sequences) if i not in val_ids]
    va_seqs = [s for i, s in enumerate(train_sequences) if i in val_ids]

    tr_batches = _make_batches(tr_seqs, skill_to_idx, device, batch_size)
    va_batches = _make_batches(va_seqs, skill_to_idx, device, batch_size)
    te_batches = _make_batches(test_sequences, skill_to_idx, device, batch_size)

    def _forward(sk, co, ma):
        # bf16 autocast: ~2x speed + lower VRAM on GPU for the VSA models (large
        # einsum/matmul). Disabled for LSTM/Transformer baselines whose cuDNN/attention
        # backward kernels are unstable under bf16 autocast (CUBLAS internal errors).
        # NOTE: forward_batch_fast runs in fp32 (no autocast) — the vectorized path
        # is already 10x+ faster than the loop, and bf16 causes NaN with HRR's FFT
        # and the cumsum trick's large intermediate values.
        # HRR excluded from fast path: FFT circular convolution produces intermediate
        # values that overflow the cumsum trick even in fp32.
        if hasattr(model, 'forward_batch_fast') and getattr(model, 'op_type', '') != 'hrr':
            return model.forward_batch_fast(sk, co, ma)
        amp_on = use_amp and device == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_on):
            return model.forward_batch(sk, co, ma)

    def _eval(batches):
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for sk, co, ma in batches:
                p = _forward(sk, co, ma).float()
                m = ma.bool()
                preds.extend(p[m].cpu().numpy().tolist())
                actuals.extend(co.float()[m].cpu().numpy().tolist())
        if len(set(actuals)) <= 1:
            return 0.5, preds, actuals
        return roc_auc_score(actuals, preds), preds, actuals

    # --- Training with early-stopping + curriculum (P0-B) ---
    # Batches are already sorted by ascending sequence length (_make_batches).
    # Curriculum: first 5 epochs ramp from 60% shortest → 100% of batches,
    # giving VSA state trackers simpler patterns to lock onto before harder ones.
    n_batches = len(tr_batches)
    curriculum_epochs = 5

    best_val = -1.0
    best_state = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        if epoch < curriculum_epochs:
            frac = 0.6 + 0.4 * (epoch / max(1, curriculum_epochs - 1))
            n_use = max(1, int(n_batches * frac))
        else:
            n_use = n_batches
        active_batches = tr_batches[:n_use]
        order = np.random.permutation(len(active_batches))
        for bi in order:
            sk, co, ma = active_batches[bi]
            optimizer.zero_grad()
            p = _forward(sk, co, ma).float().clamp(1e-6, 1 - 1e-6)
            loss = criterion(p, co.float())
            loss = (loss * ma).sum() / ma.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_auc, _, _ = _eval(va_batches)
        if val_auc > best_val + 1e-4:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Evaluation on held-out test ---
    model.eval()
    n_test = sum(b[0].shape[0] for b in te_batches)
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    test_auc, all_preds, all_actuals = _eval(te_batches)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    latency_ms = (elapsed / max(1, n_test)) * 1000.0

    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_actuals, binary_preds)

    return test_auc, acc, latency_ms


# ==============================================================================
# Main Benchmark Pipeline
# ==============================================================================

def run_rigorous_kt_benchmark():
    print("=" * 75)
    print("  Rigorous Neuro-Symbolic KT Benchmark on REAL ASSISTments 2012-2013")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    # 1. Load Data (D3 FIX: >=1000 active students with sequences >= 20)
    # D4 FIX: timestamp sort + sliding-window max_seq_len=200 (pyKT standard)
    # D5 FIX: 5000 students to give AKT/simpleKT sufficient data for Rasch embeddings
    sequences_dict, skill_set = load_assistments_real(
        max_students=5000,
        min_seq_len=20,
        max_seq_len=200,
        seed=42
    )

    skill_list = sorted(list(skill_set))
    skill_to_idx = {s: i for i, s in enumerate(skill_list)}
    num_skills = len(skill_list)
    student_sequences = list(sequences_dict.values())

    print(f"Active Skills: {num_skills} | Students: {len(student_sequences)}")
    total_inter = sum(len(s) for s in student_sequences)
    print(f"Total Interactions: {total_inter:,}")

    # 2. Define Model Configurations
    vsa_dim = 2048
    models_to_test = [
        ("DKT (LSTM Baseline)", lambda: DKT_Baseline(num_skills=num_skills, emb_dim=64, hidden_dim=128)),
        ("SAKT (Transformer Baseline)", lambda: SAKT_Baseline(num_skills=num_skills, emb_dim=64, n_heads=4)),
        ("simpleKT (Rasch+Attention)", lambda: SimpleKT_Baseline(num_skills=num_skills, emb_dim=64, n_heads=4)),
        ("AKT (Monotonic Attention)", lambda: AKT_Baseline(num_skills=num_skills, emb_dim=64)),
        ("MAP-KT (Commutative)", lambda: EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="map", device=device)),
        ("HRR-KT (Commutative)", lambda: EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="hrr", device=device)),
        ("EduHDC-KT (Non-Commutative)", lambda: EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="edubind", device=device)),
    ]

    # Measure parameter counts
    param_counts = {}
    for name, init_fn in models_to_test:
        m = init_fn()
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        param_counts[name] = params

    # 3. Run 5-Fold Cross Validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {name: {"aucs": [], "accs": [], "latencies": []} for name, _ in models_to_test}

    print(f"\nRunning {n_folds}-Fold Cross Validation across all architectures...")
    print("-" * 75)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(student_sequences)):
        train_seqs = [student_sequences[i] for i in train_idx]
        test_seqs  = [student_sequences[i] for i in test_idx]
        print(f"\n>>> Fold {fold_idx + 1}/{n_folds} (Train: {len(train_seqs)}, Test: {len(test_seqs)})")

        for name, init_fn in models_to_test:
            model = init_fn()
            # DKT/SAKT/simpleKT/AKT need lower lr for LSTM/Transformer stability; HDC models converge fast.
            # D3 FIX: 15 epochs with early-stopping (patience=3) via val-AUC.
            is_neural_baseline = any(kw in name for kw in ["DKT", "SAKT", "simpleKT", "AKT"])
            lr = 0.003 if is_neural_baseline else 0.015
            epochs = 15
            # bf16 autocast only for VSA models; neural baselines use fp32 (kernel stability)
            amp_flag = not is_neural_baseline

            _t0 = time.perf_counter()
            auc, acc, lat = train_and_eval_kt_model(
                model=model,
                train_sequences=train_seqs,
                test_sequences=test_seqs,
                skill_to_idx=skill_to_idx,
                epochs=epochs,
                lr=lr,
                device=device,
                use_amp=amp_flag
            )
            _wall = time.perf_counter() - _t0
            results[name]["aucs"].append(auc)
            results[name]["accs"].append(acc)
            results[name]["latencies"].append(lat)
            print(f"  {name:<28s} | AUC: {auc:.4f} | Acc: {acc:.4f} | "
                  f"Latency: {lat:.3f} ms/std | train {_wall:.1f}s", flush=True)

            # VRAM cleanup between models to prevent transient-peak crashes
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # JSON checkpoint after each fold — partial results survive any crash
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            ckpt = {
                "completed_folds": fold_idx + 1,
                "n_folds": n_folds,
                "num_skills": num_skills,
                "num_students": len(student_sequences),
                "total_interactions": total_inter,
                "param_counts": param_counts,
                "results": results,
            }
            with open(os.path.join(RESULTS_DIR, "kt_benchmark_results.json"), "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  [checkpoint saved: {fold_idx + 1}/{n_folds} folds]", flush=True)
        except Exception as e:
            print(f"  [checkpoint failed: {e}]", flush=True)

    # 4. Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<28s} | {'AUC-ROC (Mean ± Std)':<20s} | {'Accuracy':<12s} | {'Params':<10s} | {'Latency':<10s}")
    print("-" * 80)

    for name in results:
        aucs = np.array(results[name]["aucs"])
        accs = np.array(results[name]["accs"])
        lats = np.array(results[name]["latencies"])
        p_cnt = f"{param_counts[name]:,}"
        print(f"{name:<28s} | {aucs.mean():.4f} ± {aucs.std():.4f}       | {accs.mean():.4f}     | {p_cnt:<10s} | {lats.mean():.2f} ms")

    print("-" * 80)

    # 5. Statistical Significance vs Baselines
    edubind_aucs = np.array(results["EduHDC-KT (Non-Commutative)"]["aucs"])
    map_aucs     = np.array(results["MAP-KT (Commutative)"]["aucs"])
    dkt_aucs     = np.array(results["DKT (LSTM Baseline)"]["aucs"])

    t_vs_map, p_vs_map = scipy_stats.ttest_rel(edubind_aucs, map_aucs)
    t_vs_dkt, p_vs_dkt = scipy_stats.ttest_rel(edubind_aucs, dkt_aucs)

    print("\nStatistical Significance (Paired Student's t-test over 5 Folds):")
    print(f"  EduHDC-KT vs MAP-KT: delta = +{edubind_aucs.mean() - map_aucs.mean():.4f} AUC, t = {t_vs_map:.2f}, p-value = {p_vs_map:.4e}")
    print(f"  EduHDC-KT vs DKT:    delta = {edubind_aucs.mean() - dkt_aucs.mean():.4f} AUC, t = {t_vs_dkt:.2f}, p-value = {p_vs_dkt:.4e}")


if __name__ == "__main__":
    run_rigorous_kt_benchmark()
