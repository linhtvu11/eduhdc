"""
E2-v2 — 10-Fold KT Benchmark (strengthen p-value for EduHDC vs DKT).

Changes from kt_experiment_rigorous.py:
  - 10-fold CV (doubles degrees of freedom for paired t-test)
  - Excludes HRR-KT (established as weakest VSA, uses slow loop path)
  - 8000 students (more data → lower variance)
  - Focus: EduHDC-KT vs DKT statistical significance
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

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
from eduhdc.kt_experiment_rigorous import _make_batches, train_and_eval_kt_model

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "assistments" / \
            "2012-2013-data-with-predictions-4-final.csv"
RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")


def run_10fold_kt():
    print("=" * 75)
    print("  E2-v2: 10-Fold KT Benchmark (EduHDC vs DKT significance)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    # Load more students for lower statistical power
    sequences_dict, skill_set = load_assistments_real(
        max_students=8000,
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

    # Models: exclude HRR (slow loop, weakest VSA)
    vsa_dim = 2048
    models_to_test = [
        ("DKT (LSTM Baseline)", lambda: DKT_Baseline(num_skills=num_skills, emb_dim=64, hidden_dim=128)),
        ("SAKT (Transformer)", lambda: SAKT_Baseline(num_skills=num_skills, emb_dim=64, n_heads=4)),
        ("simpleKT (Rasch+Att)", lambda: SimpleKT_Baseline(num_skills=num_skills, emb_dim=64, n_heads=4)),
        ("AKT (Monotonic Att)", lambda: AKT_Baseline(num_skills=num_skills, emb_dim=64)),
        ("MAP-KT (Commutative)", lambda: EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="map", device=device)),
        ("EduHDC-KT (Non-Comm)", lambda: EduHDC_KT(num_skills=num_skills, vsa_dim=vsa_dim, op_type="edubind", device=device)),
    ]

    param_counts = {}
    for name, init_fn in models_to_test:
        m = init_fn()
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        param_counts[name] = params

    # 10-Fold CV
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {name: {"aucs": [], "accs": [], "latencies": []} for name, _ in models_to_test}

    print(f"\nRunning {n_folds}-Fold Cross Validation...")
    print("-" * 75)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(student_sequences)):
        train_seqs = [student_sequences[i] for i in train_idx]
        test_seqs = [student_sequences[i] for i in test_idx]
        print(f"\n>>> Fold {fold_idx + 1}/{n_folds} (Train: {len(train_seqs)}, Test: {len(test_seqs)})")

        # added 2026-08 for reproducibility; changes Table 3 numbers vs the preprint —
        # must rerun and re-verify before citing
        torch.manual_seed(42)
        np.random.seed(42)

        for name, init_fn in models_to_test:
            model = init_fn()
            is_neural_baseline = any(kw in name for kw in ["DKT", "SAKT", "simpleKT", "AKT"])
            lr = 0.003 if is_neural_baseline else 0.015
            epochs = 15
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
                use_amp=amp_flag,
                batch_size=32
            )
            _wall = time.perf_counter() - _t0
            results[name]["aucs"].append(auc)
            results[name]["accs"].append(acc)
            results[name]["latencies"].append(lat)
            print(f"  {name:<24s} | AUC: {auc:.4f} | Acc: {acc:.4f} | "
                  f"Lat: {lat:.3f} ms | {_wall:.1f}s", flush=True)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Checkpoint
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
            with open(os.path.join(RESULTS_DIR, "kt_10fold_results.json"), "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  [checkpoint: {fold_idx + 1}/{n_folds}]", flush=True)
        except Exception as e:
            print(f"  [checkpoint failed: {e}]", flush=True)

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Model':<24s} | {'AUC (Mean±Std)':<18s} | {'Accuracy':<10s} | {'Latency':<10s}")
    print("-" * 80)
    for name in results:
        aucs = np.array(results[name]["aucs"])
        accs = np.array(results[name]["accs"])
        lats = np.array(results[name]["latencies"])
        print(f"{name:<24s} | {aucs.mean():.4f}±{aucs.std():.4f} | {accs.mean():.4f}   | {lats.mean():.2f} ms")

    # Statistical tests
    edubind = np.array(results["EduHDC-KT (Non-Comm)"]["aucs"])
    dkt = np.array(results["DKT (LSTM Baseline)"]["aucs"])
    mapkt = np.array(results["MAP-KT (Commutative)"]["aucs"])

    t_dkt, p_dkt = scipy_stats.ttest_rel(edubind, dkt)
    t_map, p_map = scipy_stats.ttest_rel(edubind, mapkt)

    print(f"\nStatistical Significance (paired t-test, {n_folds} folds):")
    print(f"  EduHDC vs DKT:  ΔAUC = {edubind.mean() - dkt.mean():+.4f}, t = {t_dkt:.2f}, p = {p_dkt:.4e}")
    print(f"  EduHDC vs MAP:  ΔAUC = {edubind.mean() - mapkt.mean():+.4f}, t = {t_map:.2f}, p = {p_map:.4e}")


if __name__ == "__main__":
    run_10fold_kt()
