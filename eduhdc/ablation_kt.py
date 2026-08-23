"""
E5 — KT Ablation Sweep (Contribution C1: rescue the parameter-reduction claim).

Question: can a *lighter* EduHDC-KT (smaller vsa_dim and/or frozen VSA codebook)
keep AUC >= DKT while dropping trainable params BELOW DKT's 163K? The full config
(vsa_dim=2048, learned codebook) costs ~515K params > DKT, so the "10-50x fewer
params" claim is currently unmet. This sweep tests the honest boundary.

Design (fully vectorized, reuses the rigorous KT harness):
  - Grid: op_type in {edubind, map} x vsa_dim in {256, 512, 1024, 2048}
          x freeze_codebook in {False, True}
  - DKT baseline for reference.
  - K-fold student-level CV (default 3 folds for speed), early-stopping on val-AUC.
  - Report per-config: mean AUC/Acc, trainable #params, params-ratio vs DKT, latency.
Exports JSON to data/results/ablation_kt_results.json.
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.models import EduHDC_KT, DKT_Baseline, SAKT_Baseline
from eduhdc.data_loader_real import load_assistments_real
from eduhdc.kt_experiment_rigorous import train_and_eval_kt_model

RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "results")


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_ablation():
    print("=" * 82)
    print("  E5 — KT Ablation: lighter EduHDC-KT vs DKT (parameter-reduction probe)")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    sequences_dict, skill_set = load_assistments_real(
        max_students=1000, min_seq_len=20, seed=42)
    skill_list = sorted(list(skill_set))
    skill_to_idx = {s: i for i, s in enumerate(skill_list)}
    num_skills = len(skill_list)
    student_sequences = list(sequences_dict.values())
    print(f"Active Skills: {num_skills} | Students: {len(student_sequences)} | "
          f"Interactions: {sum(len(s) for s in student_sequences):,}")

    # Reference: DKT
    dkt_params = count_trainable(DKT_Baseline(num_skills=num_skills, emb_dim=64, hidden_dim=128))
    print(f"DKT trainable params (reference): {dkt_params:,}\n")

    # Ablation grid
    op_types = ["edubind", "map"]
    vsa_dims = [256, 512, 1024, 2048]
    freezes = [False, True]

    configs = [("DKT (LSTM Baseline)", lambda: DKT_Baseline(
        num_skills=num_skills, emb_dim=64, hidden_dim=128), "dkt", None, None),
        ("SAKT (Transformer Baseline)", lambda: SAKT_Baseline(
        num_skills=num_skills, emb_dim=64, n_heads=4), "sakt", None, None)]
    for op in op_types:
        for d in vsa_dims:
            for fz in freezes:
                tag = f"{op}-d{d}-{'frozen' if fz else 'learned'}"
                configs.append((tag,
                    (lambda op=op, d=d, fz=fz: EduHDC_KT(
                        num_skills=num_skills, vsa_dim=d, op_type=op,
                        device=device, freeze_codebook=fz)),
                    op, d, fz))

    n_folds = 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(student_sequences))

    results = {}
    t0 = time.perf_counter()
    for name, init_fn, op, d, fz in configs:
        m0 = init_fn()
        nparams = count_trainable(m0)
        del m0
        lr = 0.003 if op in ("dkt", "sakt") else 0.015
        amp_flag = op not in ("dkt", "sakt")  # bf16 only for VSA models
        aucs, accs, lats = [], [], []
        for fi, (tr, te) in enumerate(folds):
            train_seqs = [student_sequences[i] for i in tr]
            test_seqs = [student_sequences[i] for i in te]
            model = init_fn()
            auc, acc, lat = train_and_eval_kt_model(
                model, train_seqs, test_seqs, skill_to_idx,
                epochs=15, lr=lr, device=device, use_amp=amp_flag)
            aucs.append(auc); accs.append(acc); lats.append(lat)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        results[name] = {
            "op": op, "vsa_dim": d, "frozen": fz,
            "params": nparams, "params_ratio_vs_dkt": nparams / dkt_params,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)), "latency_ms": float(np.mean(lats)),
            "aucs": [float(x) for x in aucs],
        }
        beat = "YES" if np.mean(aucs) >= results["DKT (LSTM Baseline)"]["auc_mean"] else "no" \
            if "DKT (LSTM Baseline)" in results else "-"
        lighter = "LIGHTER" if nparams < dkt_params else "heavier"
        print(f"  {name:<26s} | AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
              f"params {nparams:>7,} ({nparams/dkt_params:4.2f}x, {lighter}) | "
              f"beats DKT: {beat}")

    elapsed = time.perf_counter() - t0

    # Highlight: lightest config that still beats DKT
    dkt_auc = results["DKT (LSTM Baseline)"]["auc_mean"]
    winners = [(n, r) for n, r in results.items()
               if r["op"] != "dkt" and r["auc_mean"] >= dkt_auc and r["params"] < dkt_params]
    winners.sort(key=lambda kv: kv[1]["params"])
    print("\n" + "-" * 82)
    if winners:
        n, r = winners[0]
        print(f"[WIN] Lightest EduHDC-KT beating DKT: {n}")
        print(f"      AUC {r['auc_mean']:.4f} >= DKT {dkt_auc:.4f} | "
              f"params {r['params']:,} = {r['params_ratio_vs_dkt']:.2f}x DKT "
              f"({1/r['params_ratio_vs_dkt']:.1f}x FEWER)")
    else:
        print("[HONEST] No lighter-than-DKT config reached DKT AUC. "
              "-> retract param-reduction claim; keep 'AUC >= DKT + O(1) readout'.")

    payload = {
        "config": {"n_folds": n_folds, "num_skills": num_skills,
                   "num_students": len(student_sequences),
                   "op_types": op_types, "vsa_dims": vsa_dims, "freezes": freezes,
                   "dkt_params": dkt_params},
        "results": results,
        "winners_lighter_and_better": [w[0] for w in winners],
        "elapsed_sec": elapsed,
    }
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "ablation_kt_results.json")
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[results saved: {out}]")
    except Exception as e:
        print(f"[export failed: {e}]")
    print(f"Total ablation time: {elapsed:.1f}s")
    print("=" * 82)


if __name__ == "__main__":
    run_ablation()

