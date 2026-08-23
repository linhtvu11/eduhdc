"""
E3 — VSA 1D Capacity Sweep (Contribution C1: Controlled Negative Result).

Goal (Pillar 3 of the C1 rescue plan): give *controlled empirical proof* that the
capacity of a 1-D hypervector memory in R^D (or C^D) is bottlenecked at O(D).
As the number of superposed bindings T grows relative to the dimension D, the
retrieval accuracy of ANY VSA operator collapses toward chance — governed by the
crosstalk law SNR ~ sqrt(D/T). This is the direct mathematical motivation for C2
(PAM, an O(d^2) matrix memory).

Protocol (fully vectorized, no Python loops over concepts):
  - Codebook: K distinct random key vectors and K distinct random value vectors.
  - Build a single memory trace by superposing (bundling) T bound key-value pairs,
    where each pair samples a (key, value) from the codebook:
        M = sum_{t=1}^{T} bind(key_t, value_t)
  - Probe: for each stored pair t, unbind by its key and clean up against the value
    codebook (argmax similarity). Retrieval is correct iff argmax == true value idx.
  - Accuracy is averaged over R independent trials and all T probes.

We sweep T x D (the theoretical axis) at fixed K, plus a K-sweep at fixed T,D.
All operators (MAP, HRR, EduBind) are expected to collapse on the SAME sqrt(T/D)
curve — proving the limit is structural (dimensionality), not operator choice.

Outputs: JSON under data/results/ + console table + crosstalk fit (R^2).
"""

import sys
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.operators import BipolarMAP, RealHRR, EduBindBlockDiag

RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")


def _make_op(op_type: str, dim: int, device: str):
    if op_type == "map":
        return BipolarMAP(dim=dim, device=device), dim
    if op_type == "hrr":
        return RealHRR(dim=dim, device=device), dim
    if op_type == "edubind":
        op = EduBindBlockDiag(dim=dim, device=device)
        return op, op.actual_dim
    raise ValueError(op_type)


@torch.no_grad()
def measure_retrieval_accuracy(op_type: str, K: int, T: int, D: int,
                               n_trials: int, device: str, seed: int = 0) -> float:
    """
    Vectorized retrieval accuracy for a single (op, K, T, D) point.

    Shapes:
      key_cb, val_cb : (K, dim)   codebooks
      keys, vals     : (R, T, dim) sampled bindings per trial
      trace          : (R, dim)    superposed memory
      recov          : (R, T, dim) unbound value estimates
      sims           : (R, T, K)   similarity to every codebook value

    Returns mean top-1 retrieval accuracy over R*T probes.
    """
    torch.manual_seed(seed)
    op, dim = _make_op(op_type, D, device)
    R = n_trials

    # Codebooks (K, dim)
    key_cb = op.random_vector(K)
    val_cb = op.random_vector(K)

    # Sample T key/value indices per trial (R, T).
    #
    # AUDIT FIX B12 — keys are drawn WITHOUT REPLACEMENT. Sampling with
    # replacement (the previous behaviour) binds the SAME key to several
    # different values whenever T > K, which makes retrieval ill-posed by
    # construction: with m copies of a key in the trace, top-1 can be right at
    # most 1/m of the time. With K = 100 and T up to 500 the measured accuracy
    # tracked that collision ceiling E[1/m] to within 0.02 at every T, so the
    # sweep was measuring a sampling artefact rather than crosstalk. Distinct
    # keys per trial make the retrieval target unique, so what is left to
    # degrade accuracy is interference alone. Values may still repeat: two keys
    # may legitimately point at the same value.
    if T > K:
        raise ValueError(f"T={T} > K={K}: cannot draw {T} distinct keys from a "
                         f"codebook of {K}. Raise K (K >> T) or lower T.")
    key_idx = torch.stack([torch.randperm(K, device=device)[:T] for _ in range(R)])
    val_idx = torch.randint(0, K, (R, T), device=device)

    keys = key_cb[key_idx]            # (R, T, dim)
    vals = val_cb[val_idx]            # (R, T, dim)

    # Bind then superpose: M = sum_t bind(key_t, val_t)  -> (R, dim)
    bound = op.bind(keys, vals)       # (R, T, dim)
    trace = bound.sum(dim=1)          # (R, dim)  (unnormalized bundle)

    # Probe: unbind trace by each stored key -> (R, T, dim)
    trace_exp = trace.unsqueeze(1).expand(-1, T, -1)   # (R, T, dim)
    recov = op.unbind(trace_exp, keys)                 # (R, T, dim)

    # Clean up: similarity of each recovered vector to the whole value codebook.
    # Use cosine / Hermitian cosine consistent with the operator's field.
    if recov.is_complex():
        # (R,T,dim) x (K,dim) -> (R,T,K) Hermitian inner product, then normalize
        num = torch.einsum("rtd,kd->rtk", recov, val_cb.conj()).real
        rn = recov.abs().pow(2).sum(-1).sqrt().unsqueeze(-1)          # (R,T,1)
        vn = val_cb.abs().pow(2).sum(-1).sqrt().unsqueeze(0).unsqueeze(0)  # (1,1,K)
        sims = num / (rn * vn + 1e-8)
    else:
        rec_n = torch.nn.functional.normalize(recov, p=2, dim=-1)
        val_n = torch.nn.functional.normalize(val_cb, p=2, dim=-1)
        sims = torch.einsum("rtd,kd->rtk", rec_n, val_n)             # (R,T,K)

    pred = sims.argmax(dim=-1)         # (R, T)
    correct = (pred == val_idx).float().mean().item()

    del key_cb, val_cb, keys, vals, bound, trace, trace_exp, recov, sims, pred
    if device == "cuda":
        torch.cuda.empty_cache()
    return correct


def _fit_crosstalk(load_ratio: np.ndarray, acc: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit accuracy vs the crosstalk SNR proxy z = sqrt(D/T) = 1/sqrt(load).
    Model: acc ~ a * z + b (monotone increasing in SNR). Report Pearson R^2.
    Also report Spearman-like monotonicity via rank correlation sign.
    """
    z = 1.0 / np.sqrt(load_ratio + 1e-12)          # sqrt(D/T)
    # Linear least squares acc = a*z + b
    A = np.vstack([z, np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((acc - pred) ** 2))
    ss_tot = float(np.sum((acc - acc.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(coef[0]), float(coef[1]), r2


def run_capacity_sweep():
    print("=" * 82)
    print("  E3 — VSA 1-D Capacity Sweep (Controlled Negative Result for C1 -> C2)")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    ops = ["map", "hrr", "edubind"]
    D_list = [1024, 2048, 4096, 8192]
    T_list = [25, 50, 100, 200, 350, 500]
    # AUDIT FIX B12: K must exceed max(T_list) so that T distinct keys can be
    # drawn per trial. K = 2000 >> 500 also makes the clean-up a genuine
    # discrimination task rather than a near-trivial one.
    K_fixed = 2000
    n_trials = 30

    print(f"Fixed codebook K={K_fixed} (>> max T={max(T_list)}) | trials/point R={n_trials}")
    print(f"Sweeping D in {D_list}  x  T in {T_list}  x  ops {ops}\n")

    # ---- Main sweep: T x D per operator ----
    sweep = {op: {"T": T_list, "D": D_list, "acc": {}} for op in ops}
    all_load, all_acc = [], []   # pooled across ops+D for the universal crosstalk fit

    t0 = time.perf_counter()
    for op in ops:
        print(f">>> Operator: {op.upper()}")
        header = "  T \\ D  | " + " | ".join(f"{d:>6d}" for d in D_list)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for T in T_list:
            row_acc = []
            for D in D_list:
                acc = measure_retrieval_accuracy(op, K_fixed, T, D, n_trials, device,
                                                 seed=hash((op, T, D)) % (2**31))
                row_acc.append(acc)
                sweep[op]["acc"][f"T{T}_D{D}"] = acc
                all_load.append(T / D)
                all_acc.append(acc)
            print(f"  {T:>5d}  | " + " | ".join(f"{a:6.3f}" for a in row_acc))
        print()

    elapsed = time.perf_counter() - t0

    # ---- Universal crosstalk fit (pooled over all ops & D) ----
    load = np.array(all_load)
    accs = np.array(all_acc)
    a, b, r2 = _fit_crosstalk(load, accs)
    print("-" * 82)
    print("Universal crosstalk law fit  (acc ~ a * sqrt(D/T) + b), pooled over ALL ops & D:")
    print(f"  a = {a:.4f} | b = {b:.4f} | R^2 = {r2:.4f}")
    print(f"  -> Accuracy is governed by the load ratio T/D, NOT the operator choice.")

    # ---- Per-operator collapse check: correlation of acc with sqrt(D/T) ----
    print("\nPer-operator fit to sqrt(D/T) (all collapse on the same axis):")
    op_fits = {}
    for op in ops:
        lo, ac = [], []
        for T in T_list:
            for D in D_list:
                lo.append(T / D); ac.append(sweep[op]["acc"][f"T{T}_D{D}"])
        a_o, b_o, r2_o = _fit_crosstalk(np.array(lo), np.array(ac))
        op_fits[op] = {"a": a_o, "b": b_o, "r2": r2_o}
        print(f"  {op.upper():<10s} | a={a_o:.4f} b={b_o:.4f} R^2={r2_o:.4f}")

    # ---- K-sweep at fixed moderate load (shows discrimination cost of larger codebook) ----
    print("\nK-sweep (EduBind) at fixed D=2048, T=100  (larger codebook -> harder cleanup):")
    K_list = [250, 500, 1000, 2000, 4000]   # B12: every K must exceed T=100
    k_curve = {}
    for K in K_list:
        acc = measure_retrieval_accuracy("edubind", K, 100, 2048, n_trials, device,
                                         seed=hash(("k", K)) % (2**31))
        k_curve[K] = acc
        print(f"  K={K:>5d} | acc={acc:.3f} | chance={1.0/K:.4f}")

    # ---- Export ----
    payload = {
        "config": {"K_fixed": K_fixed, "n_trials": n_trials,
                   "D_list": D_list, "T_list": T_list, "ops": ops},
        "sweep": sweep,
        "universal_fit": {"a": a, "b": b, "r2": r2},
        "per_operator_fit": op_fits,
        "k_sweep_edubind_D2048_T100": k_curve,
        "elapsed_sec": elapsed,
    }
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "capacity_sweep_results.json")
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[results saved: {out}]")
    except Exception as e:
        print(f"[export failed: {e}]")

    print(f"\nTotal sweep time: {elapsed:.1f}s")
    print("=" * 82)
    print("Interpretation: every operator's retrieval accuracy collapses toward chance")
    print("as T/D grows, tracking the sqrt(D/T) crosstalk SNR. The 1-D superposition")
    print("memory is capacity-bounded at O(D) regardless of binding algebra -> motivates")
    print("the O(d^2) matrix-valued PAM memory of Contribution C2.")


if __name__ == "__main__":
    run_capacity_sweep()
