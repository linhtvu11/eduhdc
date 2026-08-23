"""
E3-v4 — P3 DEEP INVESTIGATION: Is the O(D) negative result fundamental or an
artifact of naive retrieval?

The current negative result uses ONE-PASS unbind + argmax cleanup (the simplest
retrieval). The user suspects untapped potential. We test progressively stronger
retrieval to see whether capacity improves:

  L0  One-pass unbind + argmax            (baseline, reproduces O(D))
  L1  Iterative/resonant cleanup          (few refinement passes)
  L2  Successive Interference Cancellation (decode item, subtract, decode next)

Key scientific question:
  - If better retrieval improves only the CONSTANT (curve shifts up but still
    collapses on sqrt(D/T)), the O(D) limit is FUNDAMENTAL -> C2 motivation holds,
    and we've quantified the "retrieval gap" (potential the user sensed).
  - If better retrieval changes the SCALING (no longer collapses on sqrt(D/T)),
    the negative result is an ARTIFACT -> must revise the thesis story.

We measure retrieval accuracy vs load T/D for each level and fit the crosstalk
curve, comparing scaling exponents.
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.operators import BipolarMAP, RealHRR, EduBindBlockDiag

RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")



def _stable_seed(*parts) -> int:
    """Deterministic seed. `hash()` of a tuple containing a str is randomised
    per process (PYTHONHASHSEED), so the previous `hash(...)` seeds made this
    sweep irreproducible run to run. CRC32 of the repr is stable."""
    import zlib
    return zlib.crc32(repr(parts).encode()) % (2 ** 31)

def _make_op(op_type, dim, device):
    if op_type == "map":
        return BipolarMAP(dim=dim, device=device), dim
    if op_type == "hrr":
        return RealHRR(dim=dim, device=device), dim
    op = EduBindBlockDiag(dim=dim, device=device)
    return op, op.actual_dim


def _sims(recov, val_cb):
    if recov.is_complex():
        num = torch.einsum("...d,kd->...k", recov, val_cb.conj()).real
        rn = recov.abs().pow(2).sum(-1, keepdim=True).sqrt()
        vn = val_cb.abs().pow(2).sum(-1).sqrt().view(1, 1)
        return num / (rn * vn + 1e-8)
    rec_n = F.normalize(recov, p=2, dim=-1)
    val_n = F.normalize(val_cb, p=2, dim=-1)
    return torch.einsum("...d,kd->...k", rec_n, val_n)


@torch.no_grad()
def measure(op_type, K, T, D, n_trials, device, level=0, seed=0):
    torch.manual_seed(seed)
    op, dim = _make_op(op_type, D, device)
    R = n_trials

    key_cb = op.random_vector(K)
    val_cb = op.random_vector(K)
    # AUDIT FIX B12: distinct keys per trial (see capacity_sweep.py for why).
    if T > K:
        raise ValueError(f"T={T} > K={K}: cannot draw {T} distinct keys.")
    key_idx = torch.stack([torch.randperm(K, device=device)[:T] for _ in range(R)])
    val_idx = torch.randint(0, K, (R, T), device=device)
    keys = key_cb[key_idx]   # (R,T,dim)
    vals = val_cb[val_idx]

    bound = op.bind(keys, vals)          # (R,T,dim)
    trace = bound.sum(dim=1)             # (R,dim)

    if level == 0:
        # One-pass
        recov = op.unbind(trace.unsqueeze(1).expand(-1, T, -1), keys)
        pred = _sims(recov, val_cb).argmax(-1)
        return (pred == val_idx).float().mean().item()

    if level == 1:
        # Iterative resonant: refine each probe by re-projecting onto codebook
        recov = op.unbind(trace.unsqueeze(1).expand(-1, T, -1), keys)
        for _ in range(3):
            # clean to nearest codebook value, then re-add residual consistency
            idx = _sims(recov, val_cb).argmax(-1)              # (R,T)
            clean = val_cb[idx]                                 # (R,T,dim)
            # blend: keep cleaned estimate (resonant convergence)
            recov = clean
        pred = _sims(recov, val_cb).argmax(-1)
        return (pred == val_idx).float().mean().item()

    if level == 2:
        # Successive Interference Cancellation: decode in order, subtract contribution
        cur = trace.clone()                                    # (R,dim)
        correct = 0
        total = 0
        for t in range(T):
            k_t = keys[:, t]                                   # (R,dim)
            recov = op.unbind(cur, k_t)                        # (R,dim)
            pred = _sims(recov.unsqueeze(1), val_cb).argmax(-1).squeeze(-1)  # (R,)
            correct += (pred == val_idx[:, t]).sum().item()
            total += R
            # subtract decoded contribution
            dec = op.bind(k_t, val_cb[pred])                   # (R,dim)
            cur = cur - dec
        return correct / total

    raise ValueError(level)


def fit_scaling(load, acc):
    """Fit acc ~ f(sqrt(D/T)); return linear R^2 on z and whether monotone decay."""
    z = 1.0 / np.sqrt(load + 1e-12)
    A = np.vstack([z, np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((acc - pred) ** 2) / (np.sum((acc - acc.mean()) ** 2) + 1e-12)
    return float(r2)


def main():
    print("=" * 86)
    print("  E3-v4 — P3 Deep Investigation: retrieval gap vs fundamental O(D) limit")
    print("=" * 86)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    D = 2048
    K = 2000   # B12: must exceed max(T_list)=500
    T_list = [25, 50, 100, 200, 350, 500]
    n_trials = 20
    ops = ["map", "edubind"]

    results = {}
    for op in ops:
        print(f"\n>>> Operator: {op.upper()}  (D={D}, K={K})")
        print(f"  {'T':>5s} | {'L0 one-pass':>11s} | {'L1 resonant':>11s} | {'L2 SIC':>8s}")
        accs = {0: [], 1: [], 2: []}
        for T in T_list:
            row = []
            for lv in [0, 1, 2]:
                a = measure(op, K, T, D, n_trials, device, level=lv, seed=_stable_seed(op, T, lv))
                accs[lv].append(a)
                row.append(a)
            print(f"  {T:>5d} | {row[0]:11.3f} | {row[1]:11.3f} | {row[2]:8.3f}")
        load = np.array([T / D for T in T_list])
        r2 = {lv: fit_scaling(load, np.array(accs[lv])) for lv in [0, 1, 2]}
        results[op] = {"T": T_list, "L0": accs[0], "L1": accs[1], "L2": accs[2],
                       "r2_L0": r2[0], "r2_L1": r2[1], "r2_L2": r2[2]}
        # Gain at high load (T=200) from better retrieval
        gain = accs[2][3] - accs[0][3]
        print(f"  R^2(sqrt(D/T)): L0={r2[0]:.3f} L1={r2[1]:.3f} L2={r2[2]:.3f} | SIC gain @T=200: {gain:+.3f}")

    print("\n" + "=" * 86)
    print("INTERPRETATION:")
    for op in ops:
        r = results[op]
        still_collapses = r["L2"][-1] < 0.5  # at T=500, does SIC still collapse?
        print(f"  {op}: L2(SIC) at T=500 = {r['L2'][-1]:.3f} -> "
              + ("STILL collapses (O(D) fundamental)" if still_collapses else "does NOT collapse (artifact!)"))

    with open(Path(RESULTS_DIR) / "capacity_deep_investigation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved: {RESULTS_DIR}\\capacity_deep_investigation.json]")


if __name__ == "__main__":
    main()
