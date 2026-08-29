"""
E6 — Precision Fidelity of VSA bind/unbind under int8 / bf16 / fp16 (Contribution C1).

Research question (per prior HDC quantization literature): does low-precision
storage — especially int8 — DEGRADE the algebraic fidelity of bind/unbind? HDC/VSA
is famously robust to component noise (holographic redundancy across D dims), so
int8 quantization should leave retrieval essentially intact. This script measures
that directly and honestly, for every operator.

Two fidelity metrics per (operator, dtype):
  (1) Unbind cosine fidelity: cos( unbind(bind(k,v), k), v )  vs the fp32 truth.
  (2) Superposition retrieval accuracy: store T bound pairs in one trace, unbind
      each by its key, clean up against the value codebook (argmax). This is the
      end-to-end task metric — what actually matters downstream.

Quantization model (matches src/eduhdc/models_fast.py, the edge path):
  int8  -> symmetric per-tensor: q = round(x/scale).clamp(-127,127); x ~= q*scale
  bf16  -> torch.bfloat16 cast (8-bit exponent, 7-bit mantissa)
  fp16  -> torch.float16 cast (5-bit exponent, 10-bit mantissa)
  fp32  -> reference (fidelity 1.0 by definition)

Fully vectorized (no Python loops over concepts). Exports JSON to data/results/.
"""

import sys
import os
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.operators import BipolarMAP, RealHRR, EduBindBlockDiag

RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")


def _make_op(op_type, dim, device):
    if op_type == "map":
        return BipolarMAP(dim=dim, device=device), dim
    if op_type == "hrr":
        return RealHRR(dim=dim, device=device), dim
    if op_type == "edubind":
        op = EduBindBlockDiag(dim=dim, device=device)
        return op, op.actual_dim
    raise ValueError(op_type)


def quantize(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """Return a low-precision *rendering* of x (dequantized back to float for the
    algebra, exactly as the edge int8 path does). fp32 returns x unchanged."""
    if dtype == "fp32":
        return x
    if dtype == "bf16":
        return x.to(torch.bfloat16).float()
    if dtype == "fp16":
        return x.to(torch.float16).float()
    if dtype == "int8":
        # symmetric per-tensor scale using max-abs
        scale = x.abs().amax().clamp(min=1e-8) / 127.0
        q = (x / scale).round().clamp(-127, 127).to(torch.int8)
        return q.float() * scale
    raise ValueError(dtype)


@torch.no_grad()
def eval_precision(op_type: str, dtype: str, D: int, K: int, T: int,
                   n_trials: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    op, dim = _make_op(op_type, D, device)
    R = n_trials

    # fp32 reference codebooks, then render at target precision
    key_cb0 = op.random_vector(K)
    val_cb0 = op.random_vector(K)
    key_cb = quantize(key_cb0, dtype)
    val_cb = quantize(val_cb0, dtype)

    # ---- (1) Single-pair unbind cosine fidelity vs fp32 ground truth ----
    ki = torch.randint(0, K, (R,), device=device)
    vi = torch.randint(0, K, (R,), device=device)
    k1, v1 = key_cb[ki], val_cb[vi]
    rec_q = op.unbind(op.bind(k1, v1), k1)                    # low-precision recovery
    # fp32 truth
    k1f, v1f = key_cb0[ki], val_cb0[vi]
    rec_f = op.unbind(op.bind(k1f, v1f), k1f)
    cos = op.similarity(rec_q, v1f).mean().item()            # recovered vs true value
    cos_self = op.similarity(rec_q, rec_f).mean().item()     # vs fp32 recovery

    # ---- (2) Superposition retrieval accuracy (end-to-end task metric) ----
    # AUDIT FIX (see src/eduhdc/capacity_sweep.py B12): keys must be drawn
    # WITHOUT REPLACEMENT. Sampling with replacement binds the SAME key to
    # several different values whenever T > K/collisions occur, which makes
    # retrieval ill-posed by construction (top-1 correct at most 1/m of the
    # time with m copies of a key) and drives the measured accuracy toward
    # the collision ceiling E[1/m] rather than measuring true crosstalk.
    if T > K:
        raise ValueError(f"T={T} > K={K}: cannot draw {T} distinct keys from a "
                         f"codebook of {K}. Raise K (K >> T) or lower T.")
    val_idx = torch.randint(0, K, (R, T), device=device)
    key_idx = torch.stack([torch.randperm(K, device=device)[:T] for _ in range(R)])
    keys = key_cb[key_idx]                                    # (R,T,dim)
    vals = val_cb[val_idx]
    trace = op.bind(keys, vals).sum(dim=1)                    # (R,dim)
    recov = op.unbind(trace.unsqueeze(1).expand(-1, T, -1), keys)  # (R,T,dim)

    if recov.is_complex():
        num = torch.einsum("rtd,kd->rtk", recov, val_cb.conj()).real
        rn = recov.abs().pow(2).sum(-1).sqrt().unsqueeze(-1)
        vn = val_cb.abs().pow(2).sum(-1).sqrt().view(1, 1, -1)
        sims = num / (rn * vn + 1e-8)
    else:
        rec_n = torch.nn.functional.normalize(recov, p=2, dim=-1)
        val_n = torch.nn.functional.normalize(val_cb, p=2, dim=-1)
        sims = torch.einsum("rtd,kd->rtk", rec_n, val_n)
    pred = sims.argmax(dim=-1)
    acc = (pred == val_idx).float().mean().item()

    del key_cb0, val_cb0, key_cb, val_cb, keys, vals, trace, recov, sims, pred
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"unbind_cos_vs_true": cos, "unbind_cos_vs_fp32": cos_self, "retrieval_acc": acc}


def run_precision_eval():
    print("=" * 88)
    print("  E6 — VSA bind/unbind Fidelity under int8 / bf16 / fp16 (C1 quantization robustness)")
    print("=" * 88)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    ops = ["map", "hrr", "edubind"]
    dtypes = ["fp32", "bf16", "fp16", "int8"]
    D = 2048
    K = 100
    T = 50           # moderate superposition load
    n_trials = 200
    print(f"D={D} | K={K} | T={T} | trials={n_trials}\n")

    out = {op: {} for op in ops}
    t0 = time.perf_counter()
    for op in ops:
        print(f">>> {op.upper()}")
        print(f"  {'dtype':<6s} | {'unbind_cos(vs true)':>19s} | {'unbind_cos(vs fp32)':>19s} | {'retrieval_acc':>13s}")
        print("  " + "-" * 70)
        ref_acc = None
        for dt in dtypes:
            r = eval_precision(op, dt, D, K, T, n_trials, device,
                               # `hash()` of a tuple containing a str is randomised per
                               # process (PYTHONHASHSEED), so this made runs irreproducible.
                               # CRC32 of the repr is stable across processes.
                               seed=zlib.crc32(repr((op, dt)).encode()) % (2**31))
            out[op][dt] = r
            if dt == "fp32":
                ref_acc = r["retrieval_acc"]
            drop = "" if ref_acc is None else f"  (Δacc {r['retrieval_acc']-ref_acc:+.4f})"
            print(f"  {dt:<6s} | {r['unbind_cos_vs_true']:>19.4f} | "
                  f"{r['unbind_cos_vs_fp32']:>19.4f} | {r['retrieval_acc']:>13.4f}{drop}")
        print()
    elapsed = time.perf_counter() - t0

    # Verdict: int8 vs fp32 retrieval-accuracy gap, averaged over operators
    int8_gaps = [out[op]["int8"]["retrieval_acc"] - out[op]["fp32"]["retrieval_acc"] for op in ops]
    bf16_gaps = [out[op]["bf16"]["retrieval_acc"] - out[op]["fp32"]["retrieval_acc"] for op in ops]
    print("-" * 88)
    print(f"Mean retrieval Δacc vs fp32:  int8 = {np.mean(int8_gaps):+.4f} | bf16 = {np.mean(bf16_gaps):+.4f}")
    print("Interpretation: |Δacc| small (<~0.01) => quantization does NOT break bind/unbind;")
    print("holographic redundancy across D dims absorbs int8/bf16 rounding noise.")

    payload = {
        "config": {"D": D, "K": K, "T": T, "n_trials": n_trials, "ops": ops, "dtypes": dtypes},
        "results": out,
        "summary": {"mean_int8_delta_acc": float(np.mean(int8_gaps)),
                    "mean_bf16_delta_acc": float(np.mean(bf16_gaps))},
        "elapsed_sec": elapsed,
    }
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fp = os.path.join(RESULTS_DIR, "precision_eval_results.json")
        with open(fp, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[results saved: {fp}]")
    except Exception as e:
        print(f"[export failed: {e}]")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 88)


if __name__ == "__main__":
    run_precision_eval()
