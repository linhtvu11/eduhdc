"""
C1 Revision 4 -- E2: role-filler vs composition retrieval accuracy as a
function of chain length n, at matched dimension D and identical codebook/
accuracy units. Direct empirical test of the Separation Theorem
(main_r4.tex, "The separation is a superposition cost"): does role-filler
chain encoding degrade with n while composition-based recovery stays exact?

WHY THIS SCRIPT EXISTS
-----------------------
The paper currently anchors the Separation Theorem to two SEPARATE existing
measurements: capacity_sweep.py's generic superposition-load curve (accuracy
vs T/D, T meaning "how many things are bundled together") and
transitivity_exact_check.py's chain-error-vs-hop-length curve (float32
roundtrip error, not retrieval accuracy, and only for composition). The paper
argues "a role-filler chain of length n sits at T=n on the capacity curve"
as an inference connecting the two, rather than measuring both encodings on
the SAME x-axis (chain length n) and the SAME y-axis (top-1 retrieval
accuracy against a content codebook) in one experiment. This script closes
that gap directly.

Two encodings, both realized with the SAME operator implementations
(src/eduhdc/operators.py) used everywhere else in this codebase, at the SAME
dimension D=2048 and the SAME content codebook size K=2000:

  ROLE-FILLER  (reuses capacity_sweep.measure_retrieval_accuracy verbatim,
      at T=n): bundle n independently key-bound content values into one
      vector, unbind each by its own key, look up nearest neighbour in the
      content codebook. This is encChainRF (main_r4.tex S5.2) realized in
      floating point.

  COMPOSITION  (measure_composition_accuracy, new here): encode ONE content
      value by binding it through a chain of n independently-sampled relation
      operators (a different relation at every hop, matching chainRoundtrip's
      heterogeneous nesting -- NOT the same operator repeated), recover by
      unbinding through the identical chain in reverse, look up the nearest
      neighbour in the SAME content codebook. This is chainRoundtrip
      (main_r4.tex S5.1 / ChainTransitivity.lean) realized in floating point,
      reported in the SAME accuracy units as the role-filler side so the two
      curves are directly comparable rather than argued into alignment.

Chain lengths span the observed Junyi hop range (2..41, main_r4.tex S6.2) and
extend beyond it to make the divergence unambiguous.

Usage:  python src/eduhdc/chain_length_separation.py
Output: data/results/chain_length_separation_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.capacity_sweep import _make_op, _stable_seed, measure_retrieval_accuracy

RESULTS_DIR = str(src_dir.parent / "data" / "results")


@torch.no_grad()
def measure_composition_accuracy(op_type: str, K: int, n: int, D: int,
                                  n_trials: int, device: str, seed: int = 0) -> float:
    """Composition-chain retrieval accuracy, in the SAME units as
    measure_retrieval_accuracy: encode one content value per trial through a
    chain of n independently-sampled relation operators (a different one at
    every hop), recover by unbinding through the identical chain in reverse
    order (chainRoundtrip's nesting), and look up the nearest neighbour in a
    content codebook of size K."""
    torch.manual_seed(seed)
    op, dim = _make_op(op_type, D, device)
    R = n_trials

    val_cb = op.random_vector(K)                       # (K, dim) content codebook
    val_idx = torch.randint(0, K, (R,), device=device)
    Y = val_cb[val_idx]                                  # (R, dim) content to encode

    # n hops, a DIFFERENT relation operator sampled per hop (heterogeneous
    # chain, matching chainRoundtrip / chain_roundtrip_int -- not one operator
    # repeated). Same relation family, one draw per hop, shared across trials'
    # batch dimension so the whole batch traverses the identical relation
    # sequence at each hop (a fixed chain of relations, many contents).
    rels = [op.random_vector(1).expand(R, -1) for _ in range(n)]

    Z = Y
    for r in rels:                    # bind forward through the chain
        Z = op.bind(r, Z)
    for r in reversed(rels):          # unbind backward through the chain
        Z = op.unbind(Z, r)

    if Z.is_complex():
        num = torch.einsum("rd,kd->rk", Z, val_cb.conj()).real
        rn = Z.abs().pow(2).sum(-1).sqrt().unsqueeze(-1)
        vn = val_cb.abs().pow(2).sum(-1).sqrt().unsqueeze(0)
        sims = num / (rn * vn + 1e-8)
    else:
        rec_n = torch.nn.functional.normalize(Z, p=2, dim=-1)
        val_n = torch.nn.functional.normalize(val_cb, p=2, dim=-1)
        sims = torch.einsum("rd,kd->rk", rec_n, val_n)

    pred = sims.argmax(dim=-1)
    correct = (pred == val_idx).float().mean().item()

    del val_cb, val_idx, Y, rels, Z, sims, pred
    if device == "cuda":
        torch.cuda.empty_cache()
    return correct


def main():
    print("=" * 84)
    print("  C1 Revision 4 -- E2: role-filler vs composition, accuracy vs chain length n")
    print("=" * 84)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    ops = ["map", "hrr", "edubind"]
    D = 2048          # matches the VSA dimension used throughout the paper's KT/probe experiments
    K = 2000          # >> max(n_list), matches capacity_sweep.py's codebook size
    n_trials = 30     # matches capacity_sweep.py's R
    n_list = [2, 4, 8, 16, 24, 32, 41, 64, 100, 150]

    print(f"D={D}  K={K}  trials/point R={n_trials}")
    print(f"Chain lengths n = {n_list} (2..41 spans the observed Junyi hop range)\n")

    role_filler = {op: {} for op in ops}
    composition = {op: {} for op in ops}

    t0 = time.perf_counter()
    for op in ops:
        print(f">>> Operator: {op.upper()}")
        print("  n     | role-filler | composition")
        for n in n_list:
            rf = measure_retrieval_accuracy(op, K, n, D, n_trials, device,
                                            seed=_stable_seed("rf", op, n, D))
            co = measure_composition_accuracy(op, K, n, D, n_trials, device,
                                              seed=_stable_seed("co", op, n, D))
            role_filler[op][n] = rf
            composition[op][n] = co
            print(f"  {n:>5d} |   {rf:8.4f}  |   {co:8.4f}")
        print()
    elapsed = time.perf_counter() - t0

    payload = {
        "config": {"D": D, "K": K, "n_trials": n_trials, "n_list": n_list, "ops": ops},
        "role_filler_accuracy": role_filler,
        "composition_accuracy": composition,
        "elapsed_sec": elapsed,
        "note": ("Both curves measured with the identical operator implementations "
                 "(src/eduhdc/operators.py), dimension D, content codebook size K, and "
                 "top-1 nearest-neighbour accuracy definition -- role-filler via "
                 "capacity_sweep.measure_retrieval_accuracy at T=n (bundle n key-bound "
                 "contents), composition via measure_composition_accuracy (chain of n "
                 "independently-sampled relations applied to one content, recovered by "
                 "unbinding the identical chain in reverse). Chain lengths 2..41 span the "
                 "Junyi curriculum's observed hop range (main_r4.tex S6.2); 64/100/150 "
                 "extend beyond it to make the divergence unambiguous."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "chain_length_separation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("-" * 84)
    print(f"Total time: {elapsed:.1f}s")
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()
