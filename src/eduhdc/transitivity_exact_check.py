"""
FW3 algebraic layer — Runtime exact-unbind transitivity check.

Realizes the APPLICATION layer of the "giai phap dat tuyet doi" in
docs/temp_overview.md, complementing the Lean proof
(ChainTransitivity.lean :: chain_exact_unbind, the FORMAL backbone).

For every transitive chain length that actually occurs in the cleaned Junyi
curriculum DAG, we EXECUTE the bind-then-unbind roundtrip and check recovery:

  1. EXACT-INTEGER regime (mirrors the Lean model): 90-degree rotation block
     R=[[0,-1],[1,0]] on integer content matrices. Recovery is tested with
     strict equality (==). This is the "absolute" regime: 100% exact, no
     tolerance, for every content and every chain length.

  2. RUNTIME-FLOAT regime (the actual EduBindBlockDiag operator, float32):
     blockwise-orthogonal content + relation operators. Recovery is exact in
     exact arithmetic; we report the max float accumulation error honestly.

HONEST SCOPE (per temp_overview): this verifies OPERATOR FIDELITY for pairs
WITH a known intermediate path in the graph (problem b). It does NOT predict
relations for pairs the graph says nothing about (problem c), and does NOT
claim the curriculum labels are perfectly transitive. This is the ALGEBRAIC
layer, distinct from the statistical probe layer (FW3c 97.6%).
"""

import sys
import os
import json
import pathlib
from collections import Counter

import numpy as np
import networkx as nx

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.prereq_transitivity_v7 import load_clean_junyi
from eduhdc.operators import EduBindBlockDiag

RESULTS_DIR = str(src_dir.parent / "data" / "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# The EduBind family: integer points of O(2) — the 90-degree rotation and a
# reflection. Revision 3 of the Lean model (EduBindSelfContained.lean) verifies
# the family, not the rotation alone, because SO(2) is abelian.
ROT = np.array([[0, -1], [1, 0]], dtype=np.int64)
REF = np.array([[1, 0], [0, -1]], dtype=np.int64)
FAMILY = [ROT, REF]

N_CONTENT = 200     # random integer contents per chain length (demonstrates ∀Y)
N_FLOAT_TRIALS = 50


def chain_roundtrip_int(Y, n, rng):
    """HETEROGENEOUS chain of length n: draw an independent relation from the
    EduBind family at every hop, bind forward through the chain, then unbind
    back through it IN REVERSE ORDER — the exact nesting of `chainRoundtrip`
    in ChainTransitivity.lean:
        chainRoundtrip (i :: is) Y = inv i (chainRoundtrip is (ops i Y))
    A real prerequisite chain A -> B -> C traverses a DIFFERENT relation at
    every hop, so a homogeneous chain (the same matrix n times) does not model
    it. Integer arithmetic, so recovery is tested with strict equality."""
    idx = rng.integers(0, len(FAMILY), size=n)
    Z = Y.copy()
    for k in idx:                       # bind forward: ops i_1, ..., ops i_n
        Z = FAMILY[k] @ Z
    for k in idx[::-1]:                 # unbind backward: inv i_n, ..., inv i_1
        Z = FAMILY[k].T @ Z
    return Z, idx


def hop_length_distribution(G):
    """Count transitive pairs (shortest-path length d>=2) by d."""
    cnt = Counter()
    for u in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, u)
        for v, d in lengths.items():
            if v != u and d >= 2:
                cnt[d] += 1
    return cnt


def main():
    print("=" * 82)
    print("  FW3 algebraic layer — runtime exact-unbind transitivity check")
    print("=" * 82)
    G, cyc = load_clean_junyi()
    print(f"Junyi cleaned DAG: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (removed {len(cyc)} cycle nodes)")

    # 1. Curriculum transitive chain-length spectrum
    cnt = hop_length_distribution(G)
    lengths = sorted(cnt.keys())
    total_trans = sum(cnt.values())
    print("\nTransitive pairs (d>=2) by hop length:")
    for d in lengths:
        print(f"  hop {d}: {cnt[d]} pairs")
    print(f"  total transitive pairs: {total_trans}, "
          f"max chain length: {max(lengths)}")

    rng = np.random.default_rng(0)

    # 2. EXACT-INTEGER check (mirrors Lean chain_exact_unbind)
    print(f"\n[EXACT-INTEGER] roundtrip on {N_CONTENT} random integer contents/length:")
    int_results = {}
    all_exact = True
    for n in lengths:
        ok = 0
        distinct = 0
        for _ in range(N_CONTENT):
            Y = rng.integers(-5, 6, size=(2, 2))
            Z, idx = chain_roundtrip_int(Y, n, rng)
            if np.array_equal(Z, Y):
                ok += 1
            distinct += len(set(idx.tolist()))
        exact = (ok == N_CONTENT)
        all_exact = all_exact and exact
        int_results[n] = {"tested": N_CONTENT, "exact_recover": ok, "exact": exact,
                          "mean_distinct_relations_per_chain": distinct / N_CONTENT}
        print(f"  n={n}: {ok}/{N_CONTENT} exact {'OK' if exact else 'FAIL'}")
    print("  => EXACT-INTEGER: "
          + ("100% exact recovery for ALL curriculum chain lengths"
             if all_exact else "FAILURES detected"))

    # 3. RUNTIME-FLOAT check (actual EduBindBlockDiag operator)
    print("\n[RUNTIME-FLOAT] actual EduBindBlockDiag operator, float32:")
    op = EduBindBlockDiag(dim=1024, device="cpu")
    float_results = {}
    max_err_overall = 0.0
    for n in lengths:
        errs = []
        for _ in range(N_FLOAT_TRIALS):
            Y = op.random_vector(1)                       # blockwise-orthogonal content
            rels = [op.random_vector(1) for _ in range(n)]  # a DIFFERENT relation per hop
            Z = Y
            for Rrel in rels:                             # bind forward
                Z = op.bind(Rrel, Z)
            Yrec = Z
            for Rrel in reversed(rels):                   # unbind in reverse order
                Yrec = op.unbind(Yrec, Rrel)
            errs.append((Yrec - Y).abs().max().item())
        m = max(errs)
        max_err_overall = max(max_err_overall, m)
        float_results[n] = {"trials": N_FLOAT_TRIALS, "max_abs_err": m}
        print(f"  n={n}: max|Y_rec - Y| = {m:.3e}")
    print(f"  => RUNTIME-FLOAT: max recovery error {max_err_overall:.3e} "
          f"(float32 accumulation); exact in exact arithmetic")

    # 4. Summary
    summary = {
        "task": "FW3 algebraic layer — runtime exact-unbind transitivity check",
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                  "removed_cycle_nodes": len(cyc)},
        "hop_length_distribution": {str(d): int(cnt[d]) for d in lengths},
        "total_transitive_pairs": int(total_trans),
        "max_chain_length": int(max(lengths)),
        "chain_kind": ("heterogeneous: an independent relation is drawn per hop, "
                       "bound forward and unbound in reverse order, matching "
                       "ChainTransitivity.lean::chainRoundtrip over a List of relations"),
        "trials": {"integer_contents_per_length": N_CONTENT,
                   "float_trials_per_length": N_FLOAT_TRIALS,
                   "chain_lengths_tested": len(lengths),
                   "total_integer_trials": N_CONTENT * len(lengths),
                   "total_float_trials": N_FLOAT_TRIALS * len(lengths)},
        "exact_integer": {"all_exact": all_exact,
                          "per_length": {str(k): v for k, v in int_results.items()}},
        "runtime_float": {"max_abs_err_overall": max_err_overall,
                          "per_length": {str(k): v for k, v in float_results.items()}},
        "honest_scope": ("operator fidelity for pairs WITH a known intermediate path "
                         "(problem b); NOT prediction for unknown pairs (problem c); "
                         "algebraic layer, distinct from the statistical probe layer "
                         "(FW3c 97.6%)"),
    }
    out = os.path.join(RESULTS_DIR, "transitivity_exact_check_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 82)


if __name__ == "__main__":
    main()
