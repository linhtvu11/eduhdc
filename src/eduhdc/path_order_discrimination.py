"""C1 -- "operator carries direction" on REAL Junyi prerequisite chains.

WHY THIS SCRIPT EXISTS
-----------------------
The pair-level experiments (`prereq_probe_controls.py`, `prereq_direction_v6*.py`,
and the 48.4% cross-source transfer) all reduce to PAIR-LEVEL order recovery, which
`ChainOrder.lean` and the paper's H0 say cannot separate operators: a learned probe
recovering direction from pair embeddings is mostly recovering curriculum DEPTH from
exercise names, not relational structure. So those numbers say little about whether
the VERIFIED OPERATOR itself carries order information.

This script asks the question the chain-level theorems actually answer, and it asks
it in a TRAINING-FREE, READOUT-FREE form on REAL curriculum chains:

  Forward-compose a prerequisite path A -> ... -> B (non-commutative composition,
  `chainRoundtrip`'s nesting) into one carrier Z. Then:
    rec_true = unbind back through the SAME order (chainRoundtrip)   -> should = content(A)
    rec_rev  = unbind back through the REVERSED order (wrong order)  -> should NOT = content(A)

and measure the DIRECTION-MARGIN  sim(rec_true, enc_a) - sim(rec_rev, enc_a).

Under the machine-checked theorems:
  * `chain_exact_unbind`  (ChainTransitivity)  -- exact recovery needs only a LEFT inverse,
    so rec_true = content(A) for ANY operator with left inverses (EduBind, MAP, HRR alike).
  * `abelian_chainAct_reverse` (ChainOrder)   -- if the label algebra is ABELIAN (MAP, HRR),
    reversing the relation list leaves the action unchanged, so rec_rev == rec_true and the
    operator CANNOT tell A->B from B->A.
  * `chain_order_sensitive`   (ChainOrder)   -- non-abelian (EduBind) HAS two orderings that
    act differently, so rec_rev != rec_true and the operator DOES discriminate.

So the PREDICTION is: EduBind should show a large positive direction-margin, while MAP and
HRR should sit at ~0 (margin ~0, DirAcc ~0.5). This is the direct, readout-free proof that
the ORDER information lives in the OPERATOR, not in a learned adapter or a learned probe.
It is the counter to "EduBind is weak and needs an adapter" and the empirical backbone for
C2/C3/C4's fixed-width constant-memory chain guarantee.

There is NO trained probe and NO readout: content vectors come from a fixed random basis
(the same carrier used for every operator), and relation keys are fixed per-node. This is
what removes the H0 confound -- nothing here can learn "exercise name -> depth -> order".

Usage:  python src/eduhdc/path_order_discrimination.py
Output: data/results/path_order_discrimination_results.json
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.capacity_sweep import _make_op, _stable_seed
from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations, humanize

RESULTS_DIR = str(src_dir.parent / "data" / "results")
D = 2048          # matched VSA dimension used throughout the paper
N_SEEDS = 3       # light CI; keep it cheap, this is fast
STRATA = [("hop2-3", 2, 3), ("hop4-6", 4, 6), ("hop7+", 7, 10**9)]
MAX_PATHS_PER_STRATUM = 600     # caps runtime; strata are huge anyway


def load_junyi_dag():
    path = src_dir.parent / "data" / "junyi" / "junyi_Exercise_table.csv"
    G = nx.DiGraph()
    with open(str(path), "r", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip(); G.add_node(name)
            for x in (row.get("prerequisites") or "").split(","):
                x = x.strip()
                if x: G.add_edge(x, name)
    return G


def norm(s): return s.strip().lower().replace(" ", "_")


def build_real_paths(G, banned):
    """Return list of (source_node A, target_node B, path list) for real
    transitive prerequisite pairs, with a shortest DAG path, excluding pairs
    present in the expert annotation set (leakage guard)."""
    paths = []
    node_list = list(G.nodes)
    node_index = {n: i for i, n in enumerate(node_list)}
    for b in sorted(node_list):
        # SORTED. `nx.ancestors` returns a SET and Python randomises string
        # hashes per process, so iterating it directly makes the sampled path
        # list differ from run to run -- the same hazard as seeding from
        # `hash(...)`. Reproducible within a process, not across them.
        for a in sorted(nx.ancestors(G, b)):
            if a == b:
                continue
            if frozenset((a, b)) in banned:
                continue
            try:
                p = nx.shortest_path(G, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            k = len(p) - 1
            if k < 2:          # need a genuine multi-hop chain
                continue
            paths.append((a, b, k, p))
    return paths


def encode_names(names, device):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    return emb


def op_project(op, dense, D):
    """FIXED per-node key vectors that are genuine members of the operator's
    group, so `exact_unbind_ax` holds and a correctly-ordered roundtrip is exact.

    IMPORTANT: for EduBindBlockDiag (`bind = X @ Y`, `unbind = X^T @ Z`) the key
    MUST be a block-diagonal ORTHOGONAL matrix, otherwise `X^T` is not the inverse
    of `X` and even a correct-order unbind fails. A Gaussian (or QR) projection of
    a 384-dim embedding is NOT a group element, so we cannot use the semantic
    embedding as the key for this test. We instead draw the key from
    `op.random_vector`, seeded deterministically from the node name: this is a real
    group element, reproducible per node, and identical across operators and seeds,
    so any difference across operators comes from the binding algebra alone.

    IMPORTANT: keys are used AS the group element, NOT re-normalised. Normalising
    would break the exact-inverse property each operator's `unbind` relies on:
      - EduBindBlockDiag.bind = X @ Y,  unbind = X^T @ Z.  Exact iff X is an
        orthogonal matrix (X^T X = I).  `op.random_vector` yields block-diagonal
        orthogonal blocks; re-normalising to unit norm would scale X by 1/c and
        break X^T X = I.
      - BipolarMAP.bind = x*y,  unbind = bound * key.  Exact iff key is bipolar
        (+/-1): key*key = 1.  Re-normalising breaks key*key = 1.
      - RealHRR.unbind uses the conjugate (approximate inverse) -- that is a known
        property, not something to fix here.

    Returns a dict {node_name: key_vector(1, dim)}."""
    keys = {}
    for n in dense.keys():
        # some operators take a generator mid-signature, some do not; set the global
        # seed instead so this is uniformly reproducible across all op implementations.
        torch.manual_seed(_stable_seed("key", n))
        k = op.random_vector(1)
        keys[n] = k            # NO re-normalisation (see docstring)
    return keys


def direction_margins(op_type, D, paths, key_vectors, device, seed):
    """For each path, forward-compose then unbind in TRUE vs REVERSED order.
    Returns per-path (margin, margin_binary) and error-control (rec_true vs content)."""
    op, dim = _make_op(op_type, D, device)
    torch.manual_seed(seed)

    margins = np.zeros(len(paths))
    rec_true_sim = np.zeros(len(paths))
    for i, (a, b, k, p) in enumerate(paths):
        # carrier = normalized random content vector (same random basis for all ops)
        c = torch.randn(1, dim, device=device, generator=torch.Generator(device=device).manual_seed(_stable_seed("carrier", i, seed)))
        enc_a = torch.nn.functional.normalize(c, p=2, dim=-1)   # (1, dim)

        # keys of the hop nodes in path order v1..vk (b is target, not keyed by content b)
        key_nodes = p[1:]                          # v_1..v_k, keys along the chain
        keys = [key_vectors[n] for n in key_nodes]  # each is (1, dim) group element

        # forward composition
        Z = enc_a
        for kk in keys:
            Z = op.bind(kk, Z)

        # TRUE read = unbind in the REVERSE of the bind order, matching
        # chainRoundtrip's nesting (chain_exact_unbind). We bind v1..vk left-to-right,
        # so we must unbind vk..v1 right-to-left to recover the carrier exactly.
        rt = Z
        for kk in reversed(keys):
            rt = op.unbind(rt, kk)

        # REVERSED read = unbind in the SAME order we bound (wrong order): v1 first.
        # For an abelian family this still recovers the carrier (margin ~0), for a
        # non-abelian family it does not (margin > 0).
        rr = Z
        for kk in keys:
            rr = op.unbind(rr, kk)

        s_true = float(op.similarity(rt, enc_a)[0])
        s_rev = float(op.similarity(rr, enc_a)[0])
        margins[i] = s_true - s_rev
        rec_true_sim[i] = s_true

    return margins, rec_true_sim


def main():
    print("=" * 84)
    print("  C1 -- operator-carried direction on REAL Junyi prerequisite chains")
    print("  (training-free, readout-free; tests chain_order_sensitive vs")
    print("   abelian_chainAct_reverse on real curriculum paths)")
    print("=" * 84)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    G = load_junyi_dag()
    ann = JunyiExpertAnnotations()
    ann.load()
    banned = {frozenset((r["A"], r["B"])) for r in ann.train_rows + ann.test_rows}
    print(f"Junyi DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    rng = np.random.default_rng(0)
    paths = build_real_paths(G, banned)
    print(f"real transitive chains (hops>=2, expert pairs excluded): {len(paths)}")

    # cap per stratum for runtime
    capped = []
    per_stratum = {name: [] for name, _, _ in STRATA}
    for a, b, k, p in paths:
        for name, lo, hi in STRATA:
            if lo <= k <= hi:
                per_stratum[name].append((a, b, k, p))
                break
    for name, lo, hi in STRATA:
        lst = per_stratum[name]
        if len(lst) > MAX_PATHS_PER_STRATUM:
            idx = sorted(rng.choice(len(lst), MAX_PATHS_PER_STRATUM, replace=False))
            lst = [lst[i] for i in idx]
        capped.extend(lst)
        print(f"  {name:8s}: {len(lst):4d} paths")
    paths = capped

    # node names used -> semantic embeddings (used only to size `names`; keys are
    # drawn from the operator's group, NOT from a projection of these embeddings,
    # because a projection is not a group element and would break exact_unbind_ax).
    names = sorted(set(G.nodes) | set(ann.exercises))
    dense = encode_names(names, device)

    print("Drawing fixed per-node keys from each operator's group (deterministic)...")
    t0 = time.perf_counter()
    op_types = ["edubind", "map", "hrr"]

    results = {}
    for op_type in op_types:
        # fixed per-node key vectors (NOT per-seed, NOT learned); identical across
        # seeds so only the binding algebra differs.
        op, _ = _make_op(op_type, D, device)
        kv = op_project(op, dict(zip(names, dense)), D)

        per_seed_acc = []
        per_seed_mean_margin = []
        per_seed_acc_strict = []
        per_seed_tie_frac = []
        per_seed_rho = []
        overall_margins = []
        overall_rho = []
        for seed in range(N_SEEDS):
            margins, true_sim = direction_margins(op_type, D, paths, kv, device, seed)
            # TIE-AWARE, matching how the HRR row of the controls table is
            # reported. A strict `margin > 0` test scores an operator whose
            # margin is IDENTICALLY ZERO -- i.e. one that is order-blind, not
            # one that is wrong -- at 0.000, which reads as "worse than chance"
            # beside another arm's 1.000. MAP is exactly that case. Ties are
            # therefore credited at 0.5, and the tie fraction is reported so the
            # two situations stay distinguishable.
            tie = np.abs(margins) < 1e-9
            acc = float(((margins > 0.0) & ~tie).mean() + 0.5 * tie.mean())
            per_seed_acc_strict.append(float((margins > 0.0).mean()))
            overall_margins.extend(margins.tolist())
            per_seed_acc.append(acc)
            per_seed_tie_frac.append(float(tie.mean()))
            per_seed_rho.append(float(true_sim.mean()))
            overall_rho.extend(true_sim.tolist())
            per_seed_mean_margin.append(float(margins.mean()))
            print(f"  {op_type:8s} seed {seed}: DirAcc={acc:.4f} mean_margin={margins.mean():.4f} "
                  f"rec_true_sim={true_sim.mean():.4f}", flush=True)

        margin = np.array(overall_margins)
        results[op_type] = {
            "dir_acc_mean": float(np.mean(per_seed_acc)),
            "dir_acc_per_seed": per_seed_acc,
            "mean_margin_mean": float(np.mean(per_seed_mean_margin)),
            "mean_margin_per_seed": per_seed_mean_margin,
            "margin_std_overall": float(margin.std()),
            "frac_margin_positive": float((margin > 0.0).mean()),
            "frac_margin_gt_0.05": float((margin > 0.05).mean()),
            "dir_acc_strict_mean": float(np.mean(per_seed_acc_strict)),
            "tie_fraction_mean": float(np.mean(per_seed_tie_frac)),
            "rho_recovery_mean": float(np.mean(per_seed_rho)),
            "rho_recovery_per_seed": per_seed_rho,
        }
        # stratum breakdown on the pooled margins of the LAST seed for a clean per-op picture
        results[op_type]["strata"] = {}
        for name, lo, hi in STRATA:
            sub = [m for (a, b, k, p), m in zip(paths, margin) if lo <= k <= (hi if hi < 10**9 else 10**9)]
            if sub:
                sub_rho = [r for (a, b, k, p), r in zip(paths, np.array(overall_rho[:len(paths)]))
                           if lo <= k <= (hi if hi < 10**9 else 10**9)]
                results[op_type]["strata"][name] = {
                    "n": len(sub),
                    "mean_margin": float(np.mean(sub)),
                    "frac_positive": float((np.array(sub) > 0.0).mean()),
                    "rho_recovery": float(np.mean(sub_rho)) if sub_rho else None,
                }

    elapsed = time.perf_counter() - t0

    payload = {
        "config": {"D": D, "n_seeds": N_SEEDS, "strata": STRATA,
                   "max_paths_per_stratum": MAX_PATHS_PER_STRATUM,
                   "note": ("Training-free, readout-free. Content = fixed random carrier "
                            "basis; keys = fixed per-node group element drawn "
                            "deterministically from the node name via op.random_vector "
                            "(identical across operators and seeds). measure(sim(rec_true, "
                            "enc_a) - sim(rec_rev, enc_a)); rec_true = unbind in the "
                            "forward-bind order, rec_rev = unbind in the reverse order. "
                            "Prediction under chain_exact_unbind/abelian_chainAct_reverse/"
                            "chain_order_sensitive: EduBind margin >> 0 (DirAcc >> 0.5), "
                            "MAP & HRR margin ~0 (DirAcc ~0.5).")},
        "results": results,
        "tot_paths": len(paths),
        "elapsed_sec": elapsed,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "path_order_discrimination_results.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved: {out}]")


if __name__ == "__main__":
    main()
