"""
C1 Revision 4 -- E1: does a role-filler encoding under a COMMUTATIVE operator
distinguish pair order? (Direct empirical test of H0 / hadamard_encPair_order_sensitive.)

WHY THIS SCRIPT EXISTS
-----------------------
`prereq_probe_controls.py` (the source of Table 1 in main_r4.tex) reports MAP
tying EduBind at exactly 0.0% strict direction accuracy on every stratum, and
the paper explains this as "MAP's bind is exactly commutative, so forward and
reverse scores are numerically identical". That explanation is correct for the
PROBE ARCHITECTURE ACTUALLY USED THERE -- `EduHDC_PrereqProbe.encode_relation`
nests bind at the OUTER level:

    bind( bind(u, role_P), bind(v, role_A) )

For MAP (elementwise product, associative AND commutative) this collapses:

    u . role_P . v . role_A  ==  v . role_P . u . role_A

regardless of role_P/role_A, because MAP is a commutative RING product, not
just a commutative binding of two things. So Table 1's MAP-tie result is a
fact about NESTED BIND under an associative-commutative operator, and it is
*not* a test of the role-filler construction the paper's H0 result
(`hadamard_encPair_order_sensitive`) is actually about:

    encPair := bundle( bind(u, role_P), bind(v, role_A) )

i.e. bind each concept to its OWN role, then SUPERPOSE (sum) the two bound
terms -- no nested nesting, no nested nesting to collapse. `encChainRF`/
`encPair` in the Lean development sum, they do not bind again.

This script runs the IDENTICAL protocol as prereq_probe_controls.py (same
data, same splits, same early-stopping, same seeds) with ONLY the probe's
`encode_relation` swapped for the true role-filler construction
(`EduHDC_PrereqProbeRoleFiller` in models.py). The prediction under H0: MAP
should NO LONGER tie at 0.0% -- a commutative operator with two distinct roles
should recover a meaningful fraction of direction accuracy, because bundling
(not binding) breaks the (u, v) symmetry algebraically.

Usage:  python src/eduhdc/prereq_role_filler_probe.py [--quick]
Output: data/results/prereq_role_filler_probe_results.json

--quick runs 3 seeds instead of 10, for a fast sanity check before the full run.
"""

import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import eduhdc.prereq_probe_controls as base
from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations, humanize
from eduhdc.models import EduHDC_PrereqProbeRoleFiller
from eduhdc.prereq_transitivity_v7 import load_clean_junyi
from sentence_transformers import SentenceTransformer

RESULTS_DIR = base.RESULTS_DIR
ARMS = ["edubind", "map", "hrr"]  # concat is unaffected by this swap (no VSA bind at all)
STRATA = base.STRATA


def make_probe_role_filler(arm, emb_dim, device):
    return EduHDC_PrereqProbeRoleFiller(emb_dim, 2048, arm, device).to(device)


def main():
    quick = "--quick" in sys.argv
    n_seeds = 3 if quick else base.N_SEEDS

    print("=" * 84)
    print("  C1 Revision 4 -- E1: role-filler encoding, commutative-operator test of H0")
    print(f"  ({n_seeds} seeds{' [QUICK]' if quick else ''}, arms={ARMS})")
    print("=" * 84, flush=True)

    # Monkeypatch make_probe inside the base module: train_eval() there resolves
    # `make_probe` from its own module globals at call time, so this swap is
    # picked up by the (otherwise unmodified) shared training/eval logic.
    base.make_probe = make_probe_role_filler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, cyc = load_clean_junyi()
    ann = JunyiExpertAnnotations()
    ann.load()
    banned = {frozenset((r["A"], r["B"])) for r in ann.train_rows + ann.test_rows}
    print(f"Junyi DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(removed {len(cyc)} cyclic); device={device}", flush=True)

    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()
    ap = base.build_pairs(G, banned)
    print(f"transitive pairs available (expert pairs excluded): {len(ap)}", flush=True)

    results = {
        "protocol": {
            "note": ("IDENTICAL protocol/data/splits/seeds to prereq_probe_controls.py "
                     "(same source functions, imported not duplicated); ONLY the probe's "
                     "encode_relation is swapped for a true role-filler construction "
                     "(bundle(bind(u,role_P), bind(v,role_A))) instead of the original's "
                     "nested bind(bind(u,role_P), bind(v,role_A)). concat-MLP is not run "
                     "here since it contains no VSA bind and is unaffected by the swap; "
                     "its Table-1 numbers apply unchanged."),
            "n_seeds": n_seeds, "arms": ARMS,
            "prediction_under_H0": ("MAP should no longer tie at 0.0% strict direction "
                                     "accuracy; a commutative operator with two distinct "
                                     "roles bundled (not bound) together should recover "
                                     "a meaningful fraction of direction accuracy."),
        }
    }

    # ---------------------------------------------------------------- TRANSDUCTIVE
    print("\n--- TRANSDUCTIVE (pair-held-out) ---", flush=True)
    trans = {a: {k: [] for k in STRATA} for a in ARMS}
    node_overlap = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        strata = {k: [] for k in STRATA}
        for (u, v), d in ap.items():
            if d < 2:
                continue
            key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
            strata[key].append((u, v))
        for k in strata:
            if len(strata[k]) > base.MAX_TEST_TRANS:
                idx = sorted(rng.choice(len(strata[k]), base.MAX_TEST_TRANS, replace=False))
                strata[k] = [strata[k][i] for i in idx]
        test_set = set()
        for v in strata.values():
            test_set.update(v)
        tr_by_cat = {"direct": [(u, v) for u, v in G.edges]}
        expert_pairs = []
        for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
            if abs(sab - sba) >= 1.0 and frozenset((a, b)) not in test_set:
                expert_pairs.append((a, b) if sab > sba else (b, a))
        tr_by_cat["expert"] = expert_pairs
        for k in STRATA:
            lo, hi = base.BOUNDS[k]
            pool = [(u, v) for (u, v), d in ap.items()
                    if lo <= d <= hi and (u, v) not in test_set]
            if len(pool) > base.PER_STRATUM_TRAIN:
                idx = sorted(rng.choice(len(pool), base.PER_STRATUM_TRAIN, replace=False))
                pool = [pool[i] for i in idx]
            tr_by_cat[k] = pool
        tr = [p for pairs in tr_by_cat.values() for p in pairs]
        assert not (set(tr) & test_set), "pair leakage"
        trn, ten = set(), set()
        for p in tr:
            trn.update(p)
        for p in test_set:
            ten.update(p)
        node_overlap.append(len(ten & trn) / max(1, len(ten)))
        tr_by_cat_t = {c: base.to_tensors(p, ex_to_dense, device) for c, p in tr_by_cat.items() if p}
        st_t = {k: base.to_tensors(p, ex_to_dense, device) for k, p in strata.items() if p}
        for arm in ARMS:
            out = base.train_eval(arm, emb_dim, device, tr_by_cat_t, st_t, seed)
            for k in STRATA:
                trans[arm][k].append(out[k])
        print(f"  seed {seed}: train={len(tr)} test={len(test_set)} "
              f"node-overlap={node_overlap[-1]:.3f} | "
              + " ".join(f"{a}={trans[a]['hop2-3'][-1]['wins']/trans[a]['hop2-3'][-1]['n']:.3f}"
                         for a in ARMS), flush=True)
    results["transductive"] = {a: {k: base.summarise(trans[a][k]) for k in STRATA} for a in ARMS}
    results["transductive_test_node_overlap_mean"] = float(np.mean(node_overlap))

    # ---------------------------------------------------------------- INDUCTIVE
    print("\n--- INDUCTIVE (node-held-out; split redrawn per seed) ---", flush=True)
    ind = {a: {k: [] for k in STRATA} for a in ARMS}
    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        nodes = list(G.nodes)
        perm = rng.permutation(len(nodes))
        n_test = int(len(nodes) * base.TEST_NODE_FRAC)
        test_nodes = {nodes[i] for i in perm[:n_test]}
        train_nodes = {nodes[i] for i in perm[n_test:]}
        strata = {k: [] for k in STRATA}
        for (u, v), d in ap.items():
            if d < 2 or u not in test_nodes or v not in test_nodes:
                continue
            key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
            strata[key].append((u, v))
        for k in strata:
            if len(strata[k]) > base.MAX_TEST_IND:
                idx = sorted(rng.choice(len(strata[k]), base.MAX_TEST_IND, replace=False))
                strata[k] = [strata[k][i] for i in idx]
        tr_by_cat = {"direct": [(u, v) for u, v in G.edges
                                 if u in train_nodes and v in train_nodes]}
        expert_pairs = []
        for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
            if abs(sab - sba) >= 1.0 and a in train_nodes and b in train_nodes:
                expert_pairs.append((a, b) if sab > sba else (b, a))
        tr_by_cat["expert"] = expert_pairs
        for k in STRATA:
            lo, hi = base.BOUNDS[k]
            pool = [(u, v) for (u, v), d in ap.items()
                    if lo <= d <= hi and u in train_nodes and v in train_nodes]
            if len(pool) > base.PER_STRATUM_TRAIN:
                idx = sorted(rng.choice(len(pool), base.PER_STRATUM_TRAIN, replace=False))
                pool = [pool[i] for i in idx]
            tr_by_cat[k] = pool
        tr = [p for pairs in tr_by_cat.values() for p in pairs]
        trn, ten = set(), set()
        for p in tr:
            trn.update(p)
        for k in STRATA:
            for p in strata[k]:
                ten.update(p)
        assert not (trn & ten), "node leakage"
        tr_by_cat_t = {c: base.to_tensors(p, ex_to_dense, device) for c, p in tr_by_cat.items() if p}
        st_t = {k: base.to_tensors(p, ex_to_dense, device) for k, p in strata.items() if p}
        for arm in ARMS:
            out = base.train_eval(arm, emb_dim, device, tr_by_cat_t, st_t, seed)
            for k in STRATA:
                if k in out:
                    ind[arm][k].append(out[k])
        sizes = {k: len(strata[k]) for k in STRATA}
        print(f"  seed {seed}: train={len(tr)} test={sizes} | "
              + " ".join(f"{a}={ind[a]['hop7+'][-1]['wins']/ind[a]['hop7+'][-1]['n']:.3f}"
                         for a in ARMS), flush=True)
    results["inductive"] = {a: {k: base.summarise(ind[a][k]) for k in STRATA if ind[a][k]}
                            for a in ARMS}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "prereq_role_filler_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 84)
    for tier in ("transductive", "inductive"):
        print(f"\n{tier.upper()}  (strict / tie-broken, bootstrap CI over seeds)")
        for arm in ARMS:
            row = results[tier].get(arm, {})
            cells = []
            for k in STRATA:
                if k in row:
                    r = row[k]
                    cells.append(f"{k} {r['dir_acc_strict']:.3f}/{r['dir_acc_tiebreak']:.3f}"
                                 f" [{r['bootstrap_ci95_strict'][0]:.3f},"
                                 f"{r['bootstrap_ci95_strict'][1]:.3f}]")
            print(f"  {arm:8s} " + " | ".join(cells))
    print(f"\n[saved: {out_path}]")


if __name__ == "__main__":
    main()
