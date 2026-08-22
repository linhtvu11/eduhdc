"""
C1 — Transitivity INDUCTIVE split: loại HOÀN TOÀN trùng node giữa train/test.

Động lực: FW3c test held-out CẶP nhưng 94% test-nodes vẫn xuất hiện trong
train-pairs (transductive). Để trả lời triệt để lo ngại leak do trùng node,
inductive split hold-out hẳn một tập NODE: test trên cặp bắc cầu mà CẢ HAI node
chưa từng xuất hiện trong bất kỳ train-pair nào. Nếu probe vẫn DirAcc cao trên
node mới hoàn toàn → tín hiệu hướng khái quát từ content thật, không dựa vào
"quen node".

Protocol:
  - Chia node thành train-nodes / test-nodes (seeded).
  - Train pairs: cạnh trực tiếp + expert + sampled transitive, CHỈ cả-2-node ∈ train-nodes.
  - Test pairs: cặp bắc cầu hop≥2 với CẢ-2-node ∈ test-nodes (stratified hop).
  => train ∩ test = ∅ cả về cặp lẫn node.
"""

import sys
import os
import json
import pathlib

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations, humanize
from eduhdc.models import EduHDC_PrereqProbe
from eduhdc.prereq_transitivity_v7 import load_clean_junyi, wilson_ci
from sentence_transformers import SentenceTransformer

RESULTS_DIR = str(src_dir.parent / "data" / "results")
TEST_NODE_FRAC = 0.35
MAX_TEST_PER_STRATUM = 1500
PER_STRATUM_TRAIN = 2500
N_SEEDS = 10
N_BOOTSTRAP = 2000


def bootstrap_ci_over_seeds(seed_accs, n_bootstrap=N_BOOTSTRAP, seed=0):
    """95% CI (percentile bootstrap) for the mean of per-seed accuracies."""
    arr = np.asarray(seed_accs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    print("=" * 82)
    print("  C1 Transitivity INDUCTIVE split (hold-out node, loại hoàn toàn trùng node)")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, cyc = load_clean_junyi()
    ann = JunyiExpertAnnotations(); ann.load()
    banned = {frozenset((r["A"], r["B"])) for r in ann.train_rows + ann.test_rows}

    # ---- chia NODE ----
    nodes = list(G.nodes)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(nodes))
    n_test = int(len(nodes) * TEST_NODE_FRAC)
    test_nodes = {nodes[i] for i in perm[:n_test]}
    train_nodes = {nodes[i] for i in perm[n_test:]}
    print(f"Node split: train {len(train_nodes)} | test {len(test_nodes)} "
          f"(frac {TEST_NODE_FRAC})")

    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()

    # all ancestor pairs
    all_pairs = {}
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if frozenset((u, v)) in banned:
                continue
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            all_pairs[(u, v)] = d

    # ---- TEST pairs: cả 2 node ∈ test_nodes, hop>=2 ----
    strata = {"hop2-3": [], "hop4-6": [], "hop7+": []}
    for (u, v), d in all_pairs.items():
        if d < 2 or u not in test_nodes or v not in test_nodes:
            continue
        key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
        strata[key].append((u, v))
    rng2 = np.random.default_rng(0)
    for k in strata:
        if len(strata[k]) > MAX_TEST_PER_STRATUM:
            idx = sorted(rng2.choice(len(strata[k]), MAX_TEST_PER_STRATUM, replace=False))
            strata[k] = [strata[k][i] for i in idx]
    print("Test strata (cả 2 node MỚI):", {k: len(v) for k, v in strata.items()})
    total_test = sum(len(v) for v in strata.values())
    if total_test < 100:
        print("!! Quá ít test-pairs inductive — không đủ để kết luận.")
        return

    # ---- TRAIN pairs: cả 2 node ∈ train_nodes ----
    train_pairs = [(u, v) for u, v in G.edges if u in train_nodes and v in train_nodes]
    n_edge_tr = len(train_pairs)
    for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
        if abs(sab - sba) >= 1.0 and a in train_nodes and b in train_nodes:
            train_pairs.append((a, b) if sab > sba else (b, a))
    for key in strata:
        lo, hi = {"hop2-3": (2, 3), "hop4-6": (4, 6), "hop7+": (7, 10**9)}[key]
        pool = [(u, v) for (u, v), d in all_pairs.items()
                if lo <= d <= hi and u in train_nodes and v in train_nodes]
        if len(pool) > PER_STRATUM_TRAIN:
            idx = sorted(rng2.choice(len(pool), PER_STRATUM_TRAIN, replace=False))
            pool = [pool[i] for i in idx]
        train_pairs.extend(pool)
    print(f"Train pairs: {len(train_pairs)} (edges {n_edge_tr})")

    # verify no node overlap
    tr_n = set(); te_n = set()
    for u, v in train_pairs:
        tr_n.add(u); tr_n.add(v)
    for k in strata:
        for u, v in strata[k]:
            te_n.add(u); te_n.add(v)
    print(f"Verify node overlap train∩test: {len(tr_n & te_n)} (phải = 0)")

    def bt(pairs):
        return (torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device),
                torch.stack([ex_to_dense[v] for _, v in pairs]).to(device))

    Xu_tr, Xv_tr = bt(train_pairs)
    strata_t = {k: bt(p) for k, p in strata.items() if p}

    results = {}
    for tag, epochs in [("ep120", 120)]:
        hits = {k: 0 for k in strata}
        seed_accs = {k: [] for k in strata}
        for seed in range(N_SEEDS):
            torch.manual_seed(31 + seed * 7)
            probe = EduHDC_PrereqProbe(emb_dim, 2048, "edubind", device).to(device)
            opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(epochs):
                opt.zero_grad()
                fs = probe(Xu_tr, Xv_tr); rs = probe(Xv_tr, Xu_tr)
                loss = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
                loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                for k, (Xu, Xv) in strata_t.items():
                    f = probe(Xu, Xv).cpu().numpy(); r = probe(Xv, Xu).cpu().numpy()
                    h = int((f > r).sum())
                    hits[k] += h
                    seed_accs[k].append(h / len(strata[k]))
        res = {}
        for k, pairs in strata.items():
            if not pairs:
                continue
            n = len(pairs) * N_SEEDS
            ci_lo, ci_hi = bootstrap_ci_over_seeds(seed_accs[k])
            res[k] = {"dir_acc": hits[k] / n, "ci95": list(wilson_ci(hits[k], n)),
                      "n_pairs": len(pairs), "n_seeds": N_SEEDS,
                      "per_seed_acc": seed_accs[k],
                      "bootstrap_ci_95": [ci_lo, ci_hi]}
        strata_keys = list(res.keys())
        ov_unweighted = float(np.mean([res[k]["dir_acc"] for k in strata_keys]))
        total_hits = sum(hits[k] for k in strata_keys)
        total_n = sum(len(strata[k]) * N_SEEDS for k in strata_keys)
        ov_weighted = float(total_hits / total_n)
        results[tag] = res
        results[tag]["overall_unweighted"] = ov_unweighted
        results[tag]["overall_weighted_by_n"] = ov_weighted
        print(f"\nINDUCTIVE {tag} ({N_SEEDS} seeds): " + " | ".join(
            f"{k} {res[k]['dir_acc']:.2%} CI95=[{res[k]['bootstrap_ci_95'][0]:.2%},{res[k]['bootstrap_ci_95'][1]:.2%}]"
            for k in strata_keys) + f" || overall unweighted {ov_unweighted:.2%} / weighted {ov_weighted:.2%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_transitivity_inductive_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("INDUCTIVE node split: test pairs have BOTH nodes "
                                f"unseen in training; no node/pair overlap; {N_SEEDS} seeds "
                                "+ 95% percentile bootstrap CI over seeds (2026-08-21 update)"),
                   "test_node_frac": TEST_NODE_FRAC, "n_seeds": N_SEEDS,
                   "n_train_pairs": len(train_pairs), "results": results}, f, indent=2)
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()
