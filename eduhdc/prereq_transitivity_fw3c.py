"""
FW3c — đẩy hop2-3 (cặp gần, yếu nhất 89.7%) lên ≥95% bằng stratified sampling
giám sát theo tầng hop + train lâu hơn. FW3a cho thấy cặp gần thiếu giám sát
(sampled trans pairs phân bố nghiêng về hop xa). FW3c sample CÂN BẰNG theo tầng
hop cho training và tăng epochs. Test held-out giữ nguyên như FW3a.
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

RESULTS_DIR = str(src_dir / "results")
MAX_TEST_PER_STRATUM = 2000
PER_STRATUM_TRAIN = 2500


def main():
    print("=" * 82)
    print("  FW3c — stratified hop supervision (cân bằng cặp gần/xa) + longer train")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, cyc = load_clean_junyi()
    ann = JunyiExpertAnnotations(); ann.load()
    banned = {frozenset((r["A"], r["B"])) for r in ann.train_rows + ann.test_rows}

    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()

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

    rng = np.random.default_rng(0)
    strata = {"hop2-3": [], "hop4-6": [], "hop7+": []}
    for (u, v), d in all_pairs.items():
        if d < 2:
            continue
        key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
        strata[key].append((u, v))
    for k in strata:
        if len(strata[k]) > MAX_TEST_PER_STRATUM:
            idx = sorted(rng.choice(len(strata[k]), MAX_TEST_PER_STRATUM, replace=False))
            strata[k] = [strata[k][i] for i in idx]
    test_set = set()
    for k in strata:
        test_set.update(strata[k])
    print("Test strata:", {k: len(v) for k, v in strata.items()})

    # TRAIN: edges + expert + stratified trans pairs (cân bằng theo hop)
    train_pairs = [(u, v) for u, v in G.edges]
    for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
        if abs(sab - sba) >= 1.0 and frozenset((a, b)) not in test_set:
            train_pairs.append((a, b) if sab > sba else (b, a))
    for key in strata:
        lo, hi = {"hop2-3": (2, 3), "hop4-6": (4, 6), "hop7+": (7, 10**9)}[key]
        pool = [(u, v) for (u, v), d in all_pairs.items()
                if lo <= d <= hi and (u, v) not in test_set]
        if len(pool) > PER_STRATUM_TRAIN:
            idx = sorted(rng.choice(len(pool), PER_STRATUM_TRAIN, replace=False))
            pool = [pool[i] for i in idx]
        train_pairs.extend(pool)
        print(f"  train stratum {key}: +{len(pool)}")
    print(f"Total train pairs: {len(train_pairs)}")

    def bt(pairs):
        return (torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device),
                torch.stack([ex_to_dense[v] for _, v in pairs]).to(device))

    Xu_tr, Xv_tr = bt(train_pairs)
    strata_t = {k: bt(p) for k, p in strata.items() if p}

    results = {}
    for tag, epochs in [("ep100", 100), ("ep160", 160)]:
        hits = {k: 0 for k in strata}
        for seed in range(5):
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
                    hits[k] += int((f > r).sum())
        res = {}
        for k, pairs in strata.items():
            n = len(pairs) * 5
            res[k] = {"dir_acc": hits[k] / n, "ci95": list(wilson_ci(hits[k], n)),
                      "n_pairs": len(pairs)}
        results[tag] = res
        ov = np.mean([res[k]["dir_acc"] for k in strata])
        print(f"\nFW3c {tag}: " + " | ".join(
            f"{k} {res[k]['dir_acc']:.2%}" for k in strata) + f" || overall {ov:.2%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_transitivity_fw3c_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": "FW3c stratified hop supervision + longer train; 5 seeds",
                   "n_train_pairs": len(train_pairs), "results": results}, f, indent=2)
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()

