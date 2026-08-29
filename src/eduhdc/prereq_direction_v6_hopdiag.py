"""
C1 — Hop-stratified direction diagnostic on the independent Junyi DAG (task #36).

Motivation: prereq_direction_v6.py sampled ancestor pairs UNIFORMLY; the hop
distribution peaks at 11-13 hops, i.e. pairs so far apart in the curriculum
that direct prerequisite direction is pedagogically weak. This diagnostic asks
the scientifically sharper question: does the EduBind direction signal
generalize for CLOSE pairs (1-3 hops, where the prerequisite relation is
meaningful), and how does accuracy decay with curriculum distance?

Protocol is identical to v6 (frozen v4 config, annotation-free held-out pairs,
10 seeds, pooled-binomial Wilson CI). The ONLY change is that test pairs are
stratified by shortest-path hop distance and reported per stratum. This is a
characterization of the same held-out signal, NOT tuning on the test set.
"""

import sys
import os
import csv
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
from sentence_transformers import SentenceTransformer

HIGH, LOW = 6.0, 3.0
RESULTS_DIR = str(src_dir.parent / "data" / "results")
# hop strata: (label, lo, hi) inclusive
STRATA = [("hop1", 1, 1), ("hop2", 2, 2), ("hop3", 3, 3),
          ("hop1-3", 1, 3), ("hop4-6", 4, 6), ("hop7-12", 7, 12),
          ("hop13+", 13, 10**9)]
PER_STRATUM_CAP = 1500


def load_junyi_dag():
    path = src_dir.parent / "data" / "junyi" / "junyi_Exercise_table.csv"
    G = nx.DiGraph()
    with open(str(path), "r", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            G.add_node(name)
            for p in (row.get("prerequisites") or "").split(","):
                p = p.strip()
                if p:
                    G.add_edge(p, name)
    return G


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def main():
    print("=" * 82)
    print("  C1 — hop-stratified direction diagnostic (frozen v4 protocol)")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = JunyiExpertAnnotations()
    if not ann.load():
        raise SystemExit("[FAIL] expert annotations not found")
    G = load_junyi_dag()
    print(f"Junyi DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    banned = set()
    for r in ann.train_rows + ann.test_rows:
        banned.add(frozenset((r["A"], r["B"])))

    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()

    # ---- gather annotation-free ancestor pairs WITH hop distance ----
    print("Computing ancestor pairs + hop distances ...", flush=True)
    by_hop = {}
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if u == v or frozenset((u, v)) in banned:
                continue
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            by_hop.setdefault(d, []).append((u, v))

    # sample per stratum (fixed seed) so each stratum is represented
    rng = np.random.default_rng(0)
    strata_pairs = {}
    for label, lo, hi in STRATA:
        pool = []
        for d in range(lo, hi + 1):
            pool.extend(by_hop.get(d, []))
        if len(pool) > PER_STRATUM_CAP:
            idx = sorted(rng.choice(len(pool), PER_STRATUM_CAP, replace=False).tolist())
            pool = [pool[i] for i in idx]
        strata_pairs[label] = pool
        print(f"  stratum {label:>8s}: {len(pool)} pairs")

    # ---- frozen v4 training protocol (train split only) ----
    train_bin = ann.train_binary(HIGH, LOW)
    tr_pos = [(a, b) for a, b, y in train_bin if y == 1]
    tr_neg = [(a, b) for a, b, y in train_bin if y == 0]

    def bt(pairs):
        Xu = torch.stack([ex_to_dense[u] for u, _ in pairs])
        Xv = torch.stack([ex_to_dense[v] for _, v in pairs])
        return Xu.to(device), Xv.to(device)

    def split_asym(rows, gap=1.0):
        fwd, rev = [], []
        for a, b, sab, sba in ann.bidirectional_pairs(rows):
            if abs(sab - sba) >= gap:
                if sab >= sba:
                    fwd.append((a, b)); rev.append((b, a))
                else:
                    fwd.append((b, a)); rev.append((a, b))
        return fwd, rev

    fwd_tr, rev_tr = split_asym(ann.train_rows)
    all_fwd = fwd_tr + list(tr_pos)
    all_rev = rev_tr + [(b, a) for a, b in tr_pos]
    Xu_tr, Xv_tr = bt(tr_pos + tr_neg)
    y_tr = torch.tensor([1.] * len(tr_pos) + [0.] * len(tr_neg), device=device)
    Xu_f_tr, Xv_f_tr = bt(all_fwd)
    Xu_r_tr, Xv_r_tr = bt(all_rev)

    # pre-stack test tensors per stratum
    strata_tensors = {}
    for label, pairs in strata_pairs.items():
        if pairs:
            strata_tensors[label] = bt(pairs)

    results = {}
    for name, op in [("EduBind", "edubind"), ("MAP", "map")]:
        hits = {label: 0 for label, _, _ in STRATA}
        for seed in range(10):
            torch.manual_seed(7 + seed * 17)
            probe = EduHDC_PrereqProbe(emb_dim, 2048, op, device).to(device)
            opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()
            for _ in range(60):
                opt.zero_grad()
                loss = crit(probe(Xu_tr, Xv_tr), y_tr)
                fs = probe(Xu_f_tr, Xv_f_tr)
                rs = probe(Xu_r_tr, Xv_r_tr)
                loss = loss + 0.5 * F.margin_ranking_loss(
                    fs, rs, torch.ones_like(fs), margin=0.5)
                loss.backward()
                opt.step()
            probe.eval()
            with torch.no_grad():
                for label, (Xu, Xv) in strata_tensors.items():
                    f = probe(Xu, Xv).cpu().numpy()
                    r = probe(Xv, Xu).cpu().numpy()
                    hits[label] += int((f > r).sum())
        res = {}
        for label, pairs in strata_pairs.items():
            n = len(pairs) * 10
            if n == 0:
                continue
            acc = hits[label] / n
            ci = wilson_ci(hits[label], n)
            res[label] = {"dir_acc": acc, "ci95": list(ci),
                          "n_pairs": len(pairs), "pooled_trials": n}
        results[name] = res
        print(f"\n  {name}:")
        for label, _, _ in STRATA:
            if label in res:
                r = res[label]
                print(f"    {label:>8s}: DirAcc {r['dir_acc']:.2%} "
                      f"CI95 [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
                      f"(n={r['n_pairs']})", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_direction_v6_hopdiag_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("hop-stratified direction diagnostic on independent "
                                "Junyi-DAG; frozen v4 config; 10 seeds; pooled-binomial "
                                "Wilson CI. Characterization only, no test-set tuning."),
                   "results": results}, f, indent=2)
    print(f"\n[saved: {out}]")


if __name__ == "__main__":
    main()
