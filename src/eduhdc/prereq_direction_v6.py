"""
C1 — Direction & transitivity on an INDEPENDENT held-out test set (task #36).

Council fixes honored (E-F2: the 38-pair expert test set was tiny AND used for
model selection):
  * Protocol is FROZEN at the v4 config (w=0.5, gap=1.0, 60 epochs, lr=0.01) —
    NO hyperparameter touches the new test set.
  * New test set = Junyi curriculum prerequisite DAG (junyi_Exercise_table.csv,
    ~837 exercises / ~988 requires edges) — an INDEPENDENT human-authored
    source of direction, separate from the Chang et al. expert annotations.
  * Held-out pairs = ancestor pairs of the DAG (transitive closure), EXCLUDING
    every unordered pair that appears in ANY expert annotation row (train or
    test, any score) -> zero overlap with all supervision.
Metrics:
  * DirAcc: mean[ score(u,v) > score(v,u) ] over held-out ancestor pairs.
  * Transitivity accuracy (literal §5.2.4 criterion): over 2-hop chains
    u->w->v where (u,v) is NOT a direct edge, mean[ score(u,v) > score(v,u) ].
Statistics: pooled binomial over (pairs x 10 seeds) with Wilson CI, plus
per-seed DirAcc. Units are PAIRS, not seeds (council fix E-F1).
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
MAX_PAIRS, MAX_CHAINS = 2000, 2000
RESULTS_DIR = str(src_dir.parent / "data" / "results")


def load_junyi_dag():
    """Prerequisite DAG from junyi_Exercise_table.csv (name <- prerequisites)."""
    path = src_dir.parent / "data" / "junyi" / "junyi_Exercise_table.csv"
    G = nx.DiGraph()
    with open(str(path), "r", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            G.add_node(name)
            for p in (row.get("prerequisites") or "").split(","):
                p = p.strip()
                if p:
                    G.add_edge(p, name)   # p is prerequisite of name
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
    print("  C1 — independent held-out direction + transitivity (frozen v4 protocol)")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = JunyiExpertAnnotations()
    if not ann.load():
        raise SystemExit("[FAIL] expert annotations not found")
    G = load_junyi_dag()
    print(f"Junyi DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Every unordered pair appearing anywhere in expert annotations (any split,
    # any score) is banned from the DAG test set -> zero supervision overlap.
    banned = set()
    for r in ann.train_rows + ann.test_rows:
        banned.add(frozenset((r["A"], r["B"])))

    # Encoder vocabulary = DAG nodes U annotation exercises (anti-leak names only)
    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()

    # ---- held-out ancestor pairs (direction ground truth from curriculum) ----
    anc_pairs = []
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if u != v and frozenset((u, v)) not in banned:
                anc_pairs.append((u, v))
    rng = np.random.default_rng(0)
    if len(anc_pairs) > MAX_PAIRS:
        idx = sorted(rng.choice(len(anc_pairs), MAX_PAIRS, replace=False).tolist())
        anc_pairs = [anc_pairs[i] for i in idx]
    print(f"Held-out ancestor pairs (annotation-free): {len(anc_pairs)}")

    # ---- held-out 2-hop transitivity chains u->w->v, (u,v) not a direct edge ----
    edge_set = set(G.edges)
    chains = []
    for w in G.nodes:
        for u in G.predecessors(w):
            for v in G.successors(w):
                if (u != v and (u, v) not in edge_set
                        and frozenset((u, v)) not in banned):
                    chains.append((u, w, v))
    if len(chains) > MAX_CHAINS:
        idx = sorted(rng.choice(len(chains), MAX_CHAINS, replace=False).tolist())
        chains = [chains[i] for i in idx]
    print(f"Held-out 2-hop chains: {len(chains)}")

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

    Xu_anc, Xv_anc = bt(anc_pairs)
    Xu_chain = torch.stack([ex_to_dense[u] for u, _, _ in chains]).to(device)
    Xv_chain = torch.stack([ex_to_dense[v] for _, _, v in chains]).to(device)

    results = {}
    for name, op in [("EduBind", "edubind"), ("MAP", "map")]:
        dir_hits = trans_hits = 0
        seed_diraccs = []
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
                f = probe(Xu_anc, Xv_anc).cpu().numpy()
                r = probe(Xv_anc, Xu_anc).cpu().numpy()
                hit = (f > r)
                dir_hits += int(hit.sum())
                seed_diraccs.append(float(hit.mean()))
                fc = probe(Xu_chain, Xv_chain).cpu().numpy()
                rc = probe(Xv_chain, Xu_chain).cpu().numpy()
                trans_hits += int((fc > rc).sum())
        n_dir = len(anc_pairs) * 10
        n_tr = len(chains) * 10
        diracc = dir_hits / n_dir
        transacc = trans_hits / n_tr
        ci_d = wilson_ci(dir_hits, n_dir)
        ci_t = wilson_ci(trans_hits, n_tr)
        results[name] = {
            "dir_acc": diracc, "dir_ci95": list(ci_d),
            "trans_acc": transacc, "trans_ci95": list(ci_t),
            "n_anc_pairs": len(anc_pairs), "n_chains": len(chains),
            "pooled_trials_dir": n_dir, "pooled_trials_trans": n_tr,
            "per_seed_diracc": seed_diraccs,
        }
        print(f"  {name:<8s} | DirAcc {diracc:.2%} CI95 [{ci_d[0]:.3f},{ci_d[1]:.3f}]"
              f" | TransAcc {transacc:.2%} CI95 [{ci_t[0]:.3f},{ci_t[1]:.3f}]",
              flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_direction_v6_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("independent Junyi-DAG held-out direction + 2-hop "
                                "transitivity; frozen v4 config (no tuning on test); "
                                "10 seeds; pooled-binomial Wilson CI over pairs x seeds"),
                   "results": results}, f, indent=2)
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()