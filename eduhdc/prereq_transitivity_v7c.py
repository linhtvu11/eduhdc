"""
C1 — Transitivity v7c: ensemble EduBind-probe + curriculum-potential.

v7 cho thấy probe (tốt cặp gần) và potential (tốt cặp xa) bù trừ nhau. v7c kết
hợp hai decision function (chuẩn hóa margin bằng thống kê trên TRAIN-edges để không
chạm test) và đo DirAcc overall + từng tầng. Nếu ensemble đạt cao đồng đều → củng
cố kết luận transitivity từ content.
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
from eduhdc.prereq_transitivity_v7 import load_clean_junyi, wilson_ci
from sentence_transformers import SentenceTransformer

RESULTS_DIR = str(src_dir.parent / "data" / "results")
MAX_TEST_PER_STRATUM = 2000


def main():
    print("=" * 82)
    print("  C1 Transitivity v7c — ensemble probe + potential")
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

    edges = list(G.edges)
    edge_set = set(edges)
    strata = {"hop2-3": [], "hop4-6": [], "hop7+": []}
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if (u, v) in edge_set or frozenset((u, v)) in banned:
                continue
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            if d < 2:
                continue
            key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
            strata[key].append((u, v))
    rng = np.random.default_rng(0)
    for k in strata:
        if len(strata[k]) > MAX_TEST_PER_STRATUM:
            idx = sorted(rng.choice(len(strata[k]), MAX_TEST_PER_STRATUM, replace=False))
            strata[k] = [strata[k][i] for i in idx]

    def bt(pairs):
        return (torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device),
                torch.stack([ex_to_dense[v] for _, v in pairs]).to(device))

    Xu_e, Xv_e = bt(edges)

    # train probe (5 seeds, giữ margin train)
    probes = []
    for seed in range(5):
        torch.manual_seed(11 + seed * 23)
        pr = EduHDC_PrereqProbe(emb_dim, 2048, "edubind", device).to(device)
        opt = optim.Adam(pr.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(80):
            opt.zero_grad()
            fs = pr(Xu_e, Xv_e); rs = pr(Xv_e, Xu_e)
            loss = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
            loss.backward(); opt.step()
        pr.eval(); probes.append(pr)

    # train potential
    depth = {n: 0 for n in G.nodes}
    for n in nx.topological_sort(G):
        for s in G.successors(n):
            if depth[s] < depth[n] + 1:
                depth[s] = depth[n] + 1
    nodes = list(G.nodes)
    Xn = torch.stack([ex_to_dense[n] for n in nodes]).to(device)
    yn = torch.tensor([depth[n] for n in nodes], dtype=torch.float32, device=device)
    yn = (yn - yn.mean()) / (yn.std() + 1e-8)
    torch.manual_seed(0)
    phi = nn.Sequential(nn.Linear(emb_dim, 512), nn.GELU(), nn.Linear(512, 1)).to(device)
    optp = optim.Adam(phi.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        optp.zero_grad(); loss = F.mse_loss(phi(Xn).squeeze(-1), yn)
        loss.backward(); optp.step()
    phi.eval()

    # margin trên TRAIN edges để chuẩn hóa (không dùng test)
    with torch.no_grad():
        pm_tr = []
        for pr in probes:
            pm_tr.append((pr(Xu_e, Xv_e) - pr(Xv_e, Xu_e)).cpu().numpy())
        pm_tr = np.mean(pm_tr, axis=0)
        pot_tr = (phi(Xv_e).squeeze(-1) - phi(Xu_e).squeeze(-1)).cpu().numpy()
    pm_mu, pm_sd = pm_tr.mean(), pm_tr.std() + 1e-8
    pot_mu, pot_sd = pot_tr.mean(), pot_tr.std() + 1e-8

    results = {}
    with torch.no_grad():
        for k, pairs in strata.items():
            if not pairs:
                continue
            Xu, Xv = bt(pairs)
            pm = np.mean([(pr(Xu, Xv) - pr(Xv, Xu)).cpu().numpy() for pr in probes], axis=0)
            pot = (phi(Xv).squeeze(-1) - phi(Xu).squeeze(-1)).cpu().numpy()
            pm_z = (pm - pm_mu) / pm_sd
            pot_z = (pot - pot_mu) / pot_sd
            ens = 0.5 * pm_z + 0.5 * pot_z
            n = len(pairs)
            acc_probe = float((pm > 0).mean())
            acc_pot = float((pot > 0).mean())
            acc_ens = float((ens > 0).mean())
            results[k] = {
                "n_pairs": n,
                "probe": acc_probe, "potential": acc_pot, "ensemble": acc_ens,
                "ensemble_ci95": list(wilson_ci(int((ens > 0).sum()), n)),
            }
            print(f"  {k:>8s}: probe {acc_probe:.2%} | potential {acc_pot:.2%} | "
                  f"ENSEMBLE {acc_ens:.2%}")

    allpairs = sum(len(p) for p in strata.values())
    wens = sum(results[k]["ensemble"] * results[k]["n_pairs"] for k in results) / allpairs
    print(f"\n  Overall ensemble DirAcc (weighted): {wens:.2%}")
    results["overall_weighted_ensemble"] = float(wens)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_transitivity_v7c_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("ensemble EduBind-probe(avg5) + curriculum-potential; "
                                "margins normalized by TRAIN-edge stats; held-out "
                                "transitive pairs >=2 hops"),
                   "results": results}, f, indent=2)
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()
