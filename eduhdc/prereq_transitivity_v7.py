"""
C1 — Transitivity v7: redesign architecture+data (task #39).

Khác v6 (train trên expert annotations rồi test cross-source → chance), v7 dùng
chính 981 cạnh tiên quyết TRỰC TIẾP của Junyi curriculum làm giám sát hướng
(data enrichment — nhãn prerequisite người thật, KHÔNG phải leak vì input vẫn là
content embedding của humanized name, không dùng h_position/topic làm feature),
rồi test trên các cặp BẮC CẦU held-out (ancestor ≥2 hop, không phải cạnh trực
tiếp). Đây đúng là tiêu chí "Transitivity accuracy" §5.2.4: cho biết các tiên
quyết trực tiếp, suy ra hướng của các cặp gián tiếp.

Kiến trúc:
  (a) EduBind directional probe (toán tử VSA non-commutative) — đóng góp C1.
  (b) MAP probe — đối chứng giao hoán.
  (c) Curriculum potential φ(x)=MLP(emb) hồi quy depth — tham chiếu "transitive
      by construction" (total order), cho biết content có mã hóa depth hay không.

Clean graph: bỏ self-loop, loại các nút trong SCC có chu trình (chỉ 1 SCC 3 nút
+ 2 self-loop ở Junyi). Protocol frozen, không tune trên test.
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

RESULTS_DIR = str(src_dir / "results")
MAX_TEST_PER_STRATUM = 2000


def load_clean_junyi():
    path = src_dir / "data" / "junyi" / "junyi_Exercise_table.csv"
    G = nx.DiGraph()
    with open(str(path), encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            n = row["name"].strip(); G.add_node(n)
            for p in (row.get("prerequisites") or "").split(","):
                p = p.strip()
                if p:
                    G.add_edge(p, n)
    # bỏ self-loop
    G.remove_edges_from(nx.selfloop_edges(G))
    # loại nút trong SCC có chu trình
    cyc_nodes = set()
    for scc in nx.strongly_connected_components(G):
        if len(scc) > 1:
            cyc_nodes |= scc
    G.remove_nodes_from(cyc_nodes)
    return G, cyc_nodes


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def main():
    print("=" * 82)
    print("  C1 Transitivity v7 — DAG-edge supervision, held-out transitive pairs")
    print("=" * 82)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, cyc = load_clean_junyi()
    print(f"Clean Junyi graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(removed {len(cyc)} cyclic nodes); isDAG={nx.is_directed_acyclic_graph(G)}")

    ann = JunyiExpertAnnotations(); ann.load()
    banned = set()
    for r in ann.train_rows + ann.test_rows:
        banned.add(frozenset((r["A"], r["B"])))

    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    names = sorted(set(G.nodes) | set(ann.exercises))
    emb = enc.encode([humanize(n) for n in names], convert_to_tensor=True,
                     device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = enc.get_sentence_embedding_dimension()

    # ---- training: direct edges (fwd=prereq->dependent) ----
    edges = [(u, v) for u, v in G.edges]
    print(f"Training direction supervision: {len(edges)} direct edges")

    # ---- test: transitive pairs (ancestor >=2 hops, not direct edge, not banned) ----
    edge_set = set(G.edges)
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
        print(f"  test stratum {k}: {len(strata[k])} transitive pairs")

    def bt(pairs):
        Xu = torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device)
        Xv = torch.stack([ex_to_dense[v] for _, v in pairs]).to(device)
        return Xu, Xv

    Xu_e, Xv_e = bt(edges)                       # fwd (prereq->dep)
    strata_t = {k: bt(p) for k, p in strata.items() if p}

    # ---- (a)(b) directional probes ----
    results = {}
    for name, op in [("EduBind", "edubind"), ("MAP", "map")]:
        hits = {k: 0 for k in strata}; seed_accs = {k: [] for k in strata}
        for seed in range(5):
            torch.manual_seed(11 + seed * 23)
            probe = EduHDC_PrereqProbe(emb_dim, 2048, op, device).to(device)
            opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(80):
                opt.zero_grad()
                fs = probe(Xu_e, Xv_e); rs = probe(Xv_e, Xu_e)
                loss = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
                loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                for k, (Xu, Xv) in strata_t.items():
                    f = probe(Xu, Xv).cpu().numpy(); r = probe(Xv, Xu).cpu().numpy()
                    hit = (f > r)
                    hits[k] += int(hit.sum()); seed_accs[k].append(float(hit.mean()))
        res = {}
        for k, pairs in strata.items():
            n = len(pairs) * 5
            if n == 0:
                continue
            acc = hits[k] / n; ci = wilson_ci(hits[k], n)
            res[k] = {"dir_acc": acc, "ci95": list(ci), "n_pairs": len(pairs)}
        results[name] = res
        print(f"\n  {name} probe:")
        for k in strata:
            if k in res:
                print(f"    {k:>8s}: DirAcc {res[k]['dir_acc']:.2%} "
                      f"CI95[{res[k]['ci95'][0]:.3f},{res[k]['ci95'][1]:.3f}]")

    # ---- (c) curriculum potential reference ----
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
    phi = nn.Sequential(nn.Linear(emb_dim, 512), nn.GELU(),
                        nn.Linear(512, 1)).to(device)
    optp = optim.Adam(phi.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        optp.zero_grad()
        pred = phi(Xn).squeeze(-1)
        loss = F.mse_loss(pred, yn)
        loss.backward(); optp.step()
    with torch.no_grad():
        phi_val = {n: float(phi(ex_to_dense[n].unsqueeze(0)).item()) for n in nodes}
    res_phi = {}
    for k, pairs in strata.items():
        if not pairs:
            continue
        hit = sum(1 for u, v in pairs if phi_val[v] > phi_val[u])
        acc = hit / len(pairs); ci = wilson_ci(hit, len(pairs))
        res_phi[k] = {"dir_acc": acc, "ci95": list(ci), "n_pairs": len(pairs)}
    results["CurriculumPotential"] = res_phi
    print(f"\n  Curriculum potential φ (depth regression, ref):")
    for k in strata:
        if k in res_phi:
            print(f"    {k:>8s}: DirAcc {res_phi[k]['dir_acc']:.2%} "
                  f"CI95[{res_phi[k]['ci95'][0]:.3f},{res_phi[k]['ci95'][1]:.3f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_transitivity_v7_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("train on Junyi direct edges (direction ranking), test "
                                "held-out transitive pairs >=2 hops; frozen; 5 seeds; "
                                "pooled-binomial Wilson CI"),
                   "n_train_edges": len(edges),
                   "results": results}, f, indent=2)
    print(f"\n[saved: {out}]")


if __name__ == "__main__":
    main()

