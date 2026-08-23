"""
FW3 — Đẩy C1 operator-probe transitivity lên ≥95%.

v7: probe train trên 971 cạnh trực tiếp → 62-78% (cặp gần tốt, cặp xa yếu).
FW3a (richer supervision): train probe trên cạnh trực tiếp + expert asymmetric +
   các cặp bắc cầu sampled (KHÔNG nằm trong test) → thêm giám sát hướng, test
   held-out transitive. Đây là "direction generalization" với giám sát phong phú
   hơn (test pairs vẫn held-out, không leak).
FW3b (depth-distilled multi-task): thêm depth-head phụ trợ chia sẻ concept-encoder
   với task hướng; depth-head dạy encoder cấu trúc curriculum, gián tiếp cải thiện
   operator-based direction scoring. Test vẫn dùng direction score (operator-based).
Giữ anti-leakage (input content embedding), test held-out, không tune trên test.
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
MAX_TRANS_TRAIN = 4000


class DepthDistilledProbe(nn.Module):
    """EduHDC probe + auxiliary depth head chia sẻ concept encoder."""
    def __init__(self, base: EduHDC_PrereqProbe):
        super().__init__()
        self.base = base
        self.depth_head = nn.Sequential(
            nn.Linear(base.actual_dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u, v):
        return self.base(u, v)

    def concept(self, x):
        return self.base.encode_concept(x)

    def depth(self, x):
        # depth head đọc trên self-bind concept (operator tham gia)
        h = self.concept(x)
        bound = self.base.vsa.bind(h, self.base.role_prereq)
        return self.depth_head(bound).squeeze(-1)


def main():
    print("=" * 82)
    print("  FW3 — richer supervision (a) + depth-distilled (b) operator-probe")
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

    edge_set = set(G.edges)
    # collect all ancestor pairs with hop
    all_pairs = {}  # (u,v) -> hop
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if frozenset((u, v)) in banned:
                continue
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            all_pairs[(u, v)] = d

    # TEST: stratified transitive (hop>=2), held out
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

    # TRAIN pairs: direct edges + expert asym + sampled transitive (not in test)
    train_pairs = [(u, v) for u, v in G.edges]  # hop1
    # expert asymmetric
    for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
        if abs(sab - sba) >= 1.0 and frozenset((a, b)) not in test_set:
            train_pairs.append((a, b) if sab > sba else (b, a))
    # sampled transitive (hop>=2, not test)
    trans_pool = [(u, v) for (u, v), d in all_pairs.items()
                  if d >= 2 and (u, v) not in test_set]
    if len(trans_pool) > MAX_TRANS_TRAIN:
        idx = sorted(rng.choice(len(trans_pool), MAX_TRANS_TRAIN, replace=False))
        trans_pool = [trans_pool[i] for i in idx]
    train_pairs.extend(trans_pool)
    print(f"Train pairs: {len(train_pairs)} (edges {len(edge_set)}, "
          f"trans-sampled {len(trans_pool)})")

    def bt(pairs):
        return (torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device),
                torch.stack([ex_to_dense[v] for _, v in pairs]).to(device))

    Xu_tr, Xv_tr = bt(train_pairs)
    strata_t = {k: bt(p) for k, p in strata.items() if p}

    # depth targets
    depth = {n: 0 for n in G.nodes}
    for n in nx.topological_sort(G):
        for s in G.successors(n):
            if depth[s] < depth[n] + 1:
                depth[s] = depth[n] + 1
    nodes = list(G.nodes)
    Xn = torch.stack([ex_to_dense[n] for n in nodes]).to(device)
    yn = torch.tensor([depth[n] for n in nodes], dtype=torch.float32, device=device)
    yn = (yn - yn.mean()) / (yn.std() + 1e-8)

    def evaluate(probe_fn):
        res = {}
        with torch.no_grad():
            for k, (Xu, Xv) in strata_t.items():
                f = probe_fn(Xu, Xv).cpu().numpy()
                r = probe_fn(Xv, Xu).cpu().numpy()
                hit = int((f > r).sum()); n = len(f)
                res[k] = {"dir_acc": hit / n, "ci95": list(wilson_ci(hit, n)),
                          "n_pairs": n}
        return res

    results = {}
    # ---- FW3a: richer supervision, plain EduBind probe ----
    hits_a = {k: 0 for k in strata}
    for seed in range(5):
        torch.manual_seed(31 + seed * 7)
        probe = EduHDC_PrereqProbe(emb_dim, 2048, "edubind", device).to(device)
        opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(100):
            opt.zero_grad()
            fs = probe(Xu_tr, Xv_tr); rs = probe(Xv_tr, Xu_tr)
            loss = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
            loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            for k, (Xu, Xv) in strata_t.items():
                f = probe(Xu, Xv).cpu().numpy(); r = probe(Xv, Xu).cpu().numpy()
                hits_a[k] += int((f > r).sum())
    res_a = {}
    for k, pairs in strata.items():
        n = len(pairs) * 5
        res_a[k] = {"dir_acc": hits_a[k] / n, "ci95": list(wilson_ci(hits_a[k], n)),
                    "n_pairs": len(pairs)}
    results["FW3a_richer_supervision"] = res_a
    print("\nFW3a (richer supervision, EduBind):")
    for k in strata:
        print(f"    {k:>8s}: {res_a[k]['dir_acc']:.2%}")

    # ---- FW3b: depth-distilled multi-task ----
    hits_b = {k: 0 for k in strata}
    for seed in range(5):
        torch.manual_seed(31 + seed * 7)
        base = EduHDC_PrereqProbe(emb_dim, 2048, "edubind", device).to(device)
        probe = DepthDistilledProbe(base).to(device)
        opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(100):
            opt.zero_grad()
            fs = probe(Xu_tr, Xv_tr); rs = probe(Xv_tr, Xu_tr)
            l_rank = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
            l_depth = F.mse_loss(probe.depth(Xn), yn)
            loss = l_rank + 0.3 * l_depth
            loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            for k, (Xu, Xv) in strata_t.items():
                f = probe(Xu, Xv).cpu().numpy(); r = probe(Xv, Xu).cpu().numpy()
                hits_b[k] += int((f > r).sum())
    res_b = {}
    for k, pairs in strata.items():
        n = len(pairs) * 5
        res_b[k] = {"dir_acc": hits_b[k] / n, "ci95": list(wilson_ci(hits_b[k], n)),
                    "n_pairs": len(pairs)}
    results["FW3b_depth_distilled"] = res_b
    print("\nFW3b (depth-distilled, EduBind):")
    for k in strata:
        print(f"    {k:>8s}: {res_b[k]['dir_acc']:.2%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_transitivity_fw3_results.json")
    with open(out, "w") as f:
        json.dump({"protocol": ("FW3a richer supervision (edges+expert+sampled trans, "
                                "test held-out); FW3b depth-distilled multi-task; "
                                "5 seeds; Wilson CI"),
                   "n_train_pairs": len(train_pairs),
                   "results": results}, f, indent=2)
    print(f"\n[saved: {out}]")


if __name__ == "__main__":
    main()

