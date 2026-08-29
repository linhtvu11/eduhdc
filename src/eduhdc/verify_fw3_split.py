"""
Verification: xác nhận KHÔNG trùng lặp giữa tập train (sampled transitive pairs)
và tập test (held-out transitive pairs) của FW3c.

Kiểm tra:
  1. Trùng CẶP directed (u,v): train ∩ test phải = 0.
  2. Trùng CẶP unordered frozenset{u,v}: phải = 0 (loại cả trường hợp (u,v) vs (v,u)).
  3. Trùng NODE: đếm node xuất hiện ở cả train-pairs lẫn test-pairs. Vì probe dùng
     content embedding CỐ ĐỊNH + không có tham số per-node, trùng node KHÔNG phải
     leak — nhưng vẫn lượng hóa để báo cáo trung thực.
  4. Phân loại test-pairs theo số node đã thấy trong train (2/1/0 node).
Tái tạo ĐÚNG logic + RNG seed của prereq_transitivity_fw3c.py.
"""

import sys
import pathlib
import numpy as np
import networkx as nx

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations
from eduhdc.prereq_transitivity_v7 import load_clean_junyi

MAX_TEST_PER_STRATUM = 2000
PER_STRATUM_TRAIN = 2500


def main():
    G, cyc = load_clean_junyi()
    ann = JunyiExpertAnnotations(); ann.load()
    banned = {frozenset((r["A"], r["B"])) for r in ann.train_rows + ann.test_rows}

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

    # ---- tái tạo TEST set (đúng rng seed 0 như fw3c) ----
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
    test_pairs = []
    for k in strata:
        test_pairs.extend(strata[k])
    test_set = set(test_pairs)
    test_unordered = {frozenset(p) for p in test_pairs}

    # ---- tái tạo TRAIN transitive pairs (đúng logic fw3c) ----
    train_trans = []
    for key in strata:
        lo, hi = {"hop2-3": (2, 3), "hop4-6": (4, 6), "hop7+": (7, 10**9)}[key]
        pool = [(u, v) for (u, v), d in all_pairs.items()
                if lo <= d <= hi and (u, v) not in test_set]
        if len(pool) > PER_STRATUM_TRAIN:
            idx = sorted(rng.choice(len(pool), PER_STRATUM_TRAIN, replace=False))
            pool = [pool[i] for i in idx]
        train_trans.extend(pool)
    train_set = set(train_trans)
    train_unordered = {frozenset(p) for p in train_trans}

    print(f"Test transitive pairs : {len(test_pairs)}")
    print(f"Train transitive pairs: {len(train_trans)}")

    # 1. trùng cặp directed
    inter_dir = train_set & test_set
    print(f"\n[1] Trùng cặp directed (u,v): {len(inter_dir)}  (phải = 0)")

    # 2. trùng cặp unordered
    inter_unord = train_unordered & test_unordered
    print(f"[2] Trùng cặp unordered {{u,v}}: {len(inter_unord)}  (phải = 0)")

    # 3. trùng node
    test_nodes = set()
    for u, v in test_pairs:
        test_nodes.add(u); test_nodes.add(v)
    train_nodes = set()
    for u, v in train_trans:
        train_nodes.add(u); train_nodes.add(v)
    inter_nodes = train_nodes & test_nodes
    print(f"\n[3] Node trong test-pairs   : {len(test_nodes)}")
    print(f"    Node trong train-pairs  : {len(train_nodes)}")
    print(f"    Node trùng (train∩test) : {len(inter_nodes)} "
          f"({100*len(inter_nodes)/max(1,len(test_nodes)):.1f}% test-nodes)")

    # 4. phân loại test-pairs theo số node đã thấy trong train
    both = one = zero = 0
    for u, v in test_pairs:
        cu = u in train_nodes; cv = v in train_nodes
        if cu and cv:
            both += 1
        elif cu or cv:
            one += 1
        else:
            zero += 1
    print(f"\n[4] Test-pairs phân theo node đã thấy trong train-trans:")
    print(f"    cả 2 node đã thấy : {both} ({100*both/len(test_pairs):.1f}%)")
    print(f"    1 node đã thấy    : {one} ({100*one/len(test_pairs):.1f}%)")
    print(f"    0 node (mới hoàn toàn): {zero} ({100*zero/len(test_pairs):.1f}%)")

    # 5. kiểm tra thêm: test-pair có trùng direct-edge hay expert không
    edge_set = set(G.edges)
    n_edge = sum(1 for p in test_pairs if p in edge_set)
    n_banned = sum(1 for p in test_pairs if frozenset(p) in banned)
    print(f"\n[5] Test-pairs trùng direct-edge: {n_edge} (phải = 0, vì hop>=2)")
    print(f"    Test-pairs trùng expert-banned: {n_banned} (phải = 0)")

    print("\n" + "=" * 70)
    if len(inter_dir) == 0 and len(inter_unord) == 0:
        print("✅ KHÔNG trùng cặp (directed lẫn unordered) giữa train và test.")
    else:
        print("❌ CÓ TRÙNG CẶP — cần sửa!")
    print(f"ℹ️  Trùng node: {len(inter_nodes)} node — probe dùng content embedding CỐ ĐỊNH")
    print("   + KHÔNG có tham số per-node, nên trùng node KHÔNG phải leak (mô hình")
    print("   khái quát từ content, không 'nhớ' từng node). Lượng hóa ở [4].")


if __name__ == "__main__":
    main()
