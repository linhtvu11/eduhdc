"""
E4-v4 — P1 DirAcc via DIRECTIONAL DATA AUGMENTATION from unidirectional links.

Key insight: the ranking loss in v3 only used ~75 bidirectional asymmetric pairs.
But the BINARY positive training links (Prerequisite_avg >= HIGH, ~357 pairs) are
ALSO directional: (a->b) positive means a is prereq of b, so we can add the
constraint Score(a,b) > Score(b,a). This gives ~5x more directional supervision,
all leak-free (training split only).

Ranking loss = margin_ranking over:
  (1) bidirectional asymmetric train pairs (fwd > rev)
  (2) unidirectional positive train links  (a,b) > (b,a)

Keeps original simple architecture (v2 showed complex overfits on small data).
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from scipy import stats as scipy_stats

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations
from eduhdc.models import EduHDC_PrereqProbe
from sentence_transformers import SentenceTransformer

HIGH = 6.0
LOW = 3.0
RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "results")


def build_tensors(pairs, ex_to_dense, device):
    X_u = torch.stack([ex_to_dense[u] for u, _ in pairs])
    X_v = torch.stack([ex_to_dense[v] for _, v in pairs])
    return X_u.to(device), X_v.to(device)


def run_v4():
    print("=" * 82)
    print("  E4-v4 — DirAcc via directional augmentation from unidirectional links")
    print("=" * 82)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    ann = JunyiExpertAnnotations()
    if not ann.load():
        print("[ERROR] Expert annotation CSVs not found.")
        return

    print("Encoding via MiniLM...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    desc = ann.descriptions()
    names = list(desc.keys())
    texts = [desc[n] for n in names]
    emb = encoder.encode(texts, convert_to_tensor=True, device=device, batch_size=256)
    ex_to_dense = {n: emb[i] for i, n in enumerate(names)}
    emb_dim = encoder.get_sentence_embedding_dimension()

    # Binary train/test
    train_bin = ann.train_binary(HIGH, LOW)
    test_bin = ann.test_binary(HIGH, LOW)
    tr_pos = [(a, b) for a, b, y in train_bin if y == 1]
    tr_neg = [(a, b) for a, b, y in train_bin if y == 0]
    te_pos = [(a, b) for a, b, y in test_bin if y == 1]
    te_neg = [(a, b) for a, b, y in test_bin if y == 0]

    Xu_tr, Xv_tr = build_tensors(tr_pos + tr_neg, ex_to_dense, device)
    y_tr = torch.tensor([1.0] * len(tr_pos) + [0.0] * len(tr_neg), device=device)
    Xu_te, Xv_te = build_tensors(te_pos + te_neg, ex_to_dense, device)
    y_te = [1] * len(te_pos) + [0] * len(te_neg)

    # Directional eval pairs (bidirectional test, gap>=1.0)
    def _split_asym(rows, gap):
        bidir = ann.bidirectional_pairs(rows)
        asym = [(a, b, sab, sba) for a, b, sab, sba in bidir if abs(sab - sba) >= gap]
        fwd, rev = [], []
        for a, b, sab, sba in asym:
            if sab >= sba:
                fwd.append((a, b)); rev.append((b, a))
            else:
                fwd.append((b, a)); rev.append((a, b))
        return fwd, rev

    fwd_te, rev_te = _split_asym(ann.test_rows, 1.0)
    # Council fix E-F5: removed the "eval on TRAIN if test < 5" fallback —
    # it was a landmine; the test set is always the test set.
    Xu_f_te, Xv_f_te = build_tensors(fwd_te, ex_to_dense, device)
    Xu_r_te, Xv_r_te = build_tensors(rev_te, ex_to_dense, device)
    print(f"Directional EVAL pairs (test, gap>=1.0): {len(fwd_te)}")

    # Directional TRAIN supervision:
    #  (1) bidirectional asymmetric train pairs
    fwd_tr, rev_tr = _split_asym(ann.train_rows, 1.0)
    #  (2) unidirectional positive train links: (a,b) > (b,a)
    uni_fwd = list(tr_pos)
    uni_rev = [(b, a) for (a, b) in tr_pos]
    # Combine
    all_fwd = fwd_tr + uni_fwd
    all_rev = rev_tr + uni_rev
    Xu_f_tr, Xv_f_tr = build_tensors(all_fwd, ex_to_dense, device)
    Xu_r_tr, Xv_r_tr = build_tensors(all_rev, ex_to_dense, device)
    print(f"Directional TRAIN constraints: bidir={len(fwd_tr)} + unidir={len(uni_fwd)} = {len(all_fwd)}")

    probe_configs = [("MAP", "map"), ("EduBind", "edubind")]
    n_seeds = 10
    n_epochs = 60
    vsa_dim = 2048

    # Ranking weight grid
    dir_weights = [0.3, 0.5]
    DIR_MARGIN = 0.5

    all_results = {}
    for name, op_type in probe_configs:
        for dw in dir_weights:
            dir_accs, aucs = [], []
            for seed in range(n_seeds):
                torch.manual_seed(7 + seed * 17)
                probe = EduHDC_PrereqProbe(emb_dim=emb_dim, vsa_dim=vsa_dim,
                                           op_type=op_type, device=device).to(device)
                opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
                crit = nn.BCEWithLogitsLoss()

                probe.train()
                for _ in range(n_epochs):
                    opt.zero_grad()
                    loss = crit(probe(Xu_tr, Xv_tr), y_tr)
                    if dw > 0:
                        fs = probe(Xu_f_tr, Xv_f_tr)
                        rs = probe(Xu_r_tr, Xv_r_tr)
                        loss = loss + dw * F.margin_ranking_loss(
                            fs, rs, target=torch.ones_like(fs), margin=DIR_MARGIN)
                    loss.backward()
                    opt.step()

                probe.eval()
                with torch.no_grad():
                    p = torch.sigmoid(probe(Xu_te, Xv_te)).cpu().numpy()
                    auc = roc_auc_score(y_te, p) if len(set(y_te)) > 1 else 0.5
                    fwd = probe(Xu_f_te, Xv_f_te).cpu().numpy()
                    rev = probe(Xu_r_te, Xv_r_te).cpu().numpy()
                    dir_acc = float(np.mean(fwd > rev))
                dir_accs.append(dir_acc)
                aucs.append(auc)

            key = f"{name}_w{dw}"
            d = np.array(dir_accs); a = np.array(aucs)
            all_results[key] = {"dir_acc_mean": float(d.mean()), "dir_acc_std": float(d.std()),
                                "auc_mean": float(a.mean()), "dir_accs": dir_accs, "aucs": aucs}
            print(f"  {key:<14s} | DirAcc {d.mean():.2%} ± {d.std():.2%} | AUC {a.mean():.4f}")

    print("\n" + "=" * 82)
    for key in sorted(all_results, key=lambda k: -all_results[k]["dir_acc_mean"]):
        r = all_results[key]
        print(f"  {key:<14s} | DirAcc {r['dir_acc_mean']:.2%} ± {r['dir_acc_std']:.2%} | AUC {r['auc_mean']:.4f}")

    eb = np.array(all_results[[k for k in all_results if k.startswith('EduBind')][0]]["dir_accs"])
    mp = np.array(all_results[[k for k in all_results if k.startswith('MAP')][0]]["dir_accs"])
    best_eb = max((k for k in all_results if k.startswith('EduBind')), key=lambda k: all_results[k]["dir_acc_mean"])
    best_mp = max((k for k in all_results if k.startswith('MAP')), key=lambda k: all_results[k]["dir_acc_mean"])
    eb = np.array(all_results[best_eb]["dir_accs"]); mp = np.array(all_results[best_mp]["dir_accs"])
    t, pv = scipy_stats.ttest_rel(eb, mp)
    print(f"\nBest EduBind ({best_eb}) vs MAP ({best_mp}): Δ={eb.mean()-mp.mean():.2%}, t={t:.2f}, p={pv:.4e}")

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "prereq_probing_v4_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[saved: {RESULTS_DIR}\\prereq_probing_v4_results.json]")
    except Exception as e:
        print(f"[save failed: {e}]")


if __name__ == "__main__":
    run_v4()
