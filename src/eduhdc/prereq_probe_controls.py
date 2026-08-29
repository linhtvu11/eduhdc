"""
C1 — Transductive + inductive direction probing with OPERATOR CONTROLS.

Addresses three audit findings at once:

  B6  The transductive tier previously reported a POOLED Wilson binomial interval
      over (seeds x pairs) while describing it as a bootstrap over seeds. Pooling
      re-tested pairs as independent Bernoulli trials understates the interval by
      roughly sqrt(n_seeds). Here every run stores per-seed accuracy and the
      interval is a percentile bootstrap over seeds.

  B7  The inductive tier previously drew the node split ONCE (seed 42) and varied
      only `torch.manual_seed`, so its interval covered model-initialisation
      variance but not split variance. Here the node split is drawn inside the
      seed loop, so each seed is an independent (split, init) draw.

  M1  Neither tier had any control: only EduBind was ever run, so the reported
      accuracies could not be attributed to the verified operator. Here every
      tier runs four arms:
        edubind  - the verified non-commutative operator
        map      - commutative elementwise binding (negative control)
        hrr      - commutative circular convolution (negative control)
        concat   - an order-sensitive NON-VSA control: an MLP on [u ; v]

  D3  (2026-08-25) Neither tier had a validation split; the epoch budget was
      fixed by hand and an earlier 100-vs-160-epoch comparison had been made on
      test data. Each arm now holds out VAL_FRAC of its training pool as an
      internal validation set and early-stops on validation direction accuracy
      (patience=PATIENCE, same pattern as kt_experiment_rigorous.py's D3 fix).
      MAP and HRR bind is provably commutative AND associative, so their
      encode_relation(u,v) == encode_relation(v,u) exactly (elementwise product
      / circular convolution do not care about argument order at all): their
      validation accuracy is flat at 0.0 from epoch 1, so early stopping exits
      after MIN_EPOCHS + PATIENCE epochs for those two arms regardless of the
      data — this is the algebraic tie, not an artifact of the new stopping rule.

      The first D3 implementation split 15% of the POOLED training set at random
      and used the pooled validation accuracy as the stopping criterion. That is
      wrong here: MAX_TEST_TRANS=2000 is drawn from the hop2-3 pool BEFORE the
      training pool is built, and the hop2-3 pool is only ~2,187 pairs total, so
      only ~187 pairs remain for hop2-3 training (versus 2,500 each for hop4-6
      and hop7+, capped down from much larger pools). A flat 15% pooled split
      then puts only ~3% of the validation set in hop2-3, so the stopping
      decision is driven almost entirely by hop4-6/hop7+/direct-edge signal,
      which saturates early -- and training stops before the hop2-3 weights
      finish improving. This showed up exactly as predicted: transductive
      hop2-3 accuracy dropped ~8 points while hop7+ was essentially unchanged.
      Fixed by splitting the validation set INDEPENDENTLY per hop stratum and
      averaging the three per-stratum validation accuracies UNWEIGHTED for the
      stopping decision, so hop2-3's signal counts as much as hop7+'s despite
      having roughly 13x fewer training pairs. The "direct" (G.edges) and
      "expert" (bidirectional-annotation) categories are not part of this
      criterion and are not held out at all -- they exist to augment training
      signal, not to be evaluated per-stratum.

Reported per arm and per hop stratum: mean direction accuracy over seeds, a
percentile bootstrap 95% interval over seeds, and the win/loss/tie decomposition
of the pairwise comparison. The tie count matters for the commutative arms: their
forward and reverse scores are numerically identical, so a strict `f > r` test
scores 0% while a random tie-break scores 50%. Both are reported.

Usage:  python src/eduhdc/prereq_probe_controls.py
Output: data/results/prereq_probe_controls_results.json
"""

import json
import os
import pathlib
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader_junyi_expert import JunyiExpertAnnotations, humanize
from eduhdc.models import EduHDC_PrereqProbe
from eduhdc.prereq_transitivity_v7 import load_clean_junyi
from sentence_transformers import SentenceTransformer

RESULTS_DIR = str(src_dir.parent / "data" / "results")
MAX_TEST_TRANS = 2000       # transductive: test pairs per stratum
MAX_TEST_IND = 1500         # inductive:   test pairs per stratum
PER_STRATUM_TRAIN = 2500
TEST_NODE_FRAC = 0.35
N_SEEDS = 10
N_BOOTSTRAP = 2000
MAX_EPOCHS = 200      # early stopping cuts this short in practice
VAL_FRAC = 0.15
PATIENCE = 5
MIN_EPOCHS = 15
ARMS = ["edubind", "map", "hrr", "concat"]
STRATA = ["hop2-3", "hop4-6", "hop7+"]
BOUNDS = {"hop2-3": (2, 3), "hop4-6": (4, 6), "hop7+": (7, 10 ** 9)}


class ConcatMLPProbe(nn.Module):
    """Order-sensitive NON-VSA control: an MLP on the concatenation [u ; v].

    Concatenation breaks the (u, v) symmetry trivially, with no binding operator
    and no algebraic guarantee. If this control matches the VSA arms, the probe
    accuracies say nothing about the verified operator.
    """

    def __init__(self, emb_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, v], dim=-1)).squeeze(-1)


def make_probe(arm: str, emb_dim: int, device: str) -> nn.Module:
    if arm == "concat":
        return ConcatMLPProbe(emb_dim).to(device)
    return EduHDC_PrereqProbe(emb_dim, 2048, arm, device).to(device)


def bootstrap_ci(vals, n_boot=N_BOOTSTRAP, seed=0):
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) < 2:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def train_eval(arm, emb_dim, device, tr_by_cat, strata_t, seed,
                max_epochs=MAX_EPOCHS, val_frac=VAL_FRAC,
                patience=PATIENCE, min_epochs=MIN_EPOCHS):
    """Train one probe with a PER-STRATUM validation split + early stopping,
    then return per-stratum (wins, ties, n) on the held-out test strata.

    tr_by_cat: {category -> (Xu, Xv)} for the training pool, category in
    STRATA ("hop2-3"/"hop4-6"/"hop7+") plus "direct" and "expert" augmentation
    pairs. Only the STRATA categories are held out for validation, split
    INDEPENDENTLY per category so a small stratum (e.g. hop2-3, ~13x fewer
    pairs than hop4-6/hop7+ after test-set extraction) is not drowned out by
    the larger ones; the stopping criterion is the unweighted mean of the
    per-stratum validation accuracies. "direct"/"expert" are never held out.
    """
    torch.manual_seed(31 + seed * 7)
    probe = make_probe(arm, emb_dim, device)
    opt = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)

    tr_u_parts, tr_v_parts = [], []
    val_by_cat = {}
    for ci, cat in enumerate(sorted(tr_by_cat.keys())):
        Xu, Xv = tr_by_cat[cat]
        n = Xu.shape[0]
        if n == 0:
            continue
        if cat in STRATA and n >= 4:
            g = torch.Generator().manual_seed(500 + seed * 13 + ARMS.index(arm) * 97 + ci * 7919)
            perm = torch.randperm(n, generator=g).to(device)
            n_val = max(1, min(int(round(n * val_frac)), n - 1))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            val_by_cat[cat] = (Xu[val_idx], Xv[val_idx])
            tr_u_parts.append(Xu[tr_idx]); tr_v_parts.append(Xv[tr_idx])
        else:
            tr_u_parts.append(Xu); tr_v_parts.append(Xv)
    Xu_tr = torch.cat(tr_u_parts, dim=0)
    Xv_tr = torch.cat(tr_v_parts, dim=0)

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        probe.train()
        opt.zero_grad()
        fs = probe(Xu_tr, Xv_tr)
        rs = probe(Xv_tr, Xu_tr)
        loss = F.margin_ranking_loss(fs, rs, torch.ones_like(fs), margin=0.5)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            cat_accs = []
            for Au, Av in val_by_cat.values():
                vf = probe(Au, Av)
                vr = probe(Av, Au)
                cat_accs.append((vf > vr).float().mean().item())
            val_acc = float(np.mean(cat_accs)) if cat_accs else 0.0
        if val_acc > best_val + 1e-4:
            best_val, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= min_epochs and bad >= patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    out = {}
    with torch.no_grad():
        for k, (Au, Av) in strata_t.items():
            f = probe(Au, Av).float().cpu().numpy()
            r = probe(Av, Au).float().cpu().numpy()
            out[k] = {"wins": int((f > r).sum()), "ties": int((f == r).sum()), "n": len(f)}
    return out


def to_tensors(pairs, ex_to_dense, device):
    return (torch.stack([ex_to_dense[u] for u, _ in pairs]).to(device),
            torch.stack([ex_to_dense[v] for _, v in pairs]).to(device))


def summarise(per_seed):
    """per_seed: list over seeds of {wins, ties, n} -> summary dict."""
    strict = [d["wins"] / d["n"] for d in per_seed]
    tieb = [(d["wins"] + 0.5 * d["ties"]) / d["n"] for d in per_seed]
    lo_s, hi_s = bootstrap_ci(strict)
    lo_t, hi_t = bootstrap_ci(tieb)
    return {
        "n_pairs_mean": float(np.mean([d["n"] for d in per_seed])),
        "n_seeds": len(per_seed),
        "dir_acc_strict": float(np.mean(strict)),
        "dir_acc_strict_sd": float(np.std(strict, ddof=1)) if len(strict) > 1 else 0.0,
        "bootstrap_ci95_strict": [lo_s, hi_s],
        "dir_acc_tiebreak": float(np.mean(tieb)),
        "bootstrap_ci95_tiebreak": [lo_t, hi_t],
        "tie_fraction": float(np.mean([d["ties"] / d["n"] for d in per_seed])),
        "per_seed_strict": strict,
    }


def build_pairs(G, banned):
    ap = {}
    for v in G.nodes:
        for u in nx.ancestors(G, v):
            if frozenset((u, v)) in banned:
                continue
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            ap[(u, v)] = d
    return ap


def main():
    print("=" * 84)
    print("  C1 direction probing with operator controls (audit B6 / B7 / M1)")
    print("=" * 84, flush=True)
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
    ap = build_pairs(G, banned)
    print(f"transitive pairs available (expert pairs excluded): {len(ap)}", flush=True)

    results = {"protocol": {
        "n_seeds": N_SEEDS, "max_epochs": MAX_EPOCHS, "arms": ARMS,
        "ci": "percentile bootstrap over seeds (NOT a pooled binomial interval)",
        "transductive": "test pairs held out; every seed redraws the test subsample",
        "inductive": ("node split redrawn INSIDE the seed loop, so each seed is an "
                      "independent (split, init) draw; both nodes of every test pair unseen"),
        "validation": (f"D3 fix: {int(VAL_FRAC*100)}% held out INDEPENDENTLY per hop stratum "
                        f"(hop2-3/hop4-6/hop7+) from the training pool, NOT a single pooled split "
                        f"-- a pooled split let the ~13x-larger hop4-6/hop7+ pools drown out "
                        f"hop2-3's stopping signal. Early-stops on the unweighted mean of the "
                        f"three per-stratum validation accuracies, patience={PATIENCE} after at "
                        f"least {MIN_EPOCHS} epochs, restoring the best-validation checkpoint "
                        f"before test evaluation. 'direct'/'expert' augmentation pairs are never "
                        f"held out. Test strata are never used for the stopping decision."),
        "note": ("dir_acc_strict uses f(u,v) > f(v,u); commutative arms tie exactly, so "
                 "dir_acc_tiebreak = (wins + ties/2)/n is also reported"),
    }}

    # ---------------------------------------------------------------- TRANSDUCTIVE
    print("\n--- TRANSDUCTIVE (pair-held-out) ---", flush=True)
    trans = {a: {k: [] for k in STRATA} for a in ARMS}
    node_overlap = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        strata = {k: [] for k in STRATA}
        for (u, v), d in ap.items():
            if d < 2:
                continue
            key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
            strata[key].append((u, v))
        for k in strata:
            if len(strata[k]) > MAX_TEST_TRANS:
                idx = sorted(rng.choice(len(strata[k]), MAX_TEST_TRANS, replace=False))
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
            lo, hi = BOUNDS[k]
            pool = [(u, v) for (u, v), d in ap.items()
                    if lo <= d <= hi and (u, v) not in test_set]
            if len(pool) > PER_STRATUM_TRAIN:
                idx = sorted(rng.choice(len(pool), PER_STRATUM_TRAIN, replace=False))
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
        tr_by_cat_t = {c: to_tensors(p, ex_to_dense, device) for c, p in tr_by_cat.items() if p}
        st_t = {k: to_tensors(p, ex_to_dense, device) for k, p in strata.items() if p}
        for arm in ARMS:
            out = train_eval(arm, emb_dim, device, tr_by_cat_t, st_t, seed)
            for k in STRATA:
                trans[arm][k].append(out[k])
        print(f"  seed {seed}: train={len(tr)} test={len(test_set)} "
              f"node-overlap={node_overlap[-1]:.3f} | "
              + " ".join(f"{a}={trans[a]['hop2-3'][-1]['wins']/trans[a]['hop2-3'][-1]['n']:.3f}"
                         for a in ARMS), flush=True)
    results["transductive"] = {a: {k: summarise(trans[a][k]) for k in STRATA} for a in ARMS}
    results["transductive_test_node_overlap_mean"] = float(np.mean(node_overlap))

    # ---------------------------------------------------------------- INDUCTIVE
    print("\n--- INDUCTIVE (node-held-out; split redrawn per seed) ---", flush=True)
    ind = {a: {k: [] for k in STRATA} for a in ARMS}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(2000 + seed)
        nodes = list(G.nodes)
        perm = rng.permutation(len(nodes))
        n_test = int(len(nodes) * TEST_NODE_FRAC)
        test_nodes = {nodes[i] for i in perm[:n_test]}
        train_nodes = {nodes[i] for i in perm[n_test:]}
        strata = {k: [] for k in STRATA}
        for (u, v), d in ap.items():
            if d < 2 or u not in test_nodes or v not in test_nodes:
                continue
            key = "hop2-3" if d <= 3 else ("hop4-6" if d <= 6 else "hop7+")
            strata[key].append((u, v))
        for k in strata:
            if len(strata[k]) > MAX_TEST_IND:
                idx = sorted(rng.choice(len(strata[k]), MAX_TEST_IND, replace=False))
                strata[k] = [strata[k][i] for i in idx]
        tr_by_cat = {"direct": [(u, v) for u, v in G.edges
                                 if u in train_nodes and v in train_nodes]}
        expert_pairs = []
        for a, b, sab, sba in ann.bidirectional_pairs(ann.train_rows):
            if abs(sab - sba) >= 1.0 and a in train_nodes and b in train_nodes:
                expert_pairs.append((a, b) if sab > sba else (b, a))
        tr_by_cat["expert"] = expert_pairs
        for k in STRATA:
            lo, hi = BOUNDS[k]
            pool = [(u, v) for (u, v), d in ap.items()
                    if lo <= d <= hi and u in train_nodes and v in train_nodes]
            if len(pool) > PER_STRATUM_TRAIN:
                idx = sorted(rng.choice(len(pool), PER_STRATUM_TRAIN, replace=False))
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
        tr_by_cat_t = {c: to_tensors(p, ex_to_dense, device) for c, p in tr_by_cat.items() if p}
        st_t = {k: to_tensors(p, ex_to_dense, device) for k, p in strata.items() if p}
        for arm in ARMS:
            out = train_eval(arm, emb_dim, device, tr_by_cat_t, st_t, seed)
            for k in STRATA:
                if k in out:
                    ind[arm][k].append(out[k])
        sizes = {k: len(strata[k]) for k in STRATA}
        print(f"  seed {seed}: train={len(tr)} test={sizes} | "
              + " ".join(f"{a}={ind[a]['hop7+'][-1]['wins']/ind[a]['hop7+'][-1]['n']:.3f}"
                         for a in ARMS), flush=True)
    results["inductive"] = {a: {k: summarise(ind[a][k]) for k in STRATA if ind[a][k]}
                            for a in ARMS}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "prereq_probe_controls_results.json")
    with open(out, "w", encoding="utf-8") as f:
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
    print(f"\ntransductive test-node overlap with train: "
          f"{results['transductive_test_node_overlap_mean']:.3f}")
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()
