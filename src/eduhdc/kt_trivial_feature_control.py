"""
C1 — Trivial-feature control for the VSA ablation (review finding bg-1 F2).

WHY THIS EXISTS
---------------
`kt_vsa_ablation.py` shows that zeroing the two VSA-derived readout features
(`sim_uni`, `sim_bi`) costs +0.027 AUC. An independent reviewer objected that
this ablation cannot separate two different explanations:

  (H1) the VSA bind/unbind COMPUTATION carries the signal, or
  (H2) the readout merely benefits from having TWO MORE INPUT SLOTS of any
       kind, and the five classical KT features would route through them
       just as well.

Zeroing a feature removes both its content AND its slot, so the original
ablation confounds the two. This script adds the missing control: instead of
zeroing the two VSA slots, it fills them with NON-VSA content of matched
shape and matched trainability, so the readout keeps exactly seven live
inputs in every arm.

ARMS
----
  full        EduHDC-KT, unmodified                      (7 feats, 2 from VSA)
  zeroed      sim_uni = sim_bi = 0                       (5 live feats) [replicates M12]
  scalar      sim_uni, sim_bi <- 2 learnable scalars     (7 feats, 2 non-VSA, 2 params)
  classical   sim_uni, sim_bi <- 2 cheap causal KT stats (7 feats, 2 non-VSA, 0 params)
  DKT         LSTM reference baseline

`scalar` is the literal reading of the reviewer's question: two free
parameters, no VSA arithmetic, broadcast over every (student, step). It tests
whether the readout needs the slots or the content. Because a per-batch
constant cannot vary within a sequence, `classical` is the stronger version of
the same control: two genuinely informative non-VSA features of the same
shape as the VSA ones, so the slots carry real per-step signal that owes
nothing to binding.

The two `classical` substitutes are deliberately cheap and strictly causal,
computed from information available strictly before the predicted step:
  prev_correct  the learner's response at t-1 (0/1, 0.5 before the first step)
  global_rate   the learner's running accuracy over ALL skills up to t-1
Both are classical KT statistics. Note that the five features already present
include `crate`, the running accuracy on the CURRENT skill; `global_rate` is
its skill-agnostic counterpart, and `prev_correct` is the shortest-range
temporal signal available. If either recovers the ablation gap, the +0.027 is
about feature count and cheap recency, not about the binding algebra.

PROTOCOL
--------
Identical to `kt_vsa_ablation.py` and `kt_experiment_rigorous.py`: same data,
same 5 folds, same optimiser, same epoch budget, same early stopping, same
per-fold seed. The only thing that varies between arms is what occupies the
two readout slots.

Usage:  python src/eduhdc/kt_trivial_feature_control.py
Output: data/results/kt_trivial_feature_control_results.json
"""

import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.model_selection import KFold

src_dir = pathlib.Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader_real import load_assistments_real
from eduhdc.kt_experiment_rigorous import train_and_eval_kt_model
from eduhdc.models import DKT_Baseline, EduHDC_KT

RESULTS_DIR = str(src_dir.parent / "data" / "results")
N_STUDENTS = 5000     # same as the main 5-fold benchmark
N_FOLDS = 5
VSA_DIM = 2048
EPOCHS = 15
LR = 0.005


def _classical_features(sk, co, ma, num_skills):
    """The five classical KT features of EduHDC_KT's readout, recomputed here.

    Kept byte-for-byte identical to `NoVSA_KT.forward_batch_fast` in
    kt_vsa_ablation.py so the arms differ ONLY in the two VSA slots. All five
    are strictly causal: every cumulative quantity is shifted by one step, so
    nothing at index t uses the response at index t.
    """
    B, T = sk.shape
    dev = sk.device
    one_hot = F.one_hot(sk, num_skills).float()
    zeros_k = torch.zeros(B, 1, num_skills, device=dev)

    cum = torch.cumsum(one_hot, 1)
    cnt = torch.cat([zeros_k, cum[:, :-1]], 1).gather(2, sk.unsqueeze(-1)).squeeze(-1)

    coh = one_hot * co.float().unsqueeze(-1)
    cc = torch.cumsum(coh, 1)
    cs = torch.cat([zeros_k, cc[:, :-1]], 1).gather(2, sk.unsqueeze(-1)).squeeze(-1)

    pos = torch.arange(T, device=dev).float().view(1, T, 1)
    cm = torch.cummax((pos + 1) * one_hot, 1).values
    ls = torch.cat([zeros_k, cm[:, :-1]], 1).gather(2, sk.unsqueeze(-1)).squeeze(-1) - 1

    t_idx = torch.arange(T, device=dev, dtype=torch.float32)
    recency = torch.where(ls >= 0, (t_idx.view(1, T) - ls) / 100.0,
                          torch.ones(B, T, device=dev))
    count_norm = torch.clamp(cnt, max=20.0) / 20.0
    crate = torch.where(cnt > 0, cs / torch.clamp(cnt, min=1.0),
                        torch.full((B, T), 0.5, device=dev))
    pos_feat = t_idx.view(1, T).expand(B, T) / 100.0
    return pos_feat, recency, count_norm, crate


class ZeroedVSA_KT(EduHDC_KT):
    """sim_uni = sim_bi = 0. Replicates `NoVSA_KT` from kt_vsa_ablation.py.

    Included here rather than imported so that all four arms are trained in
    one process against one data load, removing any doubt that the replicated
    gap comes from a different data draw.
    """

    def forward_batch_fast(self, sk, co, ma):
        B, T = sk.shape
        dev = sk.device
        pos_feat, recency, count_norm, crate = _classical_features(
            sk, co, ma, self.num_skills)
        s_bias = self.skill_bias(sk).squeeze(-1)
        zero = torch.zeros(B, T, device=dev)
        feat = torch.stack(
            [zero, zero, s_bias, pos_feat, recency, count_norm, crate], -1)
        return torch.sigmoid(self.readout(feat).squeeze(-1))


class ScalarSlot_KT(EduHDC_KT):
    """The two VSA slots become two free learnable scalars — no VSA arithmetic.

    This is the reviewer's literal proposal. The readout sees seven live
    inputs, two of which are constants the optimiser may set freely. Note what
    this can and cannot do: a constant broadcast over every step cannot encode
    per-step information, so it is absorbable into the readout's bias. If this
    arm matches `full`, the VSA features were not carrying per-step signal; if
    it matches `zeroed` instead, then the slots alone buy nothing and the gap
    is about content. Either outcome is informative, which is why the weaker
    control is worth running alongside the stronger `classical` one.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.slot = nn.Parameter(torch.zeros(2))

    def forward_batch_fast(self, sk, co, ma):
        B, T = sk.shape
        pos_feat, recency, count_norm, crate = _classical_features(
            sk, co, ma, self.num_skills)
        s_bias = self.skill_bias(sk).squeeze(-1)
        s0 = self.slot[0].expand(B, T)
        s1 = self.slot[1].expand(B, T)
        feat = torch.stack(
            [s0, s1, s_bias, pos_feat, recency, count_norm, crate], -1)
        return torch.sigmoid(self.readout(feat).squeeze(-1))


class ClassicalSlot_KT(EduHDC_KT):
    """The two VSA slots become two cheap, strictly causal, non-VSA KT statistics.

    slot 0 = prev_correct : response at t-1 (0.5 at t=0, i.e. uninformative prior)
    slot 1 = global_rate  : running accuracy over ALL skills strictly before t

    Both vary per (student, step) exactly as the VSA similarities do, so the
    readout receives the same SHAPE of information without any bind/unbind.
    This is the control that actually answers the reviewer's question: if the
    +0.027 gap closes here, the ablation was measuring "two more informative
    per-step features," not "the binding algebra."
    """

    def forward_batch_fast(self, sk, co, ma):
        B, T = sk.shape
        dev = sk.device
        pos_feat, recency, count_norm, crate = _classical_features(
            sk, co, ma, self.num_skills)
        s_bias = self.skill_bias(sk).squeeze(-1)

        cof = co.float()
        maf = ma.float()
        # slot 0: previous response, masked, 0.5 before the first observed step
        prev_correct = torch.cat(
            [torch.full((B, 1), 0.5, device=dev), cof[:, :-1]], 1)
        prev_valid = torch.cat(
            [torch.zeros(B, 1, device=dev), maf[:, :-1]], 1)
        prev_correct = torch.where(prev_valid > 0, prev_correct,
                                   torch.full((B, T), 0.5, device=dev))

        # slot 1: running accuracy over all skills, strictly causal
        c_cum = torch.cumsum(cof * maf, 1)
        n_cum = torch.cumsum(maf, 1)
        c_before = torch.cat([torch.zeros(B, 1, device=dev), c_cum[:, :-1]], 1)
        n_before = torch.cat([torch.zeros(B, 1, device=dev), n_cum[:, :-1]], 1)
        global_rate = torch.where(n_before > 0,
                                  c_before / torch.clamp(n_before, min=1.0),
                                  torch.full((B, T), 0.5, device=dev))

        feat = torch.stack([prev_correct, global_rate, s_bias, pos_feat,
                            recency, count_norm, crate], -1)
        return torch.sigmoid(self.readout(feat).squeeze(-1))


def main():
    print("=" * 84)
    print("  C1 — trivial-feature control for the VSA ablation (review bg-1 F2)")
    print("=" * 84, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | torch={torch.__version__}", flush=True)

    seqs_d, skills = load_assistments_real(max_students=N_STUDENTS, min_seq_len=20,
                                           max_seq_len=200, seed=42)
    skill_list = sorted(skills)
    s2i = {s: i for i, s in enumerate(skill_list)}
    K = len(skill_list)
    seqs = list(seqs_d.values())
    print(f"students={len(seqs)} skills={K} "
          f"interactions={sum(len(s) for s in seqs):,}", flush=True)

    arms = [
        ("full (2 VSA features)",
         lambda: EduHDC_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
        ("zeroed (VSA feats = 0)",
         lambda: ZeroedVSA_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
        ("scalar slots (2 learnable)",
         lambda: ScalarSlot_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
        ("classical slots (2 KT stats)",
         lambda: ClassicalSlot_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
        ("DKT (LSTM)",
         lambda: DKT_Baseline(num_skills=K, emb_dim=64, hidden_dim=128)),
    ]

    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(seqs))
    out = {}
    for name, mk in arms:
        aucs, accs, lats = [], [], []
        for fi, (tr, te) in enumerate(folds):
            torch.manual_seed(42)
            np.random.seed(42)
            auc, acc, lat = train_and_eval_kt_model(
                mk(), [seqs[i] for i in tr], [seqs[i] for i in te], s2i,
                epochs=EPOCHS, lr=LR, device=device)
            aucs.append(auc)
            accs.append(acc)
            lats.append(lat)
            print(f"  {name:30s} fold{fi} AUC={auc:.4f} acc={acc:.4f} "
                  f"lat={lat:.4f}ms", flush=True)
        n_tr = sum(p.numel() for p in mk().parameters() if p.requires_grad)
        out[name] = {"aucs": aucs, "accs": accs, "latencies_ms": lats,
                     "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs, ddof=1)),
                     "latency_ms_mean": float(np.mean(lats)),
                     "trainable_params": int(n_tr)}
        print(f"=> {name:30s} AUC {np.mean(aucs):.4f} +- {np.std(aucs, ddof=1):.4f} "
              f"| {n_tr:,} params", flush=True)

    A = {k: np.array(v["aucs"]) for k, v in out.items() if "aucs" in v}

    def paired(a, b):
        t, p = scipy_stats.ttest_rel(a, b)
        sd = (a - b).std(ddof=1)
        dz = float((a - b).mean() / sd) if sd > 0 else 0.0
        return {"delta": float((a - b).mean()), "t": float(t), "p": float(p),
                "cohen_dz": dz}

    full = A["full (2 VSA features)"]
    zeroed = A["zeroed (VSA feats = 0)"]
    scalar = A["scalar slots (2 learnable)"]
    classical = A["classical slots (2 KT stats)"]
    dkt = A["DKT (LSTM)"]

    out["comparisons"] = {
        # the original M12 claim, replicated inside this run
        "full - zeroed  (replicates +0.027)": paired(full, zeroed),
        # does the READOUT just want two more slots?
        "full - scalar  (slots alone)": paired(full, scalar),
        "scalar - zeroed (slot value of a constant)": paired(scalar, zeroed),
        # the real test: two non-VSA per-step features of the same shape
        "full - classical (VSA vs cheap KT stats)": paired(full, classical),
        "classical - zeroed (gap recovered by non-VSA)": paired(classical, zeroed),
        # references
        "full - dkt": paired(full, dkt),
        "classical - dkt": paired(classical, dkt),
    }
    out["config"] = {"n_students": len(seqs), "n_folds": N_FOLDS, "num_skills": K,
                     "vsa_dim": VSA_DIM, "epochs": EPOCHS, "lr": LR,
                     "protocol": "identical to kt_vsa_ablation / kt_experiment_rigorous"}
    out["interpretation_guide"] = {
        "if classical - zeroed ~ full - zeroed":
            "the +0.027 is about having two informative per-step features, "
            "NOT about the binding algebra; report the ablation as a "
            "feature-count effect.",
        "if classical - zeroed << full - zeroed":
            "cheap causal statistics do NOT substitute for the VSA "
            "similarities; the ablation gap is specific to what bind/unbind "
            "computes (still not specific to non-commutativity, which the "
            "MAP substitution already addresses).",
        "if scalar ~ zeroed":
            "slot count alone buys nothing, as expected for a constant "
            "absorbable into the readout bias.",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "kt_trivial_feature_control_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 84)
    for k, v in out["comparisons"].items():
        print(f"  {k:44s} delta={v['delta']:+.5f}  p={v['p']:.4g}  "
              f"dz={v['cohen_dz']:+.2f}")
    print(f"\n[saved: {path}]")


if __name__ == "__main__":
    main()
