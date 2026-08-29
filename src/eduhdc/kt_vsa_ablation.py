"""
C1 — How much of EduHDC-KT's AUC comes from the VSA? (audit fix M12)

`EduHDC_KT.forward_batch_fast` feeds a 7-dimensional feature vector to its
readout, and only TWO of those features come from the VSA:

    [ sim_uni, sim_bi,                                <- VSA bind/unbind
      s_bias, pos_feat, recency, count_norm, crate ]  <- classical KT features

`crate` is the learner's running accuracy on the current skill, `s_bias` is a
per-skill difficulty term (an IRT intercept), and `recency`/`count_norm` are
exposure statistics. Those five alone are a strong knowledge-tracing baseline, so
parity with DKT does not by itself show that the binding operator contributes.

This script measures the contribution directly, under the SAME protocol as the
main benchmark (`kt_experiment_rigorous`): identical data, folds, optimiser,
epoch budget and early stopping. The only difference between the two VSA arms is
that one has `sim_uni` and `sim_bi` zeroed before the readout.

Arms:
  EduHDC-KT (edubind)          full model
  EduHDC-KT (no VSA features)  same architecture, sim_uni = sim_bi = 0
  DKT (LSTM)                   the reference baseline

Usage:  python src/eduhdc/kt_vsa_ablation.py
Output: data/results/kt_vsa_ablation_results.json
"""

import json
import os
import pathlib
import sys

import numpy as np
import torch
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


class NoVSA_KT(EduHDC_KT):
    """Identical to EduHDC_KT with the two VSA features zeroed before readout.

    Everything else is untouched: the same readout, the same per-skill bias, the
    same decay parameters, the same five classical features computed the same way
    (all causal, using only information strictly before the predicted step).
    """

    def forward_batch_fast(self, sk, co, ma):
        B, T = sk.shape
        dev = sk.device
        one_hot = F.one_hot(sk, self.num_skills).float()
        zeros_k = torch.zeros(B, 1, self.num_skills, device=dev)

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
        s_bias = self.skill_bias(sk).squeeze(-1)
        pos_feat = t_idx.view(1, T).expand(B, T) / 100.0
        zero = torch.zeros(B, T, device=dev)

        feat = torch.stack([zero, zero, s_bias, pos_feat, recency, count_norm, crate], -1)
        return torch.sigmoid(self.readout(feat).squeeze(-1))


def main():
    print("=" * 84)
    print("  C1 — VSA contribution ablation inside EduHDC-KT (audit fix M12)")
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
        ("EduHDC-KT (edubind, full)",
         lambda: EduHDC_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
        ("EduHDC-KT (VSA features zeroed)",
         lambda: NoVSA_KT(num_skills=K, vsa_dim=VSA_DIM, op_type="edubind", device=device)),
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
            print(f"  {name:34s} fold{fi} AUC={auc:.4f} acc={acc:.4f} "
                  f"lat={lat:.4f}ms", flush=True)
        n_tr = sum(p.numel() for p in mk().parameters() if p.requires_grad)
        out[name] = {"aucs": aucs, "accs": accs, "latencies_ms": lats,
                     "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs, ddof=1)),
                     "latency_ms_mean": float(np.mean(lats)),
                     "trainable_params": int(n_tr)}
        print(f"=> {name:34s} AUC {np.mean(aucs):.4f} +- {np.std(aucs, ddof=1):.4f} "
              f"| {n_tr:,} params | {np.mean(lats):.4f} ms/student", flush=True)

    full = np.array(out["EduHDC-KT (edubind, full)"]["aucs"])
    novsa = np.array(out["EduHDC-KT (VSA features zeroed)"]["aucs"])
    dkt = np.array(out["DKT (LSTM)"]["aucs"])

    def paired(a, b):
        t, p = scipy_stats.ttest_rel(a, b)
        dz = float((a - b).mean() / (a - b).std(ddof=1)) if (a - b).std(ddof=1) > 0 else 0.0
        return {"delta": float((a - b).mean()), "t": float(t), "p": float(p), "cohen_dz": dz}

    out["comparisons"] = {
        "vsa_contribution (full - novsa)": paired(full, novsa),
        "novsa_vs_dkt": paired(novsa, dkt),
        "full_vs_dkt": paired(full, dkt),
    }
    out["config"] = {"n_students": len(seqs), "n_folds": N_FOLDS, "num_skills": K,
                     "vsa_dim": VSA_DIM, "epochs": EPOCHS, "lr": LR,
                     "protocol": "identical to kt_experiment_rigorous"}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "kt_vsa_ablation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 84)
    for k, v in out["comparisons"].items():
        print(f"  {k:34s} delta={v['delta']:+.5f}  p={v['p']:.4g}  dz={v['cohen_dz']:+.2f}")
    print(f"[saved: {path}]")


if __name__ == "__main__":
    main()
