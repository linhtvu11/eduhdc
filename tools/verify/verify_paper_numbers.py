"""Cross-check every quantitative claim now in main.tex against the result JSONs."""
import io, pathlib, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "data" / "results"
import os
PAPER = os.environ.get("C1_PAPER", "main_r5.tex")
TEX = io.open(ROOT / "docs" / "arxiv_c1" / PAPER, encoding="utf-8").read()
J = lambda n: json.load(open(R / n))

rows = []      # (claim, paper_value, json_value, ok)
skipped = []   # checks deliberately not run for this paper, reported explicitly
def chk(claim, paper, actual, tol=None, fmt="{:.4f}"):
    if tol is None:
        ok = (str(paper) == str(actual))
    else:
        ok = abs(float(paper) - float(actual)) <= tol
    rows.append((claim, fmt.format(paper) if isinstance(paper, float) else str(paper),
                 fmt.format(actual) if isinstance(actual, float) else str(actual), ok))

def intex(sub, claim):
    rows.append((claim, "present in main.tex", "yes" if sub in TEX else "NO", sub in TEX))

# ---------------- runtime tier (transitivity_exact_check) ----------------
d = J("transitivity_exact_check_results.json")
chk("graph nodes", 832, d["graph"]["nodes"])
chk("graph edges", 971, d["graph"]["edges"])
chk("removed cycle nodes", 3, d["graph"]["removed_cycle_nodes"])
chk("transitive pairs", 56224, d["total_transitive_pairs"])
chk("max hop", 41, d["max_chain_length"])
chk("chain lengths tested", 40, d["trials"]["chain_lengths_tested"])
chk("total integer trials (paper: 8,000)", 8000, d["trials"]["total_integer_trials"])
chk("total float trials (paper: 2,000)", 2000, d["trials"]["total_float_trials"])
chk("integer trials per length (200)", 200, d["trials"]["integer_contents_per_length"])
chk("float trials per length (50)", 50, d["trials"]["float_trials_per_length"])
chk("all integer exact", True, d["exact_integer"]["all_exact"])
chk("max float32 err (paper 1.4e-06)", 1.4e-6, d["runtime_float"]["max_abs_err_overall"],
    tol=5e-8, fmt="{:.3e}")
chk("hop-2 float err (paper 2.4e-07)", 2.4e-7,
    d["runtime_float"]["per_length"]["2"]["max_abs_err"], tol=5e-9, fmt="{:.3e}")
chk("heterogeneous chains declared", True, "heterogeneous" in d["chain_kind"])

# ---------------- probe controls ----------------
p = J("prereq_probe_controls_results.json")
T_, I_ = p["transductive"], p["inductive"]
for arm, hops, vals in [("edubind", ["hop2-3","hop4-6","hop7+"], [85.6, 94.8, 98.7]),
                        ("concat",  ["hop2-3","hop4-6","hop7+"], [80.6, 92.5, 98.9]),
                        ("hrr",     ["hop2-3","hop4-6","hop7+"], [13.9, 13.8, 13.2])]:
    for h, v in zip(hops, vals):
        chk(f"transductive {arm} {h}", v, T_[arm][h]["dir_acc_strict"]*100, tol=0.06, fmt="{:.2f}")
for arm, vals in [("edubind", [69.2, 81.8, 92.6]), ("concat", [68.2, 79.0, 93.4]),
                  ("hrr", [13.4, 13.2, 14.1])]:
    for h, v in zip(["hop2-3","hop4-6","hop7+"], vals):
        chk(f"inductive {arm} {h}", v, I_[arm][h]["dir_acc_strict"]*100, tol=0.06, fmt="{:.2f}")
chk("transductive EduBind overall (93.0)", 93.0,
    np.mean([T_["edubind"][h]["dir_acc_strict"] for h in ["hop2-3","hop4-6","hop7+"]])*100,
    tol=0.06, fmt="{:.2f}")
chk("transductive concat overall (90.7)", 90.7,
    np.mean([T_["concat"][h]["dir_acc_strict"] for h in ["hop2-3","hop4-6","hop7+"]])*100,
    tol=0.06, fmt="{:.2f}")
chk("inductive EduBind unweighted (81.2)", 81.2,
    np.mean([I_["edubind"][h]["dir_acc_strict"] for h in ["hop2-3","hop4-6","hop7+"]])*100,
    tol=0.06, fmt="{:.2f}")
ns = [I_["edubind"][h]["n_pairs_mean"] for h in ["hop2-3","hop4-6","hop7+"]]
chk("inductive EduBind weighted (87.1)", 87.1,
    np.average([I_["edubind"][h]["dir_acc_strict"] for h in ["hop2-3","hop4-6","hop7+"]],
               weights=ns)*100, tol=0.06, fmt="{:.2f}")
chk("inductive concat unweighted (80.2)", 80.2,
    np.mean([I_["concat"][h]["dir_acc_strict"] for h in ["hop2-3","hop4-6","hop7+"]])*100,
    tol=0.06, fmt="{:.2f}")
chk("MAP tie fraction = 100% (exact, bipolar)", 1.0, T_["map"]["hop2-3"]["tie_fraction"])
chk("HRR tiebreak ~= chance (0.498)", 0.498, T_["hrr"]["hop2-3"]["dir_acc_tiebreak"],
    tol=0.02, fmt="{:.3f}")
chk("transductive node overlap (paper 100%)", 1.0, p["transductive_test_node_overlap_mean"])
chk("n_seeds = 10", 10, p["protocol"]["n_seeds"])
ov = sum(not (T_["edubind"][h]["bootstrap_ci95_strict"][1] < T_["concat"][h]["bootstrap_ci95_strict"][0]
              or T_["concat"][h]["bootstrap_ci95_strict"][1] < T_["edubind"][h]["bootstrap_ci95_strict"][0])
         for h in ["hop2-3","hop4-6","hop7+"]) + \
     sum(not (I_["edubind"][h]["bootstrap_ci95_strict"][1] < I_["concat"][h]["bootstrap_ci95_strict"][0]
              or I_["concat"][h]["bootstrap_ci95_strict"][1] < I_["edubind"][h]["bootstrap_ci95_strict"][0])
         for h in ["hop2-3","hop4-6","hop7+"])
chk("CI overlap count (paper: all six)", 6, ov)

# ---------------- KT ablation ----------------
k = J("kt_vsa_ablation_results.json")
chk("KT abl full AUC (0.7159)", 0.7159, k["EduHDC-KT (edubind, full)"]["auc_mean"], tol=6e-5)
chk("KT abl novsa AUC (0.6891)", 0.6891, k["EduHDC-KT (VSA features zeroed)"]["auc_mean"], tol=6e-5)
chk("KT abl DKT AUC (0.7098)", 0.7098, k["DKT (LSTM)"]["auc_mean"], tol=6e-5)
chk("VSA contribution (+0.0268)", 0.0268, k["comparisons"]["vsa_contribution (full - novsa)"]["delta"], tol=6e-5)
chk("novsa vs DKT (-0.0208)", -0.0208, k["comparisons"]["novsa_vs_dkt"]["delta"], tol=6e-5)
chk("full vs DKT (+0.0061)", 0.0061, k["comparisons"]["full_vs_dkt"]["delta"], tol=6e-5)
chk("full latency ms (0.406)", 0.406, k["EduHDC-KT (edubind, full)"]["latency_ms_mean"], tol=6e-3, fmt="{:.3f}")
chk("DKT latency ms (0.067)", 0.067, k["DKT (LSTM)"]["latency_ms_mean"], tol=6e-3, fmt="{:.3f}")
chk("KT abl n_students 5000", 5000, k["config"]["n_students"])

# ---------------- KT trivial-feature control (review bg-1 F2) ----------------
# Guards the four numbers the paper now quotes for the matched-slot controls.
# The point of these checks is that the paper must not go back to quoting the
# +0.0268 gap as the VSA-attributable margin: the classical-slot arm recovers
# most of it with no binding at all.
t_ = J("kt_trivial_feature_control_results.json")
chk("trivial-ctrl full AUC (0.7159)", 0.7159, t_["full (2 VSA features)"]["auc_mean"], tol=6e-5)
chk("trivial-ctrl zeroed AUC (0.6891)", 0.6891, t_["zeroed (VSA feats = 0)"]["auc_mean"], tol=6e-5)
chk("trivial-ctrl scalar AUC (0.6891)", 0.6891, t_["scalar slots (2 learnable)"]["auc_mean"], tol=6e-5)
chk("trivial-ctrl classical AUC (0.7087)", 0.7087, t_["classical slots (2 KT stats)"]["auc_mean"], tol=6e-5)
chk("classical recovers (+0.0196)", 0.0196,
    t_["comparisons"]["classical - zeroed (gap recovered by non-VSA)"]["delta"], tol=6e-5)
chk("VSA-specific residual (+0.0072)", 0.0072,
    t_["comparisons"]["full - classical (VSA vs cheap KT stats)"]["delta"], tol=6e-5)
chk("classical vs DKT (-0.0011)", -0.0011,
    t_["comparisons"]["classical - dkt"]["delta"], tol=6e-5)
# the recovered fraction the paper states as "73%"
_frac = (t_["comparisons"]["classical - zeroed (gap recovered by non-VSA)"]["delta"]
         / t_["comparisons"]["full - zeroed  (replicates +0.027)"]["delta"])
chk("classical recovers 73% of the gap", 0.73, _frac, tol=6e-3, fmt="{:.2f}")

# ---------------- cross-source transfer, seed-level CI (review bg-3 F5) ----------------
# The paper quotes BOTH estimators; this guards the seed-level one and the
# claim that it still excludes chance.
s_ = J("prereq_direction_v6_seed_ci.json")["arms"]["EduBind"]
chk("v6 seed-CI low (0.476)", 0.476, s_["seed_bootstrap_ci95"][0], tol=6e-4, fmt="{:.3f}")
chk("v6 seed-CI high (0.493)", 0.493, s_["seed_bootstrap_ci95"][1], tol=6e-4, fmt="{:.3f}")
chk("v6 seed-CI width ratio (1.21)", 1.21, s_["width_ratio_seed_over_pooled"], tol=6e-3, fmt="{:.2f}")
chk("v6 t vs chance (-3.45)", -3.45, s_["t_test_vs_chance"]["t"], tol=6e-3, fmt="{:.2f}")
chk("v6 seed-CI still excludes chance", True, s_["seed_bootstrap_excludes_chance"])

# ---------------- main KT benchmark ----------------
b = J("kt_benchmark_results.json")
a = np.array(b["results"]["EduHDC-KT (Non-Commutative)"]["aucs"])
dk = np.array(b["results"]["DKT (LSTM Baseline)"]["aucs"])
chk("KT 5-fold EduHDC (0.7152)", 0.7152, a.mean(), tol=6e-5)
chk("KT 5-fold DKT (0.7109)", 0.7109, dk.mean(), tol=6e-5)
from scipy import stats
t, pv = stats.ttest_rel(a, dk)
chk("KT 5-fold paired p (0.015)", 0.015, pv, tol=6e-4, fmt="{:.4f}")
chk("KT 5-fold dz (1.83)", 1.83, (a-dk).mean()/(a-dk).std(ddof=1), tol=6e-3, fmt="{:.3f}")
chk("EduHDC params 537,415", 537415, b["param_counts"]["EduHDC-KT (Non-Commutative)"])
chk("DKT params 166,276", 166276, b["param_counts"]["DKT (LSTM Baseline)"])
chk("HRR-KT latency ms (5.87)", 5.87,
    np.mean(b["results"]["HRR-KT (Commutative)"]["latencies"]), tol=6e-3, fmt="{:.3f}")
m5 = np.array(b["results"]["MAP-KT (Commutative)"]["aucs"])
chk("EduHDC-MAP delta 5k (-0.0003)", -0.0003, (a-m5).mean(), tol=6e-5)
chk("EduHDC-MAP p 5k (0.74)", 0.74, stats.ttest_rel(a, m5)[1], tol=6e-3, fmt="{:.3f}")
b10 = J("kt_10fold_results.json")
a10 = np.array(b10["results"]["EduHDC-KT (Non-Comm)"]["aucs"]); d10 = np.array(b10["results"]["DKT (LSTM Baseline)"]["aucs"])
chk("KT 10-fold EduHDC (0.7190)", 0.7190, a10.mean(), tol=6e-5)
chk("KT 10-fold DKT (0.7206)", 0.7206, d10.mean(), tol=6e-5)
chk("KT 10-fold p (0.0057)", 0.0057, stats.ttest_rel(a10, d10)[1], tol=6e-4, fmt="{:.4f}")
chk("KT 10-fold dz (-1.14)", -1.14, (a10-d10).mean()/(a10-d10).std(ddof=1), tol=6e-3, fmt="{:.3f}")
m10 = np.array(b10["results"]["MAP-KT (Commutative)"]["aucs"])
chk("EduHDC-MAP delta 8k (+0.0017)", 0.0017, (a10-m10).mean(), tol=6e-5)
chk("EduHDC-MAP p 8k (1.9e-4)", 1.9e-4, stats.ttest_rel(a10, m10)[1], tol=6e-5)

# ---------------- 828-param ablation ----------------
ab = J("ablation_kt_results.json")
chk("828 params", 828, ab["results"]["edubind-d256-frozen"]["params"])
chk("828 AUC (0.6625)", 0.6625, ab["results"]["edubind-d256-frozen"]["auc_mean"], tol=6e-5)
chk("DKT abl AUC (0.6300)", 0.6300, ab["results"]["DKT (LSTM Baseline)"]["auc_mean"], tol=6e-5)
chk("197x (163449/828)", 197, round(ab["config"]["dkt_params"]/828))
chk("828 latency ms (16.9)", 16.9, ab["results"]["edubind-d256-frozen"]["latency_ms"], tol=6e-2, fmt="{:.2f}")
chk("DKT abl latency ms (0.080)", 0.080, ab["results"]["DKT (LSTM Baseline)"]["latency_ms"], tol=6e-3, fmt="{:.3f}")

# ---------------- precision ----------------
pr = J("precision_eval_results.json")
chk("EduBind fp32 cos = 1.0", 1.0, pr["results"]["edubind"]["fp32"]["unbind_cos_vs_true"])
chk("HRR fp32 cos (0.7084)", 0.7084, pr["results"]["hrr"]["fp32"]["unbind_cos_vs_true"], tol=6e-5)
chk("EduBind int8 cos > 0.9998", True, pr["results"]["edubind"]["int8"]["unbind_cos_vs_true"] > 0.9998)
chk("precision D = 2048", 2048, pr["config"]["D"])

# ---------------- capacity ----------------
cf = J("capacity_fit_improved.json")
chk("capacity linear R2 (0.52)", 0.52, cf["r2_linear"], tol=6e-3, fmt="{:.3f}")
chk("capacity logistic R2 (0.99)", 0.99, cf["r2_logistic"], tol=6e-3, fmt="{:.3f}")
chk("capacity fit points (72)", 72, cf["n_points"])
cs = J("capacity_sweep_results.json")
pf = cs.get("per_operator_fit", {})
# main.tex reports these three to 3 decimals ("0.520, 0.520 and 0.522 for MAP,
# HRR and EduBind"); check each against its own rounded claim rather than an
# arbitrary raw spread threshold (the raw spread is ~0.00215, which rounds to
# exactly these figures but narrowly exceeds a naive 0.002 raw-spread cutoff).
for op, claim in [("map", 0.520), ("hrr", 0.520), ("edubind", 0.522)]:
    if op in pf:
        chk(f"per-operator R2 ({op})", claim, pf[op]["r2"], tol=6e-4, fmt="{:.3f}")

# ---------------- v7 baseline table (cut in Revision 4) ----------------
# Revision 4 removed this table (different protocol: frozen probe, 5 seeds,
# pooled Wilson -- not comparable to the validated-split controls run). The
# block is kept so that reinstating the table reinstates its checks, but it is
# now reported as a SKIP instead of vanishing silently: a gate on paper text
# that stops matching turns its checks into dead code that still reports PASS.
if "CurriculumPotential" in TEX:
    v7 = J("prereq_transitivity_v7_results.json")["results"]
    for m, vals in [("EduBind", [76.9, 64.9, 63.3]), ("CurriculumPotential", [64.8, 78.3, 97.1])]:
        for h, v in zip(["hop2-3","hop4-6","hop7+"], vals):
            chk(f"Table baseline {m} {h}", v, v7[m][h]["dir_acc"]*100, tol=0.06, fmt="{:.2f}")
    chk("Table baseline MAP = 0.0", 0.0, v7["MAP"]["hop2-3"]["dir_acc"])
else:
    skipped.append(f"v7 baseline table (7 checks): 'CurriculumPotential' absent from "
                   f"{PAPER} -- table not reported in the current revision")

# ---------------- cross-source ----------------
v6 = J("prereq_direction_v6_results.json")["results"]["EduBind"]
chk("cross-source 48.4%", 48.4, v6["dir_acc"]*100, tol=0.06, fmt="{:.2f}")
chk("cross-source CI lo (47.8)", 47.8, v6["dir_ci95"][0]*100, tol=0.06, fmt="{:.2f}")
chk("cross-source CI hi (49.1)", 49.1, v6["dir_ci95"][1]*100, tol=0.06, fmt="{:.2f}")
chk("cross-source CI excludes 0.5", True, v6["dir_ci95"][1] < 0.5)
chk("2-hop trans acc 43.0%", 43.0, v6["trans_acc"]*100, tol=0.06, fmt="{:.2f}")
hd = J("prereq_direction_v6_hopdiag_results.json")["results"]["EduBind"]
lo = min(v["dir_acc"] for v in hd.values())*100; hi = max(v["dir_acc"] for v in hd.values())*100
chk("hop-diag range lo (43.9)", 43.9, lo, tol=0.06, fmt="{:.2f}")
chk("hop-diag range hi (50.0)", 50.0, hi, tol=0.06, fmt="{:.2f}")

# ---------------- expert set / future work ----------------
ex = J("prereq_probing_expert_results.json")["dataset"]
chk("615 binary pairs", 615, ex["binary_train_pos"]+ex["binary_train_neg"])
chk("75 asymmetric pairs", 75, ex["directional_gold_pairs_train"])
fw = J("prereq_transitivity_fw3_results.json")["results"]["FW3a_richer_supervision"]
for h, v in zip(["hop2-3","hop4-6","hop7+"], [89.7, 92.7, 99.2]):
    chk(f"future-work {h}", v, fw[h]["dir_acc"]*100, tol=0.06, fmt="{:.2f}")

# The shuffled-label null test (0.482) and the v4 MAP direction accuracies
# (11.3--13.7%) were reported in earlier revisions and are NOT quoted anywhere in
# main_r4.tex. Checking them unconditionally inflated the headline claim count
# with claims the paper does not make, so each is gated on actually appearing.
if "0.482" in TEX or "48.2" in TEX:
    chk("null test 0.482", 0.482, J("c1_null_test_results.json")["results"]["shuffled"],
        tol=6e-4, fmt="{:.4f}")
else:
    skipped.append(f"shuffled-label null test (1 check): 0.482 not quoted in {PAPER}")
if "11.3" in TEX or "13.7" in TEX:
    v4 = J("prereq_probing_v4_results.json")
    chk("MAP 11-14% lo (11.3)", 11.3, v4["MAP_w0.3"]["dir_acc_mean"]*100, tol=0.06, fmt="{:.2f}")
    chk("MAP 11-14% hi (13.7)", 13.7, v4["MAP_w0.5"]["dir_acc_mean"]*100, tol=0.06, fmt="{:.2f}")
else:
    skipped.append(f"v4 MAP direction accuracy (2 checks): 11.3/13.7 not quoted in {PAPER}")

# ---------------- graph fragmentation claim ----------------
intex("70 weakly connected components", "Threats: 70 components claim")
intex("60 isolated nodes", "Threats: 60 isolated nodes claim")

# ---------------- headline numbers must actually appear in the paper ----------------
# `chk` above only proves the JSONs are self-consistent; without these, a number
# could be silently dropped from the prose and every check would still pass.
intex("56{,}224", "transitive pair count present")
intex("1.4\\times10^{-6}", "float32 max chain error present")
intex("2.4\\times10^{-7}", "hop-2 float32 error present")
intex("R^2 = 0.99", "capacity logistic R2 present")
intex("R^2 = 0.52", "capacity linear R2 present (both fits reported, not the better one alone)")
intex("0.7084", "HRR fp32 unbind cosine present")
intex("0.9998", "EduBind int8 fidelity present")
intex("$+0.0268$", "raw VSA ablation gap present")
intex("$+0.0072$", "matched-slot residual present -- the honest VSA-attributable margin")
intex("73\\%", "fraction of the ablation gap recovered by the non-VSA control")
intex("48.4\\%", "cross-source below-chance transfer disclosed")
intex("828", "frozen-codebook parameter count present")

# ---------------- structural claims about the paper's own argument ----------------
# The results the paper is built on must be named where a reader can find them.
intex("hadamard\\_encPair\\_order\\_sensitive", "H0 theorem named in the paper")
intex("encChainRF\\_crosstalk\\_witness", "crosstalk theorem named in the paper")
intex("encChainRF\\_crosstalk\\_witness\\_general", "general-n crosstalk theorem named in the paper")
intex("half a bit per neuron", "capacity theory corroboration cited in sec:capacity-exp")
intex("of the moments", "model-independence of retrieval accuracy stated")
intex("no\\_abelian\\_action\\_PedagogicalVSA", "generalized impossibility named in the paper")
intex("no\\_additive\\_label\\_PedagogicalVSA", "phase-composition theorem named in the paper")
intex("chain\\_exact\\_unbind", "chain theorem named in the paper")
intex("encPair\\_axioms\\_insufficient", "refutation theorem named in the paper")
# Revision 5: the chain-order theorems are what actually carry the operator's
# case (chain_exact_unbind consumes only Axiom 3, so it is not evidence for
# non-commutativity). Both must be named, and the paper must say so explicitly.
intex("chain\\_order\\_sensitive", "chain-order positive theorem named")
# NOTE (2026-08-28): this check previously demanded `abelian_chain_order_blind`,
# a name that exists in NO Lean file -- so it was enforcing a fiction, and passed
# for as long as the paper repeated the same wrong name. The real theorem is
# `abelian_chainAct_reverse` (ChainOrder.lean). check_listings.py now
# cross-checks every \texttt{} identifier against the Lean sources, which is what
# makes a repeat of this class of error detectable rather than self-confirming.
intex("abelian\\_chainAct\\_perm", "chain-order impossibility named, permutation form")
# The negative direction is only non-vacuous over ActionFamily; the paper must
# name the concrete instance and the recorded trap, or the scope argument is
# prose the reader cannot check.
intex("hadamard\\_chainAct\\_perm", "concrete order-blind instance named")
intex("abelian\\_chainAct\\_vacuous\\_over\\_VSA", "vacuity trap named")
intex("consumes exactly one axiom", "chain_exact_unbind's axiom dependency disclosed")
# Chain-order discrimination measurements (chain_order_discrimination.py).
cod = J("chain_order_discrimination_results.json")
chk("order: MAP reversed cos = 1.0 at n=41", 1.0,
    cod["reversed_cosine_to_target"]["map"]["41"], tol=1e-4)
chk("order: MAP reversed top-1 = 1.0 at n=41", 1.0,
    cod["reversed_accuracy"]["map"]["41"])
chk("order: verified forward cos = 1.0 at n=41", 1.0,
    cod["forward_cosine_to_target"]["edubind"]["41"], tol=1e-4)
chk("order: verified reversed cos <= 0.25 at n=2", True,
    cod["reversed_cosine_to_target"]["edubind"]["2"] <= 0.2500)
chk("order: verified discrimination = 1.0 at n=8", 1.0,
    cod["discrimination_accuracy"]["edubind"]["8"])
chk("order: verified reversed top-1 = 1.0 at n=4 (readout artifact)", 1.0,
    cod["reversed_accuracy"]["edubind"]["4"])
chk("order: HRR forward cos collapses by n=16", True,
    cod["forward_cosine_to_target"]["hrr"]["16"] < 0.05)
# Revision 7 reports the DIRECTION SIGNAL (cosine distance) rather than raw
# cosine, because 1.000 meant total failure there while it means exact
# recovery in every other table in the paper -- a polarity clash a reader
# did in fact trip over. The check follows the new units and additionally
# requires the polarity to be spelled out, since that was the actual defect.
intex("carry \\emph{no} direction signal", "commutative order blindness stated")
intex("direction is not representable", "the polarity of the direction signal spelled out")
intex("edubind\\_reverse\\_blind\\_at\\_length\\_three",
      "the not-sufficient counterexample named")
intex("d4\\_reverse\\_sensitive\\_general", "the algebraic repair named")
intex("chain\\_order\\_sensitive\\_general", "the general positive half named")
# The crosstalk result is general (proved for every n >= 2 via padding). The paper
# must state that scope, and must state the quantifier limit that remains: a
# witness chain at each length is not universality over contents at that length.
intex("every $n \\geq 2$", "crosstalk general-n scope disclosed")
intex("not that it fails for every choice of contents",
      "remaining crosstalk quantifier gap disclosed in Limitations")
# The Lean artifact figures quoted in the paper. Revision 5 moved these from a
# prose paragraph into tab:artifact, so check the numbers and the table, not the
# old sentence shapes -- a prose-shaped check would silently pass on absence.
intex("\\label{tab:artifact}", "artifact table present")
_lac_paper = J("lean_artifact_counts.json")
for _n, _what in [(str(_lac_paper["kernel"]["theorems"]), "kernel-tier theorem count"),
                  ("1{,}050", "kernel-tier lines of proof"),
                  (str(_lac_paper["mathlib"]["theorems"]), "Mathlib tier theorem count"),
                  (str(_lac_paper["mathlib"]["lines"]), "Mathlib tier line count"),
                  (str(_lac_paper["total"]["theorems"]), "grand total theorem count"),
                  ("1{,}391", "grand total lines of proof")]:
    intex(_n, f"{_what} present in tab:artifact")
intex("ofOrthogonalFamily", "class-membership constructor named in the paper")
intex("non-abelian subgroup of $SO(3)$", "O(3) discharge disclosed")
intex("ofUnitaryFamily", "unitary-family constructor named in the paper")
intex("nonInjMonoid\\_not\\_from\\_VSA", "monoid separation named in the paper")
intex("chainApply\\_bundle\\_distrib", "surviving chain distributivity named")

# ---------------- Revision 7: the chain-ORDER experiments ----------------
# These had ZERO numeric checks before Revision 7 -- the newest and most
# emphasised experiment in the paper was entirely unverified, and its rho column
# was not even written to a JSON (it was printed to stdout only). Both fixed.

# chain_reversal_direct.py -- measures the quantity the theorems are actually
# about (FORWARD composition, no unbinding), which is what
# path_order_discrimination.py and chain_order_discrimination.py do NOT measure.
crd = J("chain_reversal_direct_results.json")
_syn = crd["synthetic"]
# abelian_chainAct_perm: MAP and HRR are blind to reversal AND to any permutation.
for _op in ("map", "hrr"):
    for _n in ("2", "3", "41", "64"):
        chk(f"reversal: {_op} cos(fwd,rev) = 1.0 at n={_n}", 1.0,
            _syn[_op][_n]["cos_fwd_rev"], tol=1e-5)
        chk(f"permutation: {_op} cos(fwd,perm) = 1.0 at n={_n}", 1.0,
            _syn[_op][_n]["cos_fwd_perm"], tol=1e-5)
# edubind_reverse_blind_at_length_three, measured: the two-generator family that
# the KERNEL-TIER instance is built from is reversal-blind at odd lengths and
# not at even ones. This is the machine-checked prediction with a falsifiable
# runtime signature, so it is checked at both parities.
for _n in ("3", "5", "7", "41"):
    chk(f"edubind2 reversal-blind at odd n={_n}", True, _syn["edubind2"][_n]["reversal_blind"])
for _n in ("2", "4", "6", "8", "64"):
    chk(f"edubind2 NOT reversal-blind at even n={_n}", False, _syn["edubind2"][_n]["reversal_blind"])
# d4_reverse_sensitive_general: all eight elements, never blind, at any length.
for _n in ("2", "3", "5", "7", "41", "64"):
    chk(f"d4 not reversal-blind at n={_n}", False, _syn["d4"][_n]["reversal_blind"])
# On the REAL Junyi hop-length distribution the split is exactly by parity.
_real = crd["real_lengths_measured"]
_lens = [str(n) for n in crd["config"]["real_lengths"]]
chk("real lengths: MAP blind at all of them", len(_lens),
    sum(1 for n in _lens if _real["map"][n]["reversal_blind"]))
chk("real lengths: edubind2 blind at exactly the odd ones",
    sum(1 for n in _lens if int(n) % 2 == 1),
    sum(1 for n in _lens if _real["edubind2"][n]["reversal_blind"]))
chk("real lengths: d4 blind at none", 0,
    sum(1 for n in _lens if _real["d4"][n]["reversal_blind"]))
# The two counts the paper quotes in four places. They moved by one between the
# first two runs because path enumeration iterated a set; pinned here so a
# recurrence is a failure rather than a silently different paper.
chk("real lengths: 33 distinct hop lengths", 33, len(_lens))
chk("real lengths: 16 of them odd", 16, sum(1 for n in _lens if int(n) % 2 == 1))
intex("33 of 33", "the real-length reversal-blindness count reaches the page")
intex("16 odd ones", "the odd-length count reaches the page")
# The n=2 value is DERIVED, not observed: half the trace of the group commutator.
for _op, _pred in (("edubind", 0.25), ("d4", 0.25), ("edubind2", 0.0)):
    chk(f"n=2 closed form {_op} predicted", _pred, crd["n2_closed_form"][_op]["predicted"])
    chk(f"n=2 closed form {_op} measured within 3 s.e.", True,
        crd["n2_closed_form"][_op]["within_3se"])

# path_order_discrimination.py -- retained as a runtime sanity check, with the
# tie-break artefact removed. A strict `margin > 0` test scored MAP at 0.000,
# which reads as "worse than chance" when the truth is "identically tied"; that
# is the same defect the controls table already fixed for HRR.
pod = J("path_order_discrimination_results.json")
chk("path-order: MAP tie fraction = 1.0", 1.0, pod["results"]["map"]["tie_fraction_mean"], tol=1e-9)
chk("path-order: MAP tie-broken direction accuracy = 0.5", 0.5,
    pod["results"]["map"]["dir_acc_mean"], tol=1e-9)
chk("path-order: HRR tie-broken direction accuracy at chance", True,
    abs(pod["results"]["hrr"]["dir_acc_mean"] - 0.5) < 0.02)
chk("path-order: rho recorded for every arm", True,
    all("rho_recovery_mean" in v for v in pod["results"].values()))
chk("path-order: EduBind rho = 1.0", 1.0,
    pod["results"]["edubind"]["rho_recovery_mean"], tol=1e-4)
chk("path-order: seeds are genuinely independent", True,
    len(set(pod["results"]["edubind"]["mean_margin_per_seed"])) > 1)

# capacity_fit_improved.py -- the exponent is now MEASURED, not assumed. The
# earlier fit used sqrt(D/T) in both models, so it tested functional form and
# never the exponent.
cfi = J("capacity_fit_improved.json")
chk("capacity: fitted exponent alpha near 1/2", True,
    abs(cfi["exponent_study"]["alpha_best"] - 0.5) <= 0.05)
chk("capacity: probit fit quality at alpha=1/2", True,
    cfi["exponent_study"]["alpha_r2_at_half"] > 0.99)
chk("capacity: D and T enter only through their ratio (p ~ q)", True,
    abs(cfi["exponent_study"]["free_pq_best"]["p"]
        - cfi["exponent_study"]["free_pq_best"]["q"]) <= 0.05)

# The Lean artifact figures are now COUNTED, not string-matched. `intex("103")`
# only proved the digits appear somewhere in the .tex; it passed for as long as
# the number was written, right or wrong, and kept passing after a file was
# added.
lac = J("lean_artifact_counts.json")
for _tier in ("kernel", "mathlib", "total"):
    for _k in ("files", "theorems", "lines"):
        chk(f"artifact count present: {_tier}.{_k}", True, isinstance(lac[_tier][_k], int))

# ---------------- E1: role-filler probe, direct test of H0 ----------------
rfp = J("prereq_role_filler_probe_results.json")
for arm, tvals, ivals in [
    ("edubind", [78.1, 90.0, 98.0], [67.1, 77.9, 92.3]),
    ("map",     [80.5, 91.7, 98.5], [68.8, 77.9, 93.5]),
    ("hrr",     [81.7, 92.4, 98.8], [69.1, 78.7, 93.1]),
]:
    for h, v in zip(["hop2-3", "hop4-6", "hop7+"], tvals):
        chk(f"E1 transductive {arm} {h}", v,
            rfp["transductive"][arm][h]["dir_acc_strict"] * 100, tol=0.06, fmt="{:.2f}")
    for h, v in zip(["hop2-3", "hop4-6", "hop7+"], ivals):
        chk(f"E1 inductive {arm} {h}", v,
            rfp["inductive"][arm][h]["dir_acc_strict"] * 100, tol=0.06, fmt="{:.2f}")
intex("prereq\\_role\\_filler\\_probe.py", "E1 script named in the paper")

# ---------------- E2: role-filler vs composition, accuracy vs chain length ----------------
sep = J("chain_length_separation_results.json")
chk("E2 EduBind role-filler n=64", 0.9813, sep["role_filler_accuracy"]["edubind"]["64"], tol=6e-4)
chk("E2 EduBind role-filler n=100", 0.8460, sep["role_filler_accuracy"]["edubind"]["100"], tol=6e-4)
chk("E2 EduBind role-filler n=150", 0.5920, sep["role_filler_accuracy"]["edubind"]["150"], tol=6e-4)
chk("E2 EduBind composition n=150 (exact)", 1.0, sep["composition_accuracy"]["edubind"]["150"])
chk("E2 MAP composition n=150 (exact)", 1.0, sep["composition_accuracy"]["map"]["150"])
chk("E2 n_list max is 150", 150, max(sep["config"]["n_list"]))
intex("chain\\_length\\_separation.py", "E2 script named in the paper")
intex("59\\%", "role-filler n=150 collapse figure present")
intex("fig:separation", "separation figure referenced")

bad = [r for r in rows if not r[3]]
print(f"{len(rows)} claims checked, {len(bad)} MISMATCH\n")
for c, pv_, av, ok in rows:
    if not ok:
        print(f"  MISMATCH  {c}\n            paper={pv_}  json={av}")
if skipped:
    print(f"  {len(skipped)} check group(s) SKIPPED for {PAPER}:")
    for s in skipped:
        print(f"    - {s}")
if not bad:
    print("  every checked claim matches its source JSON.")
