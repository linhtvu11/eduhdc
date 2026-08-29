# Necessary but Not Sufficient — what non-commutative binding buys in vector-symbolic architectures

Release artifact for:

> **Necessary but Not Sufficient: What Non-Commutative Binding Buys in Vector-Symbolic Architectures**
> Vu Linh Nguyen-Van, Thai Son Nguyen, Bao-An Nguyen — Tra Vinh University, Tra Vinh, Vietnam

Everything needed to re-check the machine-verified claims and re-run the evaluations.
The directory layout mirrors the paths the paper itself cites, so a reference such as
`src/eduhdc/ChainOrder.lean` in the text resolves here unchanged.

| Directory | Content |
|---|---|
| `src/eduhdc` | Kernel tier: Lean 4 with **no external dependency**, checked directly by the kernel. Also the Python evaluation package. |
| `src/eduhdc_mathlib` | Mathlib tier (`mathlib4` pinned to the same toolchain): real and complex carriers. |
| `data/results` | Result JSON files the paper's numbers are checked against. |
| `docs/arxiv_c1` | Paper source, built PDF, figure generator, figures. |
| `tools/verify` | The verification suite. See *Reproducing the checks*. |

Third-party datasets (Junyi Academy, ASSISTments 2012–2013, and the Chang et al.
prerequisite annotations) are **not** redistributed here; obtain them under their own
terms. The code that consumes them, and the results derived from them, are included.

## Reproducing the checks

```bash
cd src/eduhdc         && lake build      # kernel tier: 13 jobs, ~11 s, no network
cd ../eduhdc_mathlib  && lake build      # Mathlib tier
cd ../..
C1_PAPER=main_r7.tex python tools/verify/run_all.py
```

`run_all.py` runs four checks, each of which exists because a real defect once slipped
past a review round:

| Check | Guards against |
|---|---|
| `check_listings.py` | A Lean listing in the paper that is not verbatim from a **built** file, and a Lean identifier named in prose that exists in no source. |
| `verify_paper_numbers.py` | A number left behind after an experiment was re-run. 251 claims are compared against the JSONs in `data/results`. |
| `consistency_scan.py` | One section contradicting another, and claims the project's own evidence has already refuted. |
| `check_lean_regression.py` | The "an earlier attack no longer type-checks" claim being an untested comment. It compiles the replay. |

`count_lean_artifact.py` produces the artifact table's file, theorem and line counts by
reading the built sources, so those numbers are measured rather than transcribed.

## What is proved

The kernel tier depends on no external library and uses `propext` and `Quot.sound` only;
the Mathlib tier additionally uses `Classical.choice`. No `sorry`, no `admit`, no unproved
`axiom`. Run `#print axioms <name>` on any theorem below to confirm.

| Result | File |
|---|---|
| `PedagogicalVSA` — the three-axiom specification | `src/eduhdc/EduBindSelfContained.lean` |
| `no_abelian_action_PedagogicalVSA` — **the criterion**: no action of an abelian label algebra satisfies the order axiom, for any carrier, any label type | `src/eduhdc/GroupActionSpec.lean` |
| `no_hadamard_via_action`, `no_additive_label_PedagogicalVSA` — elementwise and additive-phase families excluded as corollaries | `src/eduhdc/GroupActionSpec.lean` |
| `no_rotation_only_PedagogicalVSA` — rotation-only families excluded over the reals | `src/eduhdc_mathlib/Basic.lean`, `EduBindBlockDiag.lean` |
| `encPair_axioms_insufficient` — the axioms do **not** force a pair encoding to be order-sensitive | `src/eduhdc/EncPairSpec.lean` |
| `hadamard_encPair_order_sensitive_bipolar` — two **commutative** roles already distinguish pair order, with invertible bipolar entries | `src/eduhdc/EncPairSpec.lean` |
| `chain_exact_unbind` — exact recovery of a heterogeneous relation chain at every length, from the left-inverse axiom **alone** | `src/eduhdc/ChainTransitivity.lean` |
| `abelian_chainAct_perm` — abelian labels mean **every** permutation of a chain acts identically, at every length | `src/eduhdc/ChainOrder.lean` |
| `chain_order_sensitive_general` — the order axiom means some two orderings differ, at every length | `src/eduhdc/ChainOrder.lean` |
| `edubind_reverse_blind_at_length_three` — **the counterexample**: the verified two-generator family cannot tell a length-three traversal from its reverse | `src/eduhdc/ChainOrder.lean` |
| `d4_reverse_sensitive_general` — exposing the whole label algebra repairs it, at every length | `src/eduhdc/DihedralLabel.lean` |
| `encChainRF_crosstalk_witness_general` — role-filler chain recovery is contaminated at every length | `src/eduhdc/ChainCrosstalk.lean` |
| `ofOrthogonalFamily`, `ofUnitaryFamily` — membership from properties of the action, discharged at real O(3), block-diagonal and U(2) carriers | `src/eduhdc_mathlib/GroupActionReal.lean` |
| `nonInjMonoid_not_from_VSA` — dropping invertibility keeps composition and provably loses exact recovery | `src/eduhdc/MonoidRelaxation.lean` |

Verified instances: `edubindVSA`, `permVSA`, `d4VSA` (kernel); `edubindRealVSA`, `o3VSA`,
`ghrrVSA`, `u2VSA`, `unitaryVSA` (Mathlib). MAP is **not** among them — it is the subject
of an *impossibility* theorem, and counting it as a verified operator would count a
negative result as a positive one.

Every Lean listing printed in the paper is verbatim from these files, modulo
transliteration of Unicode connectives (`∀` to `forall`, `∃` to `exists`, `≠` to `!=`,
`→` to `->`) and condensed doc comments; `check_listings.py` enforces that mechanically.

## What is measured

Run from the repository root; each writes one JSON into `data/results`.

| Paper item | Script | Output |
|---|---|---|
| Reordering and direction against chain length, and the closed-form n = 2 prediction | `src/eduhdc/chain_reversal_direct.py` | `chain_reversal_direct_results.json` |
| Wrong-order unbinding on real prerequisite paths (runtime sanity check) | `src/eduhdc/path_order_discrimination.py` | `path_order_discrimination_results.json` |
| Chain recovery under float32, 40 hop lengths | `src/eduhdc/transitivity_exact_check.py` | `transitivity_exact_check_results.json` |
| Composition against role-filler, by chain length | `src/eduhdc/chain_length_separation.py` | `chain_length_separation_results.json` |
| Superposition sweep, and the fitted retrieval exponent | `src/eduhdc/capacity_sweep.py`, then `capacity_fit_improved.py` | `capacity_sweep_results.json`, `capacity_fit_improved.json` |
| Direction probing with operator controls, two tiers | `src/eduhdc/prereq_probe_controls.py` | `prereq_probe_controls_results.json` |
| The same protocol with the literal role-filler encoding | `src/eduhdc/prereq_role_filler_probe.py` | `prereq_role_filler_probe_results.json` |
| Knowledge tracing, 5-fold and 10-fold | `src/eduhdc/kt_experiment_rigorous.py`, `kt_10fold.py` | `kt_benchmark_results.json`, `kt_10fold_results.json` |
| How much of the knowledge-tracing margin is the binding operator | `src/eduhdc/kt_vsa_ablation.py`, `kt_trivial_feature_control.py` | `kt_vsa_ablation_results.json`, `kt_trivial_feature_control_results.json` |
| Quantization fidelity, fp32/bf16/fp16/int8 | `src/eduhdc/precision_eval.py` | `precision_eval_results.json` |
| Frozen-codebook ablation | `src/eduhdc/ablation_kt.py` | `ablation_kt_results.json` |
| Figures | `docs/arxiv_c1/make_figures.py` | `docs/arxiv_c1/figures/*.pdf` |

`src/eduhdc/prereq_transitivity_v7.py` holds `load_clean_junyi`, which builds the
832-node / 971-edge DAG used throughout the experiments.

Re-running an experiment invalidates its figure and may invalidate a number in the paper.
Regenerate the figures and re-run `run_all.py` afterwards — that is exactly what the
checks are for.

## Environment

Lean 4 `v4.33.0` via [elan](https://github.com/leanprover/elan), with `mathlib4` pinned to
the same tag (`lake exe cache get` before building the Mathlib tier). Python 3.13 with
`torch`, `scikit-learn`, `scipy`, `pandas`, `networkx` and `sentence-transformers`. The
algebraic experiments run on CPU in seconds; the probing and knowledge-tracing runs want
a GPU.

## Licence

See `LICENSE`.
