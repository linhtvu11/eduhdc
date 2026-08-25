# EDUHDC — Machine-Checked Order-Sensitive Binding for Vector-Symbolic Curriculum Representations

Release artifact accompanying the paper:

> **EDUHDC: Machine-Checked Order-Sensitive Binding for Vector-Symbolic Curriculum Representations in Lean 4**
> Vu Linh Nguyen-Van, Thai Son Nguyen, Bao-An Nguyen — Tra Vinh University, Tra Vinh, Vietnam

This repository contains everything needed to re-check the machine-verified claims and
re-run the empirical evaluations reported in the paper.

| Directory | Content |
|---|---|
| `lean/eduhdc-core` | Mathlib-free Lean 4 project; every file is in the default build target and is checked directly by the Lean 4 kernel |
| `lean/eduhdc-mathlib` | Mathlib tier (`mathlib4` pinned to `v4.33.0`): the `PedagogicalVSA` class, its real-matrix instance, and the continuous rotation/reflection family |
| `eduhdc` | Evaluation package used for the experiments section (run as `python eduhdc/<script>.py` from the repository root) |
| `results` | Result JSON files produced by those runs |
| `figures` | Figures included in the paper (PDF for LaTeX, PNG for preview) |

## Paper ↔ file map

Every row names the script that actually produces the cited number, and the JSON it
writes. Rows marked *(no archived output)* are diagnostics that print to stdout and are
not the source of any number or figure in the paper.

### Formal layer (Section 3)

| Paper item | File(s) |
|---|---|
| `PedagogicalVSA` specification, three axioms, §3.1 — kernel tier | `lean/eduhdc-core/EduBindSelfContained.lean` |
| Same specification as a Mathlib type class **with a real-matrix instance** (`edubindRealVSA`), §3.1 | `lean/eduhdc-mathlib/Basic.lean` |
| EduBind = integer O(2): `Rot`, `Ref`, orthogonality, exact unbind, `Rot_Ref_non_commutative`, §3.2 | `lean/eduhdc-core/EduBindSelfContained.lean` |
| Continuous O(2) family over ℝ: `rot`, `refl`, `rot_orthogonal`, `refl_orthogonal`, `rot_refl_non_commutative`, §3.2 | `lean/eduhdc-mathlib/EduBindBlockDiag.lean` |
| **Impossibility theorem 1** — no `PedagogicalVSA` from rotations alone (`rot_commutes`, `no_rotation_only_PedagogicalVSA`), §3.2 | `lean/eduhdc-mathlib/EduBindBlockDiag.lean`, `lean/eduhdc-mathlib/Basic.lean` |
| **Impossibility theorem 2** — no `PedagogicalVSA` from commutative elementwise binding (`had_ops_commute`, `hadFamily_not_order_sensitive`, `no_hadamard_PedagogicalVSA`), §3.2 | `lean/eduhdc-core/VSATriad.lean` |
| MAP commutativity and `map_order_indistinguishable`; Perm (signed permutations) as a second instance, §3.2 | `lean/eduhdc-core/VSATriad.lean` |
| `chain_exact_unbind` over a **list** of relations (heterogeneous chains), §3.3 | `lean/eduhdc-core/ChainTransitivity.lean` |
| Cost model, `dof1D` / `dofPAM`, both normalisations, §3.4 | `lean/eduhdc-core/CapacityCostModel.lean` |

### Empirical layer (Section 4)

| Paper item | Script | Output |
|---|---|---|
| Runtime verification + chain-error sweep, §4.2 and Figure 2 | `eduhdc/transitivity_exact_check.py` | `results/transitivity_exact_check_results.json` |
| **Transductive + inductive tiers with operator controls** (EduBind / MAP / HRR / concat-MLP), §4.3, Figure 3 | `eduhdc/prereq_probe_controls.py` | `results/prereq_probe_controls_results.json` |
| Transitive-pair protocol behind the `CurriculumPotential` comparison and the MAP negative control, §4.3–4.4 | `eduhdc/prereq_transitivity_v7.py` | `results/prereq_transitivity_v7_results.json` |
| Cross-source transfer (48.4%) and its hop-stratified diagnostic, §4.4 | `eduhdc/prereq_direction_v6.py`, `eduhdc/prereq_direction_v6_hopdiag.py` | `results/prereq_direction_v6_results.json`, `results/prereq_direction_v6_hopdiag_results.json` |
| Expert-annotation asymmetry study and label-shuffle null, §4.4 | `eduhdc/prereq_probing_expert.py`, `eduhdc/prereq_probing_v4.py`, `eduhdc/prereq_probing_v5.py` | `results/prereq_probing_expert_results.json`, `results/prereq_probing_v4_results.json`, `results/prereq_probing_v5_results.json` |
| Knowledge-tracing benchmark (DKT / SAKT / simpleKT / AKT / MAP / HRR / EduHDC), §4.5 | `eduhdc/kt_experiment_rigorous.py` | `results/kt_benchmark_results.json`, `results/kt_10fold_results.json` |
| **VSA-contribution ablation** inside EduHDC-KT (`sim_uni = sim_bi = 0`), §4.5 | `eduhdc/kt_vsa_ablation.py` | `results/kt_vsa_ablation_results.json` |
| 828-parameter frozen-codebook ablation, §4.5 | `eduhdc/ablation_kt.py` | `results/ablation_kt_results.json` |
| Multi-precision quantization study, §4.6 | `eduhdc/precision_eval.py` | `results/precision_eval_results.json` |
| Capacity sweep and functional-form comparison, §4.6 | `eduhdc/capacity_sweep.py`, `eduhdc/capacity_fit_improved.py`, `eduhdc/capacity_deep_investigation.py` | `results/capacity_sweep_results.json`, `results/capacity_fit_improved.json`, `results/capacity_deep_investigation.json` |
| Earlier supervision variants cited in Future Work | `eduhdc/prereq_transitivity_fw3.py`, `eduhdc/prereq_transitivity_fw3c.py`, `eduhdc/prereq_transitivity_inductive.py` | `results/prereq_transitivity_fw3_results.json`, `results/prereq_transitivity_fw3c_results.json`, `results/prereq_transitivity_inductive_results.json` |

### Shared modules and diagnostics

| File | Role |
|---|---|
| `eduhdc/operators.py` | VSA operator implementations (`EduBindBlockDiag`, `BipolarMAP`, `RealHRR`, `ComplexFHRR`) |
| `eduhdc/models.py` | Probe and knowledge-tracing architectures |
| `eduhdc/prereq_transitivity_v7.py` | `load_clean_junyi` — **the loader that builds the 832-node / 971-edge DAG used throughout Section 4** |
| `eduhdc/data_loader_real.py` | ASSISTments 2012–2013 loader (it also contains a Junyi content-table loader that the paper's DAG does *not* use) |
| `eduhdc/data_loader_junyi_expert.py` | Chang et al. (2015) expert prerequisite annotations |
| `eduhdc/semantic_encoder.py` | Content encoder wrapper |
| `eduhdc/benchmark_c1.py` | Synthetic operator-property benchmark *(no archived output)* |
| `eduhdc/prereq_probing_rigorous.py` | Early probing diagnostic *(no archived output)* |
| `eduhdc/edge_benchmark.py` | Edge throughput/memory benchmark *(not cited in this paper)* |
| `eduhdc/verify_setup.py` | Hardware/tensor sanity check |

## Building the Lean formalization

Toolchain: Lean 4 `v4.33.0` (`lean-toolchain` in each project); install via
[elan](https://github.com/leanprover/elan).

```bash
# Core tier — Mathlib-free, no network needed:
cd lean/eduhdc-core && lake build

# Mathlib tier — fetches mathlib4 v4.33.0 (use the cache if available):
cd lean/eduhdc-mathlib && lake exe cache get && lake build
```

Both tiers build with zero errors and zero `sorry`. Every Lean file in this repository
is listed in its project's default build target, so `lake build` checks all of them, and
no file quoted in the paper is left unbuilt. Axiom dependencies are the Lean standard
ones only — `propext` and `Quot.sound`, plus `Classical.choice` in the Mathlib tier —
which you can confirm with `#print axioms <name>` on any theorem the paper names.

Every Lean listing printed in the paper is verbatim from these files, modulo
transliteration of Unicode connectives (`∀`→`forall`, `∃`→`exists`, `≠`→`!=`, `→`→`->`)
and condensed doc comments.

## Reproducing the experiments

Python 3.13 with `torch`, `scikit-learn`, `scipy`, `pandas`, `networkx`, and
`sentence-transformers` (see each script's imports). The `eduhdc` directory is a
package: run any script from the repository root, e.g.
`python eduhdc/transitivity_exact_check.py`; datasets and outputs resolve relative to
the repository root.

Third-party datasets are not redistributed. Place the Junyi Academy curriculum graph
and the Chang et al. (2015) expert prerequisite annotations under `data/junyi/`, and
ASSISTments 2012-2013 under `data/assistments/` (exact file names are defined in
`eduhdc/data_loader_real.py` and `eduhdc/data_loader_junyi_expert.py`). Runs write their
JSON outputs to `results/`; the archived runs reported in the paper are already included
there.

GPU note: the knowledge-tracing scripts assume roughly 12 GB of VRAM at the default
settings (`vsa_dim=2048`, batch 64). Lower `batch_size` or `vsa_dim` if you have less.

## License

MIT (see `LICENSE`).
