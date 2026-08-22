# EDUHDC — Machine-Checked Order-Sensitive Binding for Vector-Symbolic Curriculum Representations

Release artifact accompanying the paper:

> **EDUHDC: Machine-Checking Order-Sensitive Binding for Vector-Symbolic Curriculum Representations in Lean 4**
> Vu Linh Nguyen-Van, Thai Son Nguyen, Bao-An Nguyen — Tra Vinh University, Tra Vinh, Vietnam

This repository contains everything needed to re-check the machine-verified claims and
re-run the empirical evaluations reported in the paper:

| Directory | Content |
|---|---|
| `lean/eduhdc-core` | Mathlib-free Lean 4 project; every file is in the default build target and is checked directly by the Lean 4 kernel |
| `lean/eduhdc-mathlib` | Mathlib tier (`mathlib4` pinned to `v4.33.0`): the three-axiom `PedagogicalVSA` specification and the rotation-block verification |
| `python` | Evaluation code used for the experiments section |
| `results` | Result JSON files produced by those runs |
| `figures` | Figures included in the paper |

## Paper ↔ file map

| Paper item | File(s) |
|---|---|
| `PedagogicalVSA` three-axiom specification (§3.1) | `lean/eduhdc-mathlib/Basic.lean` |
| Kernel-level `PedagogicalVSACore` tier (§3.1) | `lean/eduhdc-core/EduBindSelfContained.lean` |
| EduBind rotation-block verification, incl. the 2×2 non-commutativity counterexample and exact unbinding via `Real.cos_sq_add_sin_sq` (§3.2) | `lean/eduhdc-mathlib/EduBindBlockDiag.lean` |
| `map_order_indistinguishable` negative theorem for commutative (MAP) binding (§3.2) | `lean/eduhdc-core/VSATriad.lean` |
| `chain_exact_unbind` induction (§3.3) | `lean/eduhdc-core/ChainTransitivity.lean` |
| Capacity cost model, `capacity1D` / `capacityPAM` (§3.4) | `lean/eduhdc-core/CapacityCostModel.lean` |
| VSA operator implementations used by the experiments | `python/operators.py` |
| Probe models and training harness | `python/models.py` |
| Junyi-DAG curriculum loaders | `python/data_loader.py`, `python/data_loader_real.py` |
| Content encoder | `python/semantic_encoder.py` |
| Runtime/algebraic verification + chain-error sweep (Fig. 2) | `python/benchmark_c1.py` |
| Node-inductive generalization tier (Fig. 3) | `python/prereq_transitivity_inductive.py` |
| Transductive prerequisite probing | `python/prereq_probing_rigorous.py`, `python/prereq_probing_expert.py` |
| Knowledge-tracing corroboration (vs DKT/SAKT) | `python/kt_experiment_rigorous.py` |
| 828-parameter frozen-codebook ablation | `python/ablation_kt.py` |
| Multi-precision quantization study | `python/precision_eval.py` |

## Building the Lean formalization

Toolchain: Lean 4 `v4.33.0` (`lean-toolchain` in each project); install via
[elan](https://github.com/leanprover/elan).

```bash
# Core tier — Mathlib-free, no network needed:
cd lean/eduhdc-core && lake build

# Mathlib tier — fetches mathlib4 v4.33.0 (use the cache if available):
cd lean/eduhdc-mathlib && lake exe cache get && lake build
```

Both tiers build with zero errors and zero `sorry`; every theorem stated in the paper
is discharged by the kernel.

## Reproducing the experiments

Python 3.13 with `torch`, `scikit-learn`, `scipy`, `pandas`, `networkx`
(see each script's imports). Scripts resolve datasets relative to the parent of the
script directory — the Junyi Academy curriculum graph, the Chang et al. (2015) expert
prerequisite annotations, and ASSISTments 2012-2013 must be placed in a `data/`
directory next to `python/` (i.e., at the root of this repository). All datasets are
third-party and used under their original release terms; they are therefore not
redistributed here. The numbers reported in the paper were produced by the scripts in
`python/`; corresponding outputs are archived in `results/`.

## License

MIT (see `LICENSE`).