/-
C1/C2 — FW5: Storage cost model for the capacity lemma O(D) vs O(d²).

Empirical basis (E3, docs/c1_path_c_results.md): 1D holographic memory has
capacity O(D) — retrieval collapses along √(D/T) for every operator
(MAP/HRR/EduBind), so the limit is the dimension D itself. PAM (d×d complex
matrix, docs/c2_initial_results.md Exp3) has capacity O(d²): 0.754 vs 0.440
for HRR at the same load ratio.

This file formalizes the COST MODEL under the claim — the arithmetic between
the size parameter and the storage/capacity formulas — using ONLY Lean 4 core
(no Mathlib, no probability theory). The empirical √(D/T) law itself is
experimental evidence, not re-derived here (see section 4 note).

Formal statements (all kernel-checked by `lake build`):
  * storageCost1D D = D            — linear storage in the dimension
  * storageCostPAM d = 2·(d·d)     — quadratic storage; each complex entry of
                                     the PAM matrix carries 2 real DOF
  * dof_factor_two                — formalizes council finding D-F5: at matched
                                     logical dimension d², PAM stores exactly
                                     2× the real scalars of a 1D VSA
  * capacity1D / capacityPAM      — slot-count capacity model, efficiency η
                                     explicit
  * capacity_ratio_at_same_param  — PAM(d) = d × 1D(d) capacity
  * capacity_quadratic_exceeds_linear — for d ≥ 2, PAM strictly exceeds 1D
  * monotonicity in the size parameter
-/

namespace CapacityCostModel

-- ====================================================================
-- 1. Storage cost: number of real-valued scalar parameters in the state
-- ====================================================================

/-- 1D VSA memory of dimension D stores D real scalars. -/
def storageCost1D (D : Nat) : Nat := D

/-- PAM d×d complex matrix stores d² complex = 2·d² real scalars. -/
def storageCostPAM (d : Nat) : Nat := 2 * (d * d)

theorem storageCost1D_linear (D : Nat) : storageCost1D D = D := rfl

theorem storageCostPAM_quadratic (d : Nat) : storageCostPAM d = 2 * (d * d) := rfl

/-- Council finding D-F5, formalized: at the same logical dimension d², PAM
    stores exactly twice the real scalars of a 1D VSA. -/
theorem dof_factor_two (d : Nat) :
    storageCostPAM d = 2 * storageCost1D (d * d) := rfl

-- ====================================================================
-- 2. Capacity model: number of independently addressable slots, with the
--    storage efficiency η made explicit (η = 1 for orthogonal keys)
-- ====================================================================

/-- 1D capacity: each dimension is one slot. -/
def capacity1D (eta D : Nat) : Nat := eta * D

/-- PAM capacity: each matrix entry is one slot. -/
def capacityPAM (eta d : Nat) : Nat := eta * (d * d)

theorem capacity1D_linear (eta D : Nat) : capacity1D eta D = eta * D := rfl

theorem capacityPAM_quadratic (eta d : Nat) : capacityPAM eta d = eta * (d * d) := rfl

-- ====================================================================
-- 3. Comparison theorems
-- ====================================================================

/-- At the same size parameter d, PAM capacity is d times the 1D capacity
    (equal efficiency η) — the quadratic-vs-linear structural advantage. -/
theorem capacity_ratio_at_same_param (eta d : Nat) :
    capacityPAM eta d = d * capacity1D eta d := by
  unfold capacityPAM capacity1D
  ac_rfl

/-- For d ≥ 2, PAM capacity strictly exceeds 1D capacity (η = 1). -/
theorem capacity_quadratic_exceeds_linear (d : Nat) (hd : 2 ≤ d) :
    capacity1D 1 d < capacityPAM 1 d := by
  unfold capacity1D capacityPAM
  rw [Nat.one_mul, Nat.one_mul]
  -- goal: d < d * d
  have hprod : d * 1 < d * d :=
    Nat.mul_lt_mul_of_pos_left (by omega : 1 < d) (by omega : 0 < d)
  rwa [Nat.mul_one] at hprod

/-- 1D capacity is non-decreasing in the dimension. -/
theorem capacity1D_mono (eta D1 D2 : Nat) (h : D1 ≤ D2) :
    capacity1D eta D1 ≤ capacity1D eta D2 := by
  unfold capacity1D
  exact Nat.mul_le_mul_left eta h

/-- PAM capacity is non-decreasing in the size parameter. -/
theorem capacityPAM_mono (eta d1 d2 : Nat) (h : d1 ≤ d2) :
    capacityPAM eta d1 ≤ capacityPAM eta d2 := by
  unfold capacityPAM
  exact Nat.mul_le_mul_left eta (Nat.mul_le_mul h h)

-- ====================================================================
-- 4. Honesty note (not a theorem — scope boundary)
-- ====================================================================
/-
What this file does NOT prove, by design:
  * The √(D/T) crosstalk law and the O(D) capacity ceiling of 1D VSA. That is a
    probabilistic concentration statement (random hypervectors, interference
    growth) that requires probability/measure theory — beyond Lean 4 core and
    beyond the "cost model" scope stated in FW5. It is the EMPIRICAL input
    (E3), imported here only through the linearity of `capacity1D`.
  * That PAM retrieval is exact for orthogonal keys. That is an operator-level
    theorem (would require Mathlib linear algebra / the `exact_unbind` axioms
    already verified in EduBindSelfContained.lean); here we only use the
    slot-counting consequence O(d²).
-/

end CapacityCostModel
