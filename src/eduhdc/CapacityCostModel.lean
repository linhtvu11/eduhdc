/-
C1/C2 — Storage cost model: degrees of freedom in a 1-D VSA memory vs a matrix memory.

WHAT THIS FILE IS. Arithmetic. It formalizes the relationship between a memory's
size parameter and the number of real scalars it holds, using ONLY Lean 4 core
(no Mathlib, no probability theory). Every theorem below is a statement about
counting.

Revision 3 (2026-08-23, audit fix M4). Three things were wrong with the previous
revision, and all three were in the naming and framing rather than in the proofs:

  1. The definitions were called `capacity1D` / `capacityPAM` and documented as
     counting "independently addressable slots". They do not. They count real
     degrees of freedom. For an outer-product associative memory built as
     `S = sum_t |v_t> <k_t|` with orthogonal keys, the number of items that can be
     retrieved exactly is d, not d^2 — the d^2 figure is the dimension of the
     state, not the number of addressable entries. Calling it capacity promised a
     result this module does not establish, so the definitions are renamed
     `dof1D` / `dofPAM` and the word capacity is reserved for the empirical
     measurements (see the note at the end).

  2. `capacity_quadratic_exceeds_linear` compared a 1-D memory of dimension d
     (d scalars) against a d x d matrix memory (2*d^2 scalars). That is a
     comparison across DIFFERENT storage budgets, so the inequality followed from
     the budgets and not from any structural advantage. Both comparisons are now
     stated explicitly and side by side: at matched size parameter the matrix
     memory holds more, and at matched STORAGE the ordering REVERSES. Stating
     only the first would be selective.

  3. `dof_factor_two` and `capacity_ratio_at_same_param` closed by `rfl` and
     `ac_rfl`, i.e. they are definitional identities. That is fine for what they
     are, but the file now says so rather than presenting them as substantive.

Formal statements (all kernel-checked by `lake build`):
  * storageCost1D D  = D          — linear storage in the dimension
  * storageCostPAM d = 2*(d*d)    — quadratic; each complex entry carries 2 real DOF
  * dof_factor_two                — at matched logical dimension d^2, the matrix
                                    memory holds exactly 2x the real scalars
  * dof_ratio_at_same_param       — at the same size parameter d, dofPAM = d * dof1D
  * dofPAM_exceeds_dof1D_same_param      — for d >= 2, strict, at matched parameter
  * dof1D_exceeds_dofPAM_matched_storage — for d >= 1, strict, at matched STORAGE
                                           (the reverse direction)
  * monotonicity of both in the size parameter
-/

namespace CapacityCostModel

-- ====================================================================
-- 1. Storage cost: number of real-valued scalar parameters in the state
-- ====================================================================

/-- 1-D VSA memory of dimension D stores D real scalars. -/
def storageCost1D (D : Nat) : Nat := D

/-- A d x d complex matrix memory stores d^2 complex = 2*d^2 real scalars. -/
def storageCostPAM (d : Nat) : Nat := 2 * (d * d)

theorem storageCost1D_linear (D : Nat) : storageCost1D D = D := rfl

theorem storageCostPAM_quadratic (d : Nat) : storageCostPAM d = 2 * (d * d) := rfl

/-- At the same logical dimension d^2, the matrix memory stores exactly twice the
    real scalars of a 1-D memory. A definitional identity (`rfl`). -/
theorem dof_factor_two (d : Nat) :
    storageCostPAM d = 2 * storageCost1D (d * d) := rfl

-- ====================================================================
-- 2. Degrees of freedom, with the storage efficiency eta made explicit
--
--    NOTE ON NAMING (fix M4): these count real degrees of freedom that the
--    state can carry, NOT the number of items retrievable without
--    interference. The latter is an operator- and noise-dependent quantity
--    that this module does not model; it is measured empirically instead.
-- ====================================================================

/-- Degrees of freedom of a 1-D memory: eta per dimension. -/
def dof1D (eta D : Nat) : Nat := eta * D

/-- Degrees of freedom of a d x d matrix memory: eta per matrix entry. -/
def dofPAM (eta d : Nat) : Nat := eta * (d * d)

theorem dof1D_linear (eta D : Nat) : dof1D eta D = eta * D := rfl

theorem dofPAM_quadratic (eta d : Nat) : dofPAM eta d = eta * (d * d) := rfl

-- ====================================================================
-- 3. The two comparisons, stated together
-- ====================================================================

/-- At the same size parameter d and equal efficiency, the matrix memory holds
    d times the degrees of freedom. A definitional identity (`ac_rfl`); note
    that the two sides describe memories of DIFFERENT storage cost, so this is
    a statement about the size parameter, not about efficiency of storage. -/
theorem dof_ratio_at_same_param (eta d : Nat) :
    dofPAM eta d = d * dof1D eta d := by
  unfold dofPAM dof1D
  ac_rfl

/-- Direction 1 — matched SIZE PARAMETER: for d >= 2 the matrix memory strictly
    exceeds the 1-D memory. This is the comparison usually quoted. -/
theorem dofPAM_exceeds_dof1D_same_param (d : Nat) (hd : 2 ≤ d) :
    dof1D 1 d < dofPAM 1 d := by
  unfold dof1D dofPAM
  rw [Nat.one_mul, Nat.one_mul]
  have hprod : d * 1 < d * d :=
    Nat.mul_lt_mul_of_pos_left (by omega : 1 < d) (by omega : 0 < d)
  rwa [Nat.mul_one] at hprod

/-- Direction 2 — matched STORAGE, and the ordering REVERSES. Give the 1-D
    memory the same number of real scalars the matrix memory uses, namely
    `storageCostPAM d = 2*d^2`, and it carries strictly more degrees of freedom
    than the matrix memory does, for every d >= 1.

    This is the honest counterpart of the theorem above (fix M4): counting alone
    does not favour the matrix regime. Whatever advantage the matrix memory has
    must come from something this module does not formalize — how retrieval
    degrades under superposition — which is why that is measured empirically
    rather than asserted here. -/
theorem dof1D_exceeds_dofPAM_matched_storage (d : Nat) (hd : 1 ≤ d) :
    dofPAM 1 d < dof1D 1 (storageCostPAM d) := by
  unfold dofPAM dof1D storageCostPAM
  rw [Nat.one_mul, Nat.one_mul]
  have h : d * d < 2 * (d * d) := by
    have : 0 < d * d := Nat.mul_pos hd hd
    omega
  exact h

/-- Both comparisons at once, so neither can be quoted without the other. -/
theorem dof_comparison_depends_on_normalisation (d : Nat) (hd : 2 ≤ d) :
    dof1D 1 d < dofPAM 1 d ∧ dofPAM 1 d < dof1D 1 (storageCostPAM d) :=
  ⟨dofPAM_exceeds_dof1D_same_param d hd,
   dof1D_exceeds_dofPAM_matched_storage d (by omega)⟩

/-- 1-D degrees of freedom are non-decreasing in the dimension. -/
theorem dof1D_mono (eta D1 D2 : Nat) (h : D1 ≤ D2) :
    dof1D eta D1 ≤ dof1D eta D2 := by
  unfold dof1D
  exact Nat.mul_le_mul_left eta h

/-- Matrix degrees of freedom are non-decreasing in the size parameter. -/
theorem dofPAM_mono (eta d1 d2 : Nat) (h : d1 ≤ d2) :
    dofPAM eta d1 ≤ dofPAM eta d2 := by
  unfold dofPAM
  exact Nat.mul_le_mul_left eta (Nat.mul_le_mul h h)

-- ====================================================================
-- 4. Scope boundary (not theorems)
-- ====================================================================
/-
What this file does NOT prove, by design:

  * Any statement about RETRIEVAL. The number of items a memory can return
    without interference is not a counting quantity: for a 1-D superposition it
    degrades continuously with the load ratio T/D, and for an outer-product
    matrix memory with orthogonal keys the exact-retrieval limit is d, not d^2.
    Neither statement is derivable from the arithmetic above; the first is
    measured in the capacity sweep (src/eduhdc/capacity_sweep.py), the second is
    a linear-algebra fact requiring Mathlib.

  * The sqrt(D/T) crosstalk law. That is a probabilistic concentration statement
    about random hypervectors, outside Lean 4 core and outside the scope of a
    cost model. It is an EMPIRICAL input, and the sweep above is what supplies
    it: with keys drawn without replacement and K >> T, retrieval accuracy is a
    logistic function of z = sqrt(D/T) with R^2 = 0.99 over 72 (operator, T, D)
    points, and the collapse is indistinguishable across MAP, HRR and EduBind.

  * That the matrix regime is preferable. Section 3 above shows the counting
    comparison flips with the normalisation chosen, so the case for the matrix
    memory rests on the retrieval behaviour, not on degrees of freedom.
-/

end CapacityCostModel
