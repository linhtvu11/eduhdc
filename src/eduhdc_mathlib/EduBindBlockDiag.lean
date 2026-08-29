/-
C1 — The EduBind block family over the reals, verified with Mathlib (v4.33.0).

Extends the Mathlib-free core proofs (EduBindSelfContained.lean, integer O(2))
to the FULL continuous family for any real angle `t`:

  Axiom 1 (bundle distributivity)   — matrix mul distributes over add
  Axiom 2 (order sensitivity)       — rotation and reflection do not commute
  Axiom 3 (exact asymmetric unbind) — Mᵀ (M Y) = Y for orthogonal M

Revision 3 (2026-08-23, audit fix B1). Revision 2 modelled EduBind as the
rotation family `rot t` and discharged Axiom 2 with a SHEAR matrix. Both parts
were wrong for the same reason, now recorded as a theorem rather than papered
over:

  `rot_commutes` below proves that the rotation family is ABELIAN — for ALL
  real angles a, b. So a rotation-only family cannot satisfy Axiom 2 at all,
  and the shear that Revision 2 used as a witness is not a member of the
  operator family (it is not even orthogonal, so it does not satisfy Axiom 3).

  What actually makes the implemented operator order-sensitive is that
  `EduBindBlockDiag.random_vector` (src/eduhdc/operators.py) samples blocks
  `[[c, -s·sin t], [sin t, s·c]]` with `s ∈ {-1,+1}` — i.e. the full orthogonal
  group O(2), not SO(2). The `s = -1` branch is the reflection `refl` below
  (determinant -1). Reflections are orthogonal, so Axiom 3 is unaffected, and
  `rot_refl_non_commutative` discharges Axiom 2 with two genuine members of the
  sampled family.

We model a single 2x2 block; the full operator is B independent blocks acting
pointwise, which is the componentwise lifting already proved in
EduBindSelfContained (block theorems) — it lifts uniformly for any angle.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace EduBind

open Matrix

set_option linter.unnecessarySeqFocus false

/-- A 2x2 rotation block of arbitrary angle t (orthogonal, determinant +1). -/
noncomputable def rot (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ
  | 0, 0 => Real.cos t
  | 0, 1 => -Real.sin t
  | 1, 0 => Real.sin t
  | 1, 1 => Real.cos t

/-- A 2x2 reflection block of arbitrary angle t (orthogonal, determinant -1).
    This is the `s = -1` branch that the implementation actually samples. -/
noncomputable def refl (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ
  | 0, 0 => Real.cos t
  | 0, 1 => Real.sin t
  | 1, 0 => Real.sin t
  | 1, 1 => -Real.cos t

/-- Axiom 3 (part 1a): the rotation block is orthogonal for ANY angle t,
    Rᵀ R = I — using the Pythagorean identity cos²t + sin²t = 1. -/
theorem rot_orthogonal (t : ℝ) : (rot t).transpose * (rot t) = 1 := by
  ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two] <;>
    nlinarith [Real.cos_sq_add_sin_sq t]

/-- Axiom 3 (part 1b): the reflection block is orthogonal too, so extending the
    family from SO(2) to O(2) does not weaken exact unbinding. -/
theorem refl_orthogonal (t : ℝ) : (refl t).transpose * (refl t) = 1 := by
  ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [refl, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two] <;>
    nlinarith [Real.cos_sq_add_sin_sq t]

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding by a rotation, for any
    angle t and any content Y. -/
theorem exact_unbind (t : ℝ) (Y : Matrix (Fin 2) (Fin 2) ℝ) :
    (rot t).transpose * ((rot t) * Y) = Y := by
  rw [← Matrix.mul_assoc, rot_orthogonal, Matrix.one_mul]

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding by a reflection. -/
theorem exact_unbind_refl (t : ℝ) (Y : Matrix (Fin 2) (Fin 2) ℝ) :
    (refl t).transpose * ((refl t) * Y) = Y := by
  rw [← Matrix.mul_assoc, refl_orthogonal, Matrix.one_mul]

/-- Axiom 1, GENERAL: bind distributes over bundle for all X Y Z. -/
theorem bundle_distrib (X Y Z : Matrix (Fin 2) (Fin 2) ℝ) :
    X * (Y + Z) = X * Y + X * Z := Matrix.mul_add X Y Z

-- ---------------------------------------------------------------------------
-- Fix B1, part 1: the NEGATIVE result. The rotation family is abelian, so
-- rotations alone cannot carry order sensitivity. Stated as a theorem, for all
-- pairs of angles, rather than left as a remark.
-- ---------------------------------------------------------------------------

/-- THE ROTATION FAMILY IS ABELIAN: `rot a * rot b = rot b * rot a` for ALL real
    angles. Consequently a rotation-only EduBind family provably fails Axiom 2,
    exactly as a commutative (MAP-style) binding does — which is why the family
    must include reflections. -/
theorem rot_commutes (a b : ℝ) : rot a * rot b = rot b * rot a := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-- Corollary: no measurement on `rot a * rot b` can recover which rotation was
    applied first — the rotation-only analogue of `map_order_indistinguishable`. -/
theorem rot_order_indistinguishable (a b : ℝ)
    (measure : Matrix (Fin 2) (Fin 2) ℝ → ℝ) :
    measure (rot a * rot b) = measure (rot b * rot a) := by
  rw [rot_commutes]

-- ---------------------------------------------------------------------------
-- Fix B1, part 2: the POSITIVE result. Order sensitivity holds within O(2),
-- witnessed by a rotation and a reflection — both members of the family the
-- implementation samples, and both orthogonal.
-- ---------------------------------------------------------------------------

/-- Axiom 2, INTRA-FAMILY: a rotation and a reflection do not commute. -/
theorem rot_refl_non_commutative :
    rot (Real.pi / 2) * refl 0 ≠ refl 0 * rot (Real.pi / 2) := by
  intro h
  have hL : (rot (Real.pi / 2) * refl 0) 0 1 = 1 := by
    rw [Matrix.mul_apply, Fin.sum_univ_two]
    norm_num [rot, refl, Real.cos_pi_div_two, Real.sin_pi_div_two]
  have hR : (refl 0 * rot (Real.pi / 2)) 0 1 = -1 := by
    rw [Matrix.mul_apply, Fin.sum_univ_two]
    norm_num [rot, refl, Real.cos_pi_div_two, Real.sin_pi_div_two]
  have heq := congr_fun (congr_fun h 0) 1
  rw [hL, hR] at heq
  norm_num at heq

/-- Axiom 2 in existential form, over the O(2) family. -/
theorem non_commutative : ∃ (X Y : Matrix (Fin 2) (Fin 2) ℝ), X * Y ≠ Y * X :=
  ⟨rot (Real.pi / 2), refl 0, rot_refl_non_commutative⟩

end EduBind
