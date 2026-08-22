/-
FW4 — C1: General-angle rotation block, verified with Mathlib (v4.33.0).

Extends the core 90°-instance proofs (EduBindSelfContained.lean, no Mathlib)
to the FULL family `rot t` for any real angle `t`:

  Axiom 1 (bundle distributivity over bind)  — matrix mul distributes over add
  Axiom 2 (non-commutativity)                — explicit 2x2 rotation counterexample
  Axiom 3 (exact asymmetric unbinding)       — X^T (X Y) = Y for orthogonal X

The 90° block is the t = π/2 member of this family, so these proofs subsume the
core instance; the general orthogonality `rot_orthogonal t` relies on the
Pythagorean identity cos²t + sin²t = 1 (`Real.cos_sq_add_sin_sq`).

We model a single 2x2 block; the full operator is B independent blocks (pointwise),
which is the componentwise lifting already proved in EduBindSelfContained (B-block
theorems) — it lifts uniformly for any angle.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace EduBind

open Matrix

/-- A 2x2 rotation block of arbitrary angle t (orthogonal, determinant 1). -/
noncomputable def rot (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ
  | 0, 0 => Real.cos t
  | 0, 1 => -Real.sin t
  | 1, 0 => Real.sin t
  | 1, 1 => Real.cos t

/-- A shear matrix used to witness non-commutativity. -/
def S : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 1], ![0, 1]]

set_option linter.unnecessarySeqFocus false

/-- Axiom 3 (part 1): the rotation block is orthogonal for ANY angle t,
    R^T R = I — using the Pythagorean identity cos²t + sin²t = 1. -/
theorem rot_orthogonal (t : ℝ) : (rot t).transpose * (rot t) = 1 := by
  ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two] <;>
    nlinarith [Real.cos_sq_add_sin_sq t]

/-- Axiom 2: block multiplication is non-commutative (t = π/2 vs shear). -/
theorem non_commutative : ∃ (X Y : Matrix (Fin 2) (Fin 2) ℝ), X * Y ≠ Y * X := by
  use rot (Real.pi / 2), S
  intro h
  have hL : (rot (Real.pi / 2) * S) 1 1 = 1 := by
    rw [Matrix.mul_apply, Fin.sum_univ_two]
    norm_num [rot, S, Real.cos_pi_div_two, Real.sin_pi_div_two]
  have hR : (S * rot (Real.pi / 2)) 1 1 = 0 := by
    rw [Matrix.mul_apply, Fin.sum_univ_two]
    norm_num [rot, S, Real.cos_pi_div_two, Real.sin_pi_div_two]
  have heq := congr_fun (congr_fun h 1) 1
  rw [hL, hR] at heq
  norm_num at heq

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding for any angle t and
    any content Y: (rot t)^T ((rot t) Y) = Y. -/
theorem exact_unbind (t : ℝ) (Y : Matrix (Fin 2) (Fin 2) ℝ) :
    (rot t).transpose * ((rot t) * Y) = Y := by
  rw [← Matrix.mul_assoc, rot_orthogonal, Matrix.one_mul]

/-- Axiom 1, GENERAL: bind distributes over bundle for all X Y Z. -/
theorem bundle_distrib (X Y Z : Matrix (Fin 2) (Fin 2) ℝ) :
    X * (Y + Z) = X * Y + X * Z := Matrix.mul_add X Y Z

end EduBind