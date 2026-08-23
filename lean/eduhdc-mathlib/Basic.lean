import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.LinearAlgebra.Matrix.Defs
import EduBindBlockDiag

open scoped Matrix

set_option linter.unusedSectionVars false

/-!
# EduHDC Algebra: Formal Specification of Pedagogical VSA (Contribution C1)

This module formalizes the algebraic constraints required for an Educational
Vector Symbolic Architecture (VSA), and instantiates them.

Revision 3 (2026-08-23, audit fixes B4 / B11 / M10). Three changes:

  * B11 — the order-sensitivity axiom is now quantified INTRA-FAMILY. The
    binding operation is a family `ops : Nat → (V → V)` and the axiom requires
    two members OF THAT FAMILY to fail to commute. Under the previous
    formulation, whose second binding operation was a free field, a provably
    commutative binding satisfied every axiom; see
    `hadFamily_not_order_sensitive` in src/eduhdc/VSATriad.lean. The class
    below now states exactly the same three axioms as the Mathlib-free kernel
    tier (src/eduhdc/EduBindSelfContained.lean), so the two tiers are
    comparable and the negative results apply to both.

  * B4 — the class previously had NO instances: it was declared and then only
    accompanied by loose theorems. `edubindRealVSA` below is an actual
    instance over real 2×2 matrices, so the specification is discharged rather
    than merely stated.

  * M10 — the non-commutativity counterexample was previously stated over ℤ
    while the tier was described as being over real matrices, with the
    embedding into ℝ asserted in a comment rather than proved. It is now
    stated over ℝ directly, and by two ORTHOGONAL matrices (a rotation and a
    reflection), so the same pair also satisfies the exact-unbind axiom.
-/

namespace EduHDC

/-- A Vector Symbolic Architecture tailored for educational data: a family of
    binding operations, each exactly invertible, each distributing over
    bundling, and not all mutually commuting. -/
class PedagogicalVSA (V : Type) where
  /-- Additive bundling (superposition). -/
  bundle : V → V → V
  /-- The family of binding operations (one per relation). -/
  ops : Nat → (V → V)
  /-- The matching family of left inverses (unbinding). -/
  inv : Nat → (V → V)
  /-- Axiom 1: Superposition Distribution (information preservation). -/
  bundle_distrib : ∀ i X Y, ops i (bundle X Y) = bundle (ops i X) (ops i Y)
  /-- Axiom 2: Pedagogical Order Sensitivity.
      The order of concepts in a learning sequence matters, so two binding
      operations FROM THE FAMILY must fail to commute. Quantifying inside the
      family is what makes this a property of the operator rather than of an
      arbitrary auxiliary map. -/
  order_sensitive : ∃ i j Y, ops i (ops j Y) ≠ ops j (ops i Y)
  /-- Axiom 3: Exact Asymmetric Unbinding.
      Given `Z = ops i Y`, unbinding with the same relation recovers `Y`
      exactly, not merely approximately. -/
  exact_unbind : ∀ i Y, inv i (ops i Y) = Y


end EduHDC

/-!
## Implementation: real-matrix binding (GHRR)

Bundling is matrix addition, binding is matrix multiplication, and unbinding by
an orthogonal relation is left-multiplication by its transpose. We prove the
three axiom properties in general form, then discharge the class itself on the
O(2) family of `EduBindBlockDiag.lean`.
-/

namespace GHRR

variable {n : Type} [Fintype n] [DecidableEq n]

/-- Bundling is matrix addition (superposition). -/
def bundle (A B : Matrix n n ℝ) : Matrix n n ℝ := A + B

/-- Binding is matrix multiplication (non-commutative in general). -/
def bind (A B : Matrix n n ℝ) : Matrix n n ℝ := A * B

/-- Left unbinding by an orthogonal relation: left-multiply by its transpose. -/
def unbind (Z X : Matrix n n ℝ) : Matrix n n ℝ := Xᵀ * Z

/-- Axiom 1 (distribution): matrix multiplication distributes over addition. -/
theorem bundle_distrib (X Y Z : Matrix n n ℝ) :
    bind X (bundle Y Z) = bundle (bind X Y) (bind X Z) :=
  Matrix.mul_add X Y Z

/-- Axiom 3 (exact unbinding): holds whenever the relation matrix is orthogonal
    (`Xᵀ * X = 1`), since the transpose is then a left inverse. -/
theorem exact_unbind_orthogonal (X Y : Matrix n n ℝ) (h : Xᵀ * X = 1) :
    unbind (bind X Y) X = Y := by
  simp only [unbind, bind]
  rw [← Matrix.mul_assoc, h, Matrix.one_mul]

/-- Axiom 2 (order sensitivity) over ℝ, by two ORTHOGONAL 2×2 witnesses.
    Fix M10: the counterexample is now stated in the carrier the tier actually
    uses, and by matrices that also satisfy `exact_unbind_orthogonal`, rather
    than by nilpotent integer matrices that satisfy neither. -/
theorem non_commutative_real :
    ∃ X Y : Matrix (Fin 2) (Fin 2) ℝ, X * Y ≠ Y * X :=
  EduBind.non_commutative

/-- The rotation-only sub-family is abelian, so it fails Axiom 2 (fix B1). -/
theorem rotations_commute (a b : ℝ) : EduBind.rot a * EduBind.rot b
    = EduBind.rot b * EduBind.rot a :=
  EduBind.rot_commutes a b

/-- Rotations commute as binding OPERATIONS, not merely as matrices. -/
theorem rotation_ops_commute (a b : ℝ) (Y : Matrix (Fin 2) (Fin 2) ℝ) :
    EduBind.rot a * (EduBind.rot b * Y) = EduBind.rot b * (EduBind.rot a * Y) := by
  rw [← Matrix.mul_assoc, EduBind.rot_commutes, Matrix.mul_assoc]

end GHRR

/-!
## Discharging the specification (fix B4)

The EduBind family over ℝ: index 0 is a rotation, every other index is a
reflection. Both are orthogonal, so every member unbinds exactly; and they do
not commute, so the family is order-sensitive.
-/

namespace EduHDC

/-- The real EduBind generator family: a rotation and a reflection, i.e. two
    points of O(2) — the group the implementation samples. -/
noncomputable def eduGenReal : Nat → Matrix (Fin 2) (Fin 2) ℝ
  | 0 => EduBind.rot (Real.pi / 2)
  | _ => EduBind.refl 0

/-- EduBind over real 2×2 matrices is a `PedagogicalVSA`. -/
noncomputable instance edubindRealVSA :
    PedagogicalVSA (Matrix (Fin 2) (Fin 2) ℝ) where
  bundle := fun X Y => X + Y
  ops := fun i Y => eduGenReal i * Y
  inv := fun i Y => (eduGenReal i)ᵀ * Y
  bundle_distrib := by
    intro i X Y
    exact Matrix.mul_add _ X Y
  exact_unbind := by
    intro i Y
    cases i with
    | zero   => exact EduBind.exact_unbind (Real.pi / 2) Y
    | succ _ => exact EduBind.exact_unbind_refl 0 Y
  order_sensitive := by
    refine ⟨0, 1, 1, ?_⟩
    simpa [eduGenReal, Matrix.mul_one] using EduBind.rot_refl_non_commutative

/-- IMPOSSIBILITY THEOREM (the fix for B1, stated positively). No
    `PedagogicalVSA` can be built from ROTATIONS ALONE, at any angles: the
    rotation group is abelian, so the order-sensitivity axiom is contradictory
    with a rotation-only family. This is why `eduGenReal` above must include a
    reflection, and it matches what the implementation samples — O(2), not
    SO(2). Revision 2 described EduBind as a rotation family while discharging
    Axiom 2 with a shear, which this theorem shows could not have worked. -/
theorem no_rotation_only_PedagogicalVSA
    (P : PedagogicalVSA (Matrix (Fin 2) (Fin 2) ℝ)) (ts : Nat → ℝ)
    (hops : ∀ i Y, P.ops i Y = EduBind.rot (ts i) * Y) : False := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive
  rw [hops i, hops j, hops i, hops j] at hne
  exact hne (GHRR.rotation_ops_commute (ts i) (ts j) Y)

end EduHDC
