/-
C1 — Three VSA operator families, ALL kernel-verified in Lean 4 core (no Mathlib).

Council-upgrade (task #35, 2026-08-19): the criterion requires >= 3 operators
passing Lean verification. This file adds two more verified families next to
EduBind (imported from `EduBindSelfContained`):

  1. MAP  — elementwise (Hadamard) binding on integer vectors.
            Verified: COMMUTATIVE, exact unbind for ±1 roles, distributive.
            MAP is the commutative contrast operator.
  2. Perm — permutation binding via the swap block T = [[0,1],[1,0]].
            Verified: involution (T²=I) => exact self-unbind, orthogonal,
            NON-commutative with the rotation block R.

Plus the contrast theorems formalizing the central C1 claim:
a commutative bind makes ordered pairs (x,y) and (y,x) representationally
identical, so NO downstream measurement can recover prerequisite order;
EduBind/Perm break that symmetry.

Finally, a second `PedagogicalVSACore` instance is constructed for the
permutation family, so the library now contains two first-class verified
pedagogical operators plus the verified commutative baseline.
-/

import EduBindSelfContained

open Matrix2

-- ---------------------------------------------------------------------------
-- Operator family 2: MAP — elementwise (Hadamard) binding on Fin d → Int
-- ---------------------------------------------------------------------------

/-- Content hypervectors as integer vectors of dimension d. -/
def Vec (d : Nat) : Type := Fin d → Int

/-- Pointwise bundle (addition). -/
def vecAdd (d : Nat) (x y : Vec d) : Vec d := fun i => x i + y i

/-- MAP binding: elementwise multiplication (commutative). -/
def mapBind (d : Nat) (x y : Vec d) : Vec d := fun i => x i * y i

/-- MAP is COMMUTATIVE — the algebraic reason it cannot encode order. -/
theorem map_commutative (d : Nat) (x y : Vec d) :
    mapBind d x y = mapBind d y x := by
  funext i
  dsimp [mapBind]
  rw [Int.mul_comm]

/-- MAP exact unbind: a ±1 role vector is its own inverse under elementwise
    binding (x_i² = 1 pointwise). -/
theorem map_exact_unbind (d : Nat) (x y : Vec d) (h : ∀ i, x i * x i = 1) :
    mapBind d x (mapBind d x y) = y := by
  funext i
  dsimp [mapBind]
  rw [← Int.mul_assoc, h i, Int.one_mul]

/-- MAP binding distributes over bundle. -/
theorem map_distrib (d : Nat) (x y z : Vec d) :
    mapBind d x (vecAdd d y z) = vecAdd d (mapBind d x y) (mapBind d x z) := by
  funext i
  dsimp [mapBind, vecAdd]
  rw [Int.mul_add]

-- ---------------------------------------------------------------------------
-- Contrast theorems: commutative bind is order-blind (the central C1 claim)
-- ---------------------------------------------------------------------------

/-- For a commutative bind, the bound representation of (x, y) and (y, x) is
    IDENTICAL — no measurement can distinguish the two orders. -/
theorem map_order_indistinguishable (d : Nat) (x y : Vec d)
    (measure : Vec d → Int) :
    measure (mapBind d x y) = measure (mapBind d y x) := by
  rw [map_commutative]

/-- Concrete witness that MAP really is order-blind even when x ≠ y:
    with x = [1, 2], y = [3, 5] the two ordered bindings coincide. -/
def vecX : Vec 2 := fun i => if i = ⟨0, by decide⟩ then 1 else 2
def vecY : Vec 2 := fun i => if i = ⟨0, by decide⟩ then 3 else 5

theorem map_order_blind_witness :
    mapBind 2 vecX vecY = mapBind 2 vecY vecX :=
  map_commutative 2 vecX vecY

/-- EduBind, in contrast, DOES distinguish order (imported theorem):
    R*S ≠ S*R. Kept here as the paired statement for the triad. -/
theorem edubind_order_sensitive : R.mul S ≠ S.mul R := bind_non_commutative

-- ---------------------------------------------------------------------------
-- Operator family 3: Perm — permutation binding via the swap block
-- ---------------------------------------------------------------------------

/-- The swap block: permutation matrix T = [[0,1],[1,0]]. -/
def T : Matrix2 Int := { a00 := 0, a01 := 1, a10 := 1, a11 := 0 }

/-- T is an involution: T² = I. -/
theorem T_involution : T.mul T = Matrix2.one := by
  ext <;> dsimp [mul, T, one] <;> omega

/-- T is orthogonal: Tᵀ T = I. -/
theorem T_orthogonal : (T.transpose).mul T = Matrix2.one := by
  ext <;> dsimp [mul, transpose, T, one] <;> omega

/-- Perm exact unbind: because T² = I, binding twice by T recovers the
    content exactly (∀ Y). -/
theorem perm_exact_unbind (Y : Matrix2 Int) : T.mul (T.mul Y) = Y := by
  ext <;> dsimp [mul, T] <;> omega

/-- Perm binding distributes over bundle. -/
theorem perm_distrib (A Z : Matrix2 Int) :
    T.mul (A.add Z) = (T.mul A).add (T.mul Z) := by
  ext <;> dsimp [mul, add, T] <;> omega

/-- Perm is NON-commutative with rotation: T*R ≠ R*T. -/
theorem perm_non_comm_rotation : T.mul R ≠ R.mul T := by
  intro h
  have h00 : (T.mul R).a00 = (R.mul T).a00 := congrArg Matrix2.a00 h
  dsimp [mul, T, R] at h00
  omega

-- ---------------------------------------------------------------------------
-- Second PedagogicalVSACore instance: the permutation family
-- ---------------------------------------------------------------------------

/-- Bind by the swap block (left multiplication). -/
def bindT (Y : Matrix2 Int) : Matrix2 Int := T.mul Y

/-- The swap bind and the rotation bind do not commute on content. -/
theorem perm_rot_non_commutative : ∃ Y, bindT (bindR Y) ≠ bindR (bindT Y) := by
  refine ⟨Matrix2.one, ?_⟩
  intro h
  have h00 := congrArg Matrix2.a00 h
  dsimp [bindT, bindR, mul, one, T, R] at h00
  omega

/-- Perm (swap block + rotation witness) is a PedagogicalVSACore: non-commutative,
    exact unbind (involution), distributive over bundle. -/
def permPedagogicalVSA : PedagogicalVSACore where
  bind₁ := bindT
  bind₂ := bindR
  bundle := fun X Y => X.add Y
  unbind₁ := bindT
  non_commutative_ax := perm_rot_non_commutative
  exact_unbind_ax := perm_exact_unbind
  bundle_distrib_ax := perm_distrib

/-- The library now provides two first-class verified pedagogical operators. -/
def verifiedOperators : List PedagogicalVSACore :=
  [edubindPedagogicalVSA, permPedagogicalVSA]

theorem verifiedOperators_count : verifiedOperators.length = 2 := by rfl
