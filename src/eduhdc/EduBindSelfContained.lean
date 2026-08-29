/-
C1 — Self-contained formal verification of the EduBind block-diagonal operator.

NO Mathlib required: uses only Lean 4 core. We model one 2x2 block of the
block-diagonal GHRR operator over the integers and verify the three
PedagogicalVSA axioms:

  Axiom 1 (bundle distributivity)   — bind distributes over bundle
  Axiom 2 (order sensitivity)       — two members of the SAME operator family
                                      fail to commute
  Axiom 3 (exact asymmetric unbind) — each family member is orthogonal, so
                                      Mᵀ(M Y) = Y for every content Y

Revision 3 (2026-08-23, audit fixes B1 / B11).
  Two defects in Revision 2 are fixed here, and both fixes are load-bearing.

  B1 — the operator family was described as ROTATIONS, but the rotation group
  is ABELIAN: over the reals `rot a * rot b = rot b * rot a` for all angles
  (machine-checked as `EduBind.rot_commutes` in the Mathlib tier,
  src/eduhdc_mathlib/EduBindBlockDiag.lean). A rotation-only family therefore
  CANNOT satisfy Axiom 2. What makes the implemented operator order-sensitive
  is that `EduBindBlockDiag.random_vector` (src/eduhdc/operators.py) samples
  blocks `[[c, -s·sin t], [sin t, s·c]]` with `s ∈ {-1,+1}`, i.e. it samples
  the full orthogonal group O(2), not SO(2); the `s = -1` branch is a
  REFLECTION (determinant -1). The family below is therefore {rotation,
  reflection}, matching what the code actually samples. Both are orthogonal,
  so Axiom 3 is untouched.

  B11 — Revision 2 stated Axiom 2 as `∃ Y, bind₁ (bind₂ Y) ≠ bind₂ (bind₁ Y)`
  with `bind₂` a FREE field of the structure. That is not a property of the
  operator: for almost any `bind₁` some unrelated `bind₂` fails to commute
  with it, and indeed Revision 2 discharged it with a SHEAR matrix, which is
  neither orthogonal nor a member of the operator family. A commutative
  (Hadamard/MAP) binding satisfied all three axioms under that formulation —
  see `hadFamily_not_order_sensitive` in VSATriad.lean. Axiom 2 is now
  quantified INTRA-FAMILY: both operators must come from the declared family
  `ops`, so the axiom is a genuine property of the family.

All proofs are by `ext`/`funext` + definitional unfolding + `omega`. After
unfolding, the matrix entries are integer literals, so every goal is linear in
the content variables — `omega` (Lean 4 core) suffices; Mathlib's `ring` is not
available or needed.
-/

/-- A 2x2 matrix with entries in `α`. -/
@[ext]
structure Matrix2 (α : Type) where
  a00 : α
  a01 : α
  a10 : α
  a11 : α
  deriving DecidableEq

namespace Matrix2

def add [Add α] (M N : Matrix2 α) : Matrix2 α :=
  { a00 := M.a00 + N.a00, a01 := M.a01 + N.a01,
    a10 := M.a10 + N.a10, a11 := M.a11 + N.a11 }

def mul [Mul α] [Add α] (M N : Matrix2 α) : Matrix2 α :=
  { a00 := M.a00 * N.a00 + M.a01 * N.a10,
    a01 := M.a00 * N.a01 + M.a01 * N.a11,
    a10 := M.a10 * N.a00 + M.a11 * N.a10,
    a11 := M.a10 * N.a01 + M.a11 * N.a11 }

def transpose (M : Matrix2 α) : Matrix2 α :=
  { a00 := M.a00, a01 := M.a10, a10 := M.a01, a11 := M.a11 }

def one [OfNat α 1] [OfNat α 0] : Matrix2 α :=
  { a00 := 1, a01 := 0, a10 := 0, a11 := 1 }

end Matrix2

open Matrix2

-- ---------------------------------------------------------------------------
-- The EduBind operator family: the integer points of O(2).
--
-- `Rot` is the 90-degree rotation (determinant +1) and `Ref` is a reflection
-- (determinant -1). Both are orthogonal, hence both unbind exactly; and they
-- do NOT commute with each other, which is what gives the family its order
-- sensitivity (fix B1: a rotation-only family would be abelian).
-- ---------------------------------------------------------------------------

/-- Rotation block (90 degrees), determinant +1. -/
def Rot : Matrix2 Int := { a00 := 0, a01 := -1, a10 := 1, a11 := 0 }

/-- Reflection block, determinant -1 — the `s = -1` branch that
    `EduBindBlockDiag.random_vector` actually samples. -/
def Ref : Matrix2 Int := { a00 := 1, a01 := 0, a10 := 0, a11 := -1 }

/-- Axiom 3 (part 1a): the rotation block is orthogonal. -/
theorem Rot_orthogonal : (Rot.transpose).mul Rot = Matrix2.one := by
  ext <;> dsimp [mul, transpose, Rot, one] <;> omega

/-- Axiom 3 (part 1b): the reflection block is orthogonal too, so replacing
    the shear witness by a reflection does not weaken exact unbinding. -/
theorem Ref_orthogonal : (Ref.transpose).mul Ref = Matrix2.one := by
  ext <;> dsimp [mul, transpose, Ref, one] <;> omega

/-- Axiom 2, witness: rotation and reflection do NOT commute. Both are members
    of the EduBind family, so this is an intra-family statement (fix B11). -/
theorem Rot_Ref_non_commutative : Rot.mul Ref ≠ Ref.mul Rot := by
  intro h
  have h01 : (Rot.mul Ref).a01 = (Ref.mul Rot).a01 := congrArg Matrix2.a01 h
  dsimp [mul, Rot, Ref] at h01
  omega

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding by the rotation block,
    for every content matrix Y. -/
theorem Rot_exact_unbind (Y : Matrix2 Int) : (Rot.transpose).mul (Rot.mul Y) = Y := by
  ext <;> dsimp [mul, transpose, Rot] <;> omega

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding by the reflection block. -/
theorem Ref_exact_unbind (Y : Matrix2 Int) : (Ref.transpose).mul (Ref.mul Y) = Y := by
  ext <;> dsimp [mul, transpose, Ref] <;> omega

/-- Axiom 1, GENERAL: the rotation block distributes over bundle. -/
theorem Rot_bundle_distrib (A Z : Matrix2 Int) :
    Rot.mul (A.add Z) = (Rot.mul A).add (Rot.mul Z) := by
  ext <;> dsimp [mul, add, Rot] <;> omega

/-- Axiom 1, GENERAL: the reflection block distributes over bundle. -/
theorem Ref_bundle_distrib (A Z : Matrix2 Int) :
    Ref.mul (A.add Z) = (Ref.mul A).add (Ref.mul Z) := by
  ext <;> dsimp [mul, add, Ref] <;> omega

-- ---------------------------------------------------------------------------
-- Componentwise lifting: the full EduBind operator is B independent 2x2 blocks
-- acting pointwise. The axioms lift from one block to the whole block-diagonal
-- operator — proved, not asserted.
-- ---------------------------------------------------------------------------

/-- The block-diagonal bind: one orthogonal block per index, applied pointwise. -/
def blockBind (B : Nat) (Ms Xs : Fin B → Matrix2 Int) (i : Fin B) :
    Matrix2 Int := (Ms i).mul (Xs i)

/-- Bundling of block-diagonal operators: pointwise addition. -/
def blockBundle (B : Nat) (Xs Ys : Fin B → Matrix2 Int) (i : Fin B) :
    Matrix2 Int := (Xs i).add (Ys i)

/-- Axiom 3 lifts componentwise to the B-block operator (every block = Rot). -/
theorem block_exact_unbind (B : Nat) (Ys : Fin B → Matrix2 Int) :
    blockBind B (fun _ => Rot.transpose) (blockBind B (fun _ => Rot) Ys) = Ys := by
  funext i
  dsimp [blockBind]
  exact Rot_exact_unbind (Ys i)

/-- Axiom 1 lifts componentwise to the B-block operator. -/
theorem block_bundle_distrib (B : Nat) (Xs Ys : Fin B → Matrix2 Int) :
    blockBind B (fun _ => Rot) (blockBundle B Xs Ys) =
    blockBundle B (blockBind B (fun _ => Rot) Xs) (blockBind B (fun _ => Rot) Ys) := by
  funext i
  dsimp [blockBind, blockBundle]
  exact Rot_bundle_distrib (Xs i) (Ys i)

/-- Axiom 2 lifts: for any nonzero number of blocks, the B-block operator is
    order-sensitive, witnessed by two members of the family (Rot, Ref). -/
theorem block_non_commutative (B : Nat) (hB : 0 < B) :
    ∃ (Xs Ys : Fin B → Matrix2 Int), blockBind B Xs Ys ≠ blockBind B Ys Xs := by
  refine ⟨fun _ => Rot, fun _ => Ref, ?_⟩
  intro h
  have hi := congrFun h ⟨0, hB⟩
  dsimp [blockBind] at hi
  exact Rot_Ref_non_commutative hi

-- ---------------------------------------------------------------------------
-- The PedagogicalVSA specification (Revision 3).
--
-- The operator is now a FAMILY `ops : Nat → (V → V)` with a matching family of
-- left inverses `inv`, and order sensitivity is quantified over two members of
-- that same family. This is the fix for B11: under the old formulation, whose
-- second binding operation was a free field, a provably commutative binding
-- also satisfied every axiom (VSATriad.hadFamily_not_order_sensitive).
--
-- The specification is generic in the carrier `V`, so the same three axioms are
-- stated once and instantiated by every operator family, and the negative
-- results below are stated against exactly the same axiom shape.
-- ---------------------------------------------------------------------------

/-- A Vector Symbolic Architecture tailored for educational data: a family of
    binding operations, each exactly invertible, distributing over bundling,
    and not all mutually commuting. -/
structure PedagogicalVSA (V : Type) where
  /-- Additive bundling (superposition). -/
  bundle : V → V → V
  /-- The family of binding operations. -/
  ops : Nat → (V → V)
  /-- The matching family of left inverses (unbinding). -/
  inv : Nat → (V → V)
  /-- Axiom 3: every member of the family unbinds exactly, for every content. -/
  exact_unbind_ax : ∀ i Y, inv i (ops i Y) = Y
  /-- Axiom 1: every member distributes over bundling. -/
  bundle_distrib_ax : ∀ i X Y, ops i (bundle X Y) = bundle (ops i X) (ops i Y)
  /-- Axiom 2: order matters — two members OF THE FAMILY do not commute. -/
  order_sensitive_ax : ∃ i j Y, ops i (ops j Y) ≠ ops j (ops i Y)

/-- Left multiplication by a fixed matrix — the shape of every binding
    operation used here. -/
def lmul (M : Matrix2 Int) : Matrix2 Int → Matrix2 Int := fun Y => M.mul Y

/-- The EduBind generator family: index 0 is the rotation, every other index is
    the reflection. Both are integer points of O(2). -/
def eduGen : Nat → Matrix2 Int
  | 0 => Rot
  | _ => Ref

theorem eduGen_zero : eduGen 0 = Rot := rfl
theorem eduGen_succ (n : Nat) : eduGen (n + 1) = Ref := rfl

/-- Axiom 2 at the level of binding OPERATIONS: composing by the rotation and
    composing by the reflection give different results. Both operations come
    from the EduBind family `eduGen`, which is what fix B11 requires. -/
theorem lmul_Rot_Ref_non_comm :
    ∃ Y, lmul Rot (lmul Ref Y) ≠ lmul Ref (lmul Rot Y) := by
  refine ⟨Matrix2.one, ?_⟩
  intro h
  have h01 := congrArg Matrix2.a01 h
  dsimp [lmul, mul, one, Rot, Ref] at h01
  omega

/-- EduBind (integer O(2): rotation + reflection) is a PedagogicalVSA. -/
def edubindVSA : PedagogicalVSA (Matrix2 Int) where
  bundle := fun X Y => X.add Y
  ops := fun i => lmul (eduGen i)
  inv := fun i => lmul (eduGen i).transpose
  exact_unbind_ax := by
    intro i Y
    cases i with
    | zero   => exact Rot_exact_unbind Y
    | succ _ => exact Ref_exact_unbind Y
  bundle_distrib_ax := by
    intro i X Y
    cases i with
    | zero   => exact Rot_bundle_distrib X Y
    | succ _ => exact Ref_bundle_distrib X Y
  order_sensitive_ax := by
    obtain ⟨Y, hY⟩ := lmul_Rot_Ref_non_comm
    exact ⟨0, 1, Y, hY⟩

-- xem main.tex §3 lời mời người đọc chạy lệnh này
#print axioms edubindVSA
