/-
C1 — Self-contained formal verification of the EduBind block-diagonal operator.

NO Mathlib required: uses only Lean 4 core. We model one 2x2 block of the
block-diagonal GHRR operator over the integers and verify the three
PedagogicalVSACore axioms:

  Axiom 2 (non-commutativity)      — prerequisite order matters: ∃ R S, R*S ≠ S*R
  Axiom 3 (exact asymmetric unbind)— R is orthogonal (Rᵀ R = I), so Rᵀ(R Y) = Y
  Axiom 1 (bundle distributivity)  — bind distributes over bundle

Revision 2 (2026-08-19, council review fixes A-O2/O3, A-S1):
  * `exact_unbind` and `bundle_distrib` are now GENERAL (∀ Y and ∀ S Z),
    not just on concrete instances.
  * The componentwise lifting to B independent blocks is now PROVED
    (`block_exact_unbind`, `block_bundle_distrib`, `block_non_commutative`),
    not merely asserted in a comment.

The orthogonal bind block R is the 90° rotation [[0,-1],[1,0]] (integer
entries, determinant 1), the concrete instance of the generic 2x2 orthogonal
block used by `EduBindBlockDiag` (whose trigonometric family needs Mathlib).

All proofs are by `ext`/`funext` + definitional unfolding + `omega`. After
unfolding, the R entries are integer literals, so every goal is linear in the
content variables — `omega` (Lean 4 core) suffices; Mathlib's `ring` is not
available or needed.
-/

/-- A 2x2 matrix with entries in `α`. -/
@[ext]
structure Matrix2 (α : Type) where
  a00 : α
  a01 : α
  a10 : α
  a11 : α

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

/-- The orthogonal bind block: 90° rotation, integer entries. -/
def R : Matrix2 Int := { a00 := 0, a01 := -1, a10 := 1, a11 := 0 }

/-- A shear matrix used to witness non-commutativity. -/
def S : Matrix2 Int := { a00 := 1, a01 := 1, a10 := 0, a11 := 1 }

/-- Axiom 3 (part 1): the bind block is orthogonal, Rᵀ R = I. -/
theorem R_orthogonal : (R.transpose).mul R = Matrix2.one := by
  ext <;> dsimp [mul, transpose, R, one] <;> omega

/-- Axiom 2: bind is non-commutative — prerequisite order is observable. -/
theorem bind_non_commutative : R.mul S ≠ S.mul R := by
  intro h
  have h00 : (R.mul S).a00 = (S.mul R).a00 := congrArg Matrix2.a00 h
  dsimp [mul, R, S] at h00
  omega

/-- Axiom 3 (part 2), GENERAL: exact asymmetric unbinding holds for every
    content matrix Y, not just one concrete instance. -/
theorem exact_unbind (Y : Matrix2 Int) : (R.transpose).mul (R.mul Y) = Y := by
  ext <;> dsimp [mul, transpose, R] <;> omega

/-- Axiom 1, GENERAL: bind distributes over bundle for all S, Z. -/
theorem bundle_distrib (S Z : Matrix2 Int) :
    R.mul (S.add Z) = (R.mul S).add (R.mul Z) := by
  ext <;> dsimp [mul, add, R] <;> omega

-- ---------------------------------------------------------------------------
-- Componentwise lifting (council fix A-S1): the full EduBind operator is B
-- independent 2x2 blocks acting pointwise. The axioms lift from one block to
-- the whole block-diagonal operator — now proved, not just asserted.
-- ---------------------------------------------------------------------------

/-- The block-diagonal bind: one orthogonal block per index, applied pointwise. -/
def blockBind (B : Nat) (Rs Xs : Fin B → Matrix2 Int) (i : Fin B) :
    Matrix2 Int := (Rs i).mul (Xs i)

/-- Bundling of block-diagonal operators: pointwise addition. -/
def blockBundle (B : Nat) (Xs Ys : Fin B → Matrix2 Int) (i : Fin B) :
    Matrix2 Int := (Xs i).add (Ys i)

/-- Axiom 3 lifts componentwise to the B-block operator (every block = R). -/
theorem block_exact_unbind (B : Nat) (Ys : Fin B → Matrix2 Int) :
    blockBind B (fun _ => R.transpose) (blockBind B (fun _ => R) Ys) = Ys := by
  funext i
  dsimp [blockBind]
  exact exact_unbind (Ys i)

/-- Axiom 1 lifts componentwise to the B-block operator. -/
theorem block_bundle_distrib (B : Nat) (Xs Ys : Fin B → Matrix2 Int) :
    blockBind B (fun _ => R) (blockBundle B Xs Ys) =
    blockBundle B (blockBind B (fun _ => R) Xs) (blockBind B (fun _ => R) Ys) := by
  funext i
  dsimp [blockBind, blockBundle]
  exact bundle_distrib (Xs i) (Ys i)

/-- Axiom 2 lifts: for any nonzero number of blocks, the B-block operator is
    non-commutative (instantiate every block with the witnesses R, S). -/
theorem block_non_commutative (B : Nat) (hB : 0 < B) :
    ∃ (Xs Ys : Fin B → Matrix2 Int), blockBind B Xs Ys ≠ blockBind B Ys Xs := by
  refine ⟨fun _ => R, fun _ => S, ?_⟩
  intro h
  have hi := congrFun h ⟨0, hB⟩
  dsimp [blockBind] at hi
  exact bind_non_commutative hi

-- ---------------------------------------------------------------------------
-- PedagogicalVSACore instance (council fix A-S3): package the verified operator
-- as a first-class object satisfying the three axioms simultaneously.
-- ---------------------------------------------------------------------------

/-- Bind by the prerequisite block R (left multiplication). -/
def bindR (Y : Matrix2 Int) : Matrix2 Int := R.mul Y

/-- Bind by a second block S (left multiplication). -/
def bindS (Y : Matrix2 Int) : Matrix2 Int := S.mul Y

/-- The two bind operations do not commute as operations on content. -/
theorem bind_ops_non_commutative : ∃ Y, bindR (bindS Y) ≠ bindS (bindR Y) := by
  refine ⟨Matrix2.one, ?_⟩
  intro h
  have h00 := congrArg Matrix2.a00 h
  dsimp [bindR, bindS, Matrix2.mul, Matrix2.one, R, S] at h00
  omega

/-- The PedagogicalVSACore interface: two binds whose order matters, an exact
    asymmetric unbind for the primary bind, and distributivity over bundle. -/
structure PedagogicalVSACore where
  bind₁ : Matrix2 Int → Matrix2 Int
  bind₂ : Matrix2 Int → Matrix2 Int
  bundle : Matrix2 Int → Matrix2 Int → Matrix2 Int
  unbind₁ : Matrix2 Int → Matrix2 Int
  non_commutative_ax : ∃ Y, bind₁ (bind₂ Y) ≠ bind₂ (bind₁ Y)
  exact_unbind_ax : ∀ Y, unbind₁ (bind₁ Y) = Y
  bundle_distrib_ax : ∀ S Z, bind₁ (bundle S Z) = bundle (bind₁ S) (bind₁ Z)

/-- EduBind (90° rotation block, shear witness) is a PedagogicalVSACore. -/
def edubindPedagogicalVSA : PedagogicalVSACore where
  bind₁ := bindR
  bind₂ := bindS
  bundle := fun X Y => X.add Y
  unbind₁ := fun Y => R.transpose.mul Y
  non_commutative_ax := bind_ops_non_commutative
  exact_unbind_ax := exact_unbind
  bundle_distrib_ax := bundle_distrib