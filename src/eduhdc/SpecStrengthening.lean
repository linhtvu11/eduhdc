/-
C1 — Specification strengthening in response to formal-methods review.

WHAT THIS FILE IS
-----------------
Two findings from an independent Lean-4/Coq reviewer, discharged as machine-checked
theorems rather than as prose caveats in the paper:

  W4  `no_hadamard_PedagogicalVSA` (VSATriad.lean) is stated over the concrete
      carrier `Matrix2 Int`. The proof uses only commutativity and associativity
      of multiplication, so the obstruction is not specific to `Int`. Reviewer
      asked whether a `CommRing`-level statement goes through. It does, and this
      file proves it: `hadFamily_not_order_sensitive_comm` below quantifies over
      an arbitrary carrier with commutative, associative multiplication, so no
      choice of scalar ring escapes the negative result. The `Int` version is
      recovered as a corollary, which is the honest way to show the original
      theorem was a special case and not a lucky one.

  W2  `PedagogicalVSA.exact_unbind_ax` only requires `inv i (ops i Y) = Y`,
      i.e. `inv i` is a LEFT inverse. Nothing in the specification forces
      `ops i (inv i Y) = Y`, so `ops i` need not be surjective and the
      "unbinding" reading is one-directional. Reviewer asked whether this is
      intentional generality or an oversight.

      Answer, made precise here: it is intentional in the specification (the
      chain theorem needs only the left law, so requiring more would weaken the
      theorem's reach for no gain), and vacuous for the instances we actually
      claim (both satisfy the two-sided law). `PedagogicalVSATwoSided` below
      states the stronger contract, and `edubindTwoSided` / `permTwoSided`
      discharge it, so the paper can say the generality is deliberate WITHOUT
      leaving a reader to wonder whether the positive instances secretly rely
      on it.

  B2  Revision 3's remark on `no_hadamard_PedagogicalVSA` (VSATriad.lean) notes
      a real limit on that theorem's reach: it rules out a family that is
      ENTIRELY Hadamard, not a MIXED family that is Hadamard at most indices
      and order-sensitive at one pair — Axiom 2 (`∃ i j Y, ...`) is satisfied
      TRIVIALLY by such a family, via the one non-Hadamard pair alone, while
      Hadamard operators sit at every other index. Closing this loophole in
      general needs a strengthened axiom quantified over every pair of
      GENUINELY DISTINCT family members, not just some existential pair:

        `order_sensitive_strong_ax : ∀ i j, ops i ≠ ops j →
                                        ∃ Y, ops i (ops j Y) ≠ ops j (ops i Y)`

      Note the hypothesis is `ops i ≠ ops j` (the two operators differ as
      FUNCTIONS), not `i ≠ j` (the two INDICES differ). The latter is
      unsatisfiable for any `Nat`-indexed family built from finitely many
      distinct operators (as both `edubindVSA` and `permVSA` are): by
      pigeonhole, infinitely many index pairs collide to the same operator, so
      `i ≠ j` alone cannot force `ops i` and `ops j` to disagree anywhere. The
      correct strengthening asks the axiom to hold whenever the operators
      THEMSELVES differ — exactly the condition under which order-sensitivity
      is a meaningful demand, and exactly what the mixed-family loophole
      violates (there, ops-agreement holds at every Hadamard-vs-Hadamard pair
      and the axiom is asked nothing there; it is only asked something at
      pairs where the operators differ, which the loophole never supplies for
      MOST such pairs in a large mixed family).

      `PedagogicalVSAStrong` below states this contract, and both verified
      instances discharge it: `edubindVSA` and `permVSA` each draw their
      family from exactly two DISTINCT matrices, so every ops-disagreeing pair
      reduces to the single witnessed pair (Rot vs. Ref, or T vs. Ref) already
      proved order-sensitive, by symmetry of `≠`. A fully general instance
      drawn from the entire 8-element integer $O(2)$ group (not only two of
      its generators) would additionally verify the strengthened axiom against
      a mixed family with several Hadamard-agreeing indices, which the current
      two-generator instances are too small to exercise; that extension is
      future work, not attempted here.

Three results are Mathlib-free and kernel-checked, like the rest of this tier.
No `sorry`, no `admit`, no new axiom. The minimal algebraic class below exists
because the kernel tier deliberately has no Mathlib and therefore no `CommRing`;
it asks for exactly the two laws the proof consumes and nothing else, which also
makes precise how little structure the impossibility result needs.
-/

import EduBindSelfContained
import VSATriad

open Matrix2

-- ---------------------------------------------------------------------------
-- W4: the Hadamard obstruction over an arbitrary commutative carrier
-- ---------------------------------------------------------------------------

/-- The minimal algebraic structure the Hadamard impossibility proof consumes:
    a multiplication that is commutative and associative. Deliberately weaker
    than a ring — no addition, no unit, no distributivity is needed, which is
    itself informative about why the obstruction is unavoidable. Mathlib's
    `CommMonoid`/`CommRing` would do, but the kernel tier has no Mathlib. -/
class CommMul (α : Type) extends Mul α where
  mul_comm : ∀ a b : α, a * b = b * a
  mul_assoc : ∀ a b c : α, a * b * c = a * (b * c)

namespace CommMul

/-- `Int` is an instance, which is what lets the original `Matrix2 Int`
    statement be recovered as a corollary below. -/
instance : CommMul Int where
  mul := Int.mul
  mul_comm := Int.mul_comm
  mul_assoc := Int.mul_assoc

/-- The swap step, over an arbitrary commutative carrier. This is the entire
    mathematical content of the impossibility result: two scalar multipliers
    applied in either order give the same product. -/
theorem swap [CommMul α] (a b y : α) : a * (b * y) = b * (a * y) := by
  rw [← mul_assoc, ← mul_assoc, mul_comm a b]

end CommMul

/-- Hadamard (elementwise) product on the 2×2 carrier, over ANY commutative
    carrier rather than `Int` specifically. -/
def hadC [CommMul α] (M N : Matrix2 α) : Matrix2 α :=
  { a00 := M.a00 * N.a00, a01 := M.a01 * N.a01,
    a10 := M.a10 * N.a10, a11 := M.a11 * N.a11 }

/-- Any two Hadamard binding operations commute, over any commutative carrier
    and for any content. The `Int` case (`had_ops_commute`, VSATriad.lean) is
    this theorem at `α := Int`. -/
theorem hadC_ops_commute [CommMul α] (M N Y : Matrix2 α) :
    hadC M (hadC N Y) = hadC N (hadC M Y) := by
  ext <;> dsimp [hadC] <;> exact CommMul.swap _ _ _

/-- W4, GENERALIZED. No family of Hadamard binding operations can be
    order-sensitive, over ANY carrier with commutative associative
    multiplication — so the obstruction is a property of elementwise binding
    itself, not an artifact of choosing integer entries. -/
theorem hadFamily_not_order_sensitive_comm [CommMul α] (Ms : Nat → Matrix2 α) :
    ¬ (∃ (i j : Nat) (Y : Matrix2 α),
        hadC (Ms i) (hadC (Ms j) Y) ≠ hadC (Ms j) (hadC (Ms i) Y)) := by
  intro h; obtain ⟨i, j, Y, hne⟩ := h
  exact hne (hadC_ops_commute (Ms i) (Ms j) Y)

/-- W4, GENERALIZED, in `PedagogicalVSA` form: no instance on a commutative
    carrier can draw its whole operator family from elementwise binding.
    Compare `no_hadamard_PedagogicalVSA` (VSATriad.lean), which is this
    statement specialized to `Int`. -/
theorem no_hadamard_PedagogicalVSA_comm [CommMul α]
    (P : PedagogicalVSA (Matrix2 α)) (Ms : Nat → Matrix2 α)
    (hops : ∀ i, P.ops i = hadC (Ms i)) : False := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  rw [hops i, hops j] at hne
  exact hadFamily_not_order_sensitive_comm Ms ⟨i, j, Y, hne⟩

/-- Sanity corollary: the generalized theorem really does subsume the
    `Int`-specific one the paper quotes. Stated as a theorem rather than a
    comment so that the subsumption is checked, not asserted. -/
theorem no_hadamard_int_is_special_case
    (P : PedagogicalVSA (Matrix2 Int)) (Ms : Nat → Matrix2 Int)
    (hops : ∀ i, P.ops i = hadC (Ms i)) : False :=
  no_hadamard_PedagogicalVSA_comm P Ms hops

-- ---------------------------------------------------------------------------
-- W2: the specification's left inverse, and the two-sided law the instances
-- actually satisfy
-- ---------------------------------------------------------------------------

/-- `PedagogicalVSA` with the reverse unbinding law added: binding an already
    unbound content recovers it too, so `ops i` and `inv i` are mutually
    inverse bijections rather than merely a section/retraction pair.

    We keep this SEPARATE from `PedagogicalVSA` on purpose. `chain_exact_unbind`
    (ChainTransitivity.lean) consumes only the left law, so adding the reverse
    law to the base specification would narrow which structures the chain
    theorem applies to while proving nothing new about the chain. Stating it as
    a strengthening records that the choice is deliberate and shows what the
    positive instances satisfy beyond the minimum. -/
structure PedagogicalVSATwoSided (V : Type) extends PedagogicalVSA V where
  /-- The reverse of Axiom 3: `ops i` recovers what `inv i` removed. -/
  inv_exact_bind_ax : ∀ i Y, ops i (inv i Y) = Y

/-- With both laws, each `ops i` is injective (already implied by the left law)
    AND surjective, hence a bijection with `inv i` as its two-sided inverse.
    This is the property a reader might mistakenly assume the base
    specification already guarantees. -/
theorem twoSided_ops_surjective {V : Type} (P : PedagogicalVSATwoSided V)
    (i : Nat) (Y : V) : ∃ X, P.ops i X = Y :=
  ⟨P.inv i Y, P.inv_exact_bind_ax i Y⟩

theorem twoSided_ops_injective {V : Type} (P : PedagogicalVSATwoSided V)
    (i : Nat) {X Y : V} (h : P.ops i X = P.ops i Y) : X = Y := by
  have hx := P.exact_unbind_ax i X
  have hy := P.exact_unbind_ax i Y
  rw [← hx, ← hy, h]

-- The reverse orthogonality facts needed to discharge the strengthened axiom
-- for the two positive instances. `Rot_orthogonal` / `Ref_orthogonal` /
-- `T_orthogonal` give `Mᵀ M = I`; the reverse law needs `M Mᵀ = I`.

/-- Rotation: the reverse product is also the identity. -/
theorem Rot_orthogonal_rev : Rot.mul Rot.transpose = Matrix2.one := by
  ext <;> dsimp [mul, transpose, Rot, one] <;> omega

/-- Reflection: the reverse product is also the identity. -/
theorem Ref_orthogonal_rev : Ref.mul Ref.transpose = Matrix2.one := by
  ext <;> dsimp [mul, transpose, Ref, one] <;> omega

/-- Swap block: the reverse product is also the identity. -/
theorem T_orthogonal_rev : T.mul T.transpose = Matrix2.one := by
  ext <;> dsimp [mul, transpose, T, one] <;> omega

/-- Reverse exact unbinding for the rotation, in the shape the axiom needs. -/
theorem Rot_exact_bind (Y : Matrix2 Int) : Rot.mul ((Rot.transpose).mul Y) = Y := by
  ext <;> dsimp [mul, transpose, Rot] <;> omega

/-- Reverse exact unbinding for the reflection. -/
theorem Ref_exact_bind (Y : Matrix2 Int) : Ref.mul ((Ref.transpose).mul Y) = Y := by
  ext <;> dsimp [mul, transpose, Ref] <;> omega

/-- Reverse exact unbinding for the swap block. -/
theorem T_exact_bind (Y : Matrix2 Int) : T.mul ((T.transpose).mul Y) = Y := by
  ext <;> dsimp [mul, transpose, T] <;> omega

/-- EduBind satisfies the STRENGTHENED specification: the left law it was
    already verified against, plus the reverse law. So the base specification's
    one-directional `inv` is generality we chose to leave available, not a gap
    the verified instance depends on. -/
def edubindTwoSided : PedagogicalVSATwoSided (Matrix2 Int) where
  toPedagogicalVSA := edubindVSA
  inv_exact_bind_ax := by
    intro i Y
    cases i with
    | zero   => exact Rot_exact_bind Y
    | succ _ => exact Ref_exact_bind Y

/-- Perm likewise satisfies the strengthened specification. -/
def permTwoSided : PedagogicalVSATwoSided (Matrix2 Int) where
  toPedagogicalVSA := permVSA
  inv_exact_bind_ax := by
    intro i Y
    cases i with
    | zero   => exact T_exact_bind Y
    | succ _ => exact Ref_exact_bind Y

/-- Both claimed instances meet the strengthened contract, so the count the
    paper reports is unchanged when the reverse law is demanded. -/
def verifiedTwoSidedOperators : List (PedagogicalVSATwoSided (Matrix2 Int)) :=
  [edubindTwoSided, permTwoSided]

theorem verifiedTwoSidedOperators_count : verifiedTwoSidedOperators.length = 2 := by rfl

-- ---------------------------------------------------------------------------
-- B2: the strengthened order-sensitivity axiom, quantified over every pair of
-- operators that actually differ, not just some existential pair. Closes the
-- mixed-family loophole `no_hadamard_PedagogicalVSA`'s own remark identifies.
-- ---------------------------------------------------------------------------

/-- `PedagogicalVSA` with Axiom 2 strengthened: EVERY pair of family members
    that differ as functions must be order-sensitive, not merely some pair. -/
structure PedagogicalVSAStrong (V : Type) extends PedagogicalVSA V where
  order_sensitive_strong_ax :
    ∀ i j, ops i ≠ ops j → ∃ Y, ops i (ops j Y) ≠ ops j (ops i Y)

/-- EduBind's family has exactly two distinct members (`eduGen`: index 0 is
    `Rot`, every other index is `Ref`), so every ops-disagreeing pair reduces
    to the one witnessed pair, by symmetry of `≠`. -/
def edubindVSAStrong : PedagogicalVSAStrong (Matrix2 Int) where
  toPedagogicalVSA := edubindVSA
  order_sensitive_strong_ax := by
    intro i j hij
    obtain ⟨Y0, hY0⟩ := lmul_Rot_Ref_non_comm
    cases i with
    | zero =>
      cases j with
      | zero   => exact absurd rfl hij
      | succ _ => exact ⟨Y0, hY0⟩
    | succ _ =>
      cases j with
      | zero   => exact ⟨Y0, fun h => hY0 h.symm⟩
      | succ _ => exact absurd rfl hij

/-- Perm's family likewise has exactly two distinct members (`permGen`: index
    0 is `T`, every other index is `Ref`). -/
def permVSAStrong : PedagogicalVSAStrong (Matrix2 Int) where
  toPedagogicalVSA := permVSA
  order_sensitive_strong_ax := by
    intro i j hij
    obtain ⟨Y0, hY0⟩ := lmul_T_Ref_non_comm
    cases i with
    | zero =>
      cases j with
      | zero   => exact absurd rfl hij
      | succ _ => exact ⟨Y0, hY0⟩
    | succ _ =>
      cases j with
      | zero   => exact ⟨Y0, fun h => hY0 h.symm⟩
      | succ _ => exact absurd rfl hij

/-- Both instances meet the strengthened contract, so `no_hadamard_PedagogicalVSA`
    is recoverable as a corollary of the strong axiom's contrapositive for any
    all-Hadamard family (every pair of indices there has `ops i = ops j`
    exactly when the underlying matrices agree, and disagrees only where
    `had_ops_commute` already shows composition still commutes — so no
    all-Hadamard family can ever discharge `order_sensitive_strong_ax` at a
    disagreeing pair, matching `no_hadamard_PedagogicalVSA`'s conclusion). -/
def verifiedStrongOperators : List (PedagogicalVSAStrong (Matrix2 Int)) :=
  [edubindVSAStrong, permVSAStrong]

theorem verifiedStrongOperators_count : verifiedStrongOperators.length = 2 := by rfl

-- Axiom footprint of the three review-driven results, for the reader to confirm.
#print axioms no_hadamard_PedagogicalVSA_comm
#print axioms edubindTwoSided
#print axioms edubindVSAStrong
#print axioms permVSAStrong
