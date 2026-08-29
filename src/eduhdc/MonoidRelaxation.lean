/-
C1 -> C3 — what survives when invertibility is dropped.

WHY THIS FILE EXISTS
--------------------
`PedagogicalVSA` requires `inv` together with `exact_unbind_ax`, i.e. it assumes
every relation is invertible. Two independent lines of work say that assumption
is wrong for real knowledge graphs:

  * Knowledgebra (Yang et al., MAKE 2022) proves invertibility must break, by
    contradiction: with `ra = isMotherOf`, `rb = isBrotherOf` and `ra . rb = ra`,
    an inverse for `ra` forces `rb = e`. Their conclusion is that relations form
    a SEMIGROUP, not a group; non-invertible elements are exactly what represents
    N-to-1 relations.
  * KrausKGE (Chaki, arXiv:2605.10317) reaches the same place by rank: a single
    isometric operator cannot represent a relation whose empirical relation
    matrix has rank exceeding the embedding dimension.

This matters here and not only in the abstract. The Junyi prerequisite graph is
full of N-to-1 relations -- many sub-skills feeding one parent skill -- so when
C3 uses this algebra on that graph, it is using a specification its own data
violates. Either C1 says what survives without invertibility, or C3 has no
foundation for the non-invertible case.

WHAT THIS FILE FINDS
--------------------
The weakening is real, and it is a separation of exactly the same shape as the
one the paper already reports for pairs versus chains:

  SURVIVES (needs only distributivity and composition)
    `chainApply_append`          — functoriality along a chain
    `chainApply_bundle_distrib`  — Axiom 1 lifts to a whole chain
    `no_abelian_action_...`      — the impossibility theorem is untouched, so
                                   MAP, HRR and phase composition stay excluded
                                   from the WEAKER specification too

  DIES (and provably, not just "we could not prove it")
    exact recovery — `nonInjMonoid` is a genuine `PedagogicalMonoid` for which NO
    left-inverse family exists at all, so no `PedagogicalVSA` has its operators.

So composition survives the loss of invertibility and exact recovery does not.
The relation that breaks it is the simplest N-to-1 map there is: a rank-one
projection, which is what "several sub-skills, one parent skill" looks like once
written as a matrix.

TIER
----
Mathlib-free, kernel-checked. Same axiom boundary as the rest of this project.
-/
import EduBindSelfContained
import VSATriad
import GroupActionSpec

open Matrix2

-- ---------------------------------------------------------------------------
-- The weakened specification
-- ---------------------------------------------------------------------------

/-- `PedagogicalVSA` with invertibility dropped: no `inv`, no
    `exact_unbind_ax`. Bundling distributivity and intra-family order
    sensitivity are kept unchanged, so the two structures differ in exactly one
    assumption and nothing else. -/
structure PedagogicalMonoid (V : Type) where
  /-- Additive bundling (superposition). -/
  bundle : V → V → V
  /-- The family of binding operations. -/
  ops : Nat → (V → V)
  /-- Axiom 1, unchanged. -/
  bundle_distrib_ax : ∀ i X Y, ops i (bundle X Y) = bundle (ops i X) (ops i Y)
  /-- Axiom 2, unchanged. -/
  order_sensitive_ax : ∃ i j Y, ops i (ops j Y) ≠ ops j (ops i Y)

namespace PedagogicalMonoid

/-- Every `PedagogicalVSA` forgets to a `PedagogicalMonoid`, which is what makes
    this a weakening of the specification rather than a different one. -/
def ofVSA {V : Type} (P : PedagogicalVSA V) : PedagogicalMonoid V where
  bundle := P.bundle
  ops := P.ops
  bundle_distrib_ax := P.bundle_distrib_ax
  order_sensitive_ax := P.order_sensitive_ax

-- ---------------------------------------------------------------------------
-- What survives
-- ---------------------------------------------------------------------------

/-- Forward application along a chain of relations. Note what is absent: no
    `inv` appears, so this is definable in the weakened setting whereas
    `chainRoundtrip` is not. -/
def chainApply {V : Type} (P : PedagogicalMonoid V) : List Nat → V → V
  | [],      Y => Y
  | i :: is, Y => P.ops i (chainApply P is Y)

/-- Functoriality survives: applying a concatenated chain is applying the two
    pieces in turn. This needs nothing but the definition -- neither axiom, and
    certainly not invertibility. -/
theorem chainApply_append {V : Type} (P : PedagogicalMonoid V) :
    ∀ (is js : List Nat) (Y : V),
      chainApply P (is ++ js) Y = chainApply P is (chainApply P js Y) := by
  intro is
  induction is with
  | nil => intro js Y; rfl
  | cons i is ih =>
    intro js Y
    dsimp [chainApply]
    rw [ih js Y]

/-- Axiom 1 lifts to a whole chain without invertibility: superposition
    structure is preserved along arbitrarily long relation paths. This is the
    positive payload of the weakening, and it is what a non-invertible
    prerequisite relation can still be relied on to do. -/
theorem chainApply_bundle_distrib {V : Type} (P : PedagogicalMonoid V) :
    ∀ (is : List Nat) (X Y : V),
      chainApply P is (P.bundle X Y)
        = P.bundle (chainApply P is X) (chainApply P is Y) := by
  intro is
  induction is with
  | nil => intro X Y; rfl
  | cons i is ih =>
    intro X Y
    dsimp [chainApply]
    rw [ih X Y]
    exact P.bundle_distrib_ax i _ _

/-- The impossibility theorem is insensitive to the weakening: its proof reads
    only `order_sensitive_ax`. So MAP, HRR and phase-composition operators are
    excluded from the WEAKER specification too, which is what C3 needs -- giving
    up invertibility does not buy back a commutative operator. -/
theorem no_abelian_action_PedagogicalMonoid
    {G V : Type} (P : PedagogicalMonoid V) (A : PedagogicalVSA.ActionFamily G V)
    (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g)
    (hops : ∀ i, P.ops i = A.act (gen i)) : False := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  rw [hops i, hops j] at hne
  exact hne (A.commutes_of_abelian hab (gen i) (gen j) Y)

end PedagogicalMonoid

-- ---------------------------------------------------------------------------
-- What dies: a genuine monoid instance with no inverse anywhere
-- ---------------------------------------------------------------------------

/-- A rank-one projection. As a relation this is the simplest N-to-1 map there
    is: it collapses a whole line of contents onto one image, which is what
    "several sub-skills feed one parent skill" becomes once written down. -/
def Proj : Matrix2 Int := { a00 := 1, a01 := 0, a10 := 0, a11 := 0 }

/-- The projection distributes over bundling, so Axiom 1 holds for it. -/
theorem Proj_bundle_distrib (A Z : Matrix2 Int) :
    Proj.mul (A.add Z) = (Proj.mul A).add (Proj.mul Z) := by
  ext <;> dsimp [mul, add, Proj] <;> omega

/-- The projection does not commute with the rotation, so Axiom 2 holds too. -/
theorem Rot_Proj_non_commutative : Rot.mul Proj ≠ Proj.mul Rot := by
  intro h
  have h01 := congrArg Matrix2.a01 h
  dsimp [mul, Rot, Proj] at h01
  omega

/-- The generator family: index 0 is the rotation, every other index is the
    non-invertible projection. -/
def monoGen : Nat → Matrix2 Int
  | 0 => Rot
  | _ => Proj

/-- A genuine `PedagogicalMonoid`: both surviving axioms are discharged. -/
def nonInjMonoid : PedagogicalMonoid (Matrix2 Int) where
  bundle := fun X Y => X.add Y
  ops := fun i => lmul (monoGen i)
  bundle_distrib_ax := by
    intro i X Y
    cases i with
    | zero   => exact Rot_bundle_distrib X Y
    | succ _ => exact Proj_bundle_distrib X Y
  order_sensitive_ax := by
    refine ⟨0, 1, Matrix2.one, by decide⟩

/-- The projection is not injective: two different contents with one image. -/
theorem Proj_not_injective :
    ∃ Y₁ Y₂ : Matrix2 Int, Y₁ ≠ Y₂ ∧ Proj.mul Y₁ = Proj.mul Y₂ := by
  refine ⟨{ a00 := 0, a01 := 0, a10 := 1, a11 := 0 },
          { a00 := 0, a01 := 0, a10 := 0, a11 := 0 }, ?_, ?_⟩
  · intro h
    have h10 := congrArg Matrix2.a10 h
    dsimp at h10
    omega
  · ext <;> dsimp [mul, Proj] <;> omega

/-- **THE SEPARATION.** `nonInjMonoid` admits no left-inverse family whatsoever:
    Axiom 3 is not merely unproved for it, it is unsatisfiable. -/
theorem nonInjMonoid_no_inverse :
    ¬ ∃ inv : Nat → (Matrix2 Int → Matrix2 Int),
        ∀ i Y, inv i (nonInjMonoid.ops i Y) = Y := by
  rintro ⟨inv, hinv⟩
  obtain ⟨Y₁, Y₂, hne, heq⟩ := Proj_not_injective
  refine hne ?_
  have h1 : inv 1 (Proj.mul Y₁) = Y₁ := hinv 1 Y₁
  have h2 : inv 1 (Proj.mul Y₂) = Y₂ := hinv 1 Y₂
  rw [heq] at h1
  exact h1.symm.trans h2

/-- Stated the way the paper reads it: no `PedagogicalVSA` has these operators.
    Composition survives the loss of invertibility (`chainApply_append`,
    `chainApply_bundle_distrib`); exact recovery does not. -/
theorem nonInjMonoid_not_from_VSA :
    ¬ ∃ P : PedagogicalVSA (Matrix2 Int), ∀ i, P.ops i = nonInjMonoid.ops i := by
  rintro ⟨P, hops⟩
  refine nonInjMonoid_no_inverse ⟨P.inv, ?_⟩
  intro i Y
  rw [← hops i]
  exact P.exact_unbind_ax i Y

-- ---------------------------------------------------------------------------
-- Axiom footprint
-- ---------------------------------------------------------------------------

#print axioms PedagogicalMonoid.chainApply_append
#print axioms PedagogicalMonoid.chainApply_bundle_distrib
#print axioms PedagogicalMonoid.no_abelian_action_PedagogicalMonoid
#print axioms nonInjMonoid
#print axioms nonInjMonoid_no_inverse
#print axioms nonInjMonoid_not_from_VSA
