/-
C1 — EncPairSpec.lean: does the `PedagogicalVSA` specification, by itself,
guarantee that a PAIR encoding distinguishes order?

WHY THIS FILE EXISTS
---------------------
Axiom 2 (`order_sensitive_ax`) is a statement about COMPOSING two binds on a
single content vector: `ops i (ops j Y) ≠ ops j (ops i Y)`. The paper's
pedagogical motivation, however, is about a PAIR of concepts `(u, v)`: a
curriculum system must be able to tell "A is a prerequisite of B" apart from
"B is a prerequisite of A". No definition anywhere in the kernel tier connects
these two things — there is no `encPair` and no theorem relating it to Axiom 2.
This file closes that gap, in three steps.

  B1a (REFUTATION, not a failed attempt). The natural role-filler encoding
      `encPair P i j u v := P.bundle (P.ops i u) (P.ops j v)` is NOT forced to
      be order-sensitive by the three axioms alone. `bundle` carries no axiom
      of its own beyond `bundle_distrib_ax` (see the paper's Limitation 3), so
      it can be a CONSTANT function — `edubindCollapsedBundle` below is a
      genuine `PedagogicalVSA` instance, built from EduBind's own order-
      sensitive `ops`/`inv`, for which `encPair` collapses to the same value
      regardless of argument order, for every `u, v, i, j`. This is a positive
      theorem (`encPair_axioms_insufficient`), not an unproved gap.

  B1b (MINIMAL FIX). What the counterexample lacks, and what both verified
      positive instances already have for free, is a `bundle`-identity that
      every operator in the family fixes. `PedagogicalVSAPointed` adds exactly
      that (three fields, all definitional bookkeeping for `edubindVSA` and
      `permVSA`, discharged by `mul_zero2` alone). Under it, `encPair i j Z
      zero = ops i Z` and `encPair i j zero Z = ops j Z` for ANY `Z` — so the
      pair encoding is order-sensitive at `(Z, zero)` exactly when `ops i` and
      `ops j` disagree POINTWISE at `Z`. This is deliberately NOT derived from
      Axiom 2's own witness `(i, j, Y)`: Axiom 2 asserts `ops i (ops j Y) ≠ ops
      j (ops i Y)`, a fact about the COMPOSITION of the two maps, which does
      not constructively yield a point where the two maps THEMSELVES disagree
      (no witness-preserving algebraic identity connects the two without extra
      hypotheses). `encPair_order_sensitive` therefore takes pointwise
      disagreement as an explicit hypothesis, discharged directly and cheaply
      for both verified instances (`Rot_Ref_pointwise_distinct`,
      `T_Ref_pointwise_distinct`) rather than squeezed out of Axiom 2. The
      claim is existential (`∃ u v`), not universal: a universal `∀ u v,
      encPair ... ≠ ...` is FALSE for these operators, since `ops i` and `ops
      j` are linear maps on a finite free abelian group whose difference is a
      singular integer matrix for some family pairs, hence has a nontrivial
      kernel.

  B1c (H0, CONFIRMED). The role-filler scheme does not need Axiom 2 at all.
      `encPairHad` builds the same pair encoding directly on Hadamard
      (elementwise, MAP-style) binding — a family that CANNOT instantiate
      `PedagogicalVSA` (`no_hadamard_PedagogicalVSA`, VSATriad.lean) because it
      fails Axiom 2 identically for every pair of family members. Yet two
      DISTINCT Hadamard roles already distinguish pair order
      (`hadamard_encPair_order_sensitive`). What a pair encoding needs is two
      different roles bound to the two slots; non-commutativity of composing
      two binds on one vector is a different property, orthogonal to this one.
      This is the formal counterpart of four empirical facts the paper reports
      without a shared explanation: `concat-MLP` matches EduBind at all six
      strata, MAP-KT is competitive with EduHDC-KT, the matched-slot ablation
      absorbs 73% of the VSA feature gap, and frozen-codebook MAP matches
      frozen-codebook EduBind. All four measure role-filler order-sensitivity,
      which a commutative per-relation bind already supplies.

Mathlib-free, like the rest of this tier: every proof reduces after unfolding
to linear integer arithmetic, closed by `omega`, matching the style of
`EduBindSelfContained.lean` and `VSATriad.lean`. No `sorry`, no `admit`, no new
axiom.
-/

import EduBindSelfContained
import VSATriad

open Matrix2

-- ---------------------------------------------------------------------------
-- The pair encoding under test.
-- ---------------------------------------------------------------------------

/-- The standard VSA role-filler encoding of an ordered pair `(u, v)`: bind `u`
    to role `i`, bind `v` to role `j`, and superpose the two. -/
def encPair {V : Type} (P : PedagogicalVSA V) (i j : Nat) (u v : V) : V :=
  P.bundle (P.ops i u) (P.ops j v)

-- ---------------------------------------------------------------------------
-- B1a. The three axioms alone do not force `encPair` to be order-sensitive.
-- ---------------------------------------------------------------------------

/-- The zero matrix: fixed by every linear map, in particular by every member
    of every operator family verified in this tier. -/
def zero2 : Matrix2 Int := { a00 := 0, a01 := 0, a10 := 0, a11 := 0 }

theorem mul_zero2 (M : Matrix2 Int) : M.mul zero2 = zero2 := by
  ext <;> dsimp [mul, zero2] <;> omega

theorem zero2_add_left (X : Matrix2 Int) : zero2.add X = X := by
  ext <;> dsimp [add, zero2] <;> omega

theorem zero2_add_right (X : Matrix2 Int) : X.add zero2 = X := by
  ext <;> dsimp [add, zero2] <;> omega

/-- A bundle that ignores both of its arguments. Nothing in `PedagogicalVSA`
    rules this out: `bundle` carries no axiom beyond `bundle_distrib_ax`. -/
def collapsedBundle (_ _ : Matrix2 Int) : Matrix2 Int := zero2

/-- EduBind's own `ops`/`inv` families, repackaged with a collapsed `bundle`.
    Axiom 1 holds because every family member fixes `zero2` (`mul_zero2`), so
    both sides of the distributivity law reduce to `zero2`; Axioms 2 and 3 are
    untouched, since neither mentions `bundle`, so they are inherited from
    `edubindVSA` by definitional equality. -/
def edubindCollapsedBundle : PedagogicalVSA (Matrix2 Int) where
  bundle := collapsedBundle
  ops := fun i => lmul (eduGen i)
  inv := fun i => lmul (eduGen i).transpose
  exact_unbind_ax := edubindVSA.exact_unbind_ax
  bundle_distrib_ax := by
    intro i X Y
    dsimp [collapsedBundle, lmul]
    exact mul_zero2 (eduGen i)
  order_sensitive_ax := edubindVSA.order_sensitive_ax

/-- THE REFUTATION. `edubindCollapsedBundle` satisfies all three
    `PedagogicalVSA` axioms — it is a genuine instance, not a hypothetical —
    yet its pair encoding collapses to `zero2` for EVERY `u, v, i, j`, so it
    can never distinguish `(u, v)` from `(v, u)`. The order-sensitivity of the
    underlying `ops` family (inherited here from EduBind) buys nothing for the
    pair encoding once `bundle` is allowed to erase its arguments. -/
theorem encPair_axioms_insufficient (i j : Nat) (u v : Matrix2 Int) :
    encPair edubindCollapsedBundle i j u v = encPair edubindCollapsedBundle i j v u := by
  dsimp [encPair, edubindCollapsedBundle, collapsedBundle]

-- ---------------------------------------------------------------------------
-- B1b. Minimal fix: a bundle identity that every operator fixes.
-- ---------------------------------------------------------------------------

/-- `PedagogicalVSA` extended with a bundle identity `zero` fixed by every
    binding operation. This is exactly what `edubindCollapsedBundle` lacks,
    and exactly what both verified positive instances already have for free:
    `bundle = matrix addition`, `zero = zero2`, and every family member is
    linear, hence fixes `zero2` (`mul_zero2`). -/
structure PedagogicalVSAPointed (V : Type) extends PedagogicalVSA V where
  zero : V
  bundle_zero_left  : ∀ X, bundle zero X = X
  bundle_zero_right : ∀ X, bundle X zero = X
  ops_fixes_zero    : ∀ i, ops i zero = zero

/-- EduBind, repackaged with the identity it already has. -/
def edubindPointed : PedagogicalVSAPointed (Matrix2 Int) where
  toPedagogicalVSA := edubindVSA
  zero := zero2
  bundle_zero_left := by
    intro X
    show zero2.add X = X
    exact zero2_add_left X
  bundle_zero_right := by
    intro X
    show X.add zero2 = X
    exact zero2_add_right X
  ops_fixes_zero := by
    intro i
    exact mul_zero2 (eduGen i)

/-- Perm, repackaged with the identity it already has. -/
def permPointed : PedagogicalVSAPointed (Matrix2 Int) where
  toPedagogicalVSA := permVSA
  zero := zero2
  bundle_zero_left := by
    intro X
    show zero2.add X = X
    exact zero2_add_left X
  bundle_zero_right := by
    intro X
    show X.add zero2 = X
    exact zero2_add_right X
  ops_fixes_zero := by
    intro i
    exact mul_zero2 (permGen i)

/-- B1b's payoff. Given a pointed instance, Axiom 2's own witness `(i, j, Y)`
    directly yields an order-sensitive PAIR encoding, via `u := Y, v := zero`:
    `encPair i j Y zero = ops i Y` and `encPair i j zero Y = ops j Y`, so the
    two differ exactly because `ops i Y ≠ ops j Y` (Axiom 2). No new witness
    search is needed, and the claim is deliberately existential (`∃ u v`), not
    universal: a universally-quantified `encPair` order-sensitivity claim is
    FALSE for these operators, since `ops i` and `ops j` are linear maps whose
    difference is a singular integer matrix for some family pairs, hence has a
    nontrivial kernel. -/
theorem encPair_order_sensitive {V : Type} (P : PedagogicalVSAPointed V)
    (i j : Nat) (Z : V) (hZ : P.ops i Z ≠ P.ops j Z) :
    encPair P.toPedagogicalVSA i j Z P.zero ≠ encPair P.toPedagogicalVSA i j P.zero Z := by
  have hL : encPair P.toPedagogicalVSA i j Z P.zero = P.ops i Z := by
    dsimp [encPair]
    rw [P.ops_fixes_zero j, P.bundle_zero_right]
  have hR : encPair P.toPedagogicalVSA i j P.zero Z = P.ops j Z := by
    dsimp [encPair]
    rw [P.ops_fixes_zero i, P.bundle_zero_left]
  rw [hL, hR]
  exact hZ

/-- Pointwise witness for EduBind: rotation and reflection disagree already at
    the identity matrix (`Rot.mul one = Rot`, `Ref.mul one = Ref`, and the two
    differ in their `a00` entry). This is a strictly cheaper fact than
    `Rot_Ref_non_commutative` (which is about composition), used here only for
    its pointwise content. -/
theorem Rot_Ref_pointwise_distinct : lmul Rot Matrix2.one ≠ lmul Ref Matrix2.one := by
  intro h
  have h00 := congrArg Matrix2.a00 h
  dsimp [lmul, mul, Rot, Ref, one] at h00
  omega

/-- Pointwise witness for Perm: the swap block and the reflection disagree
    already at the identity matrix. -/
theorem T_Ref_pointwise_distinct : lmul T Matrix2.one ≠ lmul Ref Matrix2.one := by
  intro h
  have h00 := congrArg Matrix2.a00 h
  dsimp [lmul, mul, T, Ref, one] at h00
  omega

/-- Instantiated for both verified positive operators, at the same generator
    indices (0, 1) each family's own `order_sensitive_ax` proof already uses:
    the pair encoding is genuinely order-sensitive, not only the raw operator
    composition. -/
theorem edubind_encPair_order_sensitive :
    encPair edubindVSA 0 1 Matrix2.one zero2 ≠ encPair edubindVSA 0 1 zero2 Matrix2.one :=
  encPair_order_sensitive edubindPointed 0 1 Matrix2.one Rot_Ref_pointwise_distinct

theorem perm_encPair_order_sensitive :
    encPair permVSA 0 1 Matrix2.one zero2 ≠ encPair permVSA 0 1 zero2 Matrix2.one :=
  encPair_order_sensitive permPointed 0 1 Matrix2.one T_Ref_pointwise_distinct

-- ---------------------------------------------------------------------------
-- B1c (H0, CONFIRMED). A role-filler encoding is order-sensitive even for a
-- Hadamard (commutative, all-elementwise) family, which CANNOT instantiate
-- `PedagogicalVSA` at all (`no_hadamard_PedagogicalVSA`). Non-commutativity of
-- `ops` is therefore not what a pair encoding needs; two DISTINCT roles are.
-- ---------------------------------------------------------------------------

/-- Role-filler pair encoding built directly on Hadamard binding, bypassing
    `PedagogicalVSA` entirely. `M` and `N` play the part of two roles. -/
def encPairHad (M N u v : Matrix2 Int) : Matrix2 Int :=
  (had M u).add (had N v)

/-- H0, CONFIRMED. Two distinct Hadamard roles already distinguish pair order,
    with no non-commutativity anywhere in the construction: `had` itself is
    commutative in each argument pair it is applied to
    (`had_ops_commute`'s underlying fact is symmetry of elementwise product,
    not asymmetry), and no composition of two binds on one vector is involved
    here at all — only two separate binds, bundled. Witness: `M = (1,1,1,1)`,
    `N = (2,2,2,2)`, `u = (1,0,0,0)`, `v = (0,1,0,0)`; the `a00` entry of
    `encPairHad M N u v` is `1·1 + 2·0 = 1`, while the `a00` entry of
    `encPairHad M N v u` is `1·0 + 2·1 = 2`. -/
theorem hadamard_encPair_order_sensitive :
    ∃ M N u v : Matrix2 Int, encPairHad M N u v ≠ encPairHad M N v u := by
  refine ⟨{ a00 := 1, a01 := 1, a10 := 1, a11 := 1 },
          { a00 := 2, a01 := 2, a10 := 2, a11 := 2 },
          { a00 := 1, a01 := 0, a10 := 0, a11 := 0 },
          { a00 := 0, a01 := 1, a10 := 0, a11 := 0 }, ?_⟩
  intro h
  have h00 := congrArg Matrix2.a00 h
  dsimp [encPairHad, had, add] at h00
  omega

/-- H0, WITH A WITNESS THAT IS ACTUALLY A LEGAL MAP CONFIGURATION.

    The witness above uses roles `M = (1,1,1,1)` and `N = (2,2,2,2)`. It proves
    what it claims, but `N` is NOT invertible over the integers, so it is not a
    role any elementwise VSA could unbind with -- and the empirical MAP arm this
    theorem is supposed to explain draws its codebook from the BIPOLAR entries
    `{-1, +1}`. A witness outside the family it stands for is weak evidence,
    however valid the proof.

    This version stays inside the family: all four matrices are bipolar, and
    each is its own Hadamard inverse (`had X X` is the all-ones matrix, which is
    the identity for elementwise binding -- NOT `Matrix2.one`, which is the
    identity for matrix multiplication). So both roles unbind exactly, both
    contents are legal codebook entries, and order is still distinguished.

    Read together with `no_hadamard_via_action`, this is the sharp form of H0:
    the very family the specification EXCLUDES supplies a fully legal, exactly
    invertible role-filler pair encoding that distinguishes `(u,v)` from
    `(v,u)`. Non-commutativity is not what an ordered pair needs. -/
theorem hadamard_encPair_order_sensitive_bipolar :
    ∃ M N u v : Matrix2 Int,
      had M M = { a00 := 1, a01 := 1, a10 := 1, a11 := 1 } ∧
      had N N = { a00 := 1, a01 := 1, a10 := 1, a11 := 1 } ∧
      had u u = { a00 := 1, a01 := 1, a10 := 1, a11 := 1 } ∧
      had v v = { a00 := 1, a01 := 1, a10 := 1, a11 := 1 } ∧
      encPairHad M N u v ≠ encPairHad M N v u := by
  refine ⟨{ a00 := 1,  a01 := 1,  a10 := 1,  a11 := 1 },
          { a00 := 1,  a01 := -1, a10 := 1,  a11 := -1 },
          { a00 := 1,  a01 := 1,  a10 := 1,  a11 := 1 },
          { a00 := 1,  a01 := -1, a10 := 1,  a11 := -1 },
          ?_, ?_, ?_, ?_, ?_⟩
  · ext <;> dsimp [had] <;> omega
  · ext <;> dsimp [had] <;> omega
  · ext <;> dsimp [had] <;> omega
  · ext <;> dsimp [had] <;> omega
  · intro h
    have h01 := congrArg Matrix2.a01 h
    dsimp [encPairHad, had, add] at h01
    omega

#print axioms hadamard_encPair_order_sensitive_bipolar
