/-
C1 — Algebraic transitivity of the pedagogical VSA chain composition.

The "Transitivity ≥ 95%" criterion conflates two genuinely different layers,
and they must be kept apart:

  1. ALGEBRAIC layer — does the binding operator itself compose along a
     prerequisite chain with ZERO accumulated error? This is a theorem, not a
     measurement. Proved below by induction for ALL chains
     (`chain_exact_unbind`): exact, by the Lean kernel, for every content.

  2. EMPIRICAL layer — can a LEARNED probe/embedding recover the transitive
     direction from finite data? This is statistics. The gap to 100% lives
     ENTIRELY here.

Revision 3 (2026-08-23). Two changes, both consequences of the Revision-3
specification in `EduBindSelfContained.lean`:

  * The roundtrip is now indexed by a LIST of operator indices rather than a
    single repetition count. Revision 2 applied the SAME binding operation n
    times, which does not model a prerequisite chain: a real chain A → B → C
    traverses a DIFFERENT relation at every hop. The list formulation covers
    heterogeneous chains, so the theorem now says what §3.3 of the paper claims
    it says. The homogeneous case is recovered as `List.replicate n i`.

  * The statement is generic in the carrier, so it applies to every
    `PedagogicalVSA` instance, not only matrix-valued ones.

KEY DESIGN NOTE — why this proof needs NO matrix-mult associativity and NO
Mathlib. Forming the chain as a single product and then reassociating would
require a general associativity lemma, a cubic polynomial identity unavailable
without Mathlib's `ring` (core `omega` is linear-only). Instead the roundtrip is
defined RECURSIVELY as function composition:

    chainRoundtrip []        Y = Y
    chainRoundtrip (i :: is) Y = inv i (chainRoundtrip is (ops i Y))

Function application associates definitionally, so the induction goes through
using ONLY the single exact-unbind axiom, already kernel-checked for every
`PedagogicalVSA`.

Honesty (what this does and does NOT prove):
  * DOES prove: the operator algebra composes transitively with zero error for
    every chain, of every length, over any sequence of relations from the
    family — a decidable guarantee, strictly stronger than any percentage.
  * Does NOT prove: that a learned probe reaches 100% (empirical layer).
  * Does NOT prove: that the ground-truth curriculum labels are perfectly
    transitive (a data-quality question no formal proof can settle).
  * Does NOT prove anything about predicting a prerequisite pair for which no
    intermediate path is known; that is a separate link-prediction problem.
-/

import VSATriad

open Matrix2

namespace PedagogicalVSA

/-- Bind a content `Y` forward through the chain of relations named by `is`,
    then unbind back through the reversed chain. Defined recursively as function
    composition (association order made explicit), so the proof below needs no
    associativity of the underlying multiplication. -/
def chainRoundtrip {V : Type} (P : PedagogicalVSA V) : List Nat → V → V
  | [],      Y => Y
  | i :: is, Y => P.inv i (chainRoundtrip P is (P.ops i Y))

/-- ALGEBRAIC TRANSITIVITY (layer 1). For ANY verified pedagogical operator
    family `P`, ANY chain of relations `is` — of any length, with a different
    relation at every hop — and ANY content `Y`, binding forward through the
    chain then unbinding back through it recovers `Y` EXACTLY. Proved by
    induction from the single exact-unbind axiom. This is a formal guarantee,
    not a statistic. -/
theorem chain_exact_unbind {V : Type} (P : PedagogicalVSA V) :
    ∀ (is : List Nat) (Y : V), chainRoundtrip P is Y = Y := by
  intro is
  induction is with
  | nil => intro Y; rfl
  | cons i is ih =>
    intro Y
    dsimp [chainRoundtrip]
    rw [ih (P.ops i Y)]
    exact P.exact_unbind_ax i Y

/-- The homogeneous chain of Revision 2 is the special case in which every hop
    uses the same relation. -/
theorem chain_exact_unbind_replicate {V : Type} (P : PedagogicalVSA V)
    (n i : Nat) (Y : V) :
    chainRoundtrip P (List.replicate n i) Y = Y :=
  chain_exact_unbind P (List.replicate n i) Y

/-- Chain length is unconstrained: the guarantee is quantified over every list,
    hence over chains of every length — including the 41-hop chains observed in
    the Junyi curriculum graph. -/
theorem chain_exact_unbind_at_length {V : Type} (P : PedagogicalVSA V)
    (n : Nat) (is : List Nat) (Y : V) (_h : is.length = n) :
    chainRoundtrip P is Y = Y :=
  chain_exact_unbind P is Y

end PedagogicalVSA

-- ---------------------------------------------------------------------------
-- Concrete instantiations for the two verified operator families.
-- ---------------------------------------------------------------------------

/-- EduBind's chain composition unbinds exactly, for any chain of relations. -/
theorem edubind_chain_exact_unbind (is : List Nat) (Y : Matrix2 Int) :
    PedagogicalVSA.chainRoundtrip edubindVSA is Y = Y :=
  PedagogicalVSA.chain_exact_unbind edubindVSA is Y

/-- Perm's chain composition unbinds exactly, for any chain of relations. -/
theorem perm_chain_exact_unbind (is : List Nat) (Y : Matrix2 Int) :
    PedagogicalVSA.chainRoundtrip permVSA is Y = Y :=
  PedagogicalVSA.chain_exact_unbind permVSA is Y

/-- A genuinely heterogeneous 3-hop chain (rotation, reflection, rotation) —
    the shape of a real prerequisite path A → B → C → D, which the Revision-2
    single-operator formulation could not express. -/
theorem edubind_heterogeneous_chain (Y : Matrix2 Int) :
    PedagogicalVSA.chainRoundtrip edubindVSA [0, 1, 0] Y = Y :=
  PedagogicalVSA.chain_exact_unbind edubindVSA [0, 1, 0] Y

-- ---------------------------------------------------------------------------
-- Componentwise lifting to the B-block diagonal operator.
-- ---------------------------------------------------------------------------

/-- Block-diagonal chain roundtrip: run the EduBind chain roundtrip
    independently in each of the B blocks. -/
def blockChainRoundtrip (B : Nat) (is : List Nat) (Ys : Fin B → Matrix2 Int) :
    Fin B → Matrix2 Int :=
  fun i => PedagogicalVSA.chainRoundtrip edubindVSA is (Ys i)

/-- The B-block diagonal chain composition also unbinds exactly, for any number
    of blocks and any chain of relations. -/
theorem block_chain_exact_unbind (B : Nat) (is : List Nat)
    (Ys : Fin B → Matrix2 Int) :
    blockChainRoundtrip B is Ys = Ys := by
  funext i
  dsimp [blockChainRoundtrip]
  exact PedagogicalVSA.chain_exact_unbind edubindVSA is (Ys i)
