/-
C1 — Algebraic transitivity of the pedagogical VSA chain composition.

Motivation (docs/temp_overview.md, FW3): the "Transitivity ≥ 95%" criterion
conflates two genuinely different layers, and the thesis must keep them apart:

  1. ALGEBRAIC layer — does the binding operator itself compose along a
     prerequisite chain with ZERO accumulated error? This is a theorem, not a
     measurement. Proved below by induction for ALL chain lengths n
     (`chain_exact_unbind`): 100% by the Lean kernel, for every content.

  2. EMPIRICAL layer — can a LEARNED probe/embedding recover the transitive
     direction from finite data? This is statistics (FW3c: 97.6% transductive /
     79.3% inductive / 62–78% pure). The gap to 100% lives ENTIRELY here.

KEY DESIGN NOTE — why this proof needs NO matrix-mult associativity and NO
Mathlib. Forming the n-chain as a single power R^n and then reassociating
(Rᵀ)ⁿ·(Rⁿ·Y) would require a general associativity lemma, which is a cubic
polynomial identity unavailable without Mathlib's `ring` (core `omega` is
linear-only). Instead we define the bind-then-unbind roundtrip RECURSIVELY as
function composition:

    chainRoundtrip 0     Y = Y
    chainRoundtrip (n+1) Y = unbind₁ (chainRoundtrip n (bind₁ Y))

Function application associates definitionally, so the induction goes through
using ONLY the single exact-unbind axiom `unbind₁ (bind₁ Y) = Y`, already
kernel-checked for every `PedagogicalVSACore`. The result: for ANY chain length n
and ANY content Y, binding through the n-chain then unbinding through the
reversed n-chain recovers Y EXACTLY — no cross-talk, no accumulated error.

Honesty (what this does and does NOT prove, per temp_overview):
  * DOES prove: the operator algebra composes transitively with zero error for
    all n — a decidable, 100%-by-definition guarantee (like `lake build` = 0),
    strictly stronger than any percentage.
  * Does NOT prove: the learned FW3c probe reaches 100% (empirical layer).
  * Does NOT prove: the ground-truth curriculum labels are perfectly transitive
    (a data-quality question no formal proof can settle).
-/

import VSATriad

open Matrix2

namespace PedagogicalVSACore

/-- Bind a content `Y` through an `n`-step chain of the primary bind `bind₁`,
    then unbind through the reversed `n`-step chain `unbind₁`. Defined
    recursively as function composition (association order made explicit), so
    the proof below needs no associativity of the underlying multiplication. -/
def chainRoundtrip (V : PedagogicalVSACore) : Nat → Matrix2 Int → Matrix2 Int
  | 0, Y => Y
  | n + 1, Y => V.unbind₁ (chainRoundtrip V n (V.bind₁ Y))

/-- ALGEBRAIC TRANSITIVITY (layer 1). For ANY verified pedagogical operator `V`,
    ANY chain length `n`, and ANY content `Y`, binding through the `n`-chain
    then unbinding through the reversed `n`-chain recovers `Y` EXACTLY. Proved
    by induction from the single exact-unbind axiom. This is 100% for all `n` —
    a formal guarantee, not a statistic. -/
theorem chain_exact_unbind (V : PedagogicalVSACore) (n : Nat) :
    ∀ (Y : Matrix2 Int), chainRoundtrip V n Y = Y := by
  induction n with
  | zero => intro Y; rfl
  | succ n ih =>
    intro Y
    dsimp [chainRoundtrip]
    rw [ih (V.bind₁ Y)]
    exact V.exact_unbind_ax Y

end PedagogicalVSACore

-- ---------------------------------------------------------------------------
-- Concrete instantiations for the two verified operators.
-- ---------------------------------------------------------------------------

/-- EduBind's chain composition unbinds exactly, for any chain length `n`. -/
theorem edubind_chain_exact_unbind (n : Nat) (Y : Matrix2 Int) :
    PedagogicalVSACore.chainRoundtrip edubindPedagogicalVSA n Y = Y :=
  PedagogicalVSACore.chain_exact_unbind edubindPedagogicalVSA n Y

/-- Perm's chain composition unbinds exactly, for any chain length `n`. -/
theorem perm_chain_exact_unbind (n : Nat) (Y : Matrix2 Int) :
    PedagogicalVSACore.chainRoundtrip permPedagogicalVSA n Y = Y :=
  PedagogicalVSACore.chain_exact_unbind permPedagogicalVSA n Y

-- ---------------------------------------------------------------------------
-- Componentwise lifting to the B-block diagonal operator (mirrors
-- EduBindSelfContained.block_exact_unbind, now across an n-chain).
-- ---------------------------------------------------------------------------

/-- Block-diagonal n-chain roundtrip: run the EduBind chain roundtrip
    independently in each of the B blocks. -/
def blockChainRoundtrip (B : Nat) (n : Nat) (Ys : Fin B → Matrix2 Int) :
    Fin B → Matrix2 Int :=
  fun i => PedagogicalVSACore.chainRoundtrip edubindPedagogicalVSA n (Ys i)

/-- The B-block diagonal chain composition also unbinds exactly, for any number
    of blocks B and any chain length n. -/
theorem block_chain_exact_unbind (B : Nat) (n : Nat) (Ys : Fin B → Matrix2 Int) :
    blockChainRoundtrip B n Ys = Ys := by
  funext i
  dsimp [blockChainRoundtrip]
  exact PedagogicalVSACore.chain_exact_unbind edubindPedagogicalVSA n (Ys i)
