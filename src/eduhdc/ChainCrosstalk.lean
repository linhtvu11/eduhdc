/-
C1 — ChainCrosstalk.lean: what does non-commutative binding buy for a CHAIN
that a commutative role-filler encoding does not?

WHY THIS FILE EXISTS
---------------------
EncPairSpec.lean settles what Axiom 2 buys for a PAIR: nothing a commutative
role-filler encoding cannot already supply (`hadamard_encPair_order_sensitive`).
That result, left alone, undercuts EDUHDC's own motivation: if commutative
binding already encodes pair order, what does the verified operator buy at
all? This file answers for the object §3.4 (`ChainTransitivity.lean`) is
actually about — a CHAIN of relations, not a single pair.

Two ways to encode a chain of `n` relation/content steps exist side by side in
this codebase already:

  ROLE-FILLER (superposition): bundle `n` independently role-bound terms into
      ONE vector, `encChainRF`. This is the natural generalization of
      `encPair` to `n` slots, built entirely from `bundle` — one call per
      element.

  COMPOSITION (already verified): `chainRoundtrip` (`ChainTransitivity.lean`),
      recovering the content exactly at every length via `chain_exact_unbind`.
      Its definition is `chainRoundtrip (i :: is) Y = inv i (chainRoundtrip is
      (ops i Y))` — inspect it: it calls `ops` and `inv`, and NEVER `bundle`.
      It never holds more than one live value at a time.

THE RESULT (T1, informally: a Separation Theorem). Role-filler chain encoding
pays a superposition cost that composition does not, and the cost is not an
asymptotic estimate here — it is an algebraic fact, demonstrated concretely:
`encChainRF_naive_unbind` decomposes naive per-role recovery into the target
content PLUS a crosstalk term contributed by every OTHER bundled role, and
`encChainRF_crosstalk_witness` exhibits a case where that crosstalk term is
PROVABLY NONZERO — recovery genuinely fails, not merely degrades in a limit.
`chain_exact_unbind`, by contrast, is unconditional at every length, for the
structural reason above (no `bundle` call to accumulate crosstalk from).

This is the algebraic mechanism behind the ALREADY-MEASURED empirical fact of
§4.6: capacity-sweep retrieval accuracy is a logistic function of $\sqrt{D/T}$
($R^2 = 0.99$), where $T$ is superposition load. `encChainRF` for a length-$n$
chain bundles $n$ terms, i.e. $T = n$ in that sweep's own sense, so it inherits
that degradation; `chainRoundtrip` never bundles more than one thing, so it
never enters that regime at any $n$ — which is exactly why `chain_exact_unbind`
holds exactly at all 40 observed hop lengths in §4.2 while a role-filler
encoding of the same chains would not.

GENERALIZATION TO ARBITRARY n (added after the first version of this file,
which only checked n = 2). `encChainRF_crosstalk_witness_general` lifts the
n = 2 witness to EVERY n >= 2, and by a route simpler than first anticipated:
padding the n = 2 witness with extra `(role, zero2)` pairs at the end of the
chain leaves the crosstalk value unchanged, since every appended term is
`zero2` and every operator fixes it (`mul_zero2`) — no linear-independence
argument about the family's images is needed after all, only that the padding
is inert (`encChainRF_all_zero`, `encChainRF_pad_zero`). This closes the scope
gap the paper's Remark on this file used to flag, and directly serves C2-C4:
C2's exact-retrieval limit, C3's training-free path encoding, and C4's
fixed-width edge state all rest on chain composition never entering the
superposition regime AT ANY LENGTH, not only at length 2.

Mathlib-free, like the rest of this tier. Every new lemma reduces after
unfolding to linear integer arithmetic over LITERAL matrices (`Rot`, `Ref`),
closed by `omega`, exactly as `Rot_bundle_distrib` / `Ref_bundle_distrib`
already are — no ring-normalization machinery, no `sorry`, no new axiom.
`Matrix2` was given a `DecidableEq` instance (`EduBindSelfContained.lean`) to
prove this section: without it, Lean's equation compiler falls back to
`Classical.choice` when building the match/induction motives the padding
lemmas need, which would have been a real (if silent) breach of this tier's
Mathlib-free, classical-free axiom footprint. Confirmed clean via
`#print axioms` on every theorem below: `propext` and `Quot.sound` only.
-/

import EduBindSelfContained
import ChainTransitivity
import EncPairSpec

open Matrix2

-- ---------------------------------------------------------------------------
-- The role-filler chain encoding, indexed by the SAME `List Nat` of roles
-- `chainRoundtrip` uses, so the two encodings are directly comparable.
-- ---------------------------------------------------------------------------

/-- Bundle `n` independently role-bound content values into ONE vector — the
    role-filler generalization of `encPair` from 2 slots to a list of slots.
    Built entirely from `bundle` (here, matrix addition): one call per cons. -/
def encChainRF : List Nat → List (Matrix2 Int) → Matrix2 Int
  | [],      _       => zero2
  | _,       []       => zero2
  | i :: is, v :: vs => (lmul (eduGen i) v).add (encChainRF is vs)

-- ---------------------------------------------------------------------------
-- Two small helper facts about `eduGen`'s transpose, needed to decompose naive
-- unbinding. Both reduce to the SAME omega-closed pattern already used for
-- `Rot_bundle_distrib` / `Ref_bundle_distrib` / `Rot_exact_unbind`, just
-- packaged uniformly over the two-element generator family.
-- ---------------------------------------------------------------------------

theorem eduGenT_mul_add (i : Nat) (X Y : Matrix2 Int) :
    (eduGen i).transpose.mul (X.add Y)
      = ((eduGen i).transpose.mul X).add ((eduGen i).transpose.mul Y) := by
  cases i with
  | zero   => rw [eduGen_zero]; ext <;> dsimp [mul, add, transpose, Rot] <;> omega
  | succ n => rw [eduGen_succ n]; ext <;> dsimp [mul, add, transpose, Ref] <;> omega

theorem eduGenT_mul_eduGen (i : Nat) (v : Matrix2 Int) :
    (eduGen i).transpose.mul (lmul (eduGen i) v) = v := by
  cases i with
  | zero   => exact Rot_exact_unbind v
  | succ _ => exact Ref_exact_unbind v

-- ---------------------------------------------------------------------------
-- THE DECOMPOSITION. Naive recovery at role `i` — unbind with `(eduGen i)ᵀ`
-- directly on the bundled encoding, the only recovery route `encChainRF`
-- offers, since (unlike `chainRoundtrip`) nothing in its construction records
-- an unwind order — equals the target content `v` PLUS a crosstalk term
-- contributed by every OTHER role/content pair still bundled in `vs`.
-- ---------------------------------------------------------------------------

theorem encChainRF_naive_unbind (i : Nat) (v : Matrix2 Int)
    (is : List Nat) (vs : List (Matrix2 Int)) :
    (eduGen i).transpose.mul (encChainRF (i :: is) (v :: vs))
      = v.add ((eduGen i).transpose.mul (encChainRF is vs)) := by
  dsimp [encChainRF]
  rw [eduGenT_mul_add i (lmul (eduGen i) v) (encChainRF is vs),
      eduGenT_mul_eduGen i v]

-- ---------------------------------------------------------------------------
-- THE WITNESS. The crosstalk term is not merely present in the general
-- shape above — it is PROVABLY NONZERO for a concrete 2-role chain, so naive
-- recovery genuinely fails, not just "in the limit" or "on average". Content
-- `zero2` bound to role 0, content `Matrix2.one` bound to role 1; naive
-- recovery of role 0's content, in the presence of role 1's, is not `zero2`.
-- ---------------------------------------------------------------------------

theorem encChainRF_crosstalk_witness :
    (eduGen 0).transpose.mul (encChainRF [0, 1] [zero2, Matrix2.one]) ≠ zero2 := by
  have hval : (eduGen 0).transpose.mul (encChainRF [0, 1] [zero2, Matrix2.one])
      = Rot.transpose.mul ((Rot.mul zero2).add (Ref.mul Matrix2.one)) := rfl
  rw [hval]
  intro h
  have h01 := congrArg Matrix2.a01 h
  dsimp [mul, transpose, add, Rot, Ref, one, zero2] at h01
  omega

-- ---------------------------------------------------------------------------
-- THE GENERALIZATION TO ARBITRARY n. `encChainRF_crosstalk_witness` above is
-- checked at n = 2. This section lifts it to every n >= 2, by a MUCH simpler
-- route than the "linear-independence of the family's images" argument the
-- paper's Remark on this file first anticipated: padding the n = 2 witness
-- with extra (role, zero2) pairs at the END of the chain changes nothing,
-- because every appended term is `zero2` and every operator fixes `zero2`
-- (`mul_zero2`). So the SAME crosstalk value that makes the n = 2 case
-- nonzero survives verbatim at every larger n — no new algebraic content is
-- needed, only that the padding is inert.
-- ---------------------------------------------------------------------------

/-- Encoding an all-`zero2`-content chain, of any length and any roles,
    collapses to `zero2`: each hop contributes `eduGen _ * zero2 = zero2`
    (`mul_zero2`), and bundling `zero2` with `zero2` is `zero2`. -/
theorem encChainRF_all_zero (is : List Nat) :
    encChainRF is (List.replicate is.length zero2) = zero2 := by
  induction is with
  | nil => rfl
  | cons i is ih =>
    show (lmul (eduGen i) zero2).add (encChainRF is (List.replicate is.length zero2)) = zero2
    rw [ih]
    dsimp [lmul]
    rw [mul_zero2 (eduGen i), zero2_add_left]

/-- Appending extra `(role, zero2)` pairs to the end of a chain does not
    change what `encChainRF` encodes: the appended tail contributes `zero2` at
    every hop (`encChainRF_all_zero`), and `zero2` is a two-sided identity for
    `bundle` (matrix addition) throughout the recursion. -/
theorem encChainRF_pad_zero (is1 : List Nat) (vs1 : List (Matrix2 Int))
    (h : is1.length = vs1.length) (is2 : List Nat) :
    encChainRF (is1 ++ is2) (vs1 ++ List.replicate is2.length zero2) = encChainRF is1 vs1 := by
  induction is1 generalizing vs1 with
  | nil =>
    cases vs1 with
    | nil =>
      show encChainRF ([] ++ is2) ([] ++ List.replicate is2.length zero2) = encChainRF [] []
      rw [List.nil_append, List.nil_append, encChainRF_all_zero]
      rfl
    | cons _ _ =>
      simp only [List.length_nil, List.length_cons] at h
      omega
  | cons i is ih =>
    cases vs1 with
    | nil =>
      simp only [List.length_cons, List.length_nil] at h
      omega
    | cons v vs =>
      show (lmul (eduGen i) v).add (encChainRF (is ++ is2) (vs ++ List.replicate is2.length zero2))
         = (lmul (eduGen i) v).add (encChainRF is vs)
      have hlen : is.length = vs.length := by
        simp only [List.length_cons] at h
        omega
      rw [ih vs hlen]

/-- THE GENERAL WITNESS. For every chain length n >= 2, there is a concrete
    n-role chain (roles `0, 1` followed by n - 2 padding roles, contents
    `zero2, Matrix2.one` followed by n - 2 copies of `zero2`) of EXACTLY
    length n, for which naive unbinding at role 0 provably fails to recover
    the target content -- the SAME crosstalk value as the n = 2 case, since
    the padding is inert (`encChainRF_pad_zero`). This is the generalization
    the paper's Remark on n = 2 scope calls for, closing Future Work item 1:
    it is not an asymptotic or statistical claim, but a machine-checked family
    of counterexamples, one per n, uniform in construction. The two
    conjuncts are proved together, as `dof_comparison_depends_on_normalisation`
    conjoins its two comparisons in `CapacityCostModel.lean`, so the length
    claim and the crosstalk claim about the SAME list are never quoted apart. -/
theorem encChainRF_crosstalk_witness_general (n : Nat) (hn : 2 ≤ n) :
    (0 :: 1 :: List.replicate (n - 2) 0 : List Nat).length = n ∧
    (eduGen 0).transpose.mul
      (encChainRF (0 :: 1 :: List.replicate (n - 2) 0)
                  (zero2 :: Matrix2.one :: List.replicate (n - 2) zero2)) ≠ zero2 := by
  refine ⟨by rw [List.length_cons, List.length_cons, List.length_replicate]; omega, ?_⟩
  have hpad :
      encChainRF (0 :: 1 :: List.replicate (n - 2) 0)
                 (zero2 :: Matrix2.one :: List.replicate (n - 2) zero2)
        = encChainRF [0, 1] [zero2, Matrix2.one] := by
    have h2 : ([0, 1] : List Nat).length = ([zero2, Matrix2.one] : List (Matrix2 Int)).length := rfl
    have hraw := encChainRF_pad_zero [0, 1] [zero2, Matrix2.one] h2 (List.replicate (n - 2) 0)
    rw [List.length_replicate] at hraw
    exact hraw
  rw [hpad]
  exact encChainRF_crosstalk_witness

-- ---------------------------------------------------------------------------
-- THE CONTRAST. `chain_exact_unbind` (`ChainTransitivity.lean`) already gives
-- the composition scheme's side of the separation: recovery is EXACT at the
-- SAME 2-role chain, unconditionally, and — by the fully general theorem
-- already proved there — at every length, not only this one. Restated here at
-- matching shape for direct comparison with `encChainRF_crosstalk_witness`.
-- ---------------------------------------------------------------------------

theorem chainRoundtrip_no_crosstalk_at_matching_length (Y : Matrix2 Int) :
    PedagogicalVSA.chainRoundtrip edubindVSA [0, 1] Y = Y :=
  PedagogicalVSA.chain_exact_unbind edubindVSA [0, 1] Y
