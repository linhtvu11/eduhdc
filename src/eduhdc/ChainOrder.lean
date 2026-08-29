/-
  ChainOrder.lean — what order sensitivity buys along a CHAIN.

  This file exists to answer a question the rest of the development leaves
  open, and answers negatively if left open.

  `chain_exact_unbind` (ChainTransitivity.lean) consumes exactly one axiom:
  `exact_unbind_ax`. It never reads `order_sensitive_ax`. So exact recovery of a
  chain is available to ANY family with left inverses — including the abelian
  families that `GroupActionSpec.lean` proves cannot satisfy Axiom 2 at all
  (MAP satisfies `x * x = 1` pointwise, hence has a left inverse). Exact
  recovery therefore cannot be the thing non-commutativity buys, and the
  measured curves agree: composition is exact at every chain length for a
  commutative operator too.

  What an abelian family loses is something else, and this file isolates it: it
  cannot tell WHICH ORDER a chain was traversed in.

    `chain_order_sensitive`         Axiom 2 => some two orderings of the same
                                    multiset of relations act differently.
    `chain_order_sensitive_general` the same, AT EVERY LENGTH n >= 2, via an
                                    injective-prefix padding argument.
    `abelian_chainAct_perm`         abelian labels => EVERY reordering acts
                                    identically -- any permutation, not only
                                    reversal, at every length.
    `abelian_chainAct_reverse`      the reversal special case, kept because
                                    reversal is what prerequisite direction is.

  That pair is the separation, and it is symmetric in the quantifier:
  abelian => ALL orderings agree at EVERY length; Axiom 2 => SOME two orderings
  disagree at EVERY length.

  WHAT IS *NOT* TRUE, AND WAS ASSUMED TO BE UNTIL REVISION 7. "A non-abelian
  operator distinguishes a traversal from its REVERSE" does not follow, and is
  FALSE for this development's own verified instance at every odd length:
  `eduGen` exposes only two of its group's eight elements, and
  `edubind_reverse_blind_at_length_three` proves that all eight length-three
  words over those two generators act exactly as their own reverse. Exhaustive
  search confirms 0 of 2^n words differ at n = 3, 5, 7. Half of a curriculum's
  paths have odd hop length, so this is not an edge case. Reverse-specific
  claims must therefore be made per family, not for non-commutativity as such.

  A WARNING ABOUT WHERE THE NEGATIVE DIRECTION MAY BE STATED. It must NOT be
  stated over `PedagogicalVSA` or `PedagogicalMonoid`, because both of those
  carry `order_sensitive_ax`, and `no_abelian_action_PedagogicalVSA` /
  `no_abelian_action_PedagogicalMonoid` derive `False` from exactly the
  hypotheses (structure + abelian action + hops) such a statement would take.
  The result would type-check, `lake build` would stay green, and it would be
  VACUOUS: quantified over an empty domain, saying nothing whatsoever about MAP.
  `abelian_chainAct_vacuous_over_VSA` below records that trap as a theorem so it
  cannot be walked into again. The non-vacuous home for the statement is
  `ActionFamily` itself, which carries no order axiom, and that is where it is
  proved: `abelian_chainAct_reverse` applies to MAP, whose label algebra is
  abelian, and MAP is exactly what it needs to talk about.
-/

import MonoidRelaxation

namespace PedagogicalVSA

variable {V : Type}

-- ---------------------------------------------------------------------------
-- Positive direction: Axiom 2 separates two orderings of one relation multiset
-- ---------------------------------------------------------------------------

/-- Forward application of a relation chain by a `PedagogicalVSA`'s own family.
    No `inv` appears, so ordering questions are separable from recovery ones. -/
def chainOps (P : PedagogicalVSA V) : List Nat → V → V
  | [],      Y => Y
  | i :: is, Y => P.ops i (chainOps P is Y)

/-- Axiom 2 is exactly the statement that two orderings of the same relations
    act differently on some content. The two chains are permutations of each
    other, so the difference is attributable to ORDER and nothing else: same
    relations, same length, same content, different result. Non-vacuous: it is
    stated over `PedagogicalVSA`, which has instances (`edubindVSA`). -/
theorem chain_order_sensitive (P : PedagogicalVSA V) :
    ∃ (is js : List Nat) (Y : V),
      is.length = js.length ∧ chainOps P is Y ≠ chainOps P js Y := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  refine ⟨[i, j], [j, i], Y, rfl, ?_⟩
  dsimp [chainOps]
  exact hne

-- ---------------------------------------------------------------------------
-- Revision 7 additions: the positive direction, general in the chain length.
--
-- `chain_order_sensitive` above produces lists of length exactly TWO, and its
-- proof is `dsimp; exact hne` -- it is Axiom 2 with `chainOps` unfolded. That
-- is honest but thin, and it leaves the paper's central claim ("a non-abelian
-- label algebra buys order ALONG A CHAIN") without a statement at length n.
--
-- The universally quantified version is FALSE: a palindromic chain, or a chain
-- of repeated relations, acts identically in either direction no matter how
-- non-commutative the family is. The correct general form is EXISTENTIAL PER
-- LENGTH, exactly as `encChainRF_crosstalk_witness_general` is, and it is
-- proved here by padding a length-2 witness with a prefix that is INJECTIVE
-- rather than inert -- injectivity is free, because every `ops i` has a left
-- inverse by Axiom 3.
--
-- A WARNING THAT COST A REVISION. The general statement is about TWO ORDERINGS
-- OF THE SAME RELATIONS, not about a traversal versus its REVERSE. The reverse
-- form does not follow, and for this development's own verified instance it is
-- outright FALSE at every odd length -- see
-- `edubind_reverse_blind_at_length_three` below.
-- ---------------------------------------------------------------------------

/-- Applying a concatenated chain is applying the suffix, then the prefix.
    (`chainOps` consumes the head last, mirroring function composition.) -/
theorem chainOps_append (P : PedagogicalVSA V) :
    ∀ (l1 l2 : List Nat) (Y : V),
      chainOps P (l1 ++ l2) Y = chainOps P l1 (chainOps P l2 Y) := by
  intro l1
  induction l1 with
  | nil => intro l2 Y; rfl
  | cons i is ih =>
    intro l2 Y
    dsimp [chainOps]
    rw [ih l2 Y]

/-- Every chain acts INJECTIVELY. This consumes Axiom 3 only: each `ops i` has
    a left inverse, hence is injective, and a composition of injections is one.
    It is what lets a length-2 witness be padded to any length without the
    padding being able to collapse the difference. -/
theorem chainOps_injective (P : PedagogicalVSA V) :
    ∀ (is : List Nat) (X Y : V), chainOps P is X = chainOps P is Y → X = Y := by
  intro is
  induction is with
  | nil => intro X Y h; exact h
  | cons i is ih =>
    intro X Y h
    dsimp [chainOps] at h
    have h2 := congrArg (P.inv i) h
    rw [P.exact_unbind_ax, P.exact_unbind_ax] at h2
    exact ih X Y h2

/-- THE POSITIVE DIRECTION, GENERAL IN THE CHAIN LENGTH. For every length
    `n >= 2` there are two chains of that length that are permutations of one
    another -- the SAME relations, traversed in a different order -- and a
    content on which they act differently.

    This is the statement that pairs with `abelian_chainAct_perm` below, and
    the pair is the paper's separation:

      abelian labels  =>  EVERY ordering acts identically, at every length;
      Axiom 2         =>  at every length SOME two orderings act differently.

    Proof: take Axiom 2's witness `(i, j, Y)`, and compare
    `replicate (n-2) i ++ [i, j]` with `replicate (n-2) i ++ [j, i]`. The two
    differ only in their last two entries, which `chainOps` applies FIRST, so
    the shared prefix acts on two already-different values -- and it cannot
    identify them, because it is injective. -/
theorem chain_order_sensitive_general (P : PedagogicalVSA V) (n : Nat) (hn : 2 ≤ n) :
    ∃ (is js : List Nat) (Y : V),
      is.length = n ∧ js.length = n ∧ is.Perm js ∧
      chainOps P is Y ≠ chainOps P js Y := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  refine ⟨List.replicate (n - 2) i ++ [i, j],
          List.replicate (n - 2) i ++ [j, i], Y, ?_, ?_, ?_, ?_⟩
  · rw [List.length_append, List.length_replicate]
    exact Nat.sub_add_cancel hn
  · rw [List.length_append, List.length_replicate]
    exact Nat.sub_add_cancel hn
  · exact List.Perm.append_left _ (List.Perm.swap j i [])
  · rw [chainOps_append, chainOps_append]
    intro hcontra
    exact hne (chainOps_injective P _ _ _ hcontra)

-- ---------------------------------------------------------------------------
-- The reverse form does NOT generalize, and this instance shows why.
-- ---------------------------------------------------------------------------

/-- A DISCLOSED SCOPE LIMIT, PROVED RATHER THAN ASSERTED. The verified integer
    family `eduGen` exposes only TWO of the eight elements of its group (index
    0 is `Rot`, every other index is `Ref`). At chain length three, EVERY word
    over those two generators acts exactly as its own reverse -- all eight of
    them -- so this instance cannot distinguish a traversal from its reverse at
    length three, however non-commutative the family is.

    Exhaustive search over the same family confirms the pattern at every ODD
    length (0 of 2^n words differ at n = 3, 5, 7), while at even lengths half
    of them do. Adding a third generator (`Rot.mul Ref`) restores a witness at
    every length from 2 to 10.

    This matters because a curriculum path of odd hop length is not a special
    case: it is about half of the graph. Any claim of the form "a non-abelian
    operator distinguishes a path from its reverse" must therefore be stated
    for a family, not for non-commutativity as such --
    `chain_order_sensitive_general` above is the form that survives. -/
theorem eduGen_cases (i : Nat) : eduGen i = Rot ∨ eduGen i = Ref := by
  cases i with
  | zero   => exact Or.inl rfl
  | succ _ => exact Or.inr rfl

theorem edubind_reverse_blind_at_length_three (i j k : Nat) (Y : Matrix2 Int) :
    chainOps edubindVSA [i, j, k] Y = chainOps edubindVSA [k, j, i] Y := by
  dsimp [chainOps, edubindVSA, lmul]
  cases i <;> cases j <;> cases k <;>
    simp only [eduGen_zero, eduGen_succ] <;>
    (ext <;> dsimp [Matrix2.mul, Rot, Ref] <;> omega)


end PedagogicalVSA

namespace PedagogicalVSA.ActionFamily

variable {G V : Type}

-- ---------------------------------------------------------------------------
-- Negative direction: stated over ActionFamily, which carries NO order axiom
-- ---------------------------------------------------------------------------

/-- Forward application of a chain of relation LABELS through an action. This is
    the right home for the order question: an `ActionFamily` records how labels
    compose and imposes no order-sensitivity axiom, so a statement about abelian
    label algebras here is a statement about a NON-EMPTY class — MAP included. -/
def chainAct (A : ActionFamily G V) (gen : Nat → G) : List Nat → V → V
  | [],      Y => Y
  | i :: is, Y => A.act (gen i) (chainAct A gen is Y)

/-- One adjacent transposition is invisible to an abelian family. This is the
    inductive step of the theorem below, isolated because it is where the
    abelian hypothesis is consumed and the only place it is needed. -/
theorem chainAct_swap_of_abelian (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g) :
    ∀ (i j : Nat) (is : List Nat) (Y : V),
      chainAct A gen (i :: j :: is) Y = chainAct A gen (j :: i :: is) Y := by
  intro i j is Y
  dsimp [chainAct]
  exact A.commutes_of_abelian hab (gen i) (gen j) (chainAct A gen is Y)

/-- ORDER BLINDNESS. For a family with abelian label composition, reversing a
    chain leaves its action unchanged at every length -- so the forward and
    backward traversals of an ordered relation sequence are literally the same
    map. A commutative binding can recover a chain exactly (that needs only a
    left inverse, `chain_exact_unbind`) and still be unable to say which way it
    was walked.

    Reversal is the case that matters for prerequisite structure: it is the
    difference between "A is a prerequisite of B" read along a path and the same
    path read backwards. -/
theorem abelian_chainAct_reverse (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g) :
    ∀ (is : List Nat) (Y : V),
      chainAct A gen is.reverse Y = chainAct A gen is Y := by
  have push : ∀ (is : List Nat) (i : Nat) (Y : V),
      chainAct A gen (is ++ [i]) Y = chainAct A gen (i :: is) Y := by
    intro is
    induction is with
    | nil => intro i Y; rfl
    | cons k ks ih =>
      intro i Y
      show chainAct A gen (k :: (ks ++ [i])) Y = chainAct A gen (i :: k :: ks) Y
      dsimp [chainAct]
      rw [show chainAct A gen (ks ++ [i]) Y = chainAct A gen (i :: ks) Y from ih i Y]
      have h := chainAct_swap_of_abelian A gen hab k i ks Y
      dsimp [chainAct] at h
      exact h
  intro is
  induction is with
  | nil => intro Y; rfl
  | cons i is ih =>
    intro Y
    rw [List.reverse_cons, push is.reverse i Y]
    dsimp [chainAct]
    rw [ih Y]

/-- THE TRAP, recorded so it cannot be re-entered. Stating order blindness over
    `PedagogicalVSA` instead of `ActionFamily` yields a theorem that type-checks
    and is VACUOUS: its hypotheses are exactly those from which
    `no_abelian_action_PedagogicalVSA` derives `False`, so it quantifies over an
    empty domain and says nothing about MAP or about any concrete operator. The
    same applies to `PedagogicalMonoid`, which also carries Axiom 2
    (`no_abelian_action_PedagogicalMonoid`). Anything at all follows from those
    hypotheses -- which is what this theorem demonstrates. -/
theorem abelian_chainAct_vacuous_over_VSA
    (P : PedagogicalVSA V) (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g)
    (hops : ∀ i, P.ops i = A.act (gen i)) (Q : Prop) : Q :=
  absurd trivial (fun _ => no_abelian_action_PedagogicalVSA P A gen hab hops)

/-- The order blindness of elementwise (MAP-style) binding, as a concrete
    instance rather than a statement about a class: at every chain length, a
    chain of elementwise binds and its reverse are the same map. This is the
    fact `chain_order_discrimination.py` measures at cosine 1.0. -/
theorem hadamard_chainAct_reverse (Ms : Nat → Matrix2 Int) :
    ∀ (is : List Nat) (Y : Matrix2 Int),
      chainAct hadAction Ms is.reverse Y = chainAct hadAction Ms is Y :=
  abelian_chainAct_reverse hadAction Ms hadAction_abelian

/-- ORDER BLINDNESS, FULL STRENGTH. For a family with abelian label
    composition, ANY two chains that are permutations of one another act
    identically -- not merely a chain and its reverse.

    This is strictly stronger than `abelian_chainAct_reverse` and costs almost
    nothing, because `List.Perm` is generated by exactly the adjacent
    transposition `chainAct_swap_of_abelian` already discharges, plus `cons`,
    `trans` and `nil`. Revision 6 listed the permutation case as future work
    needing "a normal form for permutations under adjacent transpositions";
    in Lean 4 the normal form is the inductive definition itself. -/
theorem abelian_chainAct_perm (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g) :
    ∀ {is js : List Nat}, is.Perm js → ∀ (Y : V),
      chainAct A gen is Y = chainAct A gen js Y := by
  intro is js hp
  induction hp with
  | nil => intro Y; rfl
  | cons x _ ih =>
    intro Y
    dsimp [chainAct]
    rw [ih Y]
  | swap x y l => intro Y; exact chainAct_swap_of_abelian A gen hab y x l Y
  | trans _ _ ih1 ih2 => intro Y; rw [ih1 Y, ih2 Y]

/-- The concrete elementwise (MAP-style) instance of full order blindness: a
    chain of elementwise binds acts the same under every reordering of its
    relations, at every length. Reversal is the special case
    `hadamard_chainAct_reverse` reports. -/
theorem hadamard_chainAct_perm (Ms : Nat → Matrix2 Int) :
    ∀ {is js : List Nat}, is.Perm js → ∀ (Y : Matrix2 Int),
      chainAct hadAction Ms is Y = chainAct hadAction Ms js Y :=
  abelian_chainAct_perm hadAction Ms hadAction_abelian


end PedagogicalVSA.ActionFamily

#print axioms PedagogicalVSA.chain_order_sensitive
#print axioms PedagogicalVSA.chain_order_sensitive_general
#print axioms PedagogicalVSA.chainOps_injective
#print axioms PedagogicalVSA.edubind_reverse_blind_at_length_three
#print axioms PedagogicalVSA.ActionFamily.abelian_chainAct_perm
#print axioms PedagogicalVSA.ActionFamily.hadamard_chainAct_perm
#print axioms PedagogicalVSA.ActionFamily.abelian_chainAct_reverse
#print axioms PedagogicalVSA.ActionFamily.hadamard_chainAct_reverse
#print axioms PedagogicalVSA.ActionFamily.abelian_chainAct_vacuous_over_VSA
