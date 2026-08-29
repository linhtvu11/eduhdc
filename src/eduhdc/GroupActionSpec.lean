/-
C1 — The group-action layer of the specification.

WHAT THIS FILE IS
-----------------
`no_hadamard_PedagogicalVSA` (VSATriad.lean) rules out ONE operator family:
elementwise (Hadamard/MAP-style) binding. Its proof does not use anything
specific to elementwise multiplication beyond the fact that two Hadamard
operators COMMUTE, and that commutativity is itself inherited from the
commutativity of the underlying label multiplication. This file makes that
observation the theorem instead of the proof step.

The move is to name the structure the family already has. A binding family is
not merely a `Nat`-indexed bag of functions: composing two binding operations
lands back inside the family, at the composite label. That is an ACTION of a
label algebra on the carrier. Once the family is presented that way, Axiom 2
stops being a statement about the carrier and becomes a statement about the
LABEL ALGEBRA:

    no_abelian_action_PedagogicalVSA  —  no action of an abelian label algebra
                                         can satisfy Axiom 2, for any carrier,
                                         any label type, any transformations.

Three things follow, and they are the reason this file exists.

  1. `no_hadamard_PedagogicalVSA` becomes a COROLLARY (`no_hadamard_via_action`
     below), obtained by exhibiting elementwise binding as one abelian action
     among many. The Revision-4 theorem is recovered, not replaced, which is
     the honest way to show it was a special case.

  2. The paper's prose argument about RotatE — that composing two relations
     ADDS their phases, so the composition operator is commutative and the
     antisymmetry result does not bear on order-sensitivity of composition —
     becomes a machine-checked theorem (`no_additive_label_PedagogicalVSA`).
     Additive label composition is abelian; that is the whole argument, and it
     needs no property of the carrier at all.

  3. The positive side stops being a list of two instances. `ofAction` builds a
     `PedagogicalVSA` from ANY action whose label algebra is non-abelian at one
     pair and whose transformations satisfy the two computational axioms. The
     downstream chapters use operators (O(3) in C2, block-diagonal GHRR in C3)
     that are neither `edubindVSA` nor `permVSA`; membership in this class is
     what they can actually inherit, where instance-by-instance verification
     would cover neither.

TIER
----
Mathlib-free, kernel-checked, same as the rest of this project's default
target. Nothing here needs `Classical.choice`: the abelian hypothesis is used
by rewriting, not by case analysis on decidability.
-/
import EduBindSelfContained
import VSATriad

open Matrix2

namespace PedagogicalVSA

-- ---------------------------------------------------------------------------
-- The label algebra and its action
-- ---------------------------------------------------------------------------

/-- An `ActionFamily G V` presents a family of transformations of the carrier
    `V` as the action of a label algebra `G`. The single law says composing two
    transformations is applying the transformation of the composite label —
    which is exactly what makes a binding family closed under composition
    rather than an arbitrary collection of functions.

    Note what is NOT required: `comp` need not be associative, need not have a
    unit, and the action need not be faithful. The impossibility theorem below
    needs only the one law, so demanding more would shrink its reach for no
    gain. (This mirrors the deliberate one-sidedness of `exact_unbind_ax`
    documented in `SpecStrengthening.lean`.) -/
structure ActionFamily (G : Type) (V : Type) where
  /-- The transformation carried by each label. -/
  act : G → (V → V)
  /-- Composition of labels. -/
  comp : G → G → G
  /-- Composing transformations is acting by the composite label. -/
  act_comp : ∀ g h Y, act g (act h Y) = act (comp g h) Y

namespace ActionFamily

variable {G V : Type}

/-- If labels compose commutatively, the transformations commute pointwise.
    This is the entire content of the negative results below: order-sensitivity
    is destroyed at the label algebra, before the carrier is ever consulted. -/
theorem commutes_of_abelian (A : ActionFamily G V)
    (hab : ∀ g h, A.comp g h = A.comp h g) :
    ∀ g h Y, A.act g (A.act h Y) = A.act h (A.act g Y) := by
  intro g h Y
  rw [A.act_comp, A.act_comp, hab]

end ActionFamily

-- ---------------------------------------------------------------------------
-- The generalized impossibility
-- ---------------------------------------------------------------------------

/-- THE GENERALIZED NEGATIVE RESULT. No `PedagogicalVSA` can have a binding
    family that is the action of an abelian label algebra — for ANY carrier,
    ANY label type, and ANY choice of transformations. Axiom 2 is contradictory
    with abelian label composition.

    `no_hadamard_PedagogicalVSA` is the special case where the labels are 2×2
    integer matrices composing elementwise. -/
theorem no_abelian_action_PedagogicalVSA
    {G V : Type} (P : PedagogicalVSA V) (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g)
    (hops : ∀ i, P.ops i = A.act (gen i)) : False := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  rw [hops i, hops j] at hne
  exact hne (A.commutes_of_abelian hab (gen i) (gen j) Y)

/-- The same fact stated as the paper reads it: for a binding family presented
    as an action, Axiom 2 says precisely that the LABEL ALGEBRA is non-abelian.
    Order-sensitivity is not a property of the vectors, nor of the particular
    transformations — it is a property of how relation labels compose. -/
theorem label_algebra_nonabelian
    {G V : Type} (P : PedagogicalVSA V) (A : ActionFamily G V) (gen : Nat → G)
    (hops : ∀ i, P.ops i = A.act (gen i)) :
    ¬ (∀ g h, A.comp g h = A.comp h g) :=
  fun hab => no_abelian_action_PedagogicalVSA P A gen hab hops

-- ---------------------------------------------------------------------------
-- Instance 1: elementwise binding — recovering the Revision-4 theorem
-- ---------------------------------------------------------------------------

/-- Elementwise (Hadamard/MAP-style) binding, presented as an action: labels are
    2×2 integer matrices, and they compose elementwise. -/
def hadAction : ActionFamily (Matrix2 Int) (Matrix2 Int) where
  act := had
  comp := had
  act_comp := by intro M N Y; ext <;> dsimp [had] <;> rw [Int.mul_assoc]

/-- Elementwise label composition is abelian, because `Int` multiplication is. -/
theorem hadAction_abelian : ∀ M N, hadAction.comp M N = hadAction.comp N M := by
  intro M N; ext <;> dsimp [hadAction, had] <;> exact Int.mul_comm _ _

/-- `no_hadamard_PedagogicalVSA` recovered as a corollary of the general
    theorem. The Revision-4 result is a special case, not a separate fact. -/
theorem no_hadamard_via_action
    (P : PedagogicalVSA (Matrix2 Int)) (Ms : Nat → Matrix2 Int)
    (hops : ∀ i, P.ops i = had (Ms i)) : False :=
  no_abelian_action_PedagogicalVSA P hadAction Ms hadAction_abelian hops

-- ---------------------------------------------------------------------------
-- Instance 2: additive label composition — the RotatE argument, machine-checked
-- ---------------------------------------------------------------------------

/-- Labels composing by ADDITION, the discrete analogue of composing two
    relations by adding their phases. Abelian by `Int.add_comm`. -/
def addAction {V : Type} (act : Int → (V → V))
    (hcomp : ∀ a b Y, act a (act b Y) = act (a + b) Y) :
    ActionFamily Int V where
  act := act
  comp := fun a b => a + b
  act_comp := hcomp

/-- No `PedagogicalVSA` can have a binding family whose labels compose by
    addition. This is the paper's RotatE argument, discharged as a theorem: the
    composition of two phase-rotations adds their phases, addition is
    commutative, and Axiom 2 is therefore unreachable — regardless of what the
    carrier is or what the individual rotations do. Antisymmetry of a single
    relation under inversion, which RotatE does establish, is a different
    property and is untouched by this. -/
theorem no_additive_label_PedagogicalVSA
    {V : Type} (P : PedagogicalVSA V) (act : Int → (V → V)) (gen : Nat → Int)
    (hcomp : ∀ a b Y, act a (act b Y) = act (a + b) Y)
    (hops : ∀ i, P.ops i = act (gen i)) : False :=
  no_abelian_action_PedagogicalVSA P (addAction act hcomp) gen
    (fun a b => Int.add_comm a b) hops

-- ---------------------------------------------------------------------------
-- The positive side: a class, not a list of two
-- ---------------------------------------------------------------------------

/-- Build a `PedagogicalVSA` from any action whose label algebra fails to
    commute at one pair and whose transformations unbind exactly and distribute
    over bundling. This is the constructor the downstream chapters need: it
    certifies membership from properties of the action, so operators outside
    `verifiedOperators` (C2's O(3) rotations, C3's block-diagonal GHRR family)
    are covered by the specification without each needing its own instance
    proof. -/
def ofAction {G V : Type} (A : ActionFamily G V) (gen : Nat → G)
    (bundle : V → V → V) (inv : Nat → (V → V))
    (h_unbind : ∀ i Y, inv i (A.act (gen i) Y) = Y)
    (h_distrib : ∀ i X Y,
        A.act (gen i) (bundle X Y) = bundle (A.act (gen i) X) (A.act (gen i) Y))
    (h_order : ∃ i j Y,
        A.act (gen i) (A.act (gen j) Y) ≠ A.act (gen j) (A.act (gen i) Y)) :
    PedagogicalVSA V where
  bundle := bundle
  ops := fun i => A.act (gen i)
  inv := inv
  exact_unbind_ax := h_unbind
  bundle_distrib_ax := h_distrib
  order_sensitive_ax := h_order

/-- The order hypothesis of `ofAction` is implied by a purely LABEL-LEVEL
    condition: two labels that fail to commute, acting on a content the action
    separates. Stated separately because it is the form a downstream chapter
    can actually check — it asks about the relation algebra, not about the
    representation. -/
theorem order_of_label_witness {G V : Type} (A : ActionFamily G V) (gen : Nat → G)
    (i j : Nat) (Y : V)
    (hsep : A.act (A.comp (gen i) (gen j)) Y ≠ A.act (A.comp (gen j) (gen i)) Y) :
    ∃ i j Y, A.act (gen i) (A.act (gen j) Y) ≠ A.act (gen j) (A.act (gen i) Y) := by
  refine ⟨i, j, Y, ?_⟩
  rw [A.act_comp, A.act_comp]
  exact hsep

-- ---------------------------------------------------------------------------
-- Axiom footprint, for the reader to confirm this file stays in the kernel tier
-- ---------------------------------------------------------------------------

#print axioms ActionFamily.commutes_of_abelian
#print axioms no_abelian_action_PedagogicalVSA
#print axioms label_algebra_nonabelian
#print axioms no_hadamard_via_action
#print axioms no_additive_label_PedagogicalVSA
#print axioms order_of_label_witness

end PedagogicalVSA
