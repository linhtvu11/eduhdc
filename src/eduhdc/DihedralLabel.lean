/-
  DihedralLabel.lean — the label algebra as a first-class object, and the
  positive counterpart to `edubind_reverse_blind_at_length_three`.

  WHY THIS FILE EXISTS
  --------------------
  Two gaps in the development, closed by one construction.

  (1) `no_abelian_action_PedagogicalVSA` is advertised as holding "for any
      carrier, any LABEL TYPE, and any choice of transformations". Until now
      every checked `ActionFamily` had `G = V`: `hadAction` takes labels to be
      the carrier itself and composes them elementwise, so the label algebra was
      never actually separated from the space it acts on. The generality was
      stated but not exercised. Here `G = D4`, an eight-element type that is not
      the carrier and carries no matrix structure of its own — the labels
      compose by the DIHEDRAL RULE, and the matrices only realize that rule.

  (2) `edubind_reverse_blind_at_length_three` (ChainOrder.lean) proves that the
      verified family `eduGen` CANNOT tell a length-three traversal from its
      reverse — all eight words over its two generators act as their own
      reverse, and exhaustive search finds the same at every odd length. That is
      a real limitation and it is disclosed there. What it is NOT is a reason to
      bolt a learned component onto the operator: the deficiency is that
      `eduGen` exposes only two of its group's eight elements, and the repair is
      ALGEBRAIC — expose the whole group.

      `d4VSA` below is that repair, and `d4_reverse_sensitive_general` is the
      payoff: at EVERY length n >= 2 there is a chain this family distinguishes
      from its reverse. The contrast between the two instances is the point.
      Order sensitivity in the sense of Axiom 2 does not by itself buy reversal
      discrimination; how much of the label algebra the family exposes does.

  TIER
  ----
  Mathlib-free, kernel-checked. Every obligation reduces to linear integer
  arithmetic over concrete matrices, so `omega` discharges all of them.
-/
import GroupActionSpec
import ChainOrder

open Matrix2

namespace PedagogicalVSA

-- ---------------------------------------------------------------------------
-- The label type: Z/4 rotations, with a reflection flag. NOT the carrier.
-- ---------------------------------------------------------------------------

/-- Rotation exponents mod 4, as a bare four-element type. -/
inductive R4 where
  | r0 | r1 | r2 | r3
  deriving DecidableEq

namespace R4

/-- Addition mod 4. -/
def add : R4 → R4 → R4
  | r0, b  => b
  | r1, r0 => r1 | r1, r1 => r2 | r1, r2 => r3 | r1, r3 => r0
  | r2, r0 => r2 | r2, r1 => r3 | r2, r2 => r0 | r2, r3 => r1
  | r3, r0 => r3 | r3, r1 => r0 | r3, r2 => r1 | r3, r3 => r2

/-- Subtraction mod 4 — needed because a reflection conjugates a rotation to
    its inverse, which is exactly why the dihedral rule is non-abelian. -/
def sub : R4 → R4 → R4
  | a,  r0 => a
  | r0, r1 => r3 | r0, r2 => r2 | r0, r3 => r1
  | r1, r1 => r0 | r1, r2 => r3 | r1, r3 => r2
  | r2, r1 => r1 | r2, r2 => r0 | r2, r3 => r3
  | r3, r1 => r2 | r3, r2 => r1 | r3, r3 => r0

/-- The rotation matrices, written out so every obligation below is literal
    integer arithmetic. -/
def mat : R4 → Matrix2 Int
  | r0 => { a00 := 1,  a01 := 0,  a10 := 0,  a11 := 1 }
  | r1 => { a00 := 0,  a01 := -1, a10 := 1,  a11 := 0 }
  | r2 => { a00 := -1, a01 := 0,  a10 := 0,  a11 := -1 }
  | r3 => { a00 := 0,  a01 := 1,  a10 := -1, a11 := 0 }

end R4

/-- A label of the dihedral group of order eight: `Rot^rot * Ref^flip`. This is
    a LABEL, not a carrier element — it is an eight-element enumeration with no
    linear structure. The action below is what gives it matrices. -/
structure D4 where
  rot : R4
  flip : Bool
  deriving DecidableEq

namespace D4

/-- THE DIHEDRAL RULE. `(Rot^a Ref^b)(Rot^c Ref^d) = Rot^(a ± c) Ref^(b xor d)`,
    with the sign flipped exactly when the left label carries a reflection,
    because `Ref * Rot^c = Rot^(-c) * Ref`. That single sign is the whole source
    of non-commutativity, and it lives in the LABEL algebra: no matrix is
    consulted to state it. -/
def comp : D4 → D4 → D4
  | ⟨a, false⟩, ⟨c, d⟩     => ⟨R4.add a c, d⟩
  | ⟨a, true⟩,  ⟨c, false⟩ => ⟨R4.sub a c, true⟩
  | ⟨a, true⟩,  ⟨c, true⟩  => ⟨R4.sub a c, false⟩

/-- Realizing a label as a matrix. -/
def mat : D4 → Matrix2 Int
  | ⟨r, false⟩ => R4.mat r
  | ⟨r, true⟩  => (R4.mat r).mul Ref

end D4

-- ---------------------------------------------------------------------------
-- The action, and its three obligations
-- ---------------------------------------------------------------------------

/-- Composing two realizations is realizing the composite LABEL. Sixty-four
    concrete cases, each linear in the content. -/
theorem d4_act_comp (g h : D4) (Y : Matrix2 Int) :
    (D4.mat g).mul ((D4.mat h).mul Y) = (D4.mat (D4.comp g h)).mul Y := by
  obtain ⟨gr, gf⟩ := g
  obtain ⟨hr, hf⟩ := h
  cases gf <;> cases hf <;> cases gr <;> cases hr
  all_goals (ext <;> dsimp [D4.mat, D4.comp, R4.mat, R4.add, R4.sub, Ref, Matrix2.mul,
            Bool.not_false, Bool.not_true] <;> omega)

/-- The dihedral group acting on the integer plane, presented as a label
    algebra with `G` genuinely distinct from the carrier `V`. -/
def d4Action : ActionFamily D4 (Matrix2 Int) where
  act := fun g => lmul (D4.mat g)
  comp := D4.comp
  act_comp := d4_act_comp

/-- The label algebra is non-abelian, stated about `comp` alone. By
    `label_algebra_nonabelian` this is exactly what Axiom 2 demands of it. -/
theorem d4comp_non_abelian : ¬ (∀ g h : D4, D4.comp g h = D4.comp h g) := by
  intro hab
  exact absurd (hab ⟨R4.r1, false⟩ ⟨R4.r0, true⟩) (by decide)

/-- Every label acts invertibly, with the transpose as the inverse. -/
theorem d4_exact_unbind (g : D4) (Y : Matrix2 Int) :
    (D4.mat g).transpose.mul ((D4.mat g).mul Y) = Y := by
  obtain ⟨r, f⟩ := g
  cases f <;> cases r <;>
    (ext <;>
      simp only [D4.mat, R4.mat, Ref, Matrix2.mul, Matrix2.transpose] <;> omega)

/-- Every label acts linearly on superposition. -/
theorem d4_bundle_distrib (g : D4) (X Y : Matrix2 Int) :
    (D4.mat g).mul (X.add Y) = ((D4.mat g).mul X).add ((D4.mat g).mul Y) := by
  obtain ⟨r, f⟩ := g
  cases f <;> cases r <;>
    (ext <;> simp only [D4.mat, R4.mat, Ref, Matrix2.mul, Matrix2.add] <;> omega)

/-- Relation indices to labels. Unlike `eduGen`, which sends index 0 to `Rot`
    and EVERY other index to `Ref` — exposing two of the group's eight elements
    — this exposes all eight. That difference is the whole content of
    `d4_reverse_sensitive_general` versus
    `edubind_reverse_blind_at_length_three`. -/
def d4Gen : Nat → D4
  | 0 => ⟨R4.r0, false⟩
  | 1 => ⟨R4.r1, false⟩
  | 2 => ⟨R4.r2, false⟩
  | 3 => ⟨R4.r3, false⟩
  | 4 => ⟨R4.r0, true⟩
  | 5 => ⟨R4.r1, true⟩
  | 6 => ⟨R4.r2, true⟩
  | _ => ⟨R4.r3, true⟩

theorem d4Gen_0 : d4Gen 0 = ⟨R4.r0, false⟩ := rfl
theorem d4Gen_1 : d4Gen 1 = ⟨R4.r1, false⟩ := rfl
theorem d4Gen_4 : d4Gen 4 = ⟨R4.r0, true⟩ := rfl

/-- THE INSTANCE, built through `ofAction` rather than by hand: membership is
    certified from properties of the ACTION, which is the route the criterion
    advertises and which no previous instance took. -/
def d4VSA : PedagogicalVSA (Matrix2 Int) :=
  ofAction d4Action d4Gen (fun X Y => X.add Y)
    (fun i => lmul (D4.mat (d4Gen i)).transpose)
    (fun i Y => d4_exact_unbind (d4Gen i) Y)
    (fun i X Y => d4_bundle_distrib (d4Gen i) X Y)
    ⟨1, 4, Matrix2.one, by decide⟩

/-- THE CRITERION RUN IN THE POSITIVE DIRECTION, as a check that it is usable
    rather than only refutational. `label_algebra_nonabelian` reads membership
    in the specification and returns non-abelian label composition; applied to
    `d4VSA` it re-derives `d4comp_non_abelian` without touching a matrix. The
    two proofs are independent: the direct one inspects `comp` at one pair, this
    one goes through Axiom 2. -/
theorem d4_label_nonabelian_derived :
    ¬ (∀ g h : D4, d4Action.comp g h = d4Action.comp h g) :=
  label_algebra_nonabelian d4VSA d4Action d4Gen (fun _ => rfl)

-- ---------------------------------------------------------------------------
-- The payoff: reversal discrimination at EVERY length
-- ---------------------------------------------------------------------------

/-- Index 0 carries the identity label, so it acts trivially. -/
theorem d4_ops_zero (Y : Matrix2 Int) : d4VSA.ops 0 Y = Y := by
  show lmul (D4.mat (d4Gen 0)) Y = Y
  rw [d4Gen_0]
  ext <;> simp only [lmul, D4.mat, R4.mat, Matrix2.mul] <;> omega

/-- A run of identity labels is inert at any length — the padding that makes the
    general statement below cost nothing. -/
theorem chainOps_d4_replicate_zero (k : Nat) (Y : Matrix2 Int) :
    chainOps d4VSA (List.replicate k 0) Y = Y := by
  induction k with
  | zero => rfl
  | succ k ih =>
    show d4VSA.ops 0 (chainOps d4VSA (List.replicate k 0) Y) = Y
    rw [ih]
    exact d4_ops_zero Y

/-- REVERSAL DISCRIMINATION AT EVERY LENGTH, for a family that exposes its whole
    label algebra. Contrast `edubind_reverse_blind_at_length_three`: the
    two-generator family cannot do this at length three, or at any odd length.

    Both families satisfy Axiom 2. So Axiom 2 is NOT what decides reversal
    discrimination -- which is why the paper's general separation is stated over
    two orderings (`chain_order_sensitive_general`) rather than over a traversal
    and its reverse.

    The witness is `replicate (n-2) 0 ++ [1, 4]`: a run of identity labels, then
    a rotation and a reflection. Reversing it moves the identity run to the far
    end, where it is equally inert, and swaps the rotation past the reflection --
    the one place the dihedral sign rule bites. -/
theorem d4_reverse_sensitive_general (n : Nat) (hn : 2 ≤ n) :
    ∃ (is : List Nat) (Y : Matrix2 Int),
      is.length = n ∧ chainOps d4VSA is Y ≠ chainOps d4VSA is.reverse Y := by
  refine ⟨List.replicate (n - 2) 0 ++ [1, 4], Matrix2.one, ?_, ?_⟩
  · rw [List.length_append, List.length_replicate]
    exact Nat.sub_add_cancel hn
  · rw [chainOps_append, chainOps_d4_replicate_zero,
        List.reverse_append, List.reverse_replicate,
        chainOps_append, chainOps_d4_replicate_zero]
    show lmul (D4.mat (d4Gen 1)) (lmul (D4.mat (d4Gen 4)) Matrix2.one)
           ≠ lmul (D4.mat (d4Gen 4)) (lmul (D4.mat (d4Gen 1)) Matrix2.one)
    rw [d4Gen_1, d4Gen_4]
    intro h
    have h01 := congrArg Matrix2.a01 h
    simp only [lmul, D4.mat, R4.mat, Ref, Matrix2.mul, Matrix2.one] at h01
    omega

end PedagogicalVSA

#print axioms PedagogicalVSA.d4_act_comp
#print axioms PedagogicalVSA.d4comp_non_abelian
#print axioms PedagogicalVSA.d4VSA
#print axioms PedagogicalVSA.d4_label_nonabelian_derived
#print axioms PedagogicalVSA.d4_reverse_sensitive_general
