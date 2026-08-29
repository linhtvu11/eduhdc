import VSATriad
open Matrix2
/-  ADVERSARIAL TEST: replay the Revision-2 attack against the Revision-3 spec.
    A Hadamard (commutative) family still satisfies Axioms 1 and 3 — those two
    fields close below with no error. Axiom 2 is left as `sorry`, and
    `no_hadamard_PedagogicalVSA` proves that `sorry` can never be filled.      -/
def Mpm : Matrix2 Int := { a00 := 1, a01 := 1, a10 := 1, a11 := -1 }
def hadGen : Nat → Matrix2 Int := fun _ => Mpm

noncomputable def hadamardAttack : PedagogicalVSA (Matrix2 Int) where
  bundle := fun X Y => X.add Y
  ops := fun i => had (hadGen i)
  inv := fun i => had (hadGen i)
  exact_unbind_ax := by
    intro i Y; ext <;> simp [had, hadGen, Mpm]        -- Axiom 3: OK
  bundle_distrib_ax := by
    intro i X Y; ext <;> simp [had, hadGen, Mpm, add] <;> omega   -- Axiom 1: OK
  order_sensitive_ax := by sorry                      -- Axiom 2: UNFILLABLE

theorem attack_is_impossible : False :=
  no_hadamard_PedagogicalVSA hadamardAttack hadGen (fun _ => rfl)
