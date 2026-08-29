/-
C1 -> C2/C3 — discharging membership for the operators the downstream chapters
actually use.

WHY THIS FILE EXISTS
--------------------
`src/eduhdc/GroupActionSpec.lean` proves the NECESSARY direction: no action of an
abelian label algebra can satisfy Axiom 2. That settles which operators are
excluded (MAP, HRR, phase composition). It does not settle which are admitted,
and until it is settled, C2's O(3) trajectory and C3's block-diagonal GHRR
encoder inherit nothing from C1 — `verifiedOperators` contains only the two O(2)
instances, and `docs/c1_to_c2c3c4_complete_mapping.md` was, before 2026-08-28,
claiming coverage it did not have.

This file discharges the other direction, over ℝ rather than over `Int`,
because ℝ is the carrier C2 and C3 actually compute in. Proving it over the
integers would reproduce exactly the analogy-instead-of-inheritance mistake this
file exists to fix.

The load-bearing result is `ofOrthogonalFamily`: ANY family of orthogonal
matrices over ANY finite index type, with one non-commuting pair, is a
`PedagogicalVSA`. Everything after it is an instantiation:

  o3VSA    — C2's H_c carrier: 3x3 real orthogonal, two elements of SO(3).
  ghrrVSA  — C3's encoder: block-diagonal with orthogonal blocks, any block count.

Both then inherit `chain_exact_unbind` and every other theorem stated generically
over `PedagogicalVSA`, by membership rather than by resemblance.

TIER
----
Mathlib tier: `Classical.choice` is expected here and is disclosed in the paper.
The kernel tier is untouched by this file.
-/
import Basic

open Matrix

-- `ofAction` / `ofOrthogonalFamily` are CONSTRUCTORS that return a class value,
-- not instances to be found by search, so semireducibility is intended here.
set_option warn.classDefReducibility false

namespace EduHDC

variable {n : Type} [Fintype n] [DecidableEq n]

-- ---------------------------------------------------------------------------
-- The action layer, mirrored over ℝ
-- ---------------------------------------------------------------------------

/-- Mirror of the kernel-tier `PedagogicalVSA.ActionFamily`
    (`src/eduhdc/GroupActionSpec.lean`). Duplicated rather than imported because
    the two tiers are separate Lake packages, following the same pattern this
    tier already uses for `PedagogicalVSA` itself. -/
structure ActionFamily (G : Type) (V : Type) where
  /-- The transformation carried by each label. -/
  act : G → (V → V)
  /-- Composition of labels. -/
  comp : G → G → G
  /-- Composing transformations is acting by the composite label. -/
  act_comp : ∀ g h Y, act g (act h Y) = act (comp g h) Y

/-- Labels that commute give transformations that commute pointwise. -/
theorem ActionFamily.commutes_of_abelian {G V : Type} (A : ActionFamily G V)
    (hab : ∀ g h, A.comp g h = A.comp h g) :
    ∀ g h Y, A.act g (A.act h Y) = A.act h (A.act g Y) := by
  intro g h Y
  rw [A.act_comp, A.act_comp, hab]

/-- The generalized impossibility, restated in this tier so the two directions
    can be read side by side. -/
theorem no_abelian_action_PedagogicalVSA
    {G V : Type} (P : PedagogicalVSA V) (A : ActionFamily G V) (gen : Nat → G)
    (hab : ∀ g h, A.comp g h = A.comp h g)
    (hops : ∀ i, P.ops i = A.act (gen i)) : False := by
  obtain ⟨i, j, Y, hne⟩ := P.order_sensitive_ax
  rw [hops i, hops j] at hne
  exact hne (A.commutes_of_abelian hab (gen i) (gen j) Y)

/-- Build a `PedagogicalVSA` from an action satisfying the three axioms. -/
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
  bundle_distrib_ax := h_distrib
  order_sensitive_ax := h_order
  exact_unbind_ax := h_unbind

/-- Matrix multiplication is the action of the matrix monoid on itself. This is
    the label algebra every operator in this project actually uses. -/
def matAction (n R : Type) [Fintype n] [DecidableEq n] [CommRing R] :
    ActionFamily (Matrix n n R) (Matrix n n R) where
  act := fun M Y => M * Y
  comp := fun M N => M * N
  act_comp := by intro M N Y; exact (Matrix.mul_assoc M N Y).symm

-- ---------------------------------------------------------------------------
-- THE CLASS THEOREM
-- ---------------------------------------------------------------------------

/-- **Membership from properties of the action alone.** Any family of orthogonal
    real matrices, over any finite index type, that fails to commute at one pair
    is a `PedagogicalVSA`: bundling is matrix addition, binding is left
    multiplication, unbinding is left multiplication by the transpose.

    This is what the downstream chapters inherit. It is stated for an arbitrary
    index type `n` on purpose — `n := Fin 3` gives C2's trajectory carrier and
    `n := Fin 2 × o` gives C3's block-diagonal encoder, with no separate proof
    for either. -/
noncomputable def ofOrthogonalFamily (gen : Nat → Matrix n n ℝ)
    (horth : ∀ i, (gen i)ᵀ * gen i = 1)
    (hne : ∃ i j Y, gen i * (gen j * Y) ≠ gen j * (gen i * Y)) :
    PedagogicalVSA (Matrix n n ℝ) :=
  ofAction (matAction n ℝ) gen (fun X Y => X + Y) (fun i Y => (gen i)ᵀ * Y)
    (by
      intro i Y
      show (gen i)ᵀ * (gen i * Y) = Y
      rw [← Matrix.mul_assoc, horth i, Matrix.one_mul])
    (by
      intro i X Y
      exact Matrix.mul_add _ X Y)
    hne

/-- The order hypothesis reduces to non-commutativity of two family members:
    take the content to be the identity. Stated separately because it is the
    form a downstream chapter can check about its own generator set. -/
theorem order_of_matrix_witness {R : Type} [CommRing R] (gen : Nat → Matrix n n R) (i j : Nat)
    (h : gen i * gen j ≠ gen j * gen i) :
    ∃ i j Y, gen i * (gen j * Y) ≠ gen j * (gen i * Y) := by
  refine ⟨i, j, 1, ?_⟩
  simpa [Matrix.mul_one] using h

-- ---------------------------------------------------------------------------
-- C2: the O(3) trajectory carrier
-- ---------------------------------------------------------------------------

/-- Quarter turn about the z axis: an element of SO(3). -/
noncomputable def rotZ : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 1]

/-- Quarter turn about the x axis: an element of SO(3). -/
noncomputable def rotX : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

theorem rotZ_orthogonal : rotZᵀ * rotZ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rotZ, Matrix.mul_apply, Fin.sum_univ_three]

theorem rotX_orthogonal : rotXᵀ * rotX = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rotX, Matrix.mul_apply, Fin.sum_univ_three]

/-- The two quarter turns do not commute: rotating about z then x is not
    rotating about x then z. This is the O(3) analogue of
    `EduBind.rot_refl_non_commutative`, and it is why C2 could migrate from O(2)
    to O(3) without losing Axiom 2. -/
theorem rotZ_rotX_non_commutative : rotZ * rotX ≠ rotX * rotZ := by
  intro h
  have h02 := congrFun (congrFun h 0) 2
  simp [rotZ, rotX, Matrix.mul_apply, Fin.sum_univ_three] at h02

/-- C2's generator family: index 0 is the z quarter turn, every other index is
    the x quarter turn. Both are orthogonal, and they do not commute. -/
noncomputable def o3Gen : Nat → Matrix (Fin 3) (Fin 3) ℝ
  | 0 => rotZ
  | _ => rotX

theorem o3Gen_orthogonal : ∀ i, (o3Gen i)ᵀ * o3Gen i = 1 := by
  intro i
  cases i with
  | zero   => exact rotZ_orthogonal
  | succ _ => exact rotX_orthogonal

/-- **C2's H_c carrier is a `PedagogicalVSA`.** Membership, not resemblance:
    every theorem stated generically over the specification — `chain_exact_unbind`
    above all — now applies to the O(3) trajectory by instantiation. -/
noncomputable instance o3VSA : PedagogicalVSA (Matrix (Fin 3) (Fin 3) ℝ) :=
  ofOrthogonalFamily o3Gen o3Gen_orthogonal
    (order_of_matrix_witness o3Gen 0 1 (by
      show rotZ * rotX ≠ rotX * rotZ
      exact rotZ_rotX_non_commutative))

-- ---------------------------------------------------------------------------
-- C3: the block-diagonal GHRR encoder
-- ---------------------------------------------------------------------------

variable {o : Type} [Fintype o] [DecidableEq o]

/-- A block-diagonal matrix whose blocks are all orthogonal is orthogonal. This
    is the only algebraic fact the block-diagonal encoder needs that a general
    orthogonal family does not already supply. -/
theorem blockDiagonal_orthogonal (M : o → Matrix n n ℝ)
    (h : ∀ k, (M k)ᵀ * M k = 1) :
    (Matrix.blockDiagonal M)ᵀ * Matrix.blockDiagonal M = 1 := by
  rw [Matrix.blockDiagonal_transpose, ← Matrix.blockDiagonal_mul]
  have hfun : (fun k => (M k)ᵀ * M k) = fun _ : o => (1 : Matrix n n ℝ) := funext h
  have hone : (fun _ : o => (1 : Matrix n n ℝ)) = 1 := rfl
  rw [hfun, hone, Matrix.blockDiagonal_one]

/-- Block-diagonal matrices agreeing as matrices agree block by block. -/
theorem blockDiagonal_eq_at {A B : o → Matrix n n ℝ} (k : o)
    (h : Matrix.blockDiagonal A = Matrix.blockDiagonal B) : A k = B k := by
  ext a b
  have hab := congrFun (congrFun h (a, k)) (b, k)
  simpa [Matrix.blockDiagonal_apply_eq] using hab

/-- C3's generator family: the same O(2) pair replicated across every block,
    which is the shape `EduBindBlockDiag.random_vector` samples. -/
noncomputable def ghrrGen (o : Type) [Fintype o] [DecidableEq o] :
    Nat → Matrix (Fin 2 × o) (Fin 2 × o) ℝ :=
  fun i => Matrix.blockDiagonal (fun _ : o => eduGenReal i)

theorem ghrrGen_orthogonal :
    ∀ i, (ghrrGen o i)ᵀ * ghrrGen o i = 1 := by
  intro i
  refine blockDiagonal_orthogonal _ (fun _ => ?_)
  cases i with
  | zero   => exact EduBind.rot_orthogonal _
  | succ _ => exact EduBind.refl_orthogonal _

/-- The block-diagonal family inherits non-commutativity from a single block,
    provided there is at least one block. -/
theorem ghrrGen_non_commutative [Nonempty o] :
    ghrrGen o 0 * ghrrGen o 1 ≠ ghrrGen o 1 * ghrrGen o 0 := by
  intro h
  simp only [ghrrGen, ← Matrix.blockDiagonal_mul] at h
  obtain ⟨k⟩ := ‹Nonempty o›
  have hk := blockDiagonal_eq_at k h
  exact EduBind.rot_refl_non_commutative (by simpa [eduGenReal] using hk)

/-- **C3's block-diagonal encoder is a `PedagogicalVSA`,** at any block count. -/
noncomputable instance ghrrVSA [Nonempty o] :
    PedagogicalVSA (Matrix (Fin 2 × o) (Fin 2 × o) ℝ) :=
  ofOrthogonalFamily (ghrrGen o) ghrrGen_orthogonal
    (order_of_matrix_witness (ghrrGen o) 0 1 ghrrGen_non_commutative)

-- ---------------------------------------------------------------------------
-- The complex unitary case: EduBindComplexUnitary / C3's U(2) blocks
-- ---------------------------------------------------------------------------

/-! `ofOrthogonalFamily` covers real orthogonal matrices. `EduBindComplexUnitary`
(`src/eduhdc/operators.py`) is complex: it samples 2x2 blocks by Haar measure on
U(2) and unbinds with the conjugate transpose. Orthogonality and unitarity are
the same argument at two different involutions, so the construction below is
`ofOrthogonalFamily` with the transpose replaced by the conjugate transpose.

One correction to the implementation while we are here: that class's docstring
says SU(2), but its sampler draws four angles and the resulting block has
determinant `exp(i*chi)`, so it is U(2), not SU(2). We state and prove the
unitary case, which is what the code actually produces and is strictly the more
general claim. -/

/-- **Membership for complex unitary families.** Any family of unitary complex
    matrices, over any finite index type, failing to commute at one pair is a
    `PedagogicalVSA`: unbinding is left multiplication by the conjugate
    transpose. -/
noncomputable def ofUnitaryFamily (gen : Nat → Matrix n n ℂ)
    (huni : ∀ i, (gen i)ᴴ * gen i = 1)
    (hne : ∃ i j Y, gen i * (gen j * Y) ≠ gen j * (gen i * Y)) :
    PedagogicalVSA (Matrix n n ℂ) :=
  ofAction (matAction n ℂ) gen (fun X Y => X + Y) (fun i Y => (gen i)ᴴ * Y)
    (by
      intro i Y
      show (gen i)ᴴ * (gen i * Y) = Y
      rw [← Matrix.mul_assoc, huni i, Matrix.one_mul])
    (by
      intro i X Y
      exact Matrix.mul_add _ X Y)
    hne

/-- A diagonal U(2) element: `diag(i, -i)`. -/
noncomputable def uA : Matrix (Fin 2) (Fin 2) ℂ := !![Complex.I, 0; 0, -Complex.I]

/-- A real rotation seen inside U(2). -/
noncomputable def uB : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; -1, 0]

theorem uA_unitary : uAᴴ * uA = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [uA, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply, Complex.ext_iff]

theorem uB_unitary : uBᴴ * uB = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [uB, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply, Complex.ext_iff]

theorem uA_uB_non_commutative : uA * uB ≠ uB * uA := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  simp [uA, uB, Matrix.mul_apply, Fin.sum_univ_two, Complex.ext_iff] at h01
  norm_num at h01

/-- The unitary generator family: index 0 is `diag(i, -i)`, every other index is
    the rotation. Both unitary, and they do not commute. -/
noncomputable def u2Gen : Nat → Matrix (Fin 2) (Fin 2) ℂ
  | 0 => uA
  | _ => uB

theorem u2Gen_unitary : ∀ i, (u2Gen i)ᴴ * u2Gen i = 1 := by
  intro i
  cases i with
  | zero   => exact uA_unitary
  | succ _ => exact uB_unitary

/-- **`EduBindComplexUnitary`'s single-block carrier is a `PedagogicalVSA`.** -/
noncomputable instance u2VSA : PedagogicalVSA (Matrix (Fin 2) (Fin 2) ℂ) :=
  ofUnitaryFamily u2Gen u2Gen_unitary
    (order_of_matrix_witness u2Gen 0 1 uA_uB_non_commutative)

/-- A block-diagonal matrix whose blocks are all unitary is unitary. -/
theorem blockDiagonal_unitary (M : o → Matrix n n ℂ)
    (h : ∀ k, (M k)ᴴ * M k = 1) :
    (Matrix.blockDiagonal M)ᴴ * Matrix.blockDiagonal M = 1 := by
  rw [Matrix.blockDiagonal_conjTranspose, ← Matrix.blockDiagonal_mul]
  have hfun : (fun k => (M k)ᴴ * M k) = fun _ : o => (1 : Matrix n n ℂ) := funext h
  have hone : (fun _ : o => (1 : Matrix n n ℂ)) = 1 := rfl
  rw [hfun, hone, Matrix.blockDiagonal_one]

/-- Block-diagonal complex matrices agreeing as matrices agree block by block. -/
theorem blockDiagonal_eq_at_complex {A B : o → Matrix n n ℂ} (k : o)
    (h : Matrix.blockDiagonal A = Matrix.blockDiagonal B) : A k = B k := by
  ext a b
  have hab := congrFun (congrFun h (a, k)) (b, k)
  simpa [Matrix.blockDiagonal_apply_eq] using hab

/-- The block-diagonal unitary family that `EduBindComplexUnitary` actually
    samples: the same U(2) pair replicated across every block. -/
noncomputable def unitaryGen (o : Type) [Fintype o] [DecidableEq o] :
    Nat → Matrix (Fin 2 × o) (Fin 2 × o) ℂ :=
  fun i => Matrix.blockDiagonal (fun _ : o => u2Gen i)

theorem unitaryGen_unitary : ∀ i, (unitaryGen o i)ᴴ * unitaryGen o i = 1 := by
  intro i
  exact blockDiagonal_unitary _ (fun _ => u2Gen_unitary i)

theorem unitaryGen_non_commutative [Nonempty o] :
    unitaryGen o 0 * unitaryGen o 1 ≠ unitaryGen o 1 * unitaryGen o 0 := by
  intro h
  simp only [unitaryGen, ← Matrix.blockDiagonal_mul] at h
  obtain ⟨k⟩ := ‹Nonempty o›
  have hk := blockDiagonal_eq_at_complex k h
  exact uA_uB_non_commutative (by simpa [u2Gen] using hk)

/-- **`EduBindComplexUnitary` is a `PedagogicalVSA` at any block count.** The last
    of the three operators the mapping document listed as covered without proof. -/
noncomputable instance unitaryVSA [Nonempty o] :
    PedagogicalVSA (Matrix (Fin 2 × o) (Fin 2 × o) ℂ) :=
  ofUnitaryFamily (unitaryGen o) unitaryGen_unitary
    (order_of_matrix_witness (unitaryGen o) 0 1 unitaryGen_non_commutative)

-- ---------------------------------------------------------------------------
-- The inheritance, made literal rather than asserted
-- ---------------------------------------------------------------------------

/-! `chain_exact_unbind` lives in `src/eduhdc/ChainTransitivity.lean`, stated
against the KERNEL-tier `PedagogicalVSA`. This tier declares its own class of the
same name, so the kernel theorem does not literally transport across the package
boundary: saying "O(3) now inherits `chain_exact_unbind`" would be exactly the
resemblance-for-inheritance substitution this file exists to remove. We therefore
restate the roundtrip here and prove it against this tier's class, then discharge
it at both carriers. The proof is the kernel one verbatim; only the class it is
stated against differs. -/

/-- Composition-based roundtrip: bind forward along a chain of relations, unbind
    backward. Mirrors `PedagogicalVSA.chainRoundtrip` in the kernel tier. -/
def chainRoundtrip {V : Type} (P : PedagogicalVSA V) : List Nat → V → V
  | [],      Y => Y
  | i :: is, Y => P.inv i (chainRoundtrip P is (P.ops i Y))

/-- The chain guarantee, generic over this tier's specification: composition
    recovers every chain exactly, at every length, superposing nothing. -/
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

/-- **C2 pays off:** the O(3) trajectory carrier recovers every chain exactly, at
    every length. Previously this was asserted in
    `docs/c1_to_c2c3c4_complete_mapping.md` on the grounds that the kernel theorem
    is "generic in `PedagogicalVSA`" -- true, but O(3) had never been shown to BE
    a `PedagogicalVSA`. Now it has. -/
theorem o3_chain_exact_unbind (is : List Nat) (Y : Matrix (Fin 3) (Fin 3) ℝ) :
    chainRoundtrip o3VSA is Y = Y :=
  chain_exact_unbind o3VSA is Y

/-- **C3 pays off:** the block-diagonal encoder recovers every chain exactly, at
    every length and any block count. -/
theorem ghrr_chain_exact_unbind [Nonempty o] (is : List Nat)
    (Y : Matrix (Fin 2 × o) (Fin 2 × o) ℝ) :
    chainRoundtrip ghrrVSA is Y = Y :=
  chain_exact_unbind ghrrVSA is Y

/-- **The unitary carrier pays off too**, at any block count. -/
theorem unitary_chain_exact_unbind [Nonempty o] (is : List Nat)
    (Y : Matrix (Fin 2 × o) (Fin 2 × o) ℂ) :
    chainRoundtrip unitaryVSA is Y = Y :=
  chain_exact_unbind unitaryVSA is Y

-- ---------------------------------------------------------------------------
-- Axiom footprint (Mathlib tier: `Classical.choice` is expected and disclosed)
-- ---------------------------------------------------------------------------

#print axioms ofOrthogonalFamily
#print axioms o3VSA
#print axioms ghrrVSA
#print axioms o3_chain_exact_unbind
#print axioms ghrr_chain_exact_unbind
#print axioms ofUnitaryFamily
#print axioms unitaryVSA
#print axioms unitary_chain_exact_unbind

end EduHDC
