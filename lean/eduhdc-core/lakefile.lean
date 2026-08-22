import Lake
open Lake DSL

-- C1 EduHDC formal verification project.
--
-- Default target builds `EduBindSelfContained`, which is Mathlib-free and is
-- checked directly by the Lean 4 kernel (`lake build` or `lean EduBindSelfContained.lean`).
--
-- The richer Mathlib-based formalizations (`Basic.lean`, `EduBindBlockDiag.lean`)
-- live in the sibling project `src/eduhdc_mathlib` (which pins Mathlib v4.33.0);
-- they are NOT in this project's build target, which stays Mathlib-free and
-- builds without network.

package «eduhdc» where

-- require mathlib from git
--   "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «EduHDC» where
  srcDir := "."
  globs := #[.one `EduBindSelfContained, .one `VSATriad, .one `ChainTransitivity,
             .one `CapacityCostModel]
