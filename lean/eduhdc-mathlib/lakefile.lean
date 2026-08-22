import Lake
open Lake DSL

-- FW4: general-rotation formalization requires Mathlib. This is a SEPARATE lake
-- project so that the core project (src/eduhdc) stays Mathlib-free and builds
-- without network. Mathlib is pinned to v4.33.0 to match lean-toolchain.
--
-- To build on a server with Mathlib cache access (recommended):
--   cd src/eduhdc_mathlib && lake exe cache get && lake build
-- Without cache access, Mathlib is compiled from source (long).

package «eduhdc_mathlib» where
  require mathlib from git
    "https://github.com/leanprover-community/mathlib4.git" @ "v4.33.0"

@[default_target]
lean_lib «EduHDCMathlib» where
  srcDir := "."
  globs := #[.one `EduBindBlockDiag, .one `Basic]