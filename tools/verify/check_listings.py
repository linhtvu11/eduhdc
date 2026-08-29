import io, pathlib, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]
import os
PAPER = os.environ.get("C1_PAPER", "main_r5.tex")
tex = io.open(ROOT / "docs" / "arxiv_c1" / PAPER, encoding="utf-8").read()
# Listings live between the first section that introduces the specification and
# the Experiments section. Revision 5 dropped the standalone "Specification"
# section (the structure moved into Preliminaries), so try each entry point in
# order rather than assuming one exists.
_CANDIDATES = ("\\section{Preliminaries}",
               "\\section{Specification}",
               "\\subsection{The \\texttt{PedagogicalVSA} Specification}")
_START = next((c for c in _CANDIDATES if c in tex), None)
if _START is None:
    raise SystemExit(f"[check_listings] {PAPER}: no known specification-section "
                     f"marker found (tried: {', '.join(_CANDIDATES)}). "
                     "TONG DONG LECH: unknown -- refusing to report a pass.")
sec = tex[tex.index(_START):tex.index("\\section{Experiments}")]
blocks = re.findall(r"\\begin\{lstlisting\}\n(.*?)\\end\{lstlisting\}", sec, re.S)

files = [str(ROOT / "src" / "eduhdc" / "EduBindSelfContained.lean"),
         str(ROOT / "src" / "eduhdc" / "VSATriad.lean"),
         str(ROOT / "src" / "eduhdc" / "ChainTransitivity.lean"),
         str(ROOT / "src" / "eduhdc" / "CapacityCostModel.lean"),
         str(ROOT / "src" / "eduhdc" / "SpecStrengthening.lean"),
         str(ROOT / "src" / "eduhdc" / "EncPairSpec.lean"),
         str(ROOT / "src" / "eduhdc" / "ChainCrosstalk.lean"),
         str(ROOT / "src" / "eduhdc" / "GroupActionSpec.lean"),
         str(ROOT / "src" / "eduhdc" / "MonoidRelaxation.lean"),
         str(ROOT / "src" / "eduhdc" / "ChainOrder.lean"),
         str(ROOT / "src" / "eduhdc" / "DihedralLabel.lean"),
         str(ROOT / "src" / "eduhdc_mathlib" / "Basic.lean"),
         str(ROOT / "src" / "eduhdc_mathlib" / "EduBindBlockDiag.lean")]
src = "".join(io.open(f, encoding="utf-8").read() for f in files)
srcn = re.sub(r"[ \t]+", " ", src)

T = [("forall", "\u2200"), ("exists", "\u2203"), (" != ", " \u2260 "),
     ("->", "\u2192"), ("not (", "\u00ac ("), ("Real", "\u211d"),
     ("<i, j, Y, hne>", "\u27e8i, j, Y, hne\u27e9"), ("<Y, hY>", "\u27e8Y, hY\u27e9"),
     ("<[i, j], [j, i], Y, rfl, ?_>",
      "\u27e8[i, j], [j, i], Y, rfl, ?_\u27e9"),
     ("<=", "\u2264"), ("/\\\\", "\u2227"),
     # Revision 7: anonymous-constructor patterns in DihedralLabel.lean.
     # Listed explicitly rather than by a general <..> rewrite, because a
     # general rule would weaken the check.
     ("| <a, false>, <c, d>     => <R4.add a c, d>",
      "| ⟨a, false⟩, ⟨c, d⟩     => ⟨R4.add a c, d⟩"),
     ("| <a, true>,  <c, false> => <R4.sub a c, true>",
      "| ⟨a, true⟩,  ⟨c, false⟩ => ⟨R4.sub a c, true⟩"),
     ("| <a, true>,  <c, true>  => <R4.sub a c, false>",
      "| ⟨a, true⟩,  ⟨c, true⟩  => ⟨R4.sub a c, false⟩"),
     ("<dofPAM_exceeds_dof1D_same_param d hd,",
      "\u27e8dofPAM_exceeds_dof1D_same_param d hd,"),
     ("dof1D_exceeds_dofPAM_matched_storage d (by omega)>",
      "dof1D_exceeds_dofPAM_matched_storage d (by omega)\u27e9")]

print("%d listing kiem tra\n" % len(blocks))
bad = 0
for bi, b in enumerate(blocks, 1):
    lines = [l for l in b.split("\n") if l.strip() and not l.strip().startswith("--")]
    miss = []
    for l in lines:
        c = l.split(" --")[0].rstrip()
        for a, z in T:
            c = c.replace(a, z)
        c = c.replace("\u211d.cos", "Real.cos").replace("\u211d.sin", "Real.sin")
        c = re.sub(r"[ \t]+", " ", c).strip()
        if c and c not in srcn:
            miss.append((l.strip(), c))
    tag = "" if not miss else "   <<< LECH"
    print("  listing %d: %d/%d dong khop ma da build%s" % (bi, len(lines) - len(miss), len(lines), tag))
    for orig, conv in miss[:5]:
        print("        paper : %s" % orig)
        print("        -> tim: %s" % conv)
    bad += len(miss)
print("\nTONG DONG LECH: %d" % bad)

# ---------------------------------------------------------------------------
# Prose identifier existence check.
#
# WHY THIS EXISTS. Everything above only inspects text INSIDE \begin{lstlisting}.
# Lean names cited in PROSE, or in a \begin{theorem}[...] header, were checked by
# nothing at all. Revision 5 shipped four such names that did not exist in any
# Lean file -- `abelian_chain_order_blind`, `chainApply_swap_of_abelian`, a
# `chain_order_sensitive` attributed to `PedagogicalMonoid` instead of
# `PedagogicalVSA`, and a claim that both halves of that theorem live over the
# invertibility-free weakening -- while this script reported TONG DONG LECH: 0.
# The listing check cannot catch those: a wrong name in prose is not a listing.
# ---------------------------------------------------------------------------

# Names that look like Lean identifiers but legitimately are not. Keep this list
# short and justified: every entry is a hole in the check.
_NOT_LEAN = {
    "Module",         # LaTeX/Lean-agnostic prose word in a figure caption
    "encode_relation",  # Python method of EduHDC_PrereqProbe, not a Lean name
}

_lean_files = sorted((ROOT / "src" / "eduhdc").glob("*.lean")) + \
              sorted((ROOT / "src" / "eduhdc_mathlib").glob("*.lean"))
lean_src = "".join(io.open(f, encoding="utf-8").read() for f in _lean_files)

_ids = set()
for m in re.finditer(r"\\texttt\{([^{}]*)\}", tex):
    s = m.group(1).replace("\\_", "_").replace("\\allowbreak", "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.']*", s):
        continue          # paths, expressions, anything with braces or slashes
    if "_" not in s and not s[0].isupper():
        continue          # ordinary lowercase words in \texttt
    if s.endswith((".py", ".lean", ".tex")) or s in _NOT_LEAN:
        continue
    _ids.add(s)

_missing = sorted(i for i in _ids if i not in lean_src)
print("\n%d dinh danh \\texttt{} kiem tra ton tai trong %d file Lean"
      % (len(_ids), len(_lean_files)))
for i in _missing:
    print("        KHONG TON TAI: %s" % i)
print("TONG DINH DANH THIEU: %d" % len(_missing))

if bad or _missing:
    raise SystemExit(1)
