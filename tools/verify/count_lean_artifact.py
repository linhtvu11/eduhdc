"""Count the Lean artifact MECHANICALLY, so the paper's artifact table is a
measurement rather than a string that once was true.

WHY THIS EXISTS
---------------
`verify_paper_numbers.py` checked the artifact-table figures with `intex("103")`
-- i.e. "the string 103 appears somewhere in the .tex". That is a presence
check, not a value check: it passes for as long as the number is *written*,
whether or not it is *right*, and it keeps passing after a file is added. The
same class of hole let `abelian_chain_order_blind` -- a theorem name present in
no Lean file at all -- survive several verification rounds.

COUNTING RULES (stated so they are reproducible, not merely automated)
  theorems : lines whose first non-space token is `theorem` or `lemma`.
             `def`, `structure`, `instance` and `example` are NOT counted.
  lines    : non-blank, non-comment source lines. Line comments (`--`) and
             block comments (`/- ... -/`, including `/-- ... -/` docstrings)
             are stripped first. `lakefile.lean` is excluded from both tiers.

Usage:  python tools/verify/count_lean_artifact.py
Output: data/results/lean_artifact_counts.json
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TIERS = {
    "kernel": ROOT / "src" / "eduhdc",
    "mathlib": ROOT / "src" / "eduhdc_mathlib",
}
THEOREM_RE = re.compile(r"^\s*(theorem|lemma)\s")


def strip_comments(text: str) -> str:
    """Remove block comments (nested-aware) then line comments."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("/-", i):
            depth += 1
            i += 2
        elif text.startswith("-/", i) and depth:
            depth -= 1
            i += 2
        elif depth:
            i += 1
        else:
            out.append(text[i])
            i += 1
    src = "".join(out)
    return "\n".join(line.split("--")[0] for line in src.splitlines())


def count_file(path: Path):
    raw = path.read_text(encoding="utf-8")
    thms = sum(1 for ln in raw.splitlines() if THEOREM_RE.match(ln))
    lines = sum(1 for ln in strip_comments(raw).splitlines() if ln.strip())
    return thms, lines


def main():
    payload, grand_t, grand_f, grand_l = {}, 0, 0, 0
    for tier, d in TIERS.items():
        files = sorted(p for p in d.glob("*.lean") if p.name != "lakefile.lean")
        per_file, t_tot, l_tot = {}, 0, 0
        for p in files:
            t, l = count_file(p)
            per_file[p.name] = {"theorems": t, "lines": l}
            t_tot += t
            l_tot += l
        payload[tier] = {"files": len(files), "theorems": t_tot,
                         "lines": l_tot, "per_file": per_file}
        grand_t += t_tot
        grand_f += len(files)
        grand_l += l_tot
        print(f"{tier:8s}: {len(files):3d} files  {t_tot:4d} theorems  {l_tot:5d} lines")
        for name, v in per_file.items():
            print(f"           {name:32s} {v['theorems']:3d} thm  {v['lines']:4d} lines")
    payload["total"] = {"files": grand_f, "theorems": grand_t, "lines": grand_l}
    print(f"{'TOTAL':8s}: {grand_f:3d} files  {grand_t:4d} theorems  {grand_l:5d} lines")

    out = ROOT / "data" / "results" / "lean_artifact_counts.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved: {out}]")


if __name__ == "__main__":
    main()
