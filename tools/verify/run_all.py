"""Run every C1 pre-submission check and exit non-zero if any of them fails.

Usage:  python tools/verify/run_all.py

What it covers, and why each check exists (all three were written after a real
defect slipped through a review round):

  check_listings.py       Every Lean listing printed in the paper must be
                          verbatim from a file that is in a lake build target.
                          A Revision-2 listing was quoted from a duplicate file
                          that was in no build target and did not compile.
                          It ALSO checks that every Lean-looking identifier in
                          \\texttt{} anywhere in the paper (prose and theorem
                          headers, not just listings) exists in some .lean file:
                          Revision 5 cited four names that did not exist while
                          the listing check reported zero mismatches. Both
                          verdicts are required to pass.

  verify_paper_numbers.py Every quantitative claim in the paper must match the
                          JSON that produced it. Catches numbers left behind
                          when an experiment is re-run.

  consistency_scan.py     No section may contradict another. Patching prose
                          section by section twice produced contradictions
                          between the body, the Limitations and the Conclusion.

  check_lean_regression.py Actually compiles the Revision-2 attack replay
                          (tools/verify/lean_regression/AttackR3_regression_
                          rev2_attack_now_fails.lean) instead of leaving the
                          "attack no longer type-checks" claim as a comment
                          nobody runs. See that script's docstring: the real
                          assertion is "compiles only via a `sorry` warning",
                          not "fails to compile" (a lone `sorry` always
                          compiles). Skips itself (exit 0, printing a notice)
                          if no Lean toolchain is on PATH.

Not covered here (run manually): `lake build` in both Lean projects, and a
pdflatex/bibtex/pdflatex/pdflatex cycle with zero errors and zero undefined
references.
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CHECKS = [
    ("Lean listings verbatim from built files", "check_listings.py",
     ("TONG DONG LECH: 0", "TONG DINH DANH THIEU: 0")),
    ("paper numbers match result JSONs", "verify_paper_numbers.py", "0 MISMATCH"),
    # Revision 4: the scan's known false positive (a sentence that explicitly
    # NEGATES the searched pattern) is now handled inside consistency_scan.py by
    # a negation guard, so the expected count is ZERO. It was previously "1
    # issue(s)", which was a real hole: once the false positive was fixed, a
    # genuine contradiction taking its place would still have been reported as
    # a PASS.
    ("no cross-section contradiction", "consistency_scan.py", "TONG VAN DE: 0"),
    ("Revision-2 attack replay still blocked", "check_lean_regression.py", "REGRESSION OK"),
]

failed = []
for label, script, expect in CHECKS:
    out = subprocess.run([sys.executable, str(HERE / script)],
                         capture_output=True, text=True,
                         stdin=subprocess.DEVNULL)
    body = out.stdout + out.stderr
    # `expect` may be a single sentinel or a tuple of sentinels that must ALL
    # appear. check_listings.py emits two independent verdicts (listing lines,
    # prose identifiers); accepting only the first would let the second fail
    # silently, which is the exact failure mode this suite exists to prevent.
    expects = expect if isinstance(expect, tuple) else (expect,)
    ok = all(e in body for e in expects)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    # Surface deliberately-skipped check groups. A check gated on paper text
    # (e.g. a table cut in a later revision) silently becomes dead code that
    # still reports PASS; printing the skips here is what makes that visible.
    for line in body.splitlines():
        if "SKIPPED" in line or line.strip().startswith("- ") and "not quoted in" in line:
            print(f"       {line.strip()}")
        elif line.strip().startswith("- ") and "absent from" in line:
            print(f"       {line.strip()}")
    if not ok:
        failed.append((label, body.strip()))

print()
if failed:
    for label, body in failed:
        print(f"--- {label} ---")
        print(body)
        print()
    print(f"{len(failed)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
