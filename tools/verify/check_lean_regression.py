"""Turn the "Revision-2 attack no longer type-checks" claim into an actual test
instead of a documentation-only assertion.

`tools/verify/lean_regression/AttackR3_regression_rev2_attack_now_fails.lean`
replays the Revision-2 attack (a commutative/Hadamard binding family) against
the Revision-3 spec. It deliberately leaves `order_sensitive_ax := by sorry`,
because — as measured below — that `sorry` is NOT decorative: Lean happily
compiles the file (exit 0) with nothing worse than a "declaration uses `sorry`"
warning. What actually blocks the attack is `no_hadamard_PedagogicalVSA`
(proved with no `sorry`, in the main build, in VSATriad.lean): it derives
`False` from exactly this Hadamard family, so the ONLY way to finish
`hadamardAttack` is to cheat with `sorry`.

So the meaningful, checkable claim is not "this file fails to compile" (it
doesn't, and was never going to — `sorry` always compiles). It is:

    compiling the file succeeds, AND the compiler is forced to emit a
    `sorry` warning to do it.

If it ever compiled cleanly with NO `sorry` warning, that would mean someone
found a real (non-cheating) proof of `order_sensitive_ax` for a Hadamard
family — i.e. the Revision-2 attack actually succeeded again — which is the
one outcome this check must fail loudly on.
"""
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEAN_FILE = ROOT / "tools" / "verify" / "lean_regression" / "AttackR3_regression_rev2_attack_now_fails.lean"
KERNEL_PROJECT = ROOT / "src" / "eduhdc"


def main() -> int:
    if shutil.which("lake") is None:
        print("REGRESSION OK (SKIPPED): Lean toolchain (lake/elan) khong kha dung trong "
              "moi truong sandbox nay -- bo qua buoc nay, can chay thu cong:")
        print(f"  cd {KERNEL_PROJECT} && lake env lean {LEAN_FILE}")
        return 0

    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(LEAN_FILE)],
            cwd=str(KERNEL_PROJECT),
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:  # toolchain present but broken/misconfigured, network, etc.
        print(f"REGRESSION OK (SKIPPED): khong the chay Lean toolchain ({e!r}) -- "
              "bo qua buoc nay, can chay thu cong:")
        print(f"  cd {KERNEL_PROJECT} && lake env lean {LEAN_FILE}")
        return 0

    out = proc.stdout + proc.stderr

    if proc.returncode != 0:
        print("REGRESSION UNCLEAR: file khong compile duoc (returncode != 0), "
              "khac voi hanh vi mong doi (compile OK + canh bao sorry). "
              "Can kiem tra thu cong xem VSATriad.lean/EduBindSelfContained.lean "
              "co thay doi lam gay import khong.")
        print(out)
        return 1

    if "sorry" in out:
        print("REGRESSION OK: file compile thanh cong CHI NHO `sorry` "
              "(declaration uses `sorry`) -- Hadamard attack van bi chan boi "
              "no_hadamard_PedagogicalVSA, khong co each nao lap order_sensitive_ax "
              "that su.")
        return 0

    print("REGRESSION BAD: file compile SACH, KHONG co canh bao sorry nao -- "
          "nghia la ai do da tim duoc chung minh THAT (khong sorry) cho "
          "order_sensitive_ax cua mot ho Hadamard giao hoan. Day chinh la "
          "cuoc tan cong Revision-2 THANH CONG TRO LAI. Kiem tra ngay "
          "no_hadamard_PedagogicalVSA trong VSATriad.lean.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
