"""
E3 — Functional form of the capacity curve: linear-in-sqrt(D/T) vs logistic.

MODEL SELECTION, not fit maximisation (audit fix m8). The retrieval-accuracy
curve saturates near 1 at low load and near chance at high load, so a model
linear in z = sqrt(D/T) is misspecified at both ends; a logistic in the same z,
  acc = c + (d - c) / (1 + exp(-a*(z - b))),
respects the saturation. We report BOTH R^2 values and let the comparison stand,
rather than reporting whichever is larger. The earlier revision of this script
stated a target R^2 and printed a "confirmed" banner when it was met; that is
the wrong direction of inference and has been removed.

Reads existing capacity_sweep_results.json (no recompute needed).
"""

import json
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "data" / "results"


def load_points():
    with open(RESULTS / "capacity_sweep_results.json") as f:
        data = json.load(f)
    pts = []  # (load=T/D, acc)
    for op, blk in data["sweep"].items():
        acc = blk["acc"]
        for key, val in acc.items():
            # key format T{t}_D{d}
            t = int(key.split("_D")[0][1:])
            d = int(key.split("_D")[1])
            pts.append((t / d, val))
    return np.array(pts)


def fit_linear(load, acc):
    z = 1.0 / np.sqrt(load + 1e-12)
    A = np.vstack([z, np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
    pred = A @ coef
    return pred, r2(acc, pred)


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def fit_logistic(load, acc):
    """acc = c + (d-c)/(1+exp(-a*(z-b))), z=sqrt(D/T). Grid+least-squares via numpy."""
    z = 1.0 / np.sqrt(load + 1e-12)
    best = None
    # coarse grid over (a,b), then solve c,d linearly
    for a in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]:
        for b in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            s = 1.0 / (1.0 + np.exp(-a * (z - b)))
            A = np.vstack([s, np.ones_like(s)]).T  # acc = (d-c)*s + c
            coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
            pred = A @ coef
            rr = r2(acc, pred)
            if best is None or rr > best[0]:
                best = (rr, a, b, coef)
    return best


def main():
    pts = load_points()
    load, acc = pts[:, 0], pts[:, 1]
    print(f"Points: {len(load)}")

    _, r2_lin = fit_linear(load, acc)
    print(f"Linear  acc ~ a*sqrt(D/T)+b      : R^2 = {r2_lin:.4f}")

    rr, a, b, coef = fit_logistic(load, acc)
    print(f"Logistic acc ~ c+(d-c)*sig(a*(z-b)): R^2 = {rr:.4f}  (a={a}, b={b}, c={coef[1]:.3f}, d={coef[0]+coef[1]:.3f})")

    print("")
    print("Delta R^2 (logistic - linear): %+.4f" % (rr - r2_lin))
    better = "logistic" if rr > r2_lin else "linear"
    print(f"-> Better-fitting functional form on this sweep: {better} in z = sqrt(D/T).")
    print("   Both values are reported; neither is a target that was aimed at.")

    out = {"n_points": int(len(load)), "r2_linear": float(r2_lin),
           "r2_logistic": float(rr), "logistic_a": float(a), "logistic_b": float(b),
           "logistic_c": float(coef[1]), "logistic_d": float(coef[0] + coef[1])}
    with open(RESULTS / "capacity_fit_improved.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved: {RESULTS}\\capacity_fit_improved.json]")


if __name__ == "__main__":
    main()
