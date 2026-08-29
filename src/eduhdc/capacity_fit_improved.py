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


def load_points_dt():
    """Same points, but keeping D and T separately so the EXPONENTS can be
    fitted instead of assumed."""
    with open(RESULTS / "capacity_sweep_results.json") as f:
        data = json.load(f)
    D, T, A = [], [], []
    for op, blk in data["sweep"].items():
        for key, val in blk["acc"].items():
            T.append(float(key.split("_D")[0][1:]))
            D.append(float(key.split("_D")[1]))
            A.append(val)
    return np.array(D), np.array(T), np.array(A)


def _probit(x):
    """Standard normal CDF, via erf -- the link Frady et al.'s derivation
    implies, since their result is a Gaussian-tail argument about the
    signal-to-noise ratio of a superposed trace."""
    from math import sqrt
    import numpy as _np
    try:
        from scipy.special import erf
    except Exception:                      # scipy-free fallback
        def erf(z):
            t = 1.0 / (1.0 + 0.3275911 * _np.abs(z))
            y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
                        * t - 0.284496736) * t + 0.254829592) * t * _np.exp(-z * z)
            return _np.sign(z) * y
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def fit_free_exponent(D, T, acc):
    """MEASURE the exponent instead of assuming it.

    The paper reports accuracy as a function of z = sqrt(D/T) and reads that as
    Frady et al.'s signal-to-noise ratio sqrt(N/M). But the earlier fit used
    z = sqrt(D/T) in BOTH the linear and the logistic model, so the comparison
    tested the functional FORM and never the EXPONENT: no value of R^2 there
    could distinguish sqrt(D/T) from D/T or from (D/T)^(1/4).

    Frady's derivation is a Gaussian-tail argument about the signal-to-noise
    ratio of a superposed trace, so the natural link is the PROBIT: it predicts
    that the probit of accuracy is LINEAR in the SNR. Working in probit space
    turns the exponent search into a sequence of ordinary least-squares fits,
    with no nonlinear optimizer and no scale parameter to confound it -- which
    is what the first attempt at this got wrong, by fitting the scale OUTSIDE
    the link and so measuring an artefact of a hardcoded inner scaling.

    Saturated cells carry no information about the exponent (a probit of 1.0 is
    infinite), so the fit is reported on the unsaturated points, with the count
    disclosed.

    Two questions, fitted separately:
      (1) which power?   probit(acc) ~ a * (D/T)^alpha + b, alpha free.
                         The SNR reading predicts alpha = 0.5.
      (2) really a ratio? probit(acc) ~ a * D^p / T^q + b, p and q free.
                         The ratio reading predicts p = q."""
    from scipy.stats import norm

    keep = (acc > 0.02) & (acc < 0.98)
    Dk, Tk, y = D[keep], T[keep], norm.ppf(acc[keep])

    def r2_at(z):
        A_ = np.vstack([z, np.ones_like(z)]).T
        coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
        return r2(y, A_ @ coef)

    alphas = np.arange(0.10, 1.51, 0.01)
    prof = np.array([r2_at((Dk / Tk) ** a) for a in alphas])
    best_alpha = float(alphas[int(prof.argmax())])

    grid = np.arange(0.10, 1.51, 0.02)
    best_pq, best_pq_r2 = None, -np.inf
    for pp in grid:
        for qq in grid:
            rr = r2_at((Dk ** pp) / (Tk ** qq))
            if rr > best_pq_r2:
                best_pq_r2, best_pq = rr, (float(pp), float(qq))

    return {
        "link": "probit",
        "n_points_unsaturated": int(keep.sum()),
        "n_points_total": int(len(acc)),
        "alpha_best": best_alpha,
        "alpha_r2_best": float(prof.max()),
        "alpha_r2_at_half": float(prof[int(np.argmin(np.abs(alphas - 0.5)))]),
        "alpha_profile": {f"{a:.2f}": float(r) for a, r in zip(alphas, prof)},
        "free_pq_best": {"p": best_pq[0], "q": best_pq[1], "r2": float(best_pq_r2)},
    }


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

    Dv, Tv, Av = load_points_dt()
    ex = fit_free_exponent(Dv, Tv, Av)
    print("")
    print("EXPONENT, measured rather than assumed (probit link):")
    print(f"  best alpha in acc ~ link(a*(D/T)^alpha + b) : {ex['alpha_best']:.2f} "
          f"(R^2 = {ex['alpha_r2_best']:.4f})")
    print(f"  R^2 at the assumed alpha = 0.50            : {ex['alpha_r2_at_half']:.4f}")
    print(f"  free exponents p, q in D^p / T^q            : p = {ex['free_pq_best']['p']:.2f}, "
          f"q = {ex['free_pq_best']['q']:.2f} (R^2 = {ex['free_pq_best']['r2']:.4f})")
    print("  -> p ~ q supports the RATIO reading; alpha ~ 0.5 supports the sqrt(D/T) reading.")

    out = {"n_points": int(len(load)), "r2_linear": float(r2_lin),
           "r2_logistic": float(rr), "logistic_a": float(a), "logistic_b": float(b),
           "logistic_c": float(coef[1]), "logistic_d": float(coef[0] + coef[1]),
           "exponent_study": ex}
    with open(RESULTS / "capacity_fit_improved.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved: {RESULTS}\\capacity_fit_improved.json]")


if __name__ == "__main__":
    main()
