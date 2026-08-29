"""
make_figures.py — regenerate the C1 paper figures as vector PDFs.

Sources (read-only):
  fig1: redrawn here (no external source file existed)
  fig2: decuong/data/results/transitivity_exact_check_results.json
  fig3: decuong/data/results/prereq_probe_controls_results.json

Outputs:
  figures/fig1_architecture.pdf, figures/fig_chain_error.pdf,
  figures/fig_inductive_accuracy.pdf   (vector, for main.tex)
  _fig_preview/*.png                    (raster previews for eyeballing)

Run:  python make_figures.py   (from anywhere)
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

BASE = pathlib.Path(__file__).resolve().parent
FIG = BASE / "figures"
PREVIEW = BASE / "_fig_preview"
FIG.mkdir(exist_ok=True)
PREVIEW.mkdir(exist_ok=True)

EXACT_JSON = BASE.parent.parent / "data" / "results" / "transitivity_exact_check_results.json"
CONTROLS_JSON = (BASE.parent.parent / "data" / "results"
                 / "prereq_probe_controls_results.json")
SEPARATION_JSON = (BASE.parent.parent / "data" / "results"
                   / "chain_length_separation_results.json")
REVERSAL_JSON = (BASE.parent.parent / "data" / "results"
                 / "chain_reversal_direct_results.json")

# ---- unified style: serif to match LaTeX, Okabe-Ito palette, light chrome ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

BLUE, BLUE_FILL = "#0072B2", "#E8F1F8"
ORANGE, PURPLE = "#E69F00", "#CC79A7"   # Okabe-Ito, completing the arm palette
GREEN, GREEN_FILL = "#009E73", "#E6F4EE"
VERM, VERM_FILL = "#D55E00", "#FBECE3"
INK, EDGE = "#222222", "#444444"

# One colour per ARM, fixed across every figure in the paper. Guide 9.1/9.4:
# a method must not change colour between figures. Encodings (composition
# vs role-filler) are NOT arms, so they are distinguished by linestyle on a
# single neutral colour rather than by borrowing an arm colour.
ARM_COLOUR = {
    "map": BLUE,
    "hrr": VERM,
    "edubind": GREEN,     # continuous O(2), what the code samples
    "edubind2": ORANGE,   # the two generators the kernel instance uses
    "d4": PURPLE,         # all eight elements
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf (+ preview)")



# ---------------------------------------------------------------- fig direction
def fig_direction():
    """Direction signal against chain length, for the five runtime families.

    This is the paper's only categorical separation, and the shape carries it:
    the commutative arms sit flat on zero at every length (no direction
    information exists to read), the two-generator family alternates between
    zero at odd lengths and full separation at even ones, and the families that
    expose a whole non-abelian label algebra never drop. A table of four
    lengths cannot show an alternation."""
    with open(REVERSAL_JSON) as f:
        d = json.load(f)
    syn = d["synthetic"]
    ns = [int(n) for n in d["config"]["n_list"]]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 2.5))

    # MAP and HRR coincide at zero everywhere; both are drawn, HRR dashed on
    # top of MAP, so the overlap is visible rather than one arm being dropped.
    series = [
        ("MAP (commutative)", "map", ARM_COLOUR["map"], "-", 1.6, "o"),
        ("HRR (commutative)", "hrr", ARM_COLOUR["hrr"], (0, (4, 3)), 1.3, None),
        ("two generators of eight", "edubind2", ARM_COLOUR["edubind2"], "-", 1.4, "s"),
        ("all eight, dihedral labels", "d4", ARM_COLOUR["d4"], "-", 1.3, "^"),
        ("continuous $O(2)$", "edubind", ARM_COLOUR["edubind"], (0, (1, 1.6)), 1.3, None),
    ]
    for label, key, col, ls, lw, mk in series:
        y = [1.0 - syn[key][str(n)]["cos_fwd_rev"] for n in ns]
        axa.plot(ns, y, color=col, linestyle=ls, lw=lw, marker=mk, ms=3.1,
                 markerfacecolor="white", markeredgewidth=0.9)
        yp = [1.0 - syn[key][str(n)]["cos_fwd_perm"] for n in ns]
        axb.plot(ns, yp, color=col, linestyle=ls, lw=lw, marker=mk, ms=3.1,
                 markerfacecolor="white", markeredgewidth=0.9, label=label)

    for ax, title in ((axa, "(a) reversal"), (axb, "(b) random permutation")):
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 4, 8, 16, 41, 64])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("chain length $n$")
        ax.set_ylim(-0.10, 1.22)
        ax.axhline(0.0, color=EDGE, lw=0.6, ls=":")
        ax.set_title(title, loc="left", pad=3)
        ax.grid(True, which="major", axis="y", lw=0.4, alpha=0.35)
    axa.set_ylabel(r"direction signal $\Delta = 1 - \cos$")
    axa.annotate("odd $n$: no direction signal", xy=(7, 0.02), xytext=(9.5, 0.30),
                 fontsize=7.0, color=VERM, ha="left",
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8,
                                 connectionstyle="arc3,rad=-0.2"))
    axb.legend(loc="lower right", bbox_to_anchor=(1.005, 0.10), framealpha=0.9,
               handlelength=1.9, borderaxespad=0.0, labelspacing=0.35)
    save(fig, "fig_direction")


# ---------------------------------------------------------------- fig 2
def fig_chain_error():
    d = json.loads(EXACT_JSON.read_text(encoding="utf-8"))
    dist = d["hop_length_distribution"]
    fl = d["runtime_float"]["per_length"]
    hops = sorted(int(k) for k in dist)
    counts = [dist[str(h)] for h in hops]
    ehops = sorted(int(k) for k in fl)
    errs = [fl[str(h)]["max_abs_err"] for h in ehops]
    total = d["total_transitive_pairs"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.55))
    a.bar(hops, counts, width=0.82, color=BLUE, edgecolor="white", linewidth=0.4)
    a.set_yscale("log")
    a.set_xlim(1, 42)
    a.set_xlabel("Chain length (hops)")
    a.set_ylabel("# transitive pairs")
    a.set_title(f"(a) Hop-length distribution (N = {total:,} pairs)", loc="left")

    b.plot(ehops, errs, "-o", color=VERM, linewidth=1.0, markersize=2.1)
    b.set_yscale("log")
    b.set_xlabel("Chain length (hops)")
    b.set_ylabel("Max abs. float32 error")
    b.set_title(f"(b) Runtime unbind error vs. chain length (max = {max(errs):.1e})",
                loc="left")
    b.grid(True, which="major", linestyle=":", linewidth=0.5, color="#bbbbbb")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_chain_error")


# ---------------------------------------------------------------- fig 3
def fig_inductive():
    """Inductive direction accuracy: EduBind against the non-VSA concat-MLP control.

    Reads the controls run (10 seeds, node split redrawn inside the seed loop) --
    the run Section 4.3 reports.
    """
    d = json.loads(CONTROLS_JSON.read_text(encoding="utf-8"))
    ind = d["inductive"]
    keys = [("hop2-3", "2-3 hops"), ("hop4-6", "4-6 hops"), ("hop7+", "7+ hops")]
    arms = [("edubind", "EduBind (verified)", GREEN),
            ("concat", "concat-MLP (non-VSA control)", BLUE)]

    labels = []
    for k, lab in keys:
        n = int(round(ind["edubind"][k]["n_pairs_mean"]))
        labels.append(lab + chr(10) + "(n" + chr(8776) + str(n) + ")")

    series = {}
    for arm, _, _ in arms:
        acc = [ind[arm][k]["dir_acc_strict"] for k, _ in keys]
        lo = [ind[arm][k]["dir_acc_strict"] - ind[arm][k]["bootstrap_ci95_strict"][0]
              for k, _ in keys]
        hi = [ind[arm][k]["bootstrap_ci95_strict"][1] - ind[arm][k]["dir_acc_strict"]
              for k, _ in keys]
        series[arm] = (acc, lo, hi)

    fig, ax = plt.subplots(figsize=(4.5, 2.85))
    x = np.arange(len(keys))
    w = 0.36
    for i, (arm, lab, col) in enumerate(arms):
        acc, lo, hi = series[arm]
        ax.bar(x + (i - 0.5) * w, acc, width=w, color=col, edgecolor="none", label=lab,
               yerr=[lo, hi], capsize=2.5, error_kw={"linewidth": 0.8, "color": INK})
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="#888888", label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Direction accuracy (inductive, 10 seeds)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6.4)
    fig.tight_layout()
    save(fig, "fig_inductive_accuracy")


# ---------------------------------------------------------------- fig 4 (E2)
def fig_separation():
    """Role-filler vs composition retrieval accuracy as a function of chain
    length n, matched D/K/codebook (chain_length_separation.py). Panel (a):
    EduBind head-to-head -- composition flat at 1.0, role-filler degrades.
    Panel (b): role-filler degradation is operator-independent (MAP, HRR,
    EduBind collapse together), the same message as the T/D capacity sweep,
    now indexed by chain length n instead of an abstract load T.
    """
    d = json.loads(SEPARATION_JSON.read_text(encoding="utf-8"))
    n_list = d["config"]["n_list"]
    rf = d["role_filler_accuracy"]
    co = d["composition_accuracy"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.75))

    a.plot(n_list, [co["edubind"][str(n)] for n in n_list], "-o", color=INK,
           linewidth=1.3, markersize=3.0, label="composition (chainRoundtrip)")
    a.plot(n_list, [rf["edubind"][str(n)] for n in n_list], "--s", color=INK,
           linewidth=1.3, markersize=3.0, markerfacecolor="white",
           label="role-filler (encChainRF)")
    a.axvline(41, color="#999999", linestyle=":", linewidth=0.8)
    a.annotate("max observed\nhop (41)", xy=(41, 1.0), xytext=(46, 0.55),
               fontsize=6.2, color="#777777", ha="left",
               arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.6))
    a.set_xscale("log")
    a.set_xticks(n_list)
    a.set_xticklabels([str(n) for n in n_list], rotation=45, fontsize=6.2)
    a.set_xlabel("Chain length n")
    a.set_ylabel("Top-1 retrieval accuracy")
    a.set_ylim(-0.03, 1.08)
    a.set_title("(a) verified $O(2)$: composition vs. role-filler", loc="left")
    a.legend(loc="lower left", framealpha=0.9, fontsize=6.6)

    colors = {k: ARM_COLOUR[k] for k in ("edubind", "map", "hrr")}
    for op, col in colors.items():
        b.plot(n_list, [rf[op][str(n)] for n in n_list], "-o", color=col,
               linewidth=1.1, markersize=2.6, label={"edubind": "continuous $O(2)$", "map": "MAP", "hrr": "HRR"}[op])
    b.set_xscale("log")
    b.set_xticks(n_list)
    b.set_xticklabels([str(n) for n in n_list], rotation=45, fontsize=6.2)
    b.set_xlabel("Chain length n (= superposition load T)")
    b.set_ylabel("Top-1 retrieval accuracy")
    b.set_ylim(-0.03, 1.08)
    b.set_title("(b) Role-filler collapse is operator-independent", loc="left")
    b.legend(loc="lower left", framealpha=0.9, fontsize=6.6)

    for ax in (a, b):
        ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="#bbbbbb")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_separation")


# ---------------------------------------------------------------- fig 1
def _box(ax, x, y, w, h, lines, edge, fill, bold=0, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    cx, cy = x + w / 2, y + h / 2
    n = len(lines)
    for i, t in enumerate(lines):
        weight = "bold" if i < bold else "normal"
        ax.text(cx, cy + (n / 2 - 0.5 - i) * 2.6, t, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=INK)


def _arrow(ax, p1, p2, color=EDGE, style="-|>", lw=1.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=1, shrinkB=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    _box(ax, 2, 26, 19, 11, ["Formal", "Specification", "(PedagogicalVSA,", "3 axioms)"],
         BLUE, BLUE_FILL, bold=3, fs=8.2)
    ax.text(11.5, 23.2, "Operator Verification", ha="center", va="center",
            fontsize=7.5, style="italic", color="#555555")
    _box(ax, 29, 32, 20, 7.5, ["EduBind — O(2)", "(verified, all 3 axioms)"], GREEN, GREEN_FILL)
    _box(ax, 29, 22.5, 20, 7.5, ["Perm — signed perms", "(verified, all 3 axioms)"], GREEN, GREEN_FILL)
    _box(ax, 29, 13, 20, 7.5, ["MAP  ·  rotations only", "(impossibility theorems)"], VERM, VERM_FILL)
    _box(ax, 57, 29.6, 23, 11.3, ["Chain-Unbinding", "Theorem", "chain_exact_unbind", "$\\forall$ chains"],
         BLUE, BLUE_FILL, bold=2, fs=7.6)
    _box(ax, 57, 19, 23, 8, ["Capacity Cost Model", "($O(D)$ vs $O(d^2)$)"], BLUE, BLUE_FILL)
    _box(ax, 29, 2.5, 51, 7.5,
         ["no_hadamard_PedagogicalVSA   ·   no_rotation_only_PedagogicalVSA",
          "($\\forall$x,y: bind x y = bind y x  $\\Rightarrow$  order unrecoverable)"],
         VERM, VERM_FILL, fs=6.9)
    _box(ax, 86, 12, 12, 26,
         ["Runtime &", "Empirical", "Validation", "", "Junyi DAG:", "56,224 pairs",
          "chains $\\leq$41 hops", "", "(algebraic /", "transductive /", "inductive)"],
         BLUE, BLUE_FILL, fs=7.3)

    for yb in (35.75, 26.25, 16.75):
        _arrow(ax, (21, 31.5), (29, yb))
    _arrow(ax, (49, 35.75), (57, 35.2))
    _arrow(ax, (49, 26.25), (57, 33.5))
    _arrow(ax, (68.5, 30.5), (68.5, 27), style="-", lw=1.0)
    _arrow(ax, (39, 13), (39, 10), color=VERM, lw=1.2)
    _arrow(ax, (80, 23), (86, 25))
    _arrow(ax, (80, 6.25), (86, 20))
    save(fig, "fig1_architecture")


if __name__ == "__main__":
    print("Regenerating C1 figures (vector PDF)...")
    fig_chain_error()
    fig_direction()
    fig_inductive()
    fig_architecture()
    fig_separation()
    print("DONE.")
