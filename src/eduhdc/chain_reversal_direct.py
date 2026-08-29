"""C1 -- direct runtime measurement of the chain-ORDER theorems.

WHY THIS SCRIPT EXISTS (and what it replaces)
---------------------------------------------
`chain_order_discrimination.py` and `path_order_discrimination.py` both measure
"bind forward through a chain, then UNBIND in the wrong order". That is a valid
quantity, but it is NOT the quantity the theorems are about:

    abelian_chainAct_perm / abelian_chainAct_reverse   (ChainOrder.lean)
    chain_order_sensitive_general                      (ChainOrder.lean)
    d4_reverse_sensitive_general                       (DihedralLabel.lean)

are all stated over `chainAct` / `chainOps` -- FORWARD application of a relation
chain, in which `inv` never appears at all. Unbinding is a different operation
governed by a different axiom (Axiom 3), and mixing the two is what let the
following go unnoticed for a whole revision. This script measures the theorems'
own quantity:

    enc_fwd  = bind Y through the chain [r_1 .. r_n] in order
    enc_rev  = bind Y through the SAME relations in REVERSE order
    enc_perm = bind Y through a random PERMUTATION of the same relations

    report   cos(enc_fwd, enc_rev)  and  cos(enc_fwd, enc_perm)

A value of 1.0 means the encoding cannot tell the two traversals apart.

WHAT THE THEOREMS PREDICT, ARM BY ARM
-------------------------------------
  map, hrr          cos = 1.0 EXACTLY at every n, for reversal AND for every
                    permutation. Their label algebras are abelian, so
                    `abelian_chainAct_perm` applies. No readout can repair this.

  edubind2          The two-generator family {Rot, Ref} that `eduGen` actually
                    exposes -- the KERNEL-TIER VERIFIED INSTANCE.
                    `edubind_reverse_blind_at_length_three` proves it acts as
                    its own reverse at length 3, and exhaustive search finds the
                    same at every ODD length. So: cos = 1.0 at odd n, < 1 at
                    even n. This is a machine-checked prediction with a sharp,
                    falsifiable runtime signature.

  d4                All eight elements of the same group, i.e. `d4Gen`.
                    `d4_reverse_sensitive_general` proves a distinguishing chain
                    exists at EVERY n >= 2, so cos < 1 at every length --
                    including the odd lengths where edubind2 collapses.

  edubind           The CONTINUOUS O(2) family the implementation actually
                    samples (theta ~ U[0, 2pi), s in {-1,+1}). It is NOT the
                    kernel-tier instance; it is covered, if at all, by the
                    Mathlib-tier `ofOrthogonalFamily`. Reported alongside so the
                    gap between the verified family and the implemented one is
                    visible rather than implicit.

The edubind2-vs-d4 contrast is the point: BOTH satisfy Axiom 2, and only one of
them can tell a path from its reverse. Order sensitivity in the sense of Axiom 2
is therefore not what buys reversal discrimination; how much of the label
algebra the family exposes is.

Two regimes are measured: synthetic chains at n = 2..64, and REAL Junyi
prerequisite paths (the hop-length distribution comes from the curriculum DAG;
the relations are group elements, as they must be for the algebra to apply).

Usage:  python src/eduhdc/chain_reversal_direct.py
Output: data/results/chain_reversal_direct_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.capacity_sweep import _make_op, _stable_seed
from eduhdc.operators import EduBindBlockDiag

RESULTS_DIR = str(src_dir.parent / "data" / "results")

D = 2048
K = 2000
N_TRIALS = 30
N_PERM = 8      # distinct non-identity reorderings averaged per length
N_LIST = [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 41, 64]
MAX_PATHS_PER_STRATUM = 400


class _DiscreteO2(EduBindBlockDiag):
    """EduBind with each 2x2 block drawn from a FINITE subset of O(2), so the
    runtime family matches a Lean instance element for element.

    `EduBindBlockDiag.random_vector` builds each block as
        [[c, -s*sin], [sin, s*c]],  c = cos(theta), sin = sin(theta),
    so theta = pi/2, s = +1 gives Rot = [[0,-1],[1,0]] and theta = 0, s = -1
    gives Ref = [[1,0],[0,-1]]. Restricting (theta, s) to a finite set therefore
    reproduces a finite subgroup exactly, with no change to bind/unbind."""

    #: (theta, s) pairs. Subclasses set this.
    ANGLES: list = []

    def random_vector(self, n: int = 1, generator=None) -> torch.Tensor:
        idx = torch.randint(0, len(self.ANGLES), (n, self.num_blocks),
                            device=self.device, generator=generator)
        tab = torch.tensor(self.ANGLES, device=self.device, dtype=torch.float32)
        theta = tab[idx, 0]
        s = tab[idx, 1]
        c, sin_t = torch.cos(theta), torch.sin(theta)
        blocks = torch.stack([c, -s * sin_t, sin_t, s * c], dim=-1)
        return blocks.reshape(n, self.actual_dim)


class EduBind2Gen(_DiscreteO2):
    """The two elements `eduGen` exposes: Rot and Ref. This is the family the
    kernel-tier instance `edubindVSA` is built from."""
    ANGLES = [(np.pi / 2, 1.0), (0.0, -1.0)]


class EduBindD4(_DiscreteO2):
    """All eight elements of the same group -- `d4Gen` in DihedralLabel.lean.
    Four rotations (s = +1) and four reflections (s = -1)."""
    ANGLES = [(k * np.pi / 2, s)
              for s in (1.0, -1.0) for k in range(4)]


def make_op(name, dim, device):
    if name == "edubind2":
        op = EduBind2Gen(dim=dim, device=device)
        return op, op.actual_dim
    if name == "d4":
        op = EduBindD4(dim=dim, device=device)
        return op, op.actual_dim
    return _make_op(name, dim, device)


def _cos(op, X, Y):
    return float(op.similarity(X, Y).mean())


@torch.no_grad()
def measure(op_name, chain_lengths, device, seed_tag):
    """For each chain length, encode one content through a chain forward, in
    reverse, and under a random permutation; report mean cosine to the forward
    encoding. 1.0 means the encoding is blind to the reordering."""
    out = {}
    for n in chain_lengths:
        torch.manual_seed(_stable_seed("rev", op_name, n, D, seed_tag))
        op, dim = make_op(op_name, D, device)
        cb = op.random_vector(K)
        idx = torch.randint(0, K, (N_TRIALS,), device=device)
        Y = cb[idx]

        rels = [op.random_vector(1).expand(N_TRIALS, -1) for _ in range(n)]

        # SEVERAL permutations, none of them the identity.
        #
        # A single random permutation per length is not a measurement of
        # "reordering in general", and two ways of getting a vacuous cell showed
        # up when the results were plotted rather than tabulated. At n = 2 the
        # draw came out as the identity, so the cell compared the forward
        # encoding with itself. At n = 4 the draw was [0, 3, 2, 1] -- a reversal
        # of an odd-length segment -- which the two-generator family is blind to
        # for exactly the reason `edubind_reverse_blind_at_length_three` gives,
        # so that cell was measuring the reversal result a second time rather
        # than an independent reordering. Averaging over N_PERM distinct
        # non-identity permutations removes both, and the minimum is reported
        # alongside the mean because "some permutation is invisible to this
        # family" is itself a fact worth keeping visible.
        rng = np.random.default_rng(_stable_seed("perm", op_name, n))
        perms, guard = [], 0
        while len(perms) < N_PERM and guard < 200:
            guard += 1
            p = [int(x) for x in rng.permutation(n)]
            if p != list(range(n)) and p not in perms:
                perms.append(p)

        def encode(order):
            Z = Y
            for k in order:
                Z = op.bind(rels[k], Z)
            return Z

        fwd = encode(range(n))
        rev = encode(range(n - 1, -1, -1))
        perm_cos = [_cos(op, fwd, encode(p)) for p in perms]

        c_rev = _cos(op, fwd, rev)
        out[n] = {
            "cos_fwd_rev": c_rev,
            "cos_fwd_perm": float(np.mean(perm_cos)),
            "cos_fwd_perm_max": float(np.max(perm_cos)),
            "n_perms": len(perms),
            "perm_blind_any": bool(max(perm_cos) > 1.0 - 1e-5),
            "reversal_blind": bool(abs(c_rev - 1.0) < 1e-5),
        }
        del cb, Y, rels, fwd, rev
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def n2_expectation(op_name, device, n_draws=400):
    """The n = 2 value is not merely observed, it is DERIVED, so it is the one
    number in this file that tests the model rather than describing it.

    For an orthogonal codebook the cosine between the two orderings reduces,
    per 2x2 block, to (1/2) tr(g1^T g2^T g1 g2) -- half the trace of the group
    COMMUTATOR of the two relations. In O(2) the commutator is the identity
    exactly when both relations are rotations (reflections anticommute past
    rotations), and is otherwise a rotation whose trace averages to zero:

      continuous O(2) : P(both rotations) = 1/4                     -> E = 0.25
      D4 (8 elements) : commuting pairs = |G| * #classes = 8*5 = 40 of 64,
                        the other 24 give the commutator Rot^2, trace -2
                        -> E = (40*1 + 24*(-1))/64                  = 0.25
      {Rot, Ref}      : (R,R) and (F,F) commute, (R,F) and (F,R) give Rot^2
                        -> E = (1 + 1 - 1 - 1)/4                    = 0.00

    A STATISTICAL NOTE that applies to `measure` above and is easy to get wrong:
    there, one relation chain is shared across the whole trial batch, so the
    batch adds no independent samples for this statistic -- the effective
    sample size is the number of BLOCKS (D/4 = 512), giving a standard error of
    about 1/sqrt(512) = 0.044. That is why the n = 2 entries there sit a few
    hundredths off the closed form. Here the relations are redrawn every draw,
    so the estimate is genuinely independent across draws."""
    torch.manual_seed(_stable_seed("n2exp", op_name, D))
    op, dim = make_op(op_name, D, device)
    Y = op.random_vector(n_draws)
    g1 = op.random_vector(n_draws)
    g2 = op.random_vector(n_draws)
    fwd = op.bind(g2, op.bind(g1, Y))
    rev = op.bind(g1, op.bind(g2, Y))
    v = op.similarity(fwd, rev)
    return float(v.mean()), float(v.std() / np.sqrt(len(v)))


PREDICTED_N2 = {"edubind": 0.25, "d4": 0.25, "edubind2": 0.0}


def real_path_lengths():
    """Hop lengths of real Junyi transitive prerequisite paths, capped per
    stratum. Only the LENGTH distribution is taken from the curriculum: the
    relations must be group elements for the algebra to apply at all, so this
    is honestly a real-length, not a real-relation, measurement."""
    import csv
    import networkx as nx
    path = src_dir.parent / "data" / "junyi" / "junyi_Exercise_table.csv"
    G = nx.DiGraph()
    with open(str(path), "r", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            G.add_node(name)
            for x in (row.get("prerequisites") or "").split(","):
                x = x.strip()
                if x:
                    G.add_edge(x, name)
    strata = {"hop2-3": (2, 3), "hop4-6": (4, 6), "hop7+": (7, 10 ** 9)}
    buckets = {k: [] for k in strata}
    for b in sorted(G.nodes):
        # SORTED. `nx.ancestors` returns a SET, and Python randomises string
        # hashes per process, so iterating it directly makes the sampled path
        # list -- and therefore every number below -- differ from run to run.
        # This is the same hazard as seeding from `hash(...)`: reproducible
        # within a process, not across them.
        for a in sorted(nx.ancestors(G, b)):
            try:
                k = len(nx.shortest_path(G, a, b)) - 1
            except Exception:
                continue
            if k < 2:
                continue
            for name, (lo, hi) in strata.items():
                if lo <= k <= hi:
                    buckets[name].append(k)
                    break
    rng = np.random.default_rng(0)
    for name in buckets:
        v = buckets[name]
        if len(v) > MAX_PATHS_PER_STRATUM:
            buckets[name] = [v[i] for i in
                             sorted(rng.choice(len(v), MAX_PATHS_PER_STRATUM, replace=False))]
    return buckets


def main():
    print("=" * 86)
    print("  Chain reversal / permutation, measured on FORWARD composition")
    print("  (the quantity abelian_chainAct_perm and d4_reverse_sensitive_general")
    print("   are actually about -- no unbinding anywhere)")
    print("=" * 86)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | D={D} K={K} trials={N_TRIALS}\n")

    ops = ["map", "hrr", "edubind2", "d4", "edubind"]
    t0 = time.perf_counter()

    synthetic = {}
    for name in ops:
        print(f">>> {name}")
        print("   n | cos(fwd,rev) | cos(fwd,perm) | reversal-blind")
        synthetic[name] = measure(name, N_LIST, device, "syn")
        for n in N_LIST:
            r = synthetic[name][n]
            print(f"  {n:>3d} | {r['cos_fwd_rev']:12.6f} | {r['cos_fwd_perm']:13.6f} |"
                  f" {'YES' if r['reversal_blind'] else 'no'}")
        print()

    print("n = 2 closed form vs. measurement (independent relation draws):")
    print("   arm       | predicted | measured +/- s.e.")
    n2 = {}
    for name, pred in PREDICTED_N2.items():
        m, se = n2_expectation(name, device)
        n2[name] = {"predicted": pred, "measured": m, "stderr": se,
                    "within_3se": bool(abs(m - pred) <= 3 * se)}
        print(f"  {name:10s} | {pred:9.3f} | {m:8.4f} +/- {se:.4f}"
              f"   {'OK' if n2[name]['within_3se'] else 'MISMATCH'}")
    print()

    print("Real Junyi hop-length distribution:")
    buckets = real_path_lengths()
    for k, v in buckets.items():
        print(f"  {k:8s}: {len(v):4d} paths, lengths {min(v)}..{max(v)}")
    real_lengths = sorted({n for v in buckets.values() for n in v})
    print(f"  distinct lengths measured: {len(real_lengths)}\n")

    real = {}
    for name in ops:
        real[name] = measure(name, real_lengths, device, "real")
        blind = [n for n in real_lengths if real[name][n]["reversal_blind"]]
        odd_blind = [n for n in blind if n % 2 == 1]
        print(f"  {name:9s}: reversal-blind at {len(blind):3d}/{len(real_lengths)} real "
              f"lengths ({len(odd_blind)} of them odd)")

    elapsed = time.perf_counter() - t0
    payload = {
        "config": {"D": D, "K": K, "n_trials": N_TRIALS, "n_perms_per_length": N_PERM, "n_list": N_LIST,
                   "ops": ops, "real_lengths": real_lengths,
                   "max_paths_per_stratum": MAX_PATHS_PER_STRATUM},
        "synthetic": {op: {str(n): v for n, v in d.items()} for op, d in synthetic.items()},
        "real_lengths_measured": {op: {str(n): v for n, v in d.items()} for op, d in real.items()},
        "real_hop_counts": {k: len(v) for k, v in buckets.items()},
        "n2_closed_form": n2,
        "elapsed_sec": elapsed,
        "note": ("Measures FORWARD composition only: enc_fwd binds a content "
                 "through the chain in order, enc_rev through the same relations "
                 "reversed, enc_perm through a random permutation. No unbinding, "
                 "so this isolates the order question from Axiom 3 exactly as "
                 "ChainOrder.lean does. cos = 1.0 means blind to the reordering. "
                 "edubind2 is the two-element family the KERNEL-TIER instance "
                 "edubindVSA is built from; d4 is all eight elements of the same "
                 "group (d4Gen); edubind is the CONTINUOUS O(2) family the "
                 "implementation samples, which no kernel-tier theorem covers."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "chain_reversal_direct_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nTotal {elapsed:.1f}s\n[saved: {out}]")


if __name__ == "__main__":
    main()
