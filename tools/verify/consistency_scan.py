"""Cross-section consistency scan: does any part of the C1 preprint contradict another?

Targets Revision 5 (`main_r5.tex`) by default; set C1_PAPER to scan a different
file. Revision 5 restructured the paper: the Specification section became
Preliminaries, order-sensitivity got its own section stating the label-algebra
criterion, Limitations and Threats to Validity became top-level sections, and
Future Work became Outlook. Only the English paper can be scanned -- the section
markers are English titles.
"""
import io, os, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = os.environ.get("C1_PAPER", "main_r5.tex")
T = io.open(ROOT / "docs" / "arxiv_c1" / PAPER, encoding="utf-8").read()

def _find(marker):
    try:
        return T.index(marker)
    except ValueError:
        raise SystemExit(
            f"[consistency_scan] {PAPER}: section marker {marker!r} not found.\n"
            "  This scanner keys on the English section titles of the CURRENT paper\n"
            "  structure (Revision 5). Two files it therefore cannot scan: a\n"
            "  translation (different titles), and a superseded revision (different\n"
            "  section layout). Refusing to scan is the correct outcome for both --\n"
            "  previously this surfaced as a bare ValueError, which reads like a crash\n"
            "  rather than 'this file is unchecked'.")

def sec(start, end=None):
    return T[_find(start):(_find(end) if end else len(T))]

ABS   = sec(r"\begin{abstract}", r"\section{Introduction}")
INTRO = sec(r"\section{Introduction}", r"\section{Related Work}")
REL   = sec(r"\section{Related Work}", r"\section{Preliminaries}")
SPEC  = sec(r"\section{Preliminaries}", r"\section{Order Sensitivity Lives in the Label Algebra}")
CRIT  = sec(r"\section{Order Sensitivity Lives in the Label Algebra}", r"\section{Pairs:")
PAIRS = sec(r"\section{Pairs:", r"\section{Recovery Needs Only a Left Inverse}")
# Revision 7 split r6's single "Chains" section into the three requirements it
# was conflating, so the scanner must too: recovery (Axiom 3 alone), reordering
# (Axiom 2, necessary AND sufficient), and direction (Axiom 2 necessary, NOT
# sufficient). Keeping one chunk here would let a contradiction between, say,
# the reordering theorem and the direction counterexample sit inside a single
# scanned region and never be compared -- which is exactly the failure mode the
# Limitations/Threats split was introduced to fix in an earlier round.
RECOV = sec(r"\section{Recovery Needs Only a Left Inverse}", r"\section{Reordering: Necessary}")
REORD = sec(r"\section{Reordering: Necessary}", r"\section{Direction: Not Sufficient}")
DIREC = sec(r"\section{Direction: Not Sufficient}", r"\section{Superposition: The Remaining Cost}")
SUPER = sec(r"\section{Superposition: The Remaining Cost}", r"\section{Formal Artifact}")
ARTIF = sec(r"\section{Formal Artifact}", r"\section{Experiments}")
CHAIN = RECOV + REORD + DIREC + SUPER
EXP   = sec(r"\section{Experiments}", r"\section{What Is Proved, What Is Measured}")
PM    = sec(r"\section{What Is Proved, What Is Measured}", r"\section{Limitations}")
LIM   = sec(r"\section{Limitations}", r"\section{Threats to Validity}")
THR   = sec(r"\section{Threats to Validity}", r"\section{Outlook}")
FUT   = sec(r"\section{Outlook}", r"\section{Conclusion}")
CONC  = sec(r"\section{Conclusion}", r"\section*{Code and Data Availability}")
ALL = {"abstract": ABS, "intro": INTRO, "related": REL, "spec": SPEC,
       "criterion": CRIT, "pairs": PAIRS, "chains": CHAIN,
       "recovery": RECOV, "reordering": REORD, "direction": DIREC,
       "superposition": SUPER, "artifact": ARTIF, "experiments": EXP,
       "proved_measured": PM, "limitations": LIM, "threats": THR,
       "future": FUT, "conclusion": CONC}

issues = []

# A sentence that explicitly NEGATES the searched pattern is not a violation --
# the paper stating "does not assert that every commutative encoding ..." is the
# correct behaviour, not the banned claim. Look back a short window for a
# negator before flagging. (This false-positive class bit us in Revision 3.)
_NEG = re.compile(r"\b(?:not|never|no|nor|rather than|instead of|does not|cannot|"
                  r"must not|should not|without)\b", re.I)

def forbid(pat, why, where=None, allow_negated=True):
    for name, body in ALL.items():
        if where and name not in where:
            continue
        for m in re.finditer(pat, body):
            lead = body[max(0, m.start() - 120):m.start()]
            if allow_negated and _NEG.search(lead):
                continue
            issues.append((name, why, body[max(0, m.start()-70):m.start()+110].replace("\n", " ")))

# --- claims earlier revisions made that the evidence contradicts ---
forbid(r"two independent operator families", "the two instances are NOT independent (dim-2 groups coincide)")
forbid(r"verified operators (?:thus )?generalize", "generalization is not operator-specific")
forbid(r"indistinguishable from chance",
       "48.4% is significantly BELOW chance (only the 0.482 null test is at chance)",
       where=["abstract","intro","limitations","threats","conclusion"])
forbid(r"corroborate the exact-unbind guarantee downstream", "KT does not corroborate exact-unbind")
forbid(r"capacity1D|capacityPAM|PedagogicalVSACore", "renamed in Revision 3")
forbid(r"chain of arbitrary length \$n\$", "chain is indexed by a list of relations")
forbid(r"4\.9\\times10\^\{-6\}", "float error is 1.4e-06")
forbid(r"97\.6\\%|78\.5\\%|84\.3\\%|97\.3\\%|86\.3\\%",
       "superseded by the validated-split controls run")
# NOTE: 80.5% was in this list until Revision 5. It is now a live number (MAP's
# hop 2--3 transductive accuracy under the role-filler encoding, sec:h0-direct),
# and it only escaped the rule in Revision 4 because the preceding sentence
# happened to contain "does not", which the negation guard treats as a licence.
# A banned-number rule that a neighbouring word can switch off is not a rule.
forbid(r"without any formal correctness proof for the antisymmetry", "RotatE has Lemma 1")
forbid(r"independently addressable slots", "DOF, not slots")
forbid(r"every one of the 56\{,\}224", "the runtime tier samples 40 chain lengths, not every pair")

# Revision 7 re-ran path_order_discrimination.py with a TIE-AWARE direction
# accuracy (a strict `margin > 0` test scored MAP at 0.000, i.e. "worse than
# chance", when the truth is "identically tied" -- the same defect the controls
# table had already fixed for HRR). The old values are now stale and must not
# be copied forward from main_r6.tex.
forbid(r"\$0\.000\$\s*&\s*\$0\.0000\$", "stale: MAP tie-broken direction accuracy is 0.500, not 0.000")
forbid(r"\$0\.303\$", "stale: HRR tie-broken direction accuracy is 0.505 (chance), not 0.303")

# --- Revision 7: two claims that already drifted once and must not drift back ---
# (1) THE ADAPTER CLAIM. Revision 6 (commit fcf092c) removed the reading that
# the 48.4% cross-source result shows the operator is "not good enough" and
# needs a learned adapter to compensate. The measured decomposition says
# otherwise: the gap is coverage (expert annotations are short-hop, the test set
# is 85.2% hop 7+) plus convention (81% of resolvable annotation pairs run
# against the DAG), and a full consistent flip recovers only to 51.94%. Neither
# is something a binding algebra could be asked to supply, so no adapter is
# proposed. Revision 7 adds a second, independent reason to hold this line:
# `edubind_reverse_blind_at_length_three` shows the two-generator family IS
# genuinely deficient at odd chain lengths -- and the repair is ALGEBRAIC
# (expose the whole label algebra, `d4VSA`), not a learned component. A real
# deficiency with an algebraic fix is the strongest case against an adapter,
# and it would be easy to mis-cite it as the opposite.
forbid(r"(?:needs?|requires?|propose[sd]?|add(?:ing)?) an adapter",
       "Revision 6 dropped the adapter reading of 48.4%; the gap is coverage + "
       "convention, and Revision 7's reversal deficiency has an ALGEBRAIC repair")
forbid(r"48\.4\?% shows (?:the|that the) operator",
       "48.4% is a coverage/source-of-truth result, not an operator verdict")
# (2) THE REVERSAL CLAIM. `abelian_chainAct_perm` is universal on the abelian
# side, but the non-abelian side is NOT: Axiom 2 gives two orderings that differ
# (`chain_order_sensitive_general`), never a traversal-vs-reverse guarantee.
# `edubind_reverse_blind_at_length_three` machine-checks a counterexample.
# Revision 7 proofread. `edubind_reverse_blind_at_length_three` proves the
# blindness AT LENGTH THREE; every odd length is exhaustive search over words,
# not a theorem. The abstract said "provably blind to reversal at every odd
# chain length" while the introduction stated it correctly two paragraphs later,
# so the paper contradicted itself and overclaimed in the place a reviewer reads
# first. Neither check_listings (prose, not listings) nor verify_paper_numbers
# (no number was wrong) nor consistency_scan as it then stood could see it.
forbid(r"(?i)prov(?:ably|en|ed)[^.]{0,60}blind[^.]{0,40}(?:every|all) odd",
       "the odd-length pattern is exhaustive search, not a theorem; only "
       "length three is machine-checked")
# The same shape one level down: the closed-form comparison has three rows and
# the third sits at 2.5 standard errors, so a blanket "each within one standard
# error" is false.
forbid(r"each within one standard error",
       "the two-generator row of the closed-form comparison is at 2.5 s.e., "
       "not 1")
forbid(r"(?i)non-?commutativ\w*[^.]{0,80}(?:guarantees?|ensures?|buys)[^.]{0,70}from its reverse",
       "FALSE: edubind_reverse_blind_at_length_three refutes it for our own "
       "verified family at every odd length; the general form is two orderings")

# --- Revision-4 specific: the central claim must not be overstated ---
# H0 refutes a NECESSITY claim with an existential witness. It must never be
# restated as "commutative binding always encodes order".
forbid(r"every commutative (?:binding|encoding) (?:distinguishes|encodes)",
       "H0 is existential: it refutes necessity, it does not claim universality")
forbid(r"all Hadamard encodings", "H0 is an existential witness, not a universal claim")
# The crosstalk theorem is now general (encChainRF_crosstalk_witness_general,
# proved via a padding argument): claims that it holds for every n >= 2 are
# correct as of Revision 4 and must NOT be flagged. What remains genuinely
# false is claiming it holds for EVERY chain/content choice at a given n
# (the theorem gives a witness per n, not universality over contents).
forbid(r"crosstalk (?:term )?is nonzero for (?:every|all) (?:content|role)s?\b",
       "encChainRF_crosstalk_witness_general gives a witness per n, not universality over all contents")
# The Cost Model was removed in Revision 4 -- no dangling references.
forbid(r"Cost Model|dof1D|dofPAM|storageCostPAM|matched storage",
       "the Cost Model section was removed in Revision 4")
# Non-commutativity must not be credited for pair-level results anywhere.
forbid(r"non-commutativity (?:is what |)(?:enables|allows) (?:the |)pair",
       "pair-level order recovery does not require non-commutativity (H0)")

# --- Revision 5: the paper must not narrate its own revision history ---
# A reader of the preprint has never seen an earlier draft, so "an earlier
# revision did X and we fixed it" spends words on a fact that is only about us.
# State the design as it stands; disclose limits in Limitations/Threats.
forbid(r"an earlier (?:revision|version|draft)|earlier revision|"
       r"was originally machine-checked|originally machine-checked|"
       r"in an earlier version|we (?:then |later )?fixed it|"
       r"as a (?:previous|prior) revision did|Revision [0-9]",
       "the paper narrates its own revision history instead of stating the design",
       allow_negated=False)
# Internal working names for the thesis chapters must never reach the preprint.
forbid(r"\bC1\b|\bC2\b|\bC3\b|\bC4\b|companion chapter|companion paper",
       "internal chapter labels (C1-C4) are project-private, not reader-facing",
       allow_negated=False)
# The five pair-level control settings must be counted the same way everywhere;
# an earlier revision said "four" in one Remark and "five" in four other places.
forbid(r"Four independent settings", "the count is five everywhere else in the paper")
# thm:abelian is the parent theorem; the load-bearing tally must include it.
forbid(r"Seven results are load-bearing", "the tally now includes thm:abelian, i.e. eight")
# A standalone preprint must not name unpublished companion chapters by codename.
forbid(r"C[234]'s", "do not name companion chapters by codename in a standalone paper")
forbid(r"companion chapters", "do not name companion chapters in a standalone paper")

# --- numbers that must agree wherever they appear ---
def multi(pat, label):
    vals = {}
    for name, body in ALL.items():
        f = re.findall(pat, body)
        if f:
            vals[name] = sorted(set(f))
    uniq = sorted({v for vs in vals.values() for v in vs})
    if len(uniq) > 1:
        issues.append(("cross-section", f"{label} appears with differing values: {uniq}", str(vals)))
    return vals

multi(r"1\.4\\times10\^\{-6\}|1\.431", "float32 max error")
multi(r"\$R\^2 = (0\.99)\$", "capacity logistic R2")
multi(r"56\{,\}(\d+)", "transitive pair count")

# --- required cross-references that must exist and be referenced ---
# Revision 5: `sec:criterion` and `tab:criterion` are new (the label-algebra
# criterion is now a contribution in its own right); `sec:outlook` replaced
# `sec:future-work`; `rem:n2` and `thm:crosstalk-general` are gone -- the
# crosstalk result is stated once, in its general form, as thm:crosstalk.
# Revision 7 restructured the spine: r6's single "Chains" section became four
# (recovery / reordering / direction / superposition), the knowledge-tracing
# table was folded into prose, and the pair-level role-filler rerun moved into
# sec:pair-tiers. The list below tracks the CURRENT structure. Labels dropped on
# purpose -- tab:kt, sec:h0-direct, sec:chains, sec:sep-cost, thm:crosstalk --
# are removed here rather than left to fail, but the theorems they pointed at
# are still required by name in verify_paper_numbers.py, so removing a label
# cannot silently drop a result.
for lab in ["tab:controls", "tab:criterion", "tab:artifact", "tab:families",
            "tab:reversal",
            "fig:separation",
            "sec:prelim", "sec:criterion", "sec:pairs", "sec:h0",
            "sec:recovery", "sec:reorder", "sec:direction", "sec:superposition",
            "sec:artifact", "sec:order-exp",
            "sec:experiments", "sec:setup", "sec:runtime", "sec:capacity-exp",
            "sec:pair-tiers",             "sec:limitations", "sec:threats", "sec:outlook",
            "thm:insufficient", "thm:h0", "thm:abelian",
            "thm:perm", "thm:order-general", "thm:blind", "thm:d4",
            "rem:predictions", "rem:criterion", "rem:coverage"]:
    if T.count("\\label{%s}" % lab) != 1:
        issues.append(("labels", f"label {lab} defined {T.count(chr(92)+'label{'+lab+'}')} times", ""))
    if ("\\ref{%s}" % lab) not in T:
        issues.append(("labels", f"label {lab} is defined but never referenced", ""))

# --- the mathlib phrasing: the kernel tier is Mathlib-FREE ---
if re.search(r"All proofs are kernel-checked against Lean 4 \\texttt\{v4\.33\.0\} with mathlib", T):
    issues.append(("spec", "says all proofs use mathlib, but the kernel tier is Mathlib-free", ""))

# --- report ---
print(f"[consistency_scan] {PAPER}: {len(ALL)} sections scanned")
if issues:
    for where, why, ctx in issues:
        print(f"  [{where}] {why}")
        if ctx:
            print(f"      ...{ctx}...")
    print(f"\nTONG VAN DE: {len(issues)}")
    raise SystemExit(1)
print("TONG VAN DE: 0")
