"""
Junyi Expert Prerequisite Annotation Loader (Contribution C1 — E4 Ground Truth).

Replaces the heuristic (stage, difficulty) DAG (data_loader_real.py: FATAL FLAW D4)
with REAL human-annotated prerequisite labels from:

    Chang, Hsu & Chen (Academia Sinica, 2015),
    "Modeling Exercise Relationships in E-Learning: A Unified Approach."

Files (data/junyi/):
  - relationship_annotation_training.csv  (1131 pairs)
  - relationship_annotation_testing.csv   (823 pairs)
  Schema: Exercise_A, Exercise_B, Similarity_avg, Similarity_raw,
          Difficulty_avg, Difficulty_raw, Prerequisite_avg, Prerequisite_raw
  (avg = mean over many independent annotators; scores on a ~1..9 scale)

Key anti-leakage principle (fixes FLAW D5):
  Concept descriptions expose ONLY the humanized exercise NAME.
  No difficulty / stage tokens leak into the semantic prompt.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
JUNYI_DIR = DATA_ROOT / "junyi"


def humanize(name: str) -> str:
    """Convert an exercise id like 'radius_diameter_and_circumference' into a
    natural-language description for semantic encoding. NO difficulty/stage leak."""
    text = name.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return f"Mathematics exercise: {text}"


class JunyiExpertAnnotations:
    """Loads human-annotated exercise relationship scores (prerequisite/similarity/difficulty)."""

    def __init__(self,
                 train_path: Optional[Path] = None,
                 test_path: Optional[Path] = None):
        self.train_path = train_path or (JUNYI_DIR / "relationship_annotation_training.csv")
        self.test_path = test_path or (JUNYI_DIR / "relationship_annotation_testing.csv")
        self.train_rows: List[Dict] = []
        self.test_rows: List[Dict] = []
        self.exercises: List[str] = []
        self.loaded = False

    # ------------------------------------------------------------------ load
    def _read(self, path: Path) -> List[Dict]:
        rows = []
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "A": r["Exercise_A"].strip(),
                    "B": r["Exercise_B"].strip(),
                    "prereq": float(r["Prerequisite_avg"]),
                    "sim": float(r["Similarity_avg"]),
                    "diff": float(r["Difficulty_avg"]),
                })
        return rows

    def load(self) -> bool:
        if not self.train_path.exists() or not self.test_path.exists():
            print(f"[WARN] Expert annotation files not found in {JUNYI_DIR}")
            return False
        self.train_rows = self._read(self.train_path)
        self.test_rows = self._read(self.test_path)
        ex = set()
        for r in self.train_rows + self.test_rows:
            ex.add(r["A"]); ex.add(r["B"])
        self.exercises = sorted(ex)
        self.loaded = True
        return True

    # -------------------------------------------------- descriptions (encoder)
    def descriptions(self) -> Dict[str, str]:
        """Map exercise id -> anti-leakage natural-language description."""
        return {name: humanize(name) for name in self.exercises}

    # -------------------------------------------------- binary link prediction
    def binary_pairs(self,
                     rows: List[Dict],
                     high: float = 6.0,
                     low: float = 3.0) -> List[Tuple[str, str, int]]:
        """
        Convert continuous Prerequisite_avg into a clean binary link-prediction set.
        - score >= high  -> positive prerequisite link (label 1)
        - score <= low   -> negative / non-prerequisite (label 0)
        - mid zone (low, high) -> EXCLUDED (ambiguous by annotators; reported separately)
        Directionality preserved: (A, B) is 'A is prerequisite of B'.
        """
        out = []
        for r in rows:
            if r["prereq"] >= high:
                out.append((r["A"], r["B"], 1))
            elif r["prereq"] <= low:
                out.append((r["A"], r["B"], 0))
        return out

    def train_binary(self, high: float = 6.0, low: float = 3.0):
        return self.binary_pairs(self.train_rows, high, low)

    def test_binary(self, high: float = 6.0, low: float = 3.0):
        return self.binary_pairs(self.test_rows, high, low)

    # -------------------------------------------------- asymmetry probe (E1)
    def bidirectional_pairs(self, rows: Optional[List[Dict]] = None
                            ) -> List[Tuple[str, str, float, float]]:
        """
        Return pairs (A, B, score_AB, score_BA) for which BOTH directions were
        annotated. This is the gold set for directional-asymmetry probing:
        a good prerequisite operator should rank the higher-scored direction above
        the lower-scored one. (268 such pairs in train.)
        """
        rows = rows if rows is not None else self.train_rows
        d = {(r["A"], r["B"]): r["prereq"] for r in rows}
        seen = set()
        out = []
        for (a, b), s_ab in d.items():
            if (b, a) in d and (b, a) not in seen:
                out.append((a, b, s_ab, d[(b, a)]))
                seen.add((a, b))
        return out

    # -------------------------------------------------- continuous (regression)
    def continuous(self, rows: List[Dict]) -> List[Tuple[str, str, float]]:
        return [(r["A"], r["B"], r["prereq"]) for r in rows]

    def stats(self) -> Dict:
        tb = self.train_binary(); vb = self.test_binary()
        return {
            "unique_exercises": len(self.exercises),
            "train_pairs_raw": len(self.train_rows),
            "test_pairs_raw": len(self.test_rows),
            "train_binary": len(tb),
            "train_pos": sum(1 for *_, y in tb if y == 1),
            "test_binary": len(vb),
            "test_pos": sum(1 for *_, y in vb if y == 1),
            "bidirectional_pairs": len(self.bidirectional_pairs()),
        }


if __name__ == "__main__":
    ann = JunyiExpertAnnotations()
    if not ann.load():
        print("[FAIL] Could not load expert annotations.")
        raise SystemExit(1)
    s = ann.stats()
    print("=== Junyi Expert Prerequisite Annotations ===")
    for k, v in s.items():
        print(f"  {k:>22s}: {v}")
    print("\n  Sample descriptions (anti-leakage):")
    desc = ann.descriptions()
    for name in ann.exercises[:5]:
        print(f"    {name!r:>45s} -> {desc[name]!r}")
    print("\n  Sample bidirectional asymmetry pairs (A, B, score_AB, score_BA):")
    for a, b, sab, sba in ann.bidirectional_pairs()[:5]:
        print(f"    {a[:28]:28s} <> {b[:28]:28s} : {sab:.2f} vs {sba:.2f}")

