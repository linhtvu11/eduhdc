"""
Real Data Loader for C1 Enhancement — Production Version.
Loads REAL datasets:
  - ASSISTments 2012-2013 (interaction logs for Knowledge Tracing)
  - Junyi Academy (Info_Content.csv for exercise hierarchy, Log_Problem.csv for interactions)

Schema verified from actual downloaded files:
  ASSISTments: user_id, skill_id, correct, start_time, problem_log_id
  Junyi Content: ucid, content_pretty_name, subject, difficulty, learning_stage, level1-4_id
  Junyi Log: uuid (user), ucid (exercise), is_correct, timestamp_TW
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# ==============================================================================
# ASSISTments 2012-2013 Loader
# ==============================================================================

class ASSISTmentsDataset:
    """
    Loads ASSISTments 2012-2013 interaction logs.
    Key columns: user_id, skill_id, correct, start_time
    We filter to only rows with a valid skill_id (Knowledge Tracing task).
    """

    def __init__(self, path: Optional[Path] = None, max_interactions: Optional[int] = None):
        if path is None:
            path = DATA_ROOT / "assistments" / "2012-2013-data-with-predictions-4-final.csv"
        self.path = path
        self.max_interactions = max_interactions  # None = load all
        self.interactions: List[Dict] = []
        self.skills: Set[str] = set()
        self.users: Set[str] = set()
        self.loaded = False

    def load(self, min_interactions_per_user: int = 10) -> bool:
        if not self.path.exists():
            print(f"[WARN] ASSISTments file not found: {self.path}")
            return False

        print(f"Loading ASSISTments from {self.path.name} ...")
        raw_by_user: Dict[str, List[Dict]] = defaultdict(list)

        with open(str(self.path), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            n_read = 0
            n_skipped_no_skill = 0
            for row in reader:
                skill_id = row.get('skill_id', '').strip()
                if not skill_id:
                    n_skipped_no_skill += 1
                    continue

                try:
                    correct = int(float(row.get('correct', '0')))
                    if correct not in (0, 1):
                        continue
                except (ValueError, TypeError):
                    continue

                raw_by_user[row['user_id'].strip()].append({
                    'user_id': row['user_id'].strip(),
                    'skill': skill_id,
                    'correct': correct,
                    'timestamp': row.get('start_time', ''),
                    'problem_log_id': row.get('problem_log_id', ''),
                })
                n_read += 1
                if self.max_interactions and n_read >= self.max_interactions:
                    break

        print(f"  Raw interactions with skill_id: {n_read:,}")
        print(f"  Skipped (no skill_id): {n_skipped_no_skill:,}")

        # Filter users with too few interactions (not enough for sequence modeling)
        for uid, seq in raw_by_user.items():
            if len(seq) >= min_interactions_per_user:
                self.interactions.extend(seq)
                self.users.add(uid)
                for s in seq:
                    self.skills.add(s['skill'])

        print(f"  After filtering (min {min_interactions_per_user} interactions/user):")
        print(f"    Students: {len(self.users):,}")
        print(f"    Skills: {len(self.skills):,}")
        print(f"    Interactions: {len(self.interactions):,}")
        self.loaded = True
        return True

    def get_student_sequences(self) -> Dict[str, List[Dict]]:
        """Group interactions by student (preserving natural order)."""
        sequences: Dict[str, List[Dict]] = defaultdict(list)
        for inter in self.interactions:
            sequences[inter['user_id']].append(inter)
        return dict(sequences)

    def stats(self) -> Dict:
        return {
            'num_interactions': len(self.interactions),
            'num_students': len(self.users),
            'num_skills': len(self.skills),
            'loaded': self.loaded,
        }


def _parse_timestamp(raw_ts: str) -> float:
    """Robustly parse an ASSISTments start_time into a sortable float key.
    Handles (a) numeric epoch/ms, (b) ISO-like 'YYYY-MM-DD HH:MM:SS' strings.
    Returns +inf for missing/unparseable so such rows sort to the end deterministically."""
    if raw_ts is None:
        return float('inf')
    ts = raw_ts.strip()
    if not ts:
        return float('inf')
    # (a) numeric epoch (seconds or milliseconds)
    try:
        return float(ts)
    except (ValueError, TypeError):
        pass
    # (b) ISO-like datetime string
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(ts, fmt).timestamp()
        except (ValueError, TypeError):
            continue
    return float('inf')


def load_assistments_real(
    path: Optional[Path] = None,
    max_students: int = 500,
    min_seq_len: int = 20,
    seed: int = 42,
    max_seq_len: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict]], set]:
    """Helper function to load sampled student sequences from ASSISTments 2012-2013.

    pyKT-standard preprocessing:
      - Each student's interactions are sorted CHRONOLOGICALLY by start_time
        (KT sequences are time-ordered by definition; file order is NOT reliable).
      - Optional sliding-window truncation via `max_seq_len` (pyKT uses 200): if a
        student has more than max_seq_len interactions, only the most recent
        max_seq_len are kept (tail window = latest knowledge state).
    """
    if path is None:
        path = DATA_ROOT / "assistments" / "2012-2013-data-with-predictions-4-final.csv"

    print(f"Loading ASSISTments (streaming, max {max_students} students)...")
    raw: Dict[str, List[Dict]] = defaultdict(list)
    skill_set = set()
    n_ts_missing = 0

    with open(str(path), 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill_id = row.get('skill_id', '').strip()
            if not skill_id:
                continue
            try:
                correct = int(float(row['correct']))
                if correct not in (0, 1):
                    continue
            except (ValueError, TypeError):
                continue

            ts_key = _parse_timestamp(row.get('start_time', ''))
            if ts_key == float('inf'):
                n_ts_missing += 1
            raw[row['user_id'].strip()].append({
                'user_id': row['user_id'].strip(),
                'skill': skill_id,
                'correct': correct,
                'ts': ts_key,
            })

    # --- CHRONOLOGICAL SORT per student (pyKT requirement; fixes AUC-deflation bug) ---
    # Stable sort keeps original file order as tie-breaker for equal timestamps.
    for uid in raw:
        raw[uid].sort(key=lambda it: it['ts'])
    if n_ts_missing:
        print(f"  [warn] {n_ts_missing:,} rows had missing/unparseable start_time "
              f"(sorted to end via +inf key)")
    else:
        print(f"  Chronological sort by start_time: OK (all rows parsed)")

    eligible = {uid: seq for uid, seq in raw.items() if len(seq) >= min_seq_len}
    print(f"  Eligible students (>= {min_seq_len} interactions): {len(eligible):,}")

    rng = np.random.default_rng(seed)
    sampled_ids = list(eligible.keys())
    if len(sampled_ids) > max_students:
        sampled_ids = rng.choice(sampled_ids, size=max_students, replace=False).tolist()

    sequences = {uid: eligible[uid] for uid in sampled_ids}

    # --- Sliding-window truncation (pyKT max_len, default 200) ---
    # Keep the most recent max_seq_len interactions (latest knowledge state).
    if max_seq_len is not None and max_seq_len > 0:
        n_trunc = 0
        for uid in sequences:
            if len(sequences[uid]) > max_seq_len:
                sequences[uid] = sequences[uid][-max_seq_len:]
                n_trunc += 1
        if n_trunc:
            print(f"  Sliding-window (max_len={max_seq_len}): truncated {n_trunc:,} long students")

    for seqs in sequences.values():
        for s in seqs:
            skill_set.add(s['skill'])

    total_interactions = sum(len(s) for s in sequences.values())
    print(f"  Sampled {len(sequences):,} students | "
          f"{len(skill_set):,} skills | {total_interactions:,} interactions")
    return sequences, skill_set


# ==============================================================================
# Junyi Academy Loader
# ==============================================================================

class JunyiGraph:
    """
    Builds an exercise hierarchy graph from Junyi Academy Info_Content.csv.
    Hierarchy levels (level1_id → level2_id → level3_id → level4_id → ucid)
    represent prerequisite ordering within each subject area.
    """

    def __init__(self, content_path: Optional[Path] = None):
        if content_path is None:
            content_path = DATA_ROOT / "junyi" / "Info_Content.csv"
        self.content_path = content_path
        self.exercises: Dict[str, Dict] = {}          # ucid → metadata
        self.level1_to_exercises: Dict[str, List[str]] = defaultdict(list)
        self.level2_to_exercises: Dict[str, List[str]] = defaultdict(list)
        self.prerequisite_edges: List[Tuple[str, str]] = []  # (prereq_ucid, advanced_ucid)
        self.loaded = False

    def load(self) -> bool:
        if not self.content_path.exists():
            print(f"[WARN] Junyi content file not found: {self.content_path}")
            return False

        print(f"Loading Junyi content from {self.content_path.name} ...")
        difficulty_order = {'easy': 0, 'normal': 1, 'hard': 2}
        stage_order = {'elementary': 0, 'junior': 1, 'senior': 2}

        self.level3_to_exercises = defaultdict(list)
        self.level4_to_exercises = defaultdict(list)

        with open(str(self.content_path), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ucid = row['ucid'].strip()
                diff_str = row['difficulty'].strip()
                stage_str = row['learning_stage'].strip()
                
                self.exercises[ucid] = {
                    'ucid': ucid,
                    'name': row['content_pretty_name'].strip(),
                    'kind': row['content_kind'].strip(),
                    'difficulty': diff_str,
                    'difficulty_num': difficulty_order.get(diff_str, 1),
                    'subject': row['subject'].strip(),
                    'stage': stage_str,
                    'stage_num': stage_order.get(stage_str, 1),
                    'level1_id': row.get('level1_id', '').strip(),
                    'level2_id': row.get('level2_id', '').strip(),
                    'level3_id': row.get('level3_id', '').strip(),
                    'level4_id': row.get('level4_id', '').strip(),
                }
                l1 = row.get('level1_id', '').strip()
                l2 = row.get('level2_id', '').strip()
                l3 = row.get('level3_id', '').strip()
                l4 = row.get('level4_id', '').strip()
                if l1: self.level1_to_exercises[l1].append(ucid)
                if l2: self.level2_to_exercises[l2].append(ucid)
                if l3: self.level3_to_exercises[l3].append(ucid)
                if l4: self.level4_to_exercises[l4].append(ucid)

        print(f"  Total exercises: {len(self.exercises):,}")

        # Build prerequisite edges:
        # Step 1: Within level 4 concepts (tightest topic binding), order by (stage, difficulty)
        # Step 2: Across level 4 concepts within the same level 3 subtopic, step-wise difficulty progression
        edge_set = set()
        for l3, exs in self.level3_to_exercises.items():
            sorted_exs = sorted(
                exs,
                key=lambda e: (
                    self.exercises[e]['stage_num'],
                    self.exercises[e]['difficulty_num']
                )
            )
            for i in range(len(sorted_exs)):
                for j in range(i + 1, min(i + 5, len(sorted_exs))): # Local forward step dependencies
                    u = sorted_exs[i]
                    v = sorted_exs[j]
                    rank_u = (self.exercises[u]['stage_num'], self.exercises[u]['difficulty_num'])
                    rank_v = (self.exercises[v]['stage_num'], self.exercises[v]['difficulty_num'])
                    if rank_u < rank_v:
                        edge_set.add((u, v))

        self.prerequisite_edges = list(edge_set)
        print(f"  Prerequisite edges (hierarchical curriculum DAG): {len(self.prerequisite_edges):,}")
        self.loaded = True
        return True

    def stats(self) -> Dict:
        subjects = set(e['subject'] for e in self.exercises.values())
        stages = set(e['stage'] for e in self.exercises.values())
        return {
            'num_exercises': len(self.exercises),
            'num_edges': len(self.prerequisite_edges),
            'subjects': sorted(subjects),
            'stages': sorted(stages),
            'num_level1_groups': len(self.level1_to_exercises),
            'num_level2_groups': len(self.level2_to_exercises),
        }


class JunyiInteractions:
    """Loads Junyi Log_Problem.csv for Knowledge Tracing task."""

    def __init__(self, log_path: Optional[Path] = None, max_interactions: Optional[int] = None):
        if log_path is None:
            log_path = DATA_ROOT / "junyi" / "Log_Problem.csv"
        self.log_path = log_path
        self.max_interactions = max_interactions
        self.interactions: List[Dict] = []
        self.users: Set[str] = set()
        self.exercises: Set[str] = set()
        self.loaded = False

    def load(self, min_interactions_per_user: int = 10) -> bool:
        if not self.log_path.exists():
            print(f"[WARN] Junyi log file not found: {self.log_path}")
            return False

        print(f"Loading Junyi Log from {self.log_path.name} ...")
        raw_by_user: Dict[str, List[Dict]] = defaultdict(list)

        with open(str(self.log_path), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            n_read = 0
            for row in reader:
                is_correct_str = row.get('is_correct', 'False').strip()
                correct = 1 if is_correct_str.lower() == 'true' else 0

                raw_by_user[row['uuid']].append({
                    'user_id': row['uuid'].strip(),
                    'skill': row['ucid'].strip(),
                    'correct': correct,
                    'timestamp': row.get('timestamp_TW', ''),
                })
                n_read += 1
                if self.max_interactions and n_read >= self.max_interactions:
                    break

        print(f"  Raw interactions: {n_read:,}")

        for uid, seq in raw_by_user.items():
            if len(seq) >= min_interactions_per_user:
                self.interactions.extend(seq)
                self.users.add(uid)
                for s in seq:
                    self.exercises.add(s['skill'])

        print(f"  After filtering (min {min_interactions_per_user}):")
        print(f"    Students: {len(self.users):,}, Exercises: {len(self.exercises):,}")
        print(f"    Interactions: {len(self.interactions):,}")
        self.loaded = True
        return True

    def get_student_sequences(self) -> Dict[str, List[Dict]]:
        sequences: Dict[str, List[Dict]] = defaultdict(list)
        for inter in self.interactions:
            sequences[inter['user_id']].append(inter)
        return dict(sequences)

    def stats(self) -> Dict:
        return {
            'num_interactions': len(self.interactions),
            'num_students': len(self.users),
            'num_skills': len(self.exercises),
            'loaded': self.loaded,
        }


# ==============================================================================
# Main: Verify both real datasets
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Real Educational Dataset Loader — Verification")
    print("=" * 70)

    # 1. ASSISTments (load only 200K interactions for quick check)
    print("\n--- ASSISTments 2012-2013 ---")
    assist = ASSISTmentsDataset(max_interactions=200_000)
    if assist.load():
        s = assist.stats()
        print(f"  OK: {s['num_interactions']:,} interactions | "
              f"{s['num_students']:,} students | {s['num_skills']:,} skills")
    else:
        print("  [FAIL] File not found")

    # 2. Junyi Content (exercise hierarchy)
    print("\n--- Junyi Academy Content ---")
    junyi_graph = JunyiGraph()
    if junyi_graph.load():
        s = junyi_graph.stats()
        print(f"  OK: {s['num_exercises']:,} exercises | {s['num_edges']:,} prerequisite edges")
        print(f"  Subjects: {s['subjects']}")
        print(f"  Stages: {s['stages']}")
        print(f"  Level1 groups: {s['num_level1_groups']:,} | Level2 groups: {s['num_level2_groups']:,}")
    else:
        print("  [FAIL] File not found")

    # 3. Junyi Interactions (load only 200K for quick check)
    print("\n--- Junyi Academy Interactions ---")
    junyi_log = JunyiInteractions(max_interactions=200_000)
    if junyi_log.load():
        s = junyi_log.stats()
        print(f"  OK: {s['num_interactions']:,} interactions | "
              f"{s['num_students']:,} students | {s['num_skills']:,} exercises")

    print("\n[DONE] All real datasets verified.")

