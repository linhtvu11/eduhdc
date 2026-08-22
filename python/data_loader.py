"""
Unified Educational Data Loader for C1 Enhancement.
Loads prerequisite DAGs from Junyi Academy and interaction logs from ASSISTments 2009.
Provides a standardized CurriculumGraph interface for downstream experiments.
"""

import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


class CurriculumGraph:
    """
    Unified directed prerequisite graph for educational concepts.
    Nodes = concepts/skills, Directed edges = prerequisite relationships (u → v: u required before v).
    """
    
    def __init__(self, name: str):
        self.name = name
        self.concepts: List[str] = []
        self.concept_metadata: Dict[str, Dict] = {}
        self.adj: Dict[str, List[str]] = defaultdict(list)      # forward edges: prereq -> advanced
        self.rev_adj: Dict[str, List[str]] = defaultdict(list)   # backward edges: advanced -> prereq
        self.edges: List[Tuple[str, str]] = []
        
    def add_concept(self, name: str, metadata: Optional[Dict] = None):
        if name not in self.concept_metadata:
            self.concepts.append(name)
            self.concept_metadata[name] = metadata or {}
            
    def add_edge(self, prereq: str, advanced: str):
        """Add directed prerequisite edge: prereq → advanced."""
        self.adj[prereq].append(advanced)
        self.rev_adj[advanced].append(prereq)
        self.edges.append((prereq, advanced))
        
    def get_ancestors(self, concept: str) -> Set[str]:
        """BFS to find all transitive prerequisites of a concept."""
        visited = set()
        queue = deque(self.rev_adj.get(concept, []))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.rev_adj.get(node, []))
        return visited
        
    def get_descendants(self, concept: str) -> Set[str]:
        """BFS to find all concepts that transitively depend on this concept."""
        visited = set()
        queue = deque(self.adj.get(concept, []))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.adj.get(node, []))
        return visited
        
    def get_all_paths(self, max_length: int = 6) -> List[List[str]]:
        """
        Enumerate all directed prerequisite paths of length 2 to max_length.
        Returns list of paths, each path is [concept_0, concept_1, ..., concept_L].
        """
        all_paths = []
        
        def dfs(node: str, path: List[str]):
            if len(path) >= 2:
                all_paths.append(list(path))
            if len(path) >= max_length + 1:
                return
            for neighbor in self.adj.get(node, []):
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
                    
        for concept in self.concepts:
            dfs(concept, [concept])
            
        return all_paths
        
    def get_longest_paths(self, n: int = 10, min_length: int = 4) -> List[List[str]]:
        """Get the N longest prerequisite paths (for hard transitivity benchmarks)."""
        all_paths = self.get_all_paths(max_length=10)
        long_paths = [p for p in all_paths if len(p) >= min_length + 1]
        long_paths.sort(key=len, reverse=True)
        return long_paths[:n]
        
    def topological_order(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree = defaultdict(int)
        for c in self.concepts:
            in_degree[c] = 0
        for u, v in self.edges:
            in_degree[v] += 1
            
        queue = deque([c for c in self.concepts if in_degree[c] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order
        
    def domains(self) -> List[str]:
        """Get unique domain/area labels."""
        return sorted(list(set(
            m.get("area", "unknown") for m in self.concept_metadata.values()
        )))
        
    def stats(self) -> Dict:
        return {
            "name": self.name,
            "num_concepts": len(self.concepts),
            "num_edges": len(self.edges),
            "num_domains": len(self.domains()),
            "domains": self.domains(),
            "avg_in_degree": sum(len(v) for v in self.rev_adj.values()) / max(len(self.concepts), 1),
            "avg_out_degree": sum(len(v) for v in self.adj.values()) / max(len(self.concepts), 1),
        }
        
    def __repr__(self):
        s = self.stats()
        return f"CurriculumGraph('{s['name']}', concepts={s['num_concepts']}, edges={s['num_edges']}, domains={s['num_domains']})"


# ==============================================================================
# Junyi Academy Loader
# ==============================================================================

def load_junyi_graph(path: Optional[Path] = None) -> CurriculumGraph:
    """Load Junyi Academy prerequisite DAG from CSV."""
    if path is None:
        path = DATA_ROOT / "junyi" / "junyi_Exercise_table.csv"
        
    if not path.exists():
        raise FileNotFoundError(f"Junyi exercise table not found at {path}. Run download_datasets.py first.")
        
    graph = CurriculumGraph("Junyi Academy")
    
    with open(str(path), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # First pass: add all concepts
    for row in rows:
        name = row["name"].strip()
        graph.add_concept(name, {
            "topic": row.get("topic", ""),
            "area": row.get("area", "unknown"),
        })
        
    # Second pass: add prerequisite edges
    for row in rows:
        name = row["name"].strip()
        prereqs_str = row.get("prerequisites", "").strip()
        if prereqs_str:
            for prereq in prereqs_str.split(","):
                prereq = prereq.strip()
                if prereq and prereq in graph.concept_metadata:
                    graph.add_edge(prereq, name)
                    
    return graph


# ==============================================================================
# ASSISTments 2009 Loader
# ==============================================================================

class ASSISTmentsDataset:
    """Loader for ASSISTments 2009 Skill Builder data."""
    
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = DATA_ROOT / "assistments" / "skill_builder_data_2009.csv"
        self.path = path
        self.interactions = []  # List of (user_id, skill_id, correct, timestamp)
        self.skills = set()
        self.users = set()
        self.loaded = False
        
    def load(self) -> bool:
        if not self.path.exists():
            print(f"[WARN] ASSISTments data not found at {self.path}")
            return False
            
        with open(str(self.path), 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    user_id = row.get("user_id", row.get("student_id", ""))
                    skill_name = row.get("skill_name", row.get("skill", ""))
                    correct = row.get("correct", row.get("is_correct", "0"))
                    
                    if not user_id or not skill_name:
                        continue
                        
                    self.interactions.append({
                        "user_id": str(user_id).strip(),
                        "skill": str(skill_name).strip(),
                        "correct": int(float(correct)) if correct else 0,
                    })
                    self.skills.add(str(skill_name).strip())
                    self.users.add(str(user_id).strip())
                except (ValueError, KeyError):
                    continue
                    
        self.loaded = True
        return True
        
    def get_student_sequences(self) -> Dict[str, List[Dict]]:
        """Group interactions by student."""
        sequences = defaultdict(list)
        for inter in self.interactions:
            sequences[inter["user_id"]].append(inter)
        return dict(sequences)
        
    def stats(self) -> Dict:
        return {
            "num_interactions": len(self.interactions),
            "num_students": len(self.users),
            "num_skills": len(self.skills),
            "loaded": self.loaded,
        }


# ==============================================================================
# Synthetic ASSISTments Generator (for testing when real data unavailable)
# ==============================================================================

def generate_synthetic_assistments(graph: CurriculumGraph, num_students: int = 500, 
                                    interactions_per_student: int = 50, seed: int = 42) -> ASSISTmentsDataset:
    """
    Generate synthetic student interaction logs based on prerequisite DAG.
    Students who master prerequisites have higher probability of correct answers on advanced concepts.
    """
    import random
    random.seed(seed)
    
    dataset = ASSISTmentsDataset()
    topo_order = graph.topological_order()
    concept_difficulty = {c: i / len(topo_order) for i, c in enumerate(topo_order)}
    
    for student_id in range(num_students):
        # Student ability: uniform [0.3, 0.9]
        ability = 0.3 + random.random() * 0.6
        mastered = set()
        
        for _ in range(interactions_per_student):
            # Choose concept (weighted toward current frontier)
            concept = random.choice(topo_order)
            
            # Probability of correct: depends on ability, difficulty, and prerequisite mastery
            diff = concept_difficulty[concept]
            prereq_bonus = 0.0
            prereqs = graph.rev_adj.get(concept, [])
            if prereqs:
                frac_mastered = len(mastered.intersection(prereqs)) / len(prereqs)
                prereq_bonus = 0.2 * frac_mastered
                
            p_correct = min(0.95, max(0.05, ability - diff * 0.5 + prereq_bonus))
            correct = 1 if random.random() < p_correct else 0
            
            if correct:
                mastered.add(concept)
                
            dataset.interactions.append({
                "user_id": f"student_{student_id}",
                "skill": concept,
                "correct": correct,
            })
            dataset.skills.add(concept)
            dataset.users.add(f"student_{student_id}")
            
    dataset.loaded = True
    return dataset


# ==============================================================================
# Main: Verify all data sources
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Educational Data Loader Verification")
    print("=" * 70)
    
    # 1. Load Junyi Graph
    print("\n--- Junyi Academy Prerequisite DAG ---")
    try:
        junyi = load_junyi_graph()
        print(f"  {junyi}")
        stats = junyi.stats()
        print(f"  Domains: {', '.join(stats['domains'])}")
        print(f"  Avg In-Degree: {stats['avg_in_degree']:.2f}")
        print(f"  Avg Out-Degree: {stats['avg_out_degree']:.2f}")
        
        # Test path enumeration
        long_paths = junyi.get_longest_paths(n=5, min_length=4)
        print(f"\n  Top {len(long_paths)} Longest Prerequisite Paths:")
        for i, path in enumerate(long_paths):
            print(f"    {i+1}. {' → '.join(path)} (L={len(path)-1})")
            
        # Test topological order
        topo = junyi.topological_order()
        print(f"\n  Topological Order (first 10): {', '.join(topo[:10])}")
        print(f"  DAG Valid: {'Yes' if len(topo) == len(junyi.concepts) else 'No (cycle detected!)'}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        
    # 2. Load ASSISTments
    print("\n--- ASSISTments 2009 ---")
    assist = ASSISTmentsDataset()
    if assist.load():
        s = assist.stats()
        print(f"  Interactions: {s['num_interactions']:,}")
        print(f"  Students: {s['num_students']:,}")
        print(f"  Skills: {s['num_skills']}")
    else:
        print("  [INFO] Real data not available. Generating synthetic data from Junyi DAG...")
        try:
            junyi = load_junyi_graph()
            synth = generate_synthetic_assistments(junyi, num_students=500, interactions_per_student=50)
            s = synth.stats()
            print(f"  Synthetic Interactions: {s['num_interactions']:,}")
            print(f"  Synthetic Students: {s['num_students']:,}")
            print(f"  Synthetic Skills: {s['num_skills']}")
        except FileNotFoundError:
            print("  [ERROR] Cannot generate synthetic data without Junyi graph")
            
    print("\n[DONE] Data loader verification complete.")
