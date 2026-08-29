"""
Realistic STEM Educational Curriculum Ontology & Prerequisite DAG (Contribution C1).
Contains 60 structured concepts with domains, Bloom levels, prerequisites, and semantic definitions.
"""

from typing import Dict, List, Tuple

CURRICULUM_CONCEPTS: Dict[str, Dict] = {
    # --- Foundations ---
    "arithmetic": {"domain": "Foundations", "bloom": 1, "diff": 0.10, "desc": "Basic arithmetic operations including addition, subtraction, multiplication, and division of integers."},
    "fractions": {"domain": "Foundations", "bloom": 2, "diff": 0.20, "desc": "Fraction arithmetic, common denominators, and improper fractions."},
    "decimals": {"domain": "Foundations", "bloom": 2, "diff": 0.20, "desc": "Decimal representations, place value, and decimal arithmetic."},
    "ratios_proportions": {"domain": "Foundations", "bloom": 2, "diff": 0.25, "desc": "Ratios, unit rates, proportions, and scaling relationships."},
    "percentages": {"domain": "Foundations", "bloom": 2, "diff": 0.25, "desc": "Percentage calculations, percentage increase, decrease, and applications."},
    "negative_numbers": {"domain": "Foundations", "bloom": 2, "diff": 0.20, "desc": "Signed arithmetic, negative integers, and absolute values."},
    "prime_factorization": {"domain": "Foundations", "bloom": 2, "diff": 0.30, "desc": "Prime numbers, greatest common divisor, and prime factorization."},

    # --- Algebra ---
    "algebraic_expressions": {"domain": "Algebra", "bloom": 2, "diff": 0.30, "desc": "Evaluating and simplifying algebraic expressions and combining like terms."},
    "linear_equations": {"domain": "Algebra", "bloom": 3, "diff": 0.40, "desc": "Solving single-variable linear equations and inequalities."},
    "linear_systems": {"domain": "Algebra", "bloom": 3, "diff": 0.50, "desc": "Systems of two linear equations using substitution and elimination."},
    "quadratic_equations": {"domain": "Algebra", "bloom": 3, "diff": 0.60, "desc": "Solving quadratic equations via factoring, completing the square, and quadratic formula."},
    "polynomials": {"domain": "Algebra", "bloom": 3, "diff": 0.55, "desc": "Polynomial arithmetic, expansion, factoring, and polynomial division."},
    "exponents_radicals": {"domain": "Algebra", "bloom": 3, "diff": 0.45, "desc": "Laws of exponents, square roots, radical expressions, and rational exponents."},
    "rational_expressions": {"domain": "Algebra", "bloom": 4, "diff": 0.65, "desc": "Simplifying, adding, and multiplying rational algebraic fractions."},
    "matrices_basic": {"domain": "Algebra", "bloom": 3, "diff": 0.50, "desc": "Matrix addition, scalar multiplication, and matrix-matrix multiplication."},
    "vector_spaces": {"domain": "Algebra", "bloom": 4, "diff": 0.70, "desc": "Linear combinations, span, basis, linear independence, and dimension."},
    "eigenvalues_eigenvectors": {"domain": "Algebra", "bloom": 5, "diff": 0.85, "desc": "Characteristic equations, eigenvalues, eigenvectors, and diagonalization."},

    # --- Functions ---
    "function_notation": {"domain": "Functions", "bloom": 2, "diff": 0.35, "desc": "Domain, range, function evaluation, and composite functions."},
    "linear_functions": {"domain": "Functions", "bloom": 3, "diff": 0.40, "desc": "Slope-intercept form, rate of change, and graphing linear functions."},
    "quadratic_functions": {"domain": "Functions", "bloom": 3, "diff": 0.55, "desc": "Parabolas, vertex form, axis of symmetry, and extrema."},
    "exponential_functions": {"domain": "Functions", "bloom": 3, "diff": 0.55, "desc": "Exponential growth, decay, and natural base e."},
    "logarithmic_functions": {"domain": "Functions", "bloom": 3, "diff": 0.60, "desc": "Properties of logarithms, log equations, and natural logarithms."},
    "trigonometric_functions": {"domain": "Functions", "bloom": 3, "diff": 0.65, "desc": "Unit circle, sine, cosine, tangent, and trigonometric graphs."},
    "inverse_trig_functions": {"domain": "Functions", "bloom": 4, "diff": 0.70, "desc": "Arcsine, arccosine, arctangent, and restricted domains."},

    # --- Calculus ---
    "limits_continuity": {"domain": "Calculus", "bloom": 4, "diff": 0.65, "desc": "One-sided limits, limits at infinity, continuity, and squeeze theorem."},
    "derivative_definition": {"domain": "Calculus", "bloom": 4, "diff": 0.70, "desc": "Instantaneous rate of change, difference quotient, and tangent lines."},
    "derivative_rules": {"domain": "Calculus", "bloom": 3, "diff": 0.65, "desc": "Power rule, constant multiple rule, sum and difference rules."},
    "product_quotient_rule": {"domain": "Calculus", "bloom": 3, "diff": 0.70, "desc": "Product rule and quotient rule for differentiating function products."},
    "chain_rule": {"domain": "Calculus", "bloom": 4, "diff": 0.75, "desc": "Chain rule for composite functions and implicit differentiation."},
    "derivative_applications": {"domain": "Calculus", "bloom": 4, "diff": 0.80, "desc": "Optimization, related rates, Mean Value Theorem, and curve sketching."},
    "riemann_sums": {"domain": "Calculus", "bloom": 3, "diff": 0.70, "desc": "Definite integrals as limits of Riemann sums and area under curves."},
    "fundamental_theorem_calculus": {"domain": "Calculus", "bloom": 4, "diff": 0.80, "desc": "Connection between differentiation and integration."},
    "integration_by_substitution": {"domain": "Calculus", "bloom": 4, "diff": 0.75, "desc": "U-substitution technique for reversing the chain rule."},
    "integration_by_parts": {"domain": "Calculus", "bloom": 4, "diff": 0.85, "desc": "Integration by parts for products of algebraic and transcendental functions."},
    "differential_equations": {"domain": "Calculus", "bloom": 5, "diff": 0.90, "desc": "First-order separable differential equations and initial value problems."},
    "taylor_series": {"domain": "Calculus", "bloom": 5, "diff": 0.90, "desc": "Power series, Taylor and Maclaurin polynomials, and radius of convergence."},

    # --- Probability & Statistics ---
    "descriptive_statistics": {"domain": "Statistics", "bloom": 2, "diff": 0.35, "desc": "Mean, median, mode, variance, standard deviation, and interquartile range."},
    "combinatorics": {"domain": "Statistics", "bloom": 3, "diff": 0.45, "desc": "Permutations, combinations, fundamental counting principle, and factorials."},
    "probability_axioms": {"domain": "Statistics", "bloom": 3, "diff": 0.45, "desc": "Sample spaces, events, mutually exclusive events, and complement rule."},
    "conditional_probability": {"domain": "Statistics", "bloom": 4, "diff": 0.60, "desc": "Conditional probability, independence, multiplication rule, and Bayes theorem."},
    "discrete_random_variables": {"domain": "Statistics", "bloom": 3, "diff": 0.55, "desc": "Probability mass functions, expected value, variance, and binomial distribution."},
    "continuous_random_variables": {"domain": "Statistics", "bloom": 4, "diff": 0.75, "desc": "Probability density functions, normal distribution, and z-scores."},
    "sampling_distributions": {"domain": "Statistics", "bloom": 4, "diff": 0.75, "desc": "Central Limit Theorem, standard error, and sampling variability."},
    "confidence_intervals": {"domain": "Statistics", "bloom": 4, "diff": 0.80, "desc": "Point estimation, margin of error, and confidence intervals for means and proportions."},
    "hypothesis_testing": {"domain": "Statistics", "bloom": 5, "diff": 0.85, "desc": "Null/alternative hypotheses, p-values, Type I/II errors, and t-tests."},

    # --- Computer Science & Algorithms ---
    "variables_control_flow": {"domain": "ComputerScience", "bloom": 2, "diff": 0.25, "desc": "Variables, conditionals, loops, and boolean logic in programming."},
    "recursion": {"domain": "ComputerScience", "bloom": 4, "diff": 0.65, "desc": "Recursive functions, base cases, call stack, and divide-and-conquer."},
    "arrays_linked_lists": {"domain": "ComputerScience", "bloom": 3, "diff": 0.45, "desc": "Linear data structures, memory allocation, and array manipulation."},
    "trees_graphs": {"domain": "ComputerScience", "bloom": 4, "diff": 0.70, "desc": "Binary trees, tree traversals, directed and undirected graph representations."},
    "graph_algorithms": {"domain": "ComputerScience", "bloom": 5, "diff": 0.80, "desc": "Breadth-first search, depth-first search, Dijkstra shortest path, and topological sort."},
    "dynamic_programming": {"domain": "ComputerScience", "bloom": 5, "diff": 0.90, "desc": "Overlapping subproblems, optimal substructure, memoization, and bottom-up DP."},
}

# Ground-Truth Directed Prerequisite Pairs (u -> v: u is required for v)
PREREQUISITE_EDGES: List[Tuple[str, str]] = [
    # Foundations -> Foundations
    ("arithmetic", "fractions"),
    ("arithmetic", "decimals"),
    ("arithmetic", "negative_numbers"),
    ("arithmetic", "prime_factorization"),
    ("fractions", "ratios_proportions"),
    ("decimals", "percentages"),
    ("ratios_proportions", "percentages"),
    
    # Foundations -> Algebra
    ("arithmetic", "algebraic_expressions"),
    ("negative_numbers", "algebraic_expressions"),
    ("fractions", "algebraic_expressions"),
    ("algebraic_expressions", "linear_equations"),
    ("linear_equations", "linear_systems"),
    ("algebraic_expressions", "polynomials"),
    ("polynomials", "quadratic_equations"),
    ("exponents_radicals", "quadratic_equations"),
    ("polynomials", "rational_expressions"),
    ("fractions", "rational_expressions"),
    ("linear_systems", "matrices_basic"),
    ("matrices_basic", "vector_spaces"),
    ("vector_spaces", "eigenvalues_eigenvectors"),
    ("matrices_basic", "eigenvalues_eigenvectors"),
    
    # Algebra -> Functions
    ("algebraic_expressions", "function_notation"),
    ("linear_equations", "linear_functions"),
    ("function_notation", "linear_functions"),
    ("quadratic_equations", "quadratic_functions"),
    ("function_notation", "quadratic_functions"),
    ("exponents_radicals", "exponential_functions"),
    ("exponential_functions", "logarithmic_functions"),
    ("ratios_proportions", "trigonometric_functions"),
    ("trigonometric_functions", "inverse_trig_functions"),
    
    # Functions -> Calculus
    ("linear_functions", "limits_continuity"),
    ("quadratic_functions", "limits_continuity"),
    ("exponential_functions", "limits_continuity"),
    ("trigonometric_functions", "limits_continuity"),
    ("limits_continuity", "derivative_definition"),
    ("derivative_definition", "derivative_rules"),
    ("derivative_rules", "product_quotient_rule"),
    ("derivative_rules", "chain_rule"),
    ("chain_rule", "derivative_applications"),
    ("limits_continuity", "riemann_sums"),
    ("riemann_sums", "fundamental_theorem_calculus"),
    ("derivative_rules", "fundamental_theorem_calculus"),
    ("fundamental_theorem_calculus", "integration_by_substitution"),
    ("chain_rule", "integration_by_substitution"),
    ("product_quotient_rule", "integration_by_parts"),
    ("fundamental_theorem_calculus", "integration_by_parts"),
    ("derivative_applications", "differential_equations"),
    ("integration_by_substitution", "differential_equations"),
    ("derivative_rules", "taylor_series"),
    
    # Foundations & Algebra -> Statistics
    ("arithmetic", "descriptive_statistics"),
    ("fractions", "probability_axioms"),
    ("percentages", "probability_axioms"),
    ("prime_factorization", "combinatorics"),
    ("combinatorics", "probability_axioms"),
    ("probability_axioms", "conditional_probability"),
    ("conditional_probability", "discrete_random_variables"),
    ("integration_by_substitution", "continuous_random_variables"),
    ("discrete_random_variables", "continuous_random_variables"),
    ("continuous_random_variables", "sampling_distributions"),
    ("descriptive_statistics", "sampling_distributions"),
    ("sampling_distributions", "confidence_intervals"),
    ("sampling_distributions", "hypothesis_testing"),
    ("confidence_intervals", "hypothesis_testing"),
    
    # Algebra & Math -> Computer Science
    ("variables_control_flow", "arrays_linked_lists"),
    ("algebraic_expressions", "variables_control_flow"),
    ("function_notation", "recursion"),
    ("variables_control_flow", "recursion"),
    ("arrays_linked_lists", "trees_graphs"),
    ("trees_graphs", "graph_algorithms"),
    ("recursion", "dynamic_programming"),
    ("arrays_linked_lists", "dynamic_programming"),
]

def get_ontology_stats():
    return {
        "num_concepts": len(CURRICULUM_CONCEPTS),
        "num_prerequisites": len(PREREQUISITE_EDGES),
        "domains": list(set(d["domain"] for d in CURRICULUM_CONCEPTS.values()))
    }

if __name__ == "__main__":
    stats = get_ontology_stats()
    print("STEM Curriculum Ontology Statistics:")
    print(f"  Concepts: {stats['num_concepts']}")
    print(f"  Prerequisite Directed Edges: {stats['num_prerequisites']}")
    print(f"  Domains: {', '.join(stats['domains'])}")
