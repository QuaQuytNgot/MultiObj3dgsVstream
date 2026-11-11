"""Two independent exact multiple-choice allocation algorithms.

Costs retain their original integer bytes. Utilities are converted from finite
binary floats to exact rational numbers, so pruning cannot erase an improving
path through floating-point rounding. No budget grid or quality monotonicity is
assumed. The intended experiment has only three objects and four alternatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Sequence


@dataclass(frozen=True)
class SolverResult:
    status: str
    selection: tuple[int, ...] | None
    cost_bytes: int | None
    utility: float | None
    diagnostics: dict = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return self.status in ("optimal", "infeasible")

    def to_dict(self) -> dict:
        return {**asdict(self), "exact": self.exact}


def _inputs(costs: Sequence[Sequence[int]], utilities: Sequence[Sequence[float]],
            budget_bytes: int):
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, Integral) or budget_bytes < 0:
        raise ValueError("budget_bytes must be a nonnegative integer")
    if not costs or len(costs) != len(utilities):
        raise ValueError("Require equally sized, nonempty cost and utility matrices")
    sizes = [len(row) for row in costs]
    if any(size == 0 for size in sizes) or any(len(row) != size for row, size in zip(utilities, sizes)):
        raise ValueError("Each object needs matching nonempty cost/utility alternatives")
    for row in costs:
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in row):
            raise ValueError("Representation costs must be nonnegative integer bytes")
    for row in utilities:
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in row):
            raise ValueError("All optimization utilities must be finite real values")
    return (tuple(tuple(int(v) for v in row) for row in costs),
            tuple(tuple(Fraction(float(v)) for v in row) for row in utilities), int(budget_bytes))


def _limit(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def exact_byte_dp(costs: Sequence[Sequence[int]], utilities: Sequence[Sequence[float]],
                  budget_bytes: int, *, max_states: int | None = None) -> SolverResult:
    """Sparse byte DP shared by the two MSE policies.

    A layer maps each reachable physical-byte sum to the best prefix. After
    expansion, lower-byte prefixes with at least the same utility dominate.
    Ties select minimum total bytes, then lexicographically smallest indices.
    A state limit explicitly returns incomplete without a fallback allocation.
    """
    costs, values, budget = _inputs(costs, utilities, budget_bytes)
    _limit(max_states, "max_states")
    states = {0: (Fraction(0), ())}
    stats = {"algorithm": "exact_integer_byte_sparse_dp", "expanded": 0,
             "pruned_feasibility": 0, "pruned_dominance": 0, "peak_states": 1}
    for row_cost, row_value in zip(costs, values):
        expanded = {}
        for previous_cost, (previous_value, path) in states.items():
            for k, (cost, value) in enumerate(zip(row_cost, row_value)):
                stats["expanded"] += 1
                total = previous_cost + cost
                if total > budget:
                    stats["pruned_feasibility"] += 1
                    continue
                proposal = (previous_value + value, path + (k,))
                old = expanded.get(total)
                if old is None or proposal[0] > old[0] or (proposal[0] == old[0] and proposal[1] < old[1]):
                    expanded[total] = proposal
                if max_states is not None and len(expanded) > max_states:
                    stats["detail"] = "configured reachable-state cap; no exact allocation certified"
                    return SolverResult("incomplete", None, None, None, stats)
        states = {}
        best = None
        for cost, state in sorted(expanded.items()):
            if best is None or state[0] > best:
                states[cost] = state
                best = state[0]
            else:
                stats["pruned_dominance"] += 1
        stats["peak_states"] = max(stats["peak_states"], len(states))
        if not states:
            return SolverResult("infeasible", None, None, None, stats)
    cost, (value, path) = min(states.items(), key=lambda item: (-item[1][0], item[0], item[1][1]))
    return SolverResult("optimal", path, cost, float(value), stats)


def nava_graph_dp(costs: Sequence[Sequence[int]], utilities: Sequence[Sequence[float]],
                  budget_bytes: int, *, max_labels: int | None = None,
                  max_nodes: int | None = None) -> SolverResult:
    """NAVA III-G1/Algorithm 1 graph pulse with three pruning checks.

    Each (object, representation) is a node with its own prefix Pareto labels;
    edges to the next layer add node-entry bytes and weighted quality. Bounds
    use minimum remaining bytes and maximum independent remaining utility.
    Caps return incomplete, marking any attached incumbent as unproven.
    """
    costs, values, budget = _inputs(costs, utilities, budget_bytes)
    _limit(max_labels, "max_labels")
    _limit(max_nodes, "max_nodes")
    n = len(costs)
    minimum_suffix = [0] * (n + 1)
    maximum_suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        minimum_suffix[i] = min(costs[i]) + minimum_suffix[i + 1]
        maximum_suffix[i] = max(values[i]) + maximum_suffix[i + 1]
    labels = [[{} for _ in row] for row in costs]
    stats = {"algorithm": "nava_layered_graph_pulse_dp", "visited": 0,
             "pruned_dominance": 0, "pruned_feasibility": 0,
             "pruned_bounds": 0, "peak_live_labels": 0}
    live = 0
    stopped = False
    incumbent: tuple[Fraction, int, tuple[int, ...]] | None = None

    def pulse(layer: int, representation: int, cost_before: int,
              utility_before: Fraction, path: tuple[int, ...]) -> None:
        nonlocal live, stopped, incumbent
        if stopped:
            return
        if max_nodes is not None and stats["visited"] >= max_nodes:
            stopped = True
            stats["detail"] = "configured pulse-node cap; attached incumbent is unproven"
            return
        stats["visited"] += 1
        cost = cost_before + costs[layer][representation]
        utility = utility_before + values[layer][representation]
        # This inexpensive prefix check also protects nonnegative budgets.
        if cost > budget:
            stats["pruned_feasibility"] += 1
            return
        frontier = labels[layer][representation]
        if any(c <= cost and u >= utility for c, u in frontier.items()):
            stats["pruned_dominance"] += 1
            return
        for dominated in [c for c, u in frontier.items() if c >= cost and u <= utility]:
            del frontier[dominated]
            live -= 1
        if max_labels is not None and live >= max_labels:
            stopped = True
            stats["detail"] = "configured live-label cap; attached incumbent is unproven"
            return
        frontier[cost] = utility
        live += 1
        stats["peak_live_labels"] = max(stats["peak_live_labels"], live)
        if cost + minimum_suffix[layer + 1] > budget:
            stats["pruned_feasibility"] += 1
            return
        if incumbent is not None and utility + maximum_suffix[layer + 1] < incumbent[0]:
            stats["pruned_bounds"] += 1
            return
        path = path + (representation,)
        if layer + 1 == n:
            proposal = (utility, cost, path)
            if incumbent is None or utility > incumbent[0] or (
                    utility == incumbent[0] and (cost, path) < (incumbent[1], incumbent[2])):
                incumbent = proposal
            return
        for next_representation in range(len(costs[layer + 1])):
            pulse(layer + 1, next_representation, cost, utility, path)
            if stopped:
                return

    for representation in range(len(costs[0])):
        pulse(0, representation, 0, Fraction(0), ())
        if stopped:
            break
    if incumbent is None:
        return SolverResult("incomplete" if stopped else "infeasible", None, None, None, stats)
    value, cost, path = incumbent
    return SolverResult("incomplete" if stopped else "optimal", path, cost, float(value), stats)
