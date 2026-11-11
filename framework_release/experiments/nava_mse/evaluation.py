"""Independent joint scoring and literal N3 enumeration, downstream of selection.

No policy imports this module. The oracle accepts a joint evaluator callback and
physical costs only; its measurements are never part of PolicyInputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
from .metrics import MetricStats, foreground_metrics, pool_metrics


def joint_quality(reference, candidate, *, alpha_threshold=0.01):
    if 'metadata' in reference or 'metadata' in candidate:
        from .rendering import validate_render_alignment
        validate_render_alignment(reference['metadata'], candidate['metadata'])
    return foreground_metrics(reference['rgb'], candidate['rgb'], reference['alpha'],
                              alpha_threshold=alpha_threshold, normalization='clip')


def enumerate_allocations(object_ids, crfs, costs):
    if len(object_ids) != 3 or len(crfs) != 4 or len(costs) != 3 or any(len(r) != 4 for r in costs):
        raise ValueError('N3 oracle requires exactly three objects and four representations')
    if len(set(object_ids)) != 3 or len(set(crfs)) != 4:
        raise ValueError('Oracle object and CRF identities must be unique')
    if any(type(v) is not int or v < 0 for row in costs for v in row):
        raise ValueError('Oracle requires nonnegative integer byte costs')
    for indices in itertools.product(range(4), repeat=3):
        yield dict(allocation=dict(zip(object_ids, (crfs[q] for q in indices))),
                   cost_bytes=sum(row[q] for row, q in zip(costs, indices)))


@dataclass(frozen=True)
class OracleResult:
    allocation: dict
    cost_bytes: int
    metrics: MetricStats
    feasible_count: int

    def to_dict(self):
        return dict(policy='EXACT_JOINT_SSE_ORACLE', allocation=self.allocation,
                    cost_bytes=self.cost_bytes, metrics=self.metrics.to_dict(),
                    feasible_count=self.feasible_count, evaluation_reference_only=True)


def exact_joint_oracle(object_ids, crfs, costs, budget, evaluate):
    """Evaluate every physically feasible allocation, then minimize pooled SSE."""
    if type(budget) is not int or budget < 0:
        raise ValueError('Oracle budget must be nonnegative integer bytes')
    best, count = None, 0
    for candidate in enumerate_allocations(object_ids, crfs, costs):
        if candidate['cost_bytes'] > budget:
            continue
        count += 1
        metric = evaluate(candidate['allocation'])
        if not isinstance(metric, MetricStats):
            raise TypeError('Joint evaluator must return pooled MetricStats')
        if best is not None and metric_protocol(metric) != metric_protocol(best.metrics):
            raise ValueError('Oracle candidates must share fixed joint reference masks')
        if best is None or metric.sse < best.metrics.sse:
            best = OracleResult(candidate['allocation'], candidate['cost_bytes'], metric, 0)
    if best is None:
        raise ValueError('No physically feasible oracle allocation')
    return OracleResult(best.allocation, best.cost_bytes, best.metrics, count)


def oracle_gap(metric, oracle):
    if metric_protocol(metric) != metric_protocol(oracle.metrics):
        raise ValueError('Oracle gap requires identical fixed reference masks and metric protocol')
    gap = metric.sse - oracle.metrics.sse
    if gap < -max(1e-10, abs(oracle.metrics.sse) * 1e-12):
        raise ValueError('Selected allocation beats enumerated feasible oracle: inconsistent evaluation')
    gap = max(0., gap)
    return dict(oracle_gap_sse=gap,
                oracle_gap_relative=None if oracle.metrics.sse == 0 else gap / oracle.metrics.sse,
                oracle_sse_zero=oracle.metrics.sse == 0,
                allocation_agrees_with_oracle=None)


def metric_protocol(metric):
    return (metric.foreground_pixels, metric.channels, metric.data_range, metric.normalization)
