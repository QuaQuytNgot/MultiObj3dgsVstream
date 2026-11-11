"""Visibility × MSE kernel copied from the validated experiment objective.

Source: adaptation_baselines.methods.construct_utilities (MSE branch) and
nava_mse.policies.projected_aabb_importance. Temporal multiplication and fsum
order are deliberately identical. No experiment code or trajectory is imported.
"""
from __future__ import annotations

import math


def construct_mse_utilities(mse_rows, area_samples, temporal_weights):
    """Return (utility rows, importance samples, integrated weights).

    The first candidate is CRF40/Q0. Signed, nonmonotonic measured gains remain
    signed. A measured empty viewport has zero importance for every instance.
    """
    rows = tuple(tuple(row) for row in mse_rows)
    areas = tuple(tuple(row) for row in area_samples)
    temporal = tuple(temporal_weights)
    if not rows or not areas or len(temporal) != len(areas):
        raise ValueError('complete MSE rows and temporal visibility are required')
    if any(not row or any(isinstance(v, bool) or not math.isfinite(v) or v < 0
                          for v in row) for row in rows):
        raise ValueError('MSE measurements must be finite and nonnegative')
    if any(len(row) != len(rows) or any(isinstance(v, bool) or not math.isfinite(v) or v < 0
                                       for v in row) for row in areas):
        raise ValueError('visibility samples must cover every instance with finite nonnegative areas')
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in temporal):
        raise ValueError('temporal weights must be finite and nonnegative')
    total = math.fsum(temporal)
    if not math.isclose(total, 1., abs_tol=1e-12, rel_tol=0):
        raise ValueError('complete normalized temporal weights are required')
    temporal = tuple(w / total for w in temporal)
    importance = []
    for row in areas:
        total = math.fsum(row)
        importance.append(tuple(v / total if total else 0.0 for v in row))
    importance = tuple(importance)
    weights = tuple(math.fsum(w * sample[i] for w, sample in zip(temporal, importance))
                    for i in range(len(rows)))
    utilities = tuple(tuple(math.fsum(w * sample[i] * (row[0] - distortion)
                                      for w, sample in zip(temporal, importance))
                            for distortion in row) for i, row in enumerate(rows))
    return utilities, importance, weights
