"""GOP allocation from explicit per-object inputs, independent of evaluation.

This module deliberately imports neither renderers nor oracle/joint metrics.
The only view-conditioned quality information is the declared in-sample
per-object exact profile. One choice per object covers the complete GOP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral

from .solvers import SolverResult, exact_byte_dp, nava_graph_dp

NAVA_STREAMSTGS_ADAPTED = "NAVA_STREAMSTGS_ADAPTED"
AVERAGE_MSE = "AVERAGE_MSE"
EXACT_VIEW_PROFILE = "EXACT_VIEW_PROFILE"
POLICIES = (NAVA_STREAMSTGS_ADAPTED, AVERAGE_MSE, EXACT_VIEW_PROFILE)
NAVA_PROXY_REFERENCE = "original_ply_synthetic_rgb"


class PolicyUnavailable(ValueError):
    """A policy has no complete, correctly bound quality information."""


@dataclass(frozen=True)
class PolicyInputs:
    object_ids: tuple[str, ...]
    crfs: tuple[int, ...]
    costs_bytes: tuple[tuple[int, ...], ...]
    budget_bytes: int
    projected_areas: tuple[tuple[float, ...], ...]
    sample_weights: tuple[float, ...]
    average_mse: tuple[tuple[float, ...], ...] | None = None
    exact_mse: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    nava_psnr: tuple[tuple[float, ...], ...] | None = None
    proxy_reference: str = NAVA_PROXY_REFERENCE
    gop: int = 0

    def __post_init__(self):
        if (not self.object_ids or len(set(self.object_ids)) != len(self.object_ids)
                or any(not isinstance(v, str) or not v for v in self.object_ids)):
            raise ValueError("Require distinct nonempty object identities")
        if not self.crfs or len(set(self.crfs)) != len(self.crfs) or 40 not in self.crfs:
            raise ValueError("Distinct CRFs including baseline CRF40 are required")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in self.crfs):
            raise ValueError("CRF identities must be integers")
        if isinstance(self.gop, bool) or not isinstance(self.gop, Integral) or self.gop < 0:
            raise ValueError("gop must be a nonnegative integer")
        n, k = len(self.object_ids), len(self.crfs)
        if len(self.costs_bytes) != n or any(len(row) != k for row in self.costs_bytes):
            raise ValueError("Physical cost matrix must cover every object and CRF")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for row in self.costs_bytes for v in row):
            raise ValueError("Physical costs must be nonnegative integer bytes")
        if isinstance(self.budget_bytes, bool) or not isinstance(self.budget_bytes, Integral) or self.budget_bytes < 0:
            raise ValueError("budget_bytes must be a nonnegative integer")
        if not self.projected_areas or len(self.sample_weights) != len(self.projected_areas):
            raise ValueError("Require matching area and temporal sample rows")
        if any(len(row) != n for row in self.projected_areas):
            raise ValueError("Each projected-area row must cover all objects")
        if any(not math.isfinite(v) or v < 0 for row in self.projected_areas for v in row):
            raise ValueError("Projected areas must be finite and nonnegative")
        if any(not math.isfinite(v) or v < 0 for v in self.sample_weights) or math.fsum(self.sample_weights) <= 0:
            raise ValueError("Temporal weights must be finite, nonnegative and have positive sum")


@dataclass(frozen=True)
class PolicyDecision:
    policy: str
    gop: int
    object_ids: tuple[str, ...]
    selected_crfs: tuple[int, ...] | None
    utilities: tuple[tuple[float, ...], ...]
    importance: tuple[tuple[float, ...], ...]
    temporal_weights: tuple[float, ...]
    solver: SolverResult
    quality_information: str

    @property
    def selection(self):
        return self.solver.selection

    @property
    def status(self):
        return self.solver.status

    @property
    def cost_bytes(self):
        return self.solver.cost_bytes

    def to_dict(self) -> dict:
        result = asdict(self)
        result["solver"] = self.solver.to_dict()
        result["allocation"] = (dict(zip(self.object_ids, self.selected_crfs))
                                if self.selected_crfs is not None else None)
        return result


def projected_aabb_importance(projected_areas) -> tuple[tuple[float, ...], ...]:
    """Paper Eq. 8: normalize each frame's area, preserve all-invisible zeros."""
    result = []
    for row in projected_areas:
        if any(not math.isfinite(v) or v < 0 for v in row):
            raise ValueError("Projected areas must be finite and nonnegative")
        total = math.fsum(row)
        result.append(tuple(v / total if total else 0.0 for v in row))
    return tuple(result)


def _matrix(value, inputs: PolicyInputs, name: str, *, nonnegative: bool):
    if value is None:
        raise PolicyUnavailable(f"{name} profile unavailable; no substitution or nearest-view lookup")
    if len(value) != len(inputs.object_ids) or any(len(row) != len(inputs.crfs) for row in value):
        raise PolicyUnavailable(f"{name} profile must cover all objects and CRFs")
    if any(not math.isfinite(v) or (nonnegative and v < 0) for row in value for v in row):
        raise PolicyUnavailable(f"{name} profile requires finite {'nonnegative ' if nonnegative else ''}values")
    return value


def allocate(policy: str, inputs: PolicyInputs, **solver_options) -> PolicyDecision:
    """Select one representation per object, after aggregating the whole GOP.

    MSE is normalized per foreground RGB scalar, not SSE. Multiplying its
    baseline reduction by normalized projected AABB area supplies exactly one
    area factor. There is no further foreground-pixel or visibility multiplier.
    Frame weights are normalized once and applied identically to all policies.
    """
    if not isinstance(inputs, PolicyInputs):
        raise TypeError("Policies require explicit PolicyInputs, never evaluation/oracle objects")
    if policy not in POLICIES:
        raise ValueError(f"Unknown allocation policy {policy!r}")
    importance = projected_aabb_importance(inputs.projected_areas)
    total_time = math.fsum(inputs.sample_weights)
    temporal = tuple(v / total_time for v in inputs.sample_weights)
    n, k, baseline = len(inputs.object_ids), len(inputs.crfs), inputs.crfs.index(40)
    if policy == NAVA_STREAMSTGS_ADAPTED:
        if inputs.proxy_reference != NAVA_PROXY_REFERENCE:
            raise PolicyUnavailable("NAVA proxy requires original_ply_synthetic_rgb provenance, not CRF20 self-reference")
        quality = _matrix(inputs.nava_psnr, inputs, "original-PLY image-PSNR", nonnegative=False)
        utility = tuple(tuple(math.fsum(w * row[i] * quality[i][q] for w, row in zip(temporal, importance))
                              for q in range(k)) for i in range(n))
        solve = nava_graph_dp
        information = "finite original_ply_synthetic_rgb image-PSNR metadata proxy; not geometry PSNR"
    else:
        if policy == AVERAGE_MSE:
            average = _matrix(inputs.average_mse, inputs, "camera-independent average-MSE", nonnegative=True)
            samples = (average,) * len(temporal)
            information = "camera-independent pooled per-object MSE"
        else:
            if inputs.exact_mse is None or len(inputs.exact_mse) != len(temporal):
                raise PolicyUnavailable("Exact camera/time profile unavailable; every GOP sample is required")
            samples = tuple(_matrix(row, inputs, "exact camera/time MSE", nonnegative=True) for row in inputs.exact_mse)
            information = "in-sample exact per-object camera/time MSE; not joint-SSE oracle"
        utility = tuple(tuple(math.fsum(w * areas[i] * (sample[i][baseline] - sample[i][q])
                                         for w, areas, sample in zip(temporal, importance, samples))
                              for q in range(k)) for i in range(n))
        solve = exact_byte_dp
    result = solve(inputs.costs_bytes, utility, inputs.budget_bytes, **solver_options)
    selected = tuple(inputs.crfs[q] for q in result.selection) if result.selection is not None else None
    return PolicyDecision(policy, inputs.gop, inputs.object_ids, selected, utility, importance,
                          temporal, result, information)
