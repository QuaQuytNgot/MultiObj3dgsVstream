"""Visibility from renderer-measured alpha-compositing contributions.

Gaussian snapshots must already be resolved at the requested media timestamp.
The renderer owns transforms, projection, compositing, and GPU map reduction;
this module converts its scalar measurements into per-object occlusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import copy
from numbers import Real
from time import perf_counter
from typing import Any

from src.renderer.analysis import (
    ContributionDiagnostics,
    VisibilityObject,
    measure_contributions,
)

__all__ = [
    "ObjectVisibility", "VisibilityObject", "VisibilityResult", "analyze_visibility",
]


@dataclass(frozen=True, slots=True)
class ObjectVisibility:
    """Alpha-weighted pixel areas and suppression by other scene objects.

    ``occlusion_ratio`` is undefined when the solo contribution is at or below
    epsilon. ``out_of_view`` also covers states with no effective contribution,
    even when their geometry falls inside the camera frustum.
    """

    object_id: str
    A_visible: float
    A_solo: float
    occlusion_ratio: float | None
    out_of_view: bool


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    timestamp: float
    resolution: tuple[int, int]  # (width, height)
    epsilon: float
    per_object: dict[str, ObjectVisibility]
    diagnostics: ContributionDiagnostics | None
    metadata: dict[str, Any]


def _object_visibility(object_id: str, visible: float, solo: float,
                       epsilon: float) -> ObjectVisibility:
    visible, solo = float(visible), float(solo)
    if not math.isfinite(visible) or not math.isfinite(solo):
        raise ValueError(f"{object_id}: contribution areas must be finite")

    # The absolute term covers near-zero reductions; the relative term scales
    # with the solo area. Only numerical excursions are corrected.
    tolerance = 1e-6 + 1e-5 * max(0.0, solo)
    if visible < -tolerance or solo < -tolerance:
        raise ValueError(f"{object_id}: contribution areas must be nonnegative")
    visible, solo = max(0.0, visible), max(0.0, solo)
    if visible > solo + tolerance:
        raise ValueError(f"{object_id}: A_visible exceeds A_solo")
    visible = min(visible, solo)

    out_of_view = solo <= epsilon
    occlusion = None if out_of_view else 1.0 - visible / solo
    return ObjectVisibility(object_id, visible, solo, occlusion, out_of_view)


def analyze_visibility(
    objects: Sequence[VisibilityObject],
    camera: Mapping[str, Any],
    timestamp: float,
    *,
    epsilon: float = 1e-6,
    diagnostics: bool = False,
    renderer_settings: Mapping[str, Any] | None = None,
    profile: bool = False,
) -> VisibilityResult:
    """Measure any resolved Gaussian state's joint/solo contributions.

    Areas are unnormalized alpha-weighted pixel sums. There is no additional
    distance weighting. The solo pass includes self-occlusion, so the ratio
    measures suppression caused by other scene objects only.

    ``epsilon`` is a finite, nonnegative threshold in alpha-weighted pixel
    units. Full contribution maps are returned only with ``diagnostics=True``.
    Renderer input validation and backend errors propagate to the caller.
    Source metadata is returned as provenance and never controls compositing.
    ``profile=True`` records synchronized phase timing and allocator peaks.
    """
    wall_start = perf_counter()
    try:
        valid_epsilon = (
            isinstance(epsilon, Real)
            and not isinstance(epsilon, bool)
            and math.isfinite(epsilon)
            and epsilon >= 0
        )
    except (TypeError, ValueError, OverflowError):
        valid_epsilon = False
    if not valid_epsilon:
        raise ValueError("epsilon must be finite and nonnegative")
    epsilon = float(epsilon)

    profile_options = {"profile": profile} if profile is not False else {}
    measurements = measure_contributions(
        objects, camera, timestamp,
        diagnostics=diagnostics, renderer_settings=renderer_settings,
        **profile_options,
    )
    per_object = {}
    for object_id, measured in measurements.per_object.items():
        if measured.object_id != object_id:
            raise ValueError(f"{object_id}: renderer contribution object_id does not match its key")
        per_object[object_id] = _object_visibility(
            object_id, measured.A_visible, measured.A_solo, epsilon,
        )
    metadata = dict(measurements.metadata)
    sources = {obj.object_id: copy.deepcopy(obj.source_metadata) for obj in objects
               if getattr(obj, "source_metadata", None)}
    if sources:
        metadata["sources"] = sources
    if profile:
        metadata["profile"] = dict(metadata["profile"])
        metadata["profile"]["analyze_visibility_wall_ms"] = (perf_counter() - wall_start) * 1000
    return VisibilityResult(
        timestamp=measurements.timestamp,
        resolution=measurements.resolution,
        epsilon=epsilon,
        per_object=per_object,
        diagnostics=measurements.diagnostics,
        metadata=metadata,
    )
