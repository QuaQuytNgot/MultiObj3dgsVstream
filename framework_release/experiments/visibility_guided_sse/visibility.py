"""Independent CRF40 visibility and time aggregation for the frozen experiment.

This module reuses the production VisibilityAnalyzer's constant-colour Gaussian
contribution passes; it does not approximate them with AABBs or thresholded
foreground pixel counts. Importing it never imports torch or executes a GPU
operation. A native call is explicit and is reserved for Task 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.occlusion_joint_sse.visibility import ProxyFrame, build_proxy


ORACLE_CAMERA_TRACE = "ORACLE_CAMERA_TRACE"
VISIBILITY_PROXY = "VISIBILITY_PROXY"
JOINT_IMAGE_GROUND_TRUTH = "JOINT_IMAGE_GROUND_TRUTH"
JOINT_ALLOCATION_ORACLE = "JOINT_ALLOCATION_ORACLE"
CONTRIBUTION_CONVENTION = "CRF40_continuous_alpha_weighted_pixels_with_actual_frustum_clipping"
_ALLOWED_ROLES = {ORACLE_CAMERA_TRACE, VISIBILITY_PROXY}
_TOL = 1e-6


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolution(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in value):
        raise ValueError("Resolution requires positive integer (width,height)")
    return int(value[0]), int(value[1])


def _weights(values: Sequence[float], count: int) -> tuple[float, ...]:
    result = tuple(_finite(x, "Temporal weight") for x in values)
    if len(result) != count or not result or any(v < 0 for v in result):
        raise ValueError("One nonnegative normalized temporal weight per sample is required")
    if not math.isclose(math.fsum(result), 1.0, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Temporal weights must sum to one; implicit normalization is forbidden")
    return result


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    return value


def _roles(values: Sequence[str]) -> tuple[str, ...]:
    roles = tuple(values)
    if len(roles) != len(set(roles)) or VISIBILITY_PROXY not in roles or any(v not in _ALLOWED_ROLES for v in roles):
        raise ValueError("Weights require independent VISIBILITY_PROXY provenance; evaluation targets are forbidden")
    return roles


@dataclass(frozen=True)
class ContributionMass:
    A_solo: float
    A_vis: float
    frustum_included: bool | None = None

    def __post_init__(self) -> None:
        solo, visible = _finite(self.A_solo, "A_solo"), _finite(self.A_vis, "A_vis")
        tolerance = _TOL + 1e-5 * max(0.0, solo)
        if solo < -tolerance or visible < -tolerance or visible > solo + tolerance:
            raise ValueError("Contributions must satisfy 0 <= A_vis <= A_solo")
        # Match the existing analyzer's explicitly numerical-only correction.
        solo, visible = max(0.0, solo), max(0.0, visible)
        visible = min(visible, solo)
        if self.frustum_included is not None and not isinstance(self.frustum_included, bool):
            raise ValueError("Frustum inclusion must be boolean or unavailable")
        if self.frustum_included is False and solo > _TOL:
            raise ValueError("Outside-frustum object cannot have positive clipped contribution")
        object.__setattr__(self, "A_solo", solo)
        object.__setattr__(self, "A_vis", visible)

    @property
    def occlusion_ratio(self) -> float | None:
        return 1 - self.A_vis / self.A_solo if self.A_solo > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {"A_solo": self.A_solo, "A_vis": self.A_vis,
                "occlusion_ratio": self.occlusion_ratio,
                "frustum_included": self.frustum_included,
                "has_viewport_support": self.A_solo > 0}


@dataclass(frozen=True)
class VisibilitySample:
    timestamp: float
    per_object: Mapping[str, ContributionMass]
    resolution: tuple[int, int]
    proxy_identity: str
    provenance: tuple[str, ...] = (VISIBILITY_PROXY,)
    quality: int = 40
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        object.__setattr__(self, "resolution", _resolution(self.resolution))
        object.__setattr__(self, "proxy_identity", _identity(self.proxy_identity, "proxy identity"))
        object.__setattr__(self, "provenance", _roles(self.provenance))
        if isinstance(self.quality, bool) or self.quality != 40:
            raise ValueError("All policy visibility must use the fixed independent CRF40 proxy")
        masses = dict(self.per_object)
        if not masses or any(not isinstance(v, ContributionMass) for v in masses.values()):
            raise ValueError("One validated contribution record per independent object is required")
        width, height = self.resolution
        for key, value in masses.items():
            _identity(key, "object")
            if value.A_solo > width * height + _TOL + 1e-5 * width * height:
                raise ValueError("Continuous alpha mass cannot exceed viewport pixel count")
        metadata = dict(self.metadata)
        if metadata.get("candidate_independent", True) is not True:
            raise ValueError("Candidate-specific visibility is forbidden")
        if metadata.get("candidate_joint_RGB_or_SSE_used", False) is not False:
            raise ValueError("Candidate joint RGB/SSE is evaluation-only")
        if metadata.get("contribution_convention", CONTRIBUTION_CONVENTION) != CONTRIBUTION_CONVENTION:
            raise ValueError("Visibility must preserve continuous CRF40 alpha contribution")
        for name in ("analysis_time_ms", "proxy_acquisition_time_ms"):
            if name in metadata and _finite(metadata[name], name) < 0:
                raise ValueError("Measured timing must be nonnegative")
        for name in ("joint_passes", "solo_passes"):
            if name in metadata and (isinstance(metadata[name], (bool, np.bool_)) or
                                     not isinstance(metadata[name], (int, np.integer)) or metadata[name] < 0):
                raise ValueError("Raster-pass counts must be nonnegative integers")
        metadata.update(candidate_independent=True, candidate_joint_RGB_or_SSE_used=False,
                        contribution_convention=CONTRIBUTION_CONVENTION)
        object.__setattr__(self, "per_object", MappingProxyType(masses))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "resolution": list(self.resolution),
                "proxy_identity": self.proxy_identity, "provenance": list(self.provenance),
                "quality": self.quality, "per_object": {k: v.to_dict() for k, v in self.per_object.items()},
                "metadata": dict(self.metadata)}


def _consistent_samples(samples: Sequence[VisibilitySample]) -> tuple[VisibilitySample, ...]:
    rows = tuple(samples)
    if not rows or any(not isinstance(v, VisibilitySample) for v in rows):
        raise ValueError("Validated independent visibility samples are required")
    ids = set(rows[0].per_object)
    if any(set(row.per_object) != ids or row.resolution != rows[0].resolution for row in rows):
        raise ValueError("Object and resolution identities must match at every sample")
    if len({row.proxy_identity for row in rows}) != len(rows):
        raise ValueError("Every camera/time visibility sample needs a distinct proxy identity")
    return rows


@dataclass(frozen=True)
class OfflineContributionProfile:
    """Alpha-only denominator bound to the exact offline SSE distribution.

    Builders are required to provide every expected source camera/time proxy
    identity in order, including samples whose solo contribution is zero.
    CRF20 reference masks and joint-evaluation RGB are never accepted inputs.
    """
    values: Mapping[str, float]
    distribution_identity: str
    resolution: tuple[int, int]
    sample_identities: tuple[str, ...]
    temporal_weights: tuple[float, ...]
    provenance: tuple[str, ...] = (VISIBILITY_PROXY,)
    quality: int = 40
    convention: str = CONTRIBUTION_CONVENTION

    def __post_init__(self) -> None:
        values = {k: _finite(v, "Offline mean contribution") for k, v in self.values.items()}
        if not values or any(v <= 0 for v in values.values()):
            raise ValueError("P2 rejects zero or invalid offline mean contribution; no fallback weight")
        for key in values:
            _identity(key, "object")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "distribution_identity", _identity(self.distribution_identity, "Offline distribution"))
        object.__setattr__(self, "resolution", _resolution(self.resolution))
        viewport_pixels = self.resolution[0] * self.resolution[1]
        if any(v > viewport_pixels + _TOL + 1e-5 * viewport_pixels for v in values.values()):
            raise ValueError("Offline continuous alpha contribution exceeds viewport size")
        ids = tuple(self.sample_identities)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Distinct offline proxy identities are required")
        for identity in ids:
            _identity(identity, "Offline proxy")
        object.__setattr__(self, "sample_identities", ids)
        object.__setattr__(self, "temporal_weights", _weights(self.temporal_weights, len(ids)))
        object.__setattr__(self, "provenance", _roles(self.provenance))
        if self.quality != 40 or isinstance(self.quality, bool) or self.convention != CONTRIBUTION_CONVENTION:
            raise ValueError("Offline denominators require the independent continuous CRF40 alpha convention")

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "distribution_identity": self.distribution_identity,
                "resolution": list(self.resolution), "sample_identities": list(self.sample_identities),
                "temporal_weights": list(self.temporal_weights), "provenance": list(self.provenance),
                "quality": self.quality, "convention": self.convention,
                "derivation": "sum_normalized_offline_camera_time_weights_times_continuous_CRF40_solo_alpha_mass"}


def aggregate_offline_contribution(samples: Sequence[VisibilitySample], temporal_weights: Sequence[float], *,
                                   distribution_identity: str,
                                   expected_proxy_identities: Sequence[str]) -> OfflineContributionProfile:
    rows = _consistent_samples(samples)
    weights = _weights(temporal_weights, len(rows))
    actual = tuple(row.proxy_identity for row in rows)
    if actual != tuple(expected_proxy_identities):
        raise ValueError("Offline camera/time coverage differs from the declared SSE sampling distribution")
    return OfflineContributionProfile(
        {key: math.fsum(w * row.per_object[key].A_solo for row, w in zip(rows, weights)) for key in rows[0].per_object},
        distribution_identity, rows[0].resolution, actual, weights)


@dataclass(frozen=True)
class GOPObjectVisibility:
    A_solo: float
    A_vis: float
    r: float
    w: float
    current_A_solo: float
    current_A_vis: float
    entire_gop_invisible: bool
    entry_exit_events: tuple[dict[str, Any], ...]
    time_series: tuple[dict[str, Any], ...]

    @property
    def occlusion_ratio(self) -> float | None:
        return 1 - self.r if self.A_solo > 0 else None

    @property
    def relative_projected_scale(self) -> float:
        # A_solo/offline mean, distinguished from occlusion-weighted w.
        return self._scale

    @property
    def _scale(self) -> float:
        return math.fsum(row["weight"] * row["offline_relative_solo"] for row in self.time_series)

    def to_dict(self) -> dict[str, Any]:
        return {"A_solo": self.A_solo, "A_vis": self.A_vis, "r": self.r, "w": self.w,
                "occlusion_ratio": self.occlusion_ratio, "relative_projected_scale": self._scale,
                "current_A_solo": self.current_A_solo, "current_A_vis": self.current_A_vis,
                "entire_gop_invisible": self.entire_gop_invisible,
                "entry_exit_events": list(self.entry_exit_events), "time_series": list(self.time_series)}


@dataclass(frozen=True)
class GOPVisibility:
    per_object: Mapping[str, GOPObjectVisibility]
    timestamps: tuple[float, ...]
    temporal_weights: tuple[float, ...]
    proxy_identities: tuple[str, ...]
    offline_distribution_identity: str
    aggregation_time_ms: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"per_object": {k: v.to_dict() for k, v in self.per_object.items()},
                "timestamps": list(self.timestamps), "temporal_weights": list(self.temporal_weights),
                "proxy_identities": list(self.proxy_identities),
                "offline_distribution_identity": self.offline_distribution_identity,
                "aggregation_time_ms": self.aggregation_time_ms, "metadata": dict(self.metadata)}


def aggregate_gop(samples: Sequence[VisibilitySample], timestamps: Sequence[float],
                  temporal_weights: Sequence[float], offline_means: OfflineContributionProfile, *,
                  expected_offline_distribution_identity: str) -> GOPVisibility:
    """P1 is ratio of time aggregates; P2 is visible mean/offline solo mean.

    Expected timestamps must contain the entire frozen upcoming GOP trace,
    not just the decision-instant snapshot. Uniform [0,.5,1,1.5] samples use
    weights [.25,.25,.25,.25] (normalized left-rectangle integration).
    """
    started = time.perf_counter()
    rows = _consistent_samples(samples)
    times = tuple(_finite(v, "Trace timestamp") for v in timestamps)
    if len(times) < 2 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("A GOP requires multiple strictly increasing frozen trace timestamps")
    if tuple(row.timestamp for row in rows) != times:
        raise ValueError("Visibility must cover the complete frozen upcoming GOP trace in order")
    weights = _weights(temporal_weights, len(rows))
    if not isinstance(offline_means, OfflineContributionProfile):
        raise TypeError("P2 needs a validated independently derived offline contribution profile")
    if offline_means.distribution_identity != _identity(expected_offline_distribution_identity, "Expected offline distribution"):
        raise ValueError("Offline contribution distribution does not match the common solo SSE profile")
    if offline_means.resolution != rows[0].resolution or set(offline_means.values) != set(rows[0].per_object):
        raise ValueError("Offline and GOP object/resolution identities differ")
    objects = {}
    for key in rows[0].per_object:
        solo = math.fsum(w * row.per_object[key].A_solo for row, w in zip(rows, weights))
        visible = math.fsum(w * row.per_object[key].A_vis for row, w in zip(rows, weights))
        ratio = visible / solo if solo > 0 else 0.0
        series = tuple({"timestamp": row.timestamp, "weight": weight, **row.per_object[key].to_dict(),
                        "offline_relative_solo": row.per_object[key].A_solo / offline_means.values[key]}
                       for row, weight in zip(rows, weights))
        # Entry/exit means measurable viewport support, distinct from zero
        # visible mass due solely to occlusion. No hand-tuned selection cutoff.
        events = tuple({"timestamp": times[i], "event": "entry" if rows[i].per_object[key].A_solo > 0 else "exit"}
                       for i in range(1, len(rows))
                       if (rows[i].per_object[key].A_solo > 0) != (rows[i - 1].per_object[key].A_solo > 0))
        objects[key] = GOPObjectVisibility(solo, visible, ratio, visible / offline_means.values[key],
                                           rows[0].per_object[key].A_solo, rows[0].per_object[key].A_vis,
                                           all(row.per_object[key].A_vis == 0 for row in rows), events, series)
    metadata = {"formula_P1": "sum_t(weight*A_vis)/sum_t(weight*A_solo); zero_solo=>0",
                "formula_P2": "sum_t(weight*A_vis)/offline_mean_solo_contribution",
                "full_upcoming_trace": True, "cross_object_normalization": False,
                "P2_clamped": False, "provenance": [ORACLE_CAMERA_TRACE, VISIBILITY_PROXY],
                "joint_target_or_candidate_used": False, "contribution_convention": CONTRIBUTION_CONVENTION,
                "joint_passes": sum(int(row.metadata.get("joint_passes", 0)) for row in rows),
                "solo_passes": sum(int(row.metadata.get("solo_passes", 0)) for row in rows),
                "visibility_analysis_time_ms": math.fsum(float(row.metadata.get("analysis_time_ms", 0.0)) for row in rows),
                "proxy_acquisition_time_ms": math.fsum(float(row.metadata.get("proxy_acquisition_time_ms", 0.0)) for row in rows)}
    return GOPVisibility(MappingProxyType(objects), times, weights, tuple(row.proxy_identity for row in rows),
                         offline_means.distribution_identity, (time.perf_counter() - started) * 1000,
                         MappingProxyType(metadata))


class VisibilityAnalyzerAdapter:
    """Reuse production analysis or validated cached CRF40 contribution maps.

    The default analyzer is imported and constructed only on native analyze().
    CPU tests inject an analyzer; no task-1 command needs a GPU backend.
    """
    def __init__(self, analyzer: Any = None) -> None:
        self._analyzer = analyzer

    def analyze(self, objects: Mapping[str, Any], camera: Any, timestamp: float, *,
                proxy_identity: str, quality: int = 40,
                provenance: tuple[str, ...] = (ORACLE_CAMERA_TRACE, VISIBILITY_PROXY),
                proxy_acquisition_time_ms: float = 0.0,
                frustum_inclusion: Mapping[str, bool] | None = None) -> VisibilitySample:
        _roles(provenance)
        _identity(proxy_identity, "proxy")
        _finite(timestamp, "timestamp")
        _resolution((camera.width, camera.height))
        if quality != 40 or isinstance(quality, bool):
            raise ValueError("Native visibility requires independently acquired CRF40 states")
        if not objects:
            raise ValueError("Independent object states are required")
        if frustum_inclusion is not None and set(frustum_inclusion) != set(objects):
            raise ValueError("Frustum audit must cover every independent object")
        acquisition_ms = _finite(proxy_acquisition_time_ms, "Proxy acquisition time")
        if acquisition_ms < 0:
            raise ValueError("Proxy acquisition time cannot be negative")
        if self._analyzer is None:
            from multiobj3dgs.client.visibility.analyzer import VisibilityAnalyzer
            self._analyzer = VisibilityAnalyzer()
        started = time.perf_counter()
        result = self._analyzer.analyze(objects, camera, timestamp, epsilon=0.0,
                                        diagnostics=False, revision=proxy_identity)
        elapsed = (time.perf_counter() - started) * 1000
        if set(result.per_object) != set(objects) or result.timestamp != timestamp:
            raise ValueError("Analyzer object/time identities differ from independent Q0 request")
        expected_resolution = (camera.width, camera.height)
        if tuple(result.resolution) != expected_resolution:
            raise ValueError("Analyzer resolution differs from frozen camera")
        per_object = {key: ContributionMass(row.A_solo, row.A_visible,
                                           frustum_inclusion[key] if frustum_inclusion is not None else None)
                      for key, row in result.per_object.items()}
        return VisibilitySample(timestamp, per_object, expected_resolution, proxy_identity, provenance, quality,
                                {**result.metadata, "analysis_time_ms": elapsed,
                                 "proxy_acquisition_time_ms": acquisition_ms,
                                 "adapter": "existing_VisibilityAnalyzer_constant_colour_CRF40_contribution",
                                 "frustum_clipping": "native_rasterizer_exact_viewport_clip"})

    def from_cached_proxy(self, proxy: ProxyFrame, *, object_ids: Sequence[str], timestamp: float,
                          provenance: tuple[str, ...] = (ORACLE_CAMERA_TRACE, VISIBILITY_PROXY),
                          frustum_inclusion: Mapping[str, bool] | None = None) -> VisibilitySample:
        """Validate effective T and contribution partition, then sum alpha.

        No thresholded pixel count enters any policy weight. This checks the
        cache's maps rather than trusting its precomputed visible fractions.
        """
        if not isinstance(proxy, ProxyFrame) or proxy.quality != 40:
            raise ValueError("A separately acquired CRF40 ProxyFrame is required")
        _roles(provenance)
        ids = tuple(object_ids)
        if not ids or len(set(ids)) != len(ids) or len(ids) != len(proxy.solo_alpha):
            raise ValueError("Cached contribution object identities/order differ")
        if proxy.statistics.get("candidate_independent") is not True or proxy.statistics.get("candidate_joint_RGB_or_SSE_used") is not False:
            raise ValueError("Cached visibility must be candidate-independent")
        rebuilt = build_proxy(proxy.solo_alpha, proxy.contributions, proxy.joint_alpha,
                              source_identity=proxy.source_identity, quality=proxy.quality, method=proxy.method)
        if not np.allclose(np.sum(rebuilt.contributions, axis=0), rebuilt.joint_alpha, rtol=1e-5, atol=1e-4):
            raise ValueError("Object contribution sum is inconsistent with joint alpha")
        if proxy.transmittance.shape != rebuilt.transmittance.shape or proxy.visible_fraction.shape != rebuilt.visible_fraction.shape or not np.allclose(proxy.transmittance, rebuilt.transmittance, rtol=1e-5, atol=1e-4) or not np.allclose(proxy.visible_fraction, rebuilt.visible_fraction, rtol=1e-5, atol=1e-4):
            raise ValueError("Cached effective transmittance/visible fraction is inconsistent with alpha")
        if frustum_inclusion is not None and set(frustum_inclusion) != set(ids):
            raise ValueError("Frustum audit must cover all cached objects")
        solo = np.sum(rebuilt.solo_alpha, axis=(1, 2), dtype=np.float64)
        visible = np.sum(rebuilt.contributions, axis=(1, 2), dtype=np.float64)
        height, width = rebuilt.joint_alpha.shape
        return VisibilitySample(timestamp,
                                {key: ContributionMass(solo[i], visible[i], None if frustum_inclusion is None else frustum_inclusion[key])
                                 for i, key in enumerate(ids)},
                                (width, height), proxy.source_identity, provenance, proxy.quality,
                                {"adapter": "validated_existing_ProxyFrame", "cache_reused": True,
                                 "joint_passes": 0, "solo_passes": 0, "analysis_time_ms": 0.0,
                                 "contribution_partition_max_abs_error": float(np.max(np.abs(rebuilt.contributions.sum(axis=0) - rebuilt.joint_alpha))),
                                 "frustum_clipping": "preserved_from_cached_native_CRF40_pixel_maps"})
