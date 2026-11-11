"""Independent, actual viewport-frame quality inputs; never policy utility.

The measurement interface intentionally knows no adaptation planner.  A later
renderer can implement ``QualityProvider``; existing *joint viewport* renders
can be supplied as validated scalar SSE/count cache records.  Object PSNR
profiles and utility scores cannot be used as viewport measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from experiments.nava_mse.metrics import MetricStats, pool_metrics
from .timeline import exact_time


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _time(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return number


@dataclass(frozen=True)
class FrameQualityContext:
    """Exact frame/allocation identity, including immutable scene and trace IDs.

    ``global_time_s`` is the continuous media timeline, not download wall time.
    ``camera_sample_id`` must identify the exact pose within ``trace_id``.
    Cache lookups include the global segment/loop: no accidental reuse across
    repeated segments or different selected allocations is permitted.
    """

    scene_id: str
    content_revision: str
    trace_id: str
    camera_sample_id: str
    global_segment_id: str
    content_gop_id: str
    loop_index: int
    global_time_s: float
    content_time_s: float
    selected_quality: Mapping[str, int | str]
    frame_duration_s: float

    def __post_init__(self) -> None:
        for name in ("scene_id", "content_revision", "trace_id", "camera_sample_id",
                     "global_segment_id", "content_gop_id"):
            _nonempty(getattr(self, name), name)
        if type(self.loop_index) is not int or self.loop_index < 0:
            raise ValueError("loop_index must be a nonnegative integer")
        for name in ("global_time_s", "content_time_s", "frame_duration_s"):
            object.__setattr__(self, name, _time(getattr(self, name), name, positive=name == "frame_duration_s"))
        if not isinstance(self.selected_quality, Mapping) or not self.selected_quality:
            raise ValueError("selected_quality must identify every selected object")
        for instance, quality in self.selected_quality.items():
            _nonempty(instance, "instance_id")
            if type(quality) not in (str, int) or isinstance(quality, str) and not quality:
                raise ValueError("Selected quality identifiers must be strings or integers")
        object.__setattr__(self, "selected_quality", dict(sorted(self.selected_quality.items())))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FrameQualityMeasurement:
    context: FrameQualityContext
    metrics: MetricStats
    reference_identity: str
    mask_mode: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        _nonempty(self.reference_identity, "reference_identity")
        if not isinstance(self.metrics, MetricStats):
            raise ValueError("Actual frame measurements require MetricStats SSE and RGB counts")
        if self.mask_mode not in {"full_viewport", "reference_foreground"}:
            raise ValueError("mask_mode must declare full_viewport or reference_foreground")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Actual frame measurements require provenance")
        for name in ("reference_frame_id", "candidate_frame_id", "measurement_source"):
            _nonempty(self.provenance.get(name), name)

    def to_dict(self) -> dict[str, Any]:
        return {**self.context.to_dict(), "available": True,
                "reference_identity": self.reference_identity, "mask_mode": self.mask_mode,
                "evaluated_pixel_count": self.metrics.foreground_pixels,
                "rgb_sample_count": self.metrics.sample_count,
                **self.metrics.to_dict(), "provenance": dict(self.provenance)}


class QualityProvider(Protocol):
    def measure(self, context: FrameQualityContext) -> FrameQualityMeasurement | None:
        """Return actual measurements for this exact frame/allocation, or None."""
        ...


class NullQualityProvider:
    def measure(self, context: FrameQualityContext) -> None:
        return None


class CachedQualityProvider:
    """Strict JSON cache of actual *joint* viewport measurements.

    Top-level fields: schema_version=1,
    information_role='ACTUAL_VIEWPORT_FRAME_MEASUREMENTS',
    measurement_kind='joint_viewport_rgb', reference_identity, protocol, frames.
    Protocol declares mask_mode, channels=3, data_range, normalization. Each
    frame contains context, MetricStats-compatible metrics, and provenance.
    Missing keys are unavailable; duplicate or conflicting records are errors.
    """

    def __init__(self, document: Mapping[str, Any]):
        if (type(document.get("schema_version")) is not int or document.get("schema_version") != 1 or
                document.get("information_role") != "ACTUAL_VIEWPORT_FRAME_MEASUREMENTS" or
                document.get("measurement_kind") != "joint_viewport_rgb"):
            raise ValueError("Quality cache must declare actual joint viewport frame measurements")
        reference = _nonempty(document.get("reference_identity"), "reference_identity")
        protocol = document.get("protocol", {})
        if not isinstance(protocol, Mapping) or set(protocol) != {"mask_mode", "channels", "data_range", "normalization"}:
            raise ValueError("Cache protocol requires mask_mode, channels, data_range and normalization")
        if protocol["channels"] != 3:
            raise ValueError("Viewport measurements must include all three RGB channels")
        # Validate protocol even for an empty cache (before any frame is read).
        MetricStats(0., 1, channels=protocol["channels"], data_range=protocol["data_range"],
                    normalization=protocol["normalization"])
        if protocol["mask_mode"] not in {"full_viewport", "reference_foreground"}:
            raise ValueError("mask_mode must declare full_viewport or reference_foreground")
        frames = document.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Quality cache frames must be a list")
        self._frames: dict[str, FrameQualityMeasurement] = {}
        for record in frames:
            context = FrameQualityContext(**record["context"])
            stats = MetricStats.from_dict(record["metrics"])
            if (stats.channels, stats.data_range, stats.normalization) != (
                    protocol["channels"], protocol["data_range"], protocol["normalization"]):
                raise ValueError("Cached frame normalization differs from the declared protocol")
            measurement = FrameQualityMeasurement(context, stats, reference, protocol["mask_mode"], record["provenance"])
            if context.key in self._frames:
                raise ValueError("Duplicate quality cache frame/allocation identity")
            self._frames[context.key] = measurement

    @classmethod
    def from_json(cls, path: str | Path) -> "CachedQualityProvider":
        with Path(path).open(encoding="utf8") as source:
            document = json.load(source)
        if not isinstance(document, Mapping):
            raise ValueError("Quality cache must be a JSON object")
        return cls(document)

    def measure(self, context: FrameQualityContext) -> FrameQualityMeasurement | None:
        return self._frames.get(context.key)


def contexts_for_playback(*, scene_id, content_revision, trace_provider, timeline,
                          decisions, played_duration_s, source_gop_ids,
                          mode="trace_samples", fps=None) -> list[FrameQualityContext]:
    """Bind independent measurement requests to actual played allocations.

    The caller supplies decisions whose scene downloads completed. A context
    represents a frame sample, never an inferred image-quality measurement.
    ``trace_samples`` preserves the supplied (potentially sparse) pose trace;
    its durations are temporal support intervals, not new generated frames.
    ``frame_grid`` requires an exact pose for every frame on a rational FPS
    grid. It never interpolates a pose or silently substitutes a nearby sample.
    """
    if mode not in {"trace_samples", "frame_grid"}:
        raise ValueError("Quality sampling mode must be trace_samples or frame_grid")
    played_end = exact_time(played_duration_s, "played_duration_s")
    if played_end > exact_time(timeline.total_duration_s):
        raise ValueError("played_duration_s exceeds the experiment duration")
    frame_rate = None
    if mode == "frame_grid":
        frame_rate = exact_time(fps, "fps", positive=True)
        for name in ("source_duration_s", "gop_duration_s"):
            if (exact_time(getattr(timeline, name)) * frame_rate).denominator != 1:
                raise ValueError(f"frame_grid requires frame-aligned {name}")
    if not isinstance(source_gop_ids, (list, tuple)) or not source_gop_ids:
        raise ValueError("source_gop_ids must map content GOP indices to source profile IDs")
    selected_decisions = {}
    for decision in decisions:
        index = decision["global_segment_index"]
        if type(index) is not int or not 0 <= index < len(timeline):
            raise ValueError("Decision global_segment_index is outside the media timeline")
        if index in selected_decisions:
            raise ValueError("Duplicate quality decision for a global segment")
        selected_decisions[index] = decision
    contexts = []
    for index, decision in sorted(selected_decisions.items()):
        if decision.get("selected_quality") is None:
            continue
        segment = timeline.segments[index]
        start, end = timeline.segment_bounds_exact(segment)
        end = min(end, played_end)
        if end <= start:
            continue
        if segment.source_gop_index >= len(source_gop_ids):
            raise ValueError("source_gop_ids does not cover the played source GOP")
        poses = trace_provider.get_trajectory(segment.gop_id)
        if not poses:
            raise ValueError("Played quality context requires an exact camera trace")
        pose_by_time = {float(p.timestamp): p for p in poses}
        if len(pose_by_time) != len(poses):
            raise ValueError("Camera trace contains duplicate pose timestamps")
        if mode == "frame_grid":
            # Fraction boundaries avoid an extra tiny frame at an exact end.
            times, k = [], 0
            while start + k / frame_rate < end:
                times.append(start + k / frame_rate)
                k += 1
        else:
            times = [exact_time(p.timestamp) for p in poses if start <= exact_time(p.timestamp) < end]
        for position, timestamp in enumerate(times):
            pose = pose_by_time.get(float(timestamp))
            if pose is None:
                raise ValueError(f"Missing exact camera pose for frame_grid timestamp {float(timestamp)}")
            boundary = min(end, timestamp + 1 / frame_rate) if mode == "frame_grid" else (
                times[position + 1] if position + 1 < len(times) else end)
            if boundary <= timestamp:
                raise ValueError("Frame quality context must have positive temporal support")
            contexts.append(FrameQualityContext(
                scene_id=scene_id, content_revision=content_revision, trace_id=trace_provider.trace_id,
                camera_sample_id=pose.identity, global_segment_id=segment.request_id,
                content_gop_id=str(source_gop_ids[segment.source_gop_index]), loop_index=segment.loop_index,
                global_time_s=float(timestamp), content_time_s=timeline.content_time(timestamp),
                selected_quality=decision["selected_quality"], frame_duration_s=float(boundary - timestamp)))
    return contexts


def _psnr_value(value: float | None) -> dict[str, Any]:
    return {"db": None if value is None or math.isinf(value) else value,
            "infinite": value is not None and math.isinf(value)}


def collect_quality(contexts: Iterable[FrameQualityContext], provider: QualityProvider | None = None) -> dict[str, Any]:
    """Frame arithmetic mean, duration mean, and pooled RGB-SSE PSNR.

    Pooled PSNR sums SSE and evaluated RGB counts, not dB values.  Arithmetic
    and duration means are over available measurements only, with explicit
    coverage. Perfect frames preserve positive infinity using a JSON-safe flag.
    Mixed reference/mask/normalization protocols are rejected, never pooled.
    """
    provider = provider or NullQualityProvider()
    frames, measurements, identities = [], [], set()
    duration_total = duration_measured = 0.0
    for context in contexts:
        if context.key in identities:
            raise ValueError("Duplicate requested frame/allocation identity")
        identities.add(context.key)
        duration_total += context.frame_duration_s
        measurement = provider.measure(context)
        if measurement is None:
            frames.append({**context.to_dict(), "available": False, "psnr_db": None,
                           "psnr_infinite": False, "sse": None, "rgb_sample_count": None,
                           "unavailable_reason": "No actual joint viewport frame measurement supplied"})
            continue
        if measurement.context.key != context.key:
            raise ValueError("Quality provider returned a different frame, pose, content or allocation")
        if measurements and (measurement.reference_identity, measurement.mask_mode) != (
                measurements[0].reference_identity, measurements[0].mask_mode):
            raise ValueError("Cannot combine frames with different reference identities or mask modes")
        measurements.append(measurement)
        duration_measured += context.frame_duration_s
        frames.append(measurement.to_dict())
    pooled = pool_metrics(m.metrics for m in measurements) if measurements else None
    infinite = any(math.isinf(m.metrics.psnr) for m in measurements)
    mean = (math.inf if infinite else math.fsum(m.metrics.psnr for m in measurements) / len(measurements)) if measurements else None
    weighted = (math.inf if infinite else math.fsum(m.metrics.psnr * m.context.frame_duration_s for m in measurements) / duration_measured) if measurements else None
    mean_value, weighted_value, pooled_value = _psnr_value(mean), _psnr_value(weighted), _psnr_value(pooled.psnr if pooled else None)
    return {"frames": frames, "summary": {
        "quality_source": "actual_joint_viewport_measurements", "frame_count": len(frames),
        "measured_frame_count": len(measurements), "unavailable_frame_count": len(frames) - len(measurements),
        "measurement_coverage": len(measurements) / len(frames) if frames else None,
        "measured_duration_s": duration_measured, "requested_duration_s": duration_total,
        "duration_coverage": duration_measured / duration_total if duration_total else None,
        "aggregation_scope": "available_measured_frames_only",
        "average_frame_psnr_db": mean_value["db"], "average_frame_psnr_infinite": mean_value["infinite"],
        "duration_weighted_psnr_db": weighted_value["db"], "duration_weighted_psnr_infinite": weighted_value["infinite"],
        "pooled_psnr_db": pooled_value["db"], "pooled_psnr_infinite": pooled_value["infinite"],
        "pooled_sse": pooled.sse if pooled else None,
        "pooled_rgb_sample_count": pooled.sample_count if pooled else None,
        "reference_identity": measurements[0].reference_identity if measurements else None,
        "mask_mode": measurements[0].mask_mode if measurements else None,
        "protocol": {"channels": pooled.channels, "data_range": pooled.data_range,
                     "normalization": pooled.normalization} if pooled else None,
    }}
