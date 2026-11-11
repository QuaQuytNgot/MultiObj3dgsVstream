"""Average and exact profiles built from one provenance-bound image dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .metrics import MetricStats, pool_metrics


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    # Accept both flattened matrices and nested JSON/numpy matrices.
    def flatten(item: Any) -> Iterable[float]:
        if isinstance(item, (str, bytes)):
            raise ValueError(f"{name} must contain numeric values")
        try:
            iterator = iter(item)
        except TypeError:
            yield float(item)
        else:
            for element in iterator:
                yield from flatten(element)
    result = tuple(flatten(value))
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class DistortionSample:
    object_id: str
    gop: int
    camera_id: str
    camera_pose: tuple[float, ...]
    timestamp: float
    frame_index: int
    crf: int
    metrics: MetricStats
    projected_scale: float | None = None
    camera_relative_position: tuple[float, ...] | None = None
    camera_relative_orientation: tuple[float, ...] | None = None
    normalized_media_time: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id or not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("Object and camera identities must be nonempty strings")
        for name in ("gop", "frame_index", "crf"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        object.__setattr__(self, "camera_pose", _vector(self.camera_pose, 16, "camera_pose"))
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if self.projected_scale is not None and (not math.isfinite(self.projected_scale) or self.projected_scale < 0):
            raise ValueError("Projected scale must be finite and nonnegative when available")
        if self.camera_relative_position is not None:
            object.__setattr__(self, "camera_relative_position", _vector(self.camera_relative_position, 3, "camera_relative_position"))
        if self.camera_relative_orientation is not None:
            object.__setattr__(self, "camera_relative_orientation", _vector(self.camera_relative_orientation, 9, "camera_relative_orientation"))
        if self.normalized_media_time is not None and (not math.isfinite(self.normalized_media_time) or not 0 <= self.normalized_media_time <= 1):
            raise ValueError("Normalized media time must be in [0, 1]")
        # Fail on non-JSON provenance immediately; never stringify missing identities.
        copied = json.loads(_canonical(dict(self.provenance)))
        object.__setattr__(self, "provenance", copied)
        if not isinstance(self.metrics, MetricStats):
            raise ValueError("metrics must be MetricStats")

    @property
    def view_key(self) -> tuple[Any, ...]:
        return (self.object_id, self.gop, self.camera_id, self.camera_pose, self.timestamp, self.projected_scale)

    @property
    def exact_key(self) -> tuple[Any, ...]:
        return (*self.view_key, self.crf)

    def to_dict(self) -> dict[str, Any]:
        return {"object_id": self.object_id, "gop": self.gop, "camera_id": self.camera_id,
                "camera_pose": list(self.camera_pose), "timestamp": self.timestamp, "frame_index": self.frame_index,
                "crf": self.crf, "projected_scale": self.projected_scale, "metrics": self.metrics.to_dict(),
                "camera_relative_position": None if self.camera_relative_position is None else list(self.camera_relative_position),
                "camera_relative_orientation": None if self.camera_relative_orientation is None else list(self.camera_relative_orientation),
                "normalized_media_time": self.normalized_media_time, "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistortionSample":
        fields = dict(value)
        fields["metrics"] = MetricStats.from_dict(fields["metrics"])
        return cls(**fields)


@dataclass(frozen=True)
class ProfileLookup:
    available: bool
    metrics: MetricStats | None = None
    reason: str | None = None
    sample_count: int = 0
    aggregation: str = "none"

    def require(self) -> MetricStats:
        if not self.available or self.metrics is None:
            raise LookupError(self.reason or "Profile sample unavailable")
        return self.metrics

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "metrics": None if self.metrics is None else self.metrics.to_dict(),
                "reason": self.reason, "sample_count": self.sample_count, "aggregation": self.aggregation}


def _validated_dataset(samples: Iterable[DistortionSample]) -> tuple[DistortionSample, ...]:
    values = tuple(samples)
    if not values:
        raise ValueError("A profile dataset cannot be empty")
    keys: set[tuple[Any, ...]] = set()
    views: dict[tuple[Any, ...], DistortionSample] = {}
    # These bindings describe a comparison, not the CRF-specific encoded file.
    binding_fields = ("comparison_signature", "config_sha256", "reference_identity", "source_identity",
                      "object_transform", "camera_intrinsics", "source_camera_conditioning", "background",
                      "renderer_identity", "decoder_identity", "reference_mask_sha256")
    for sample in values:
        if sample.exact_key in keys:
            raise ValueError("Duplicate exact object/camera/pose/GOP/time/scale/CRF sample")
        keys.add(sample.exact_key)
        if sample.crf == 20 and sample.metrics.sse != 0:
            raise ValueError("CRF20 compression-relative reference must have zero SSE")
        prior = views.get(sample.view_key)
        if prior is not None:
            if (sample.frame_index, sample.metrics.foreground_pixels, sample.metrics.channels, sample.metrics.data_range,
                sample.metrics.normalization, sample.camera_relative_position, sample.camera_relative_orientation,
                sample.normalized_media_time) != (prior.frame_index, prior.metrics.foreground_pixels, prior.metrics.channels,
                prior.metrics.data_range, prior.metrics.normalization, prior.camera_relative_position,
                prior.camera_relative_orientation, prior.normalized_media_time):
                raise ValueError("CRF samples disagree on reference mask, normalization, frame, or camera-relative metadata")
            for key in binding_fields:
                if sample.provenance.get(key) != prior.provenance.get(key):
                    raise ValueError(f"CRF samples disagree on provenance binding: {key}")
        else:
            views[sample.view_key] = sample
    return tuple(sorted(values, key=lambda sample: _canonical(sample.to_dict())))


def dataset_fingerprint(samples: Sequence[DistortionSample]) -> str:
    records = sorted((_canonical(sample.to_dict()) for sample in samples))
    return hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest()


class AverageProfile:
    """Camera-independent SSE / total RGB observations, never mean frame PSNR."""
    aggregation = "pooled_sse_over_foreground_rgb_count"

    def __init__(self, samples: Iterable[DistortionSample]) -> None:
        self.samples = _validated_dataset(samples)
        self.dataset_sha256 = dataset_fingerprint(self.samples)
        groups: dict[tuple[str, int, int], list[MetricStats]] = {}
        for sample in self.samples:
            groups.setdefault((sample.object_id, sample.gop, sample.crf), []).append(sample.metrics)
        self.entries = {key: ProfileLookup(True, pool_metrics(group), sample_count=len(group), aggregation=self.aggregation)
                        for key, group in groups.items()}
        # Every CRF in an average profile must use the same offline camera/time
        # population. Missing qualities may not silently alter the average.
        populations: dict[tuple[str, int], dict[int, set[tuple[Any, ...]]]] = {}
        for sample in self.samples:
            populations.setdefault((sample.object_id, sample.gop), {}).setdefault(sample.crf, set()).add(sample.view_key)
        for qualities in populations.values():
            first = next(iter(qualities.values()))
            if any(population != first for population in qualities.values()):
                raise ValueError("Average profile CRFs do not share the same offline camera/time dataset")

    def lookup(self, object_id: str, gop: int, crf: int) -> ProfileLookup:
        return self.entries.get((object_id, gop, crf), ProfileLookup(False, reason="No average profile for exact object/GOP/CRF"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "profile_type": "camera_independent_average", "dataset_sha256": self.dataset_sha256,
                "aggregation": self.aggregation, "entries": [{"object_id": key[0], "gop": key[1], "crf": key[2], **value.to_dict()}
                   for key, value in sorted(self.entries.items())]}


class ExactProfile:
    """Exact pose/time/scale identity only: no nearest neighbor or interpolation."""
    def __init__(self, samples: Iterable[DistortionSample]) -> None:
        self.samples = _validated_dataset(samples)
        self.dataset_sha256 = dataset_fingerprint(self.samples)
        self.entries = {sample.exact_key: sample for sample in self.samples}

    def lookup(self, *, object_id: str, gop: int, camera_id: str, camera_pose: Any, timestamp: float,
               crf: int, projected_scale: float | None = None) -> ProfileLookup:
        pose = _vector(camera_pose, 16, "camera_pose")
        key = (object_id, gop, camera_id, pose, timestamp, projected_scale, crf)
        sample = self.entries.get(key)
        if sample is None:
            return ProfileLookup(False, reason="Exact camera/pose/GOP/time/scale/CRF sample unavailable; fallback is disabled")
        return ProfileLookup(True, sample.metrics, sample_count=1, aggregation="exact_single_sample")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "profile_type": "exact_view_conditioned", "dataset_sha256": self.dataset_sha256,
                "lookup": "exact_no_interpolation_no_fallback", "samples": [sample.to_dict() for sample in self.samples]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactProfile":
        profile = cls(DistortionSample.from_dict(sample) for sample in value["samples"])
        if value.get("dataset_sha256", profile.dataset_sha256) != profile.dataset_sha256:
            raise ValueError("Profile dataset checksum mismatch")
        return profile


def build_profiles(samples: Iterable[DistortionSample]) -> tuple[AverageProfile, ExactProfile]:
    values = tuple(samples)
    average, exact = AverageProfile(values), ExactProfile(values)
    assert average.dataset_sha256 == exact.dataset_sha256
    return average, exact
