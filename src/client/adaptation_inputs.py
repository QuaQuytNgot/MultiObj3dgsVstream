"""Immutable planner contracts, without MPD parsing, decoding, or rendering.

Adapters must supply verified content/cache identities. Sizes are whole HTTP
body bytes, never serialized decoded-state sizes or nominal MPD bandwidth.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol

from .request_api import RequestSpec


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def finite(value: Any, name: str, minimum: float | None = None) -> float:
    try:
        valid = (not isinstance(value, bool) and isinstance(value, Real)
                 and math.isfinite(value) and (minimum is None or value >= minimum))
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return float(value)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def freeze(value: Any) -> Any:
    """Snapshot camera/metadata containers, including array-like matrices."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if hasattr(value, "tolist"):
        return freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported mutable context value: {type(value).__name__}")


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.),
            (0., 0., 1., 0.), (0., 0., 0., 1.))


@dataclass(frozen=True, slots=True)
class RequestUnit:
    unit_id: str
    object_id: str
    size_bytes: int | None
    request: RequestSpec
    checksum: str

    def __post_init__(self):
        _text(self.unit_id, "unit_id")
        _text(self.object_id, "object_id")
        _text(self.checksum, "checksum")
        if not isinstance(self.request, RequestSpec):
            raise ValueError("request must be a resolved RequestSpec")
        if self.request.object_id not in (None, self.object_id):
            raise ValueError("RequestSpec object identity disagrees with request unit")
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", integer(self.size_bytes, "size_bytes"))
            if (self.request.byte_range is not None
                    and self.size_bytes != self.request.byte_range[1] - self.request.byte_range[0] + 1):
                raise ValueError("size_bytes must match the inclusive request range")


@dataclass(frozen=True, slots=True)
class PayloadDescriptor:
    payload_id: str
    object_id: str
    frame: int
    quality: int
    epoch: int
    media_time_s: float
    request_unit_id: str
    decoded_state_hash: str
    dependencies: tuple[str, ...] = ()
    parent_state_hash: str | None = None
    state_version: str = "gaussian-state-v1"
    codec_id: str = "default"

    def __post_init__(self):
        for name in ("payload_id", "object_id", "request_unit_id", "decoded_state_hash", "state_version", "codec_id"):
            _text(getattr(self, name), name)
        integer(self.frame, "frame")
        integer(self.epoch, "epoch")
        if integer(self.quality, "quality") > 3:
            raise ValueError("quality must be Q0..Q3")
        finite(self.media_time_s, "media_time_s", 0)
        deps = tuple(self.dependencies)
        if len(set(deps)) != len(deps):
            raise ValueError("Duplicate payload dependencies")
        for dep in deps:
            _text(dep, "dependency")
        object.__setattr__(self, "dependencies", deps)


@dataclass(frozen=True, slots=True)
class DecodedState:
    """A caller-verified retained state, not a claim inferred from quality."""
    payload_id: str
    object_id: str
    frame: int
    quality: int
    epoch: int
    decoded_state_hash: str
    state_version: str = "gaussian-state-v1"
    codec_id: str = "default"

    def __post_init__(self):
        for name in ("payload_id", "object_id", "decoded_state_hash", "state_version", "codec_id"):
            _text(getattr(self, name), name)
        for name in ("frame", "quality", "epoch"):
            object.__setattr__(self, name, integer(getattr(self, name), name))
        if self.quality > 3:
            raise ValueError("quality must be Q0..Q3")

    @classmethod
    def from_payload(cls, payload: PayloadDescriptor) -> DecodedState:
        return cls(**{name: getattr(payload, name) for name in cls.__dataclass_fields__})

    def matches(self, payload: PayloadDescriptor) -> bool:
        return all(getattr(self, name) == getattr(payload, name)
                   for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class AvailabilitySnapshot:
    content_revision: str
    snapshot_id: str
    cached_units: Mapping[str, str] = field(default_factory=dict)
    decoded_states: Mapping[str, DecodedState] = field(default_factory=dict)

    def __post_init__(self):
        _text(self.content_revision, "content_revision")
        _text(self.snapshot_id, "snapshot_id")
        cached, states = dict(self.cached_units), dict(self.decoded_states)
        if any(not isinstance(v, str) or not v for v in cached.values()):
            raise ValueError("Cached units require verified checksums")
        if any(not isinstance(v, DecodedState) or k != v.payload_id for k, v in states.items()):
            raise ValueError("Decoded state keys must match payload identities")
        object.__setattr__(self, "cached_units", MappingProxyType(cached))
        object.__setattr__(self, "decoded_states", MappingProxyType(states))


@dataclass(frozen=True, slots=True)
class ContentIndexSnapshot:
    content_revision: str
    object_ids: tuple[str, ...]
    payloads: Mapping[str, PayloadDescriptor]
    request_units: Mapping[str, RequestUnit]
    profile_urls: Mapping[str, str] = field(default_factory=dict)
    _targets: Mapping[tuple[str, int, int], str] = field(init=False, repr=False)

    def __post_init__(self):
        _text(self.content_revision, "content_revision")
        objects = tuple(self.object_ids)
        if len(set(objects)) != len(objects):
            raise ValueError("Object identities must be unique and ordered")
        for obj in objects:
            _text(obj, "object_id")
        payloads, units, targets = dict(self.payloads), dict(self.request_units), {}
        for key, payload in payloads.items():
            if not isinstance(payload, PayloadDescriptor) or key != payload.payload_id or payload.object_id not in objects:
                raise ValueError("Payload identity is inconsistent with content index")
            target = (payload.object_id, payload.frame, payload.quality)
            if target in targets:
                raise ValueError("Ambiguous object/frame/quality payload")
            targets[target] = key
        for key, unit in units.items():
            if not isinstance(unit, RequestUnit) or key != unit.unit_id or unit.object_id not in objects:
                raise ValueError("Request unit identity is inconsistent with content index")
        object.__setattr__(self, "object_ids", objects)
        object.__setattr__(self, "payloads", MappingProxyType(payloads))
        object.__setattr__(self, "request_units", MappingProxyType(units))
        object.__setattr__(self, "profile_urls", MappingProxyType(dict(self.profile_urls)))
        object.__setattr__(self, "_targets", MappingProxyType(targets))

    def payload_for(self, object_id: str, frame: int, quality: int) -> PayloadDescriptor:
        return self.payloads[self._targets[(object_id, frame, quality)]]


@dataclass(frozen=True, slots=True)
class QualityTarget:
    object_id: str
    frames: tuple[int, ...]
    quality: int

    def __post_init__(self):
        _text(self.object_id, "object_id")
        if integer(self.quality, "quality") > 3:
            raise ValueError("quality must be Q0..Q3")
        frames = tuple(integer(frame, "frame") for frame in self.frames)
        if not frames or len(set(frames)) != len(frames):
            raise ValueError("Target frames must be nonempty and unique")
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True, slots=True)
class RequirementsResult:
    valid: bool
    missing_units: tuple[RequestUnit, ...] = ()
    decode_order: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def required_bytes(self) -> int:
        return sum(unit.size_bytes for unit in self.missing_units)  # type: ignore[misc]


class RequirementsResolver:
    """Pure progressive closure; cached raw children never bypass parents."""
    def __init__(self, index: ContentIndexSnapshot):
        self.index = index

    def required_units(self, target: QualityTarget, snapshot: AvailabilitySnapshot) -> RequirementsResult:
        if snapshot.content_revision != self.index.content_revision:
            return RequirementsResult(False, reason="content_revision_mismatch")
        seen, visiting, missing, order = set(), set(), {}, []
        physical_units: dict[tuple[str, tuple[int, int] | None], RequestUnit] = {}
        cached_locations = {
            (unit.request.url, unit.request.byte_range, unit.checksum)
            for unit in self.index.request_units.values()
            if (unit.object_id == target.object_id
                and snapshot.cached_units.get(unit.unit_id) == unit.checksum)
        }

        def visit(payload: PayloadDescriptor):
            key = payload.payload_id
            if payload.object_id != target.object_id:
                raise ValueError("cross_object_dependency")
            if key in visiting:
                raise ValueError("cyclic_dependency")
            if key in seen:
                return
            retained = snapshot.decoded_states.get(key)
            if retained is not None and retained.matches(payload):
                seen.add(key)
                return
            visiting.add(key)
            parents = [self.index.payloads[dep] for dep in payload.dependencies]
            if payload.quality:
                expected = self.index.payload_for(payload.object_id, payload.frame, payload.quality - 1)
                if expected.payload_id not in payload.dependencies:
                    raise ValueError("missing_progressive_parent")
                if expected.epoch != payload.epoch:
                    raise ValueError("parent_epoch_mismatch")
                if expected.media_time_s != payload.media_time_s:
                    raise ValueError("progressive_parent_timestamp_mismatch")
                if payload.parent_state_hash is not None and payload.parent_state_hash != expected.decoded_state_hash:
                    raise ValueError("parent_state_hash_mismatch")
            for parent in parents:
                visit(parent)
            visiting.remove(key)
            seen.add(key)
            order.append(key)
            unit = self.index.request_units[payload.request_unit_id]
            if unit.object_id != target.object_id:
                raise ValueError("cross_object_request_unit")
            location = (unit.request.url, unit.request.byte_range)
            previous = physical_units.get(location)
            if previous is not None:
                if previous.checksum != unit.checksum:
                    raise ValueError("ambiguous_request_unit_checksum")
                if (previous.size_bytes is not None and unit.size_bytes is not None
                        and previous.size_bytes != unit.size_bytes):
                    raise ValueError("ambiguous_request_unit_size")
            else:
                physical_units[location] = unit
            if (*location, unit.checksum) not in cached_locations:
                if unit.size_bytes is None:
                    raise ValueError("exact_size_unavailable")
                missing.setdefault(location, unit)

        try:
            for frame in target.frames:
                visit(self.index.payload_for(target.object_id, frame, target.quality))
        except KeyError as exc:
            return RequirementsResult(False, reason=f"missing_index_record:{exc.args[0]}")
        except ValueError as exc:
            return RequirementsResult(False, reason=str(exc))
        return RequirementsResult(True, tuple(missing.values()), tuple(order))


@dataclass(frozen=True, slots=True)
class ProfileCoordinates:
    azimuth_deg: float
    elevation_deg: float
    projected_scale: float
    media_time_s: float
    calibration_id: str

    def __post_init__(self):
        finite(self.azimuth_deg, "azimuth_deg")
        finite(self.elevation_deg, "elevation_deg")
        if finite(self.projected_scale, "projected_scale", 0) == 0:
            raise ValueError("projected_scale must be positive")
        finite(self.media_time_s, "media_time_s", 0)
        _text(self.calibration_id, "calibration_id")


@dataclass(frozen=True, slots=True)
class DensityCapability:
    content_revision: str
    calibration_id: str
    resolution: tuple[int, int]
    renderer_id: str
    rgb_convention: str
    visibility_representation: str = "decoded_Q0"
    rgb_distortion_reference: str = "decoded_Q3"
    normalization_basis: str = "decoded_Q0_solo_alpha_area"
    pixel_support: str = "full_image"
    alpha_area_epsilon: float = 1e-6

    def __post_init__(self):
        for name in ("content_revision", "calibration_id", "renderer_id", "rgb_convention"):
            _text(getattr(self, name), name)
        resolution = tuple(integer(v, "resolution", 1) for v in self.resolution)
        if len(resolution) != 2:
            raise ValueError("resolution must contain width, height")
        object.__setattr__(self, "resolution", resolution)
        finite(self.alpha_area_epsilon, "alpha_area_epsilon", 0)

    def validate(self):
        if (self.visibility_representation != "decoded_Q0"
                or self.rgb_distortion_reference != "decoded_Q3"
                or self.normalization_basis != "decoded_Q0_solo_alpha_area"
                or self.pixel_support != "full_image"):
            raise ValueError("unsupported_density_capability")


@dataclass(frozen=True, slots=True)
class DensityValue:
    value: float | None
    q0_solo_alpha_area: float | None
    reason: str | None = None


class DistortionDensityProvider(Protocol):
    capability: DensityCapability

    def get_density(self, object_id: str, quality: int,
                    coordinates: ProfileCoordinates) -> DensityValue: ...


def density_from_mse(mse: float, pixel_count: int, q0_solo_alpha_area: float,
                     *, epsilon: float = 1e-6) -> DensityValue:
    """Explicit scalar conversion ONLY after matched full-image MSE validation.

    This helper does not validate legacy profile provenance or enrich files.
    The numerical guard is not empirically calibrated. Never clamp its divisor.
    """
    mse = finite(mse, "mse", 0)
    pixels = integer(pixel_count, "pixel_count", 1)
    area = finite(q0_solo_alpha_area, "q0_solo_alpha_area", 0)
    epsilon = finite(epsilon, "epsilon", 0)
    if area > pixels * (1 + 1e-5):
        raise ValueError("q0_solo_alpha_area exceeds image pixel support")
    if area <= epsilon:
        return DensityValue(None, area, "q0_support_below_guard")
    value = mse * (pixels / area)
    if not math.isfinite(value):
        return DensityValue(None, area, "nonfinite_density")
    return DensityValue(value, area)


@dataclass(frozen=True, slots=True)
class ObjectViewContext:
    frame: int
    coordinates: ProfileCoordinates
    q0_state_hash: str
    object_to_world: tuple[tuple[float, ...], ...] = IDENTITY

    def __post_init__(self):
        integer(self.frame, "frame")
        if not isinstance(self.coordinates, ProfileCoordinates):
            raise ValueError("coordinates must be immutable ProfileCoordinates")
        _text(self.q0_state_hash, "q0_state_hash")
        matrix = tuple(tuple(finite(v, "object_to_world") for v in row)
                       for row in self.object_to_world)
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("object_to_world must be 4x4")
        object.__setattr__(self, "object_to_world", matrix)


@dataclass(frozen=True, slots=True)
class ContributionSnapshot:
    A_visible: float
    A_solo: float


@dataclass(frozen=True, slots=True)
class VisibilitySnapshot:
    """Scalar snapshot of an existing VisibilityResult; no renderer imports."""
    timestamp: float
    resolution: tuple[int, int]
    per_object: Mapping[str, ContributionSnapshot]
    metadata: Mapping[str, Any]

    @classmethod
    def from_result(cls, result: Any) -> VisibilitySnapshot:
        return cls(result.timestamp, tuple(result.resolution),
                   {key: ContributionSnapshot(value.A_visible, value.A_solo)
                    for key, value in result.per_object.items()}, result.metadata)

    def __post_init__(self):
        finite(self.timestamp, "visibility timestamp", 0)
        resolution = tuple(integer(v, "resolution", 1) for v in self.resolution)
        if len(resolution) != 2:
            raise ValueError("resolution must contain width, height")
        areas = {}
        pixels = resolution[0] * resolution[1]
        for object_id, area in self.per_object.items():
            _text(object_id, "visibility object_id")
            visible = finite(area.A_visible, "A_visible", 0)
            solo = finite(area.A_solo, "A_solo", 0)
            if visible > solo or solo > pixels * (1 + 1e-5):
                raise ValueError("Contribution areas exceed their physical support")
            areas[object_id] = ContributionSnapshot(visible, solo)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "per_object", MappingProxyType(areas))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class SegmentViewSample:
    timestamp_s: float
    camera: Mapping[str, Any]
    visibility: VisibilitySnapshot
    objects: Mapping[str, ObjectViewContext]
    weight: float = 1.0

    def __post_init__(self):
        finite(self.timestamp_s, "timestamp_s", 0)
        finite(self.weight, "weight", 0)
        if not isinstance(self.visibility, VisibilitySnapshot):
            object.__setattr__(self, "visibility", VisibilitySnapshot.from_result(self.visibility))
        object.__setattr__(self, "camera", freeze(self.camera))
        objects = dict(self.objects)
        if any(not isinstance(value, ObjectViewContext) for value in objects.values()):
            raise ValueError("objects must contain ObjectViewContext values")
        for object_id in objects:
            _text(object_id, "view object_id")
        object.__setattr__(self, "objects", MappingProxyType(objects))


@dataclass(frozen=True, slots=True)
class PlanningContext:
    index: ContentIndexSnapshot
    availability: AvailabilitySnapshot
    frames_by_object: Mapping[str, tuple[int, ...]]
    view_samples: tuple[SegmentViewSample, ...]
    density_provider: DistortionDensityProvider
    transfer_budget_seconds: float

    def __post_init__(self):
        finite(self.transfer_budget_seconds, "transfer_budget_seconds", 0)
        frames = {key: tuple(integer(f, "frame") for f in values)
                  for key, values in self.frames_by_object.items()}
        object.__setattr__(self, "frames_by_object", MappingProxyType(frames))
        views = tuple(self.view_samples)
        if any(not isinstance(view, SegmentViewSample) for view in views):
            raise ValueError("view_samples must contain SegmentViewSample values")
        object.__setattr__(self, "view_samples", views)


@dataclass(frozen=True, slots=True)
class PlanningInputs:
    context: PlanningContext
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class UtilityTable:
    values: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[bool, ...], ...]
    reasons: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True, slots=True)
class CostTable:
    values: tuple[tuple[int, ...], ...]
    valid: tuple[tuple[bool, ...], ...]
    reasons: tuple[tuple[str | None, ...], ...]
    requirements: tuple[tuple[RequirementsResult, ...], ...]
