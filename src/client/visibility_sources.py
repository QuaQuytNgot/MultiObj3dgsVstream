"""Resolve visibility representations before entering the common renderer.

Decoded NPZ/PLY sizes are serialized *input* sizes, not delivery estimates.
Proxy bytes remain extra bytes even when cached. High-quality references are
offline oracles, so their online extra-byte cost is deliberately undefined.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from tools.content_preparation.assets import (
    GaussianState, canonical_json, file_hash, load_state, state_hash,
)
from tools.content_preparation.proxy import load_proxy


SOURCE_TYPES = frozenset({"proxy", "decoded_quality", "high_quality_reference"})


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    source_type: str
    asset_path: Path
    frame: int
    timestamp: float
    quality_id: str | None = None
    already_available: bool = False
    streaming_bytes: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceAsset:
    """One unique physical input file, independent of delivery accounting."""
    key: str
    path: str
    role: str
    asset_size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StreamingCost:
    """One declared delivery unit; None means unknown or offline-only."""
    key: str
    additional_streaming_bytes: int | None
    basis: str


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    gaussian_state: GaussianState
    source_metadata: dict[str, Any]
    assets: tuple[SourceAsset, ...]
    streaming_costs: tuple[StreamingCost, ...]

    def visibility_object(self, object_id: str, object_to_world=None):
        """Attach an identity/transform without loading or rendering again."""
        from src.renderer.analysis import VisibilityObject
        return VisibilityObject(
            object_id=object_id, gaussian_state=self.gaussian_state,
            timestamp=self.source_metadata["timestamp"],
            object_to_world=np.eye(4) if object_to_world is None else object_to_world,
            source_metadata=dict(self.source_metadata),
        )


def _validate_spec(spec: RepresentationSpec) -> Path:
    if not isinstance(spec, RepresentationSpec):
        raise TypeError("Expected a RepresentationSpec")
    if not isinstance(spec.source_type, str) or spec.source_type not in SOURCE_TYPES:
        raise ValueError(f"Unsupported visibility source_type: {spec.source_type!r}")
    if not isinstance(spec.frame, Integral) or isinstance(spec.frame, (bool, np.bool_)):
        raise ValueError("frame must be an integer")
    if not isinstance(spec.timestamp, Real) or isinstance(spec.timestamp, (bool, np.bool_)):
        raise ValueError("timestamp must be finite")
    try:
        finite = math.isfinite(spec.timestamp)
    except (ValueError, OverflowError):
        finite = False
    if not finite:
        raise ValueError("timestamp must be finite")
    if not isinstance(spec.already_available, bool):
        raise ValueError("already_available must be a boolean")
    if spec.quality_id is not None and (not isinstance(spec.quality_id, str) or not spec.quality_id):
        raise ValueError("quality_id must be a nonempty string or None")
    if spec.source_type != "proxy" and spec.quality_id is None:
        raise ValueError("Decoded visibility sources require an explicit quality_id")
    if spec.streaming_bytes is not None and (
        not isinstance(spec.streaming_bytes, Integral)
        or isinstance(spec.streaming_bytes, (bool, np.bool_)) or spec.streaming_bytes < 0
    ):
        raise ValueError("streaming_bytes must be a nonnegative integer or None")
    if not isinstance(spec.provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    canonical_json(dict(spec.provenance))
    path = Path(spec.asset_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Visibility asset must be a file: {path}")
    return path


def _asset(path: Path, role: str) -> SourceAsset:
    digest = file_hash(path)
    return SourceAsset(f"{path}:{digest}", str(path), role, path.stat().st_size, digest)


def _assert_hash(expected, actual: str, name: str):
    if expected is not None and expected != actual:
        raise ValueError(f"Visibility source {name} checksum mismatch")


def _check_snapshot_metadata(metadata: Mapping, spec: RepresentationSpec, label: str):
    if "frame" in metadata:
        frame = metadata["frame"]
        if not isinstance(frame, Integral) or isinstance(frame, (bool, np.bool_)) or frame != spec.frame:
            raise ValueError(f"{label} frame does not match requested frame {spec.frame}")
    if "timestamp" in metadata:
        value = metadata["timestamp"]
        try:
            matches = (not isinstance(value, bool) and math.isfinite(value)
                       and abs(float(value) - float(spec.timestamp)) <= 1e-9)
        except (TypeError, ValueError, OverflowError):
            matches = False
        if not matches:
            raise ValueError(f"{label} timestamp does not match requested timestamp")
    # A dedicated proxy may retain the quality label of its selection source.
    if spec.source_type != "proxy":
        for key in ("quality", "quality_id"):
            if key in metadata and metadata[key] != spec.quality_id:
                raise ValueError(f"{label} quality does not match requested {spec.quality_id}")


def resolve_source(spec: RepresentationSpec) -> ResolvedSource:
    """Load exactly the specified existing representation, with no fallback.

    Proxy JSON resolves only ``frame`` and validates its payload hashes through
    ``load_proxy``. NPZ has an embedded numeric checksum; optional provenance
    keys ``asset_sha256``/``descriptor_sha256``/``state_sha256`` add external
    checks. Frame/time/quality metadata is checked when present. When timestamp
    metadata is absent, the caller's explicit frame/time mapping is recorded.
    """
    path = _validate_spec(spec)
    assets = [_asset(path, "proxy_descriptor" if spec.source_type == "proxy" and path.suffix.lower() == ".json" else "gaussian_state")]
    _assert_hash(spec.provenance.get("asset_sha256"), assets[0].sha256, "asset")
    descriptor_record = None
    if spec.source_type == "proxy":
        if path.suffix.lower() == ".json":
            _assert_hash(spec.provenance.get("descriptor_sha256"), assets[0].sha256, "descriptor")
            descriptor = json.loads(path.read_text(encoding="utf-8"))
            records = [row for row in descriptor.get("frames", []) if row.get("frame") == spec.frame]
            if not records:
                raise KeyError(f"Proxy frame {spec.frame} is unavailable")
            if len(records) != 1:
                raise ValueError(f"Proxy descriptor has duplicate frame {spec.frame}")
            descriptor_record = records[0]
            relative = Path(descriptor_record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Proxy payload paths must remain relative to the descriptor")
            payload = (path.parent / relative).resolve(strict=True)
            if not payload.is_relative_to(path.parent):
                raise ValueError("Proxy payload escapes the descriptor directory")
            assets.append(_asset(payload, "proxy_payload"))
            _check_snapshot_metadata(descriptor_record, spec, "Proxy descriptor")
        state = load_proxy(path, frame=int(spec.frame))
    else:
        if path.suffix.lower() not in {".npz", ".ply"}:
            raise ValueError("Decoded visibility input must be an existing Gaussian NPZ or PLY")
        state = load_state(path)
    _check_snapshot_metadata(state.metadata, spec, "Gaussian state")
    numeric_hash = state_hash(state)
    _assert_hash(spec.provenance.get("state_sha256"), numeric_hash, "numeric state")
    _assert_hash(spec.provenance.get("state_hash"), numeric_hash, "numeric state")
    if len(assets) > 1:
        _assert_hash(spec.provenance.get("payload_sha256"), assets[1].sha256, "payload")
    _check_snapshot_metadata(spec.provenance, spec, "Source provenance")

    bundle_key = hashlib.sha256(canonical_json([asset.key for asset in assets])).hexdigest()
    if spec.source_type == "high_quality_reference":
        costs = (StreamingCost(bundle_key, None, "offline_reference_not_online_delivery"),)
    elif spec.source_type == "decoded_quality" and spec.already_available:
        costs = (StreamingCost(bundle_key, 0, "already_required_and_decoded_for_playback"),)
    elif spec.streaming_bytes is not None:
        costs = (StreamingCost(bundle_key, int(spec.streaming_bytes), "explicit_declared_delivery_bytes"),)
    elif spec.source_type == "proxy":
        costs = tuple(StreamingCost(asset.key, asset.asset_size_bytes, "serialized_proxy_auxiliary_asset") for asset in assets)
    else:
        costs = (StreamingCost(bundle_key, None, "delivery_bytes_not_declared_for_decoded_input"),)
    additional = None if any(cost.additional_streaming_bytes is None for cost in costs) else sum(cost.additional_streaming_bytes for cost in costs)
    asset_bytes = sum(asset.asset_size_bytes for asset in assets)
    timestamp_from_metadata = ("timestamp" in state.metadata
                               or descriptor_record is not None and "timestamp" in descriptor_record)
    metadata = {
        "source_type": spec.source_type, "quality_id": spec.quality_id,
        "frame": int(spec.frame), "timestamp": float(spec.timestamp),
        "asset_path": str(path), "asset_size_bytes": asset_bytes,
        "asset_bytes": asset_bytes,
        "additional_streaming_bytes": additional,
        "already_available": spec.already_available,
        "offline_reference": spec.source_type == "high_quality_reference",
        "provenance": {**dict(spec.provenance), "state_sha256": numeric_hash,
                       "assets": [asdict(asset) for asset in assets],
                       "state_metadata": dict(state.metadata)},
        "streaming_accounting_basis": [cost.basis for cost in costs],
        "serialized_input_is_total_media_delivery": False,
        "timestamp_validation": "asset_metadata" if timestamp_from_metadata else "explicit_caller_frame_timestamp_mapping",
    }
    return ResolvedSource(state, metadata, tuple(assets), costs)


def resolve_visibility_object(object_id: str, spec: RepresentationSpec, object_to_world=None):
    return resolve_source(spec).visibility_object(object_id, object_to_world)


def account_sources(sources: Sequence[ResolvedSource], visibility_sample_count: int) -> dict[str, Any]:
    """Union inputs/cost units across repeated objects, times and viewpoints.

    ``visibility_sample_count`` counts analyzer invocations, not object rows.
    Pass all sources used by those invocations; a descriptor reused by several
    frame payloads is counted once. Explicit streaming overrides describe a
    delivery bundle and must consistently identify the same cost on reuse.
    """
    if (not isinstance(visibility_sample_count, Integral)
            or isinstance(visibility_sample_count, (bool, np.bool_))
            or visibility_sample_count <= 0):
        raise ValueError("visibility_sample_count must be a positive integer")
    assets, costs = {}, {}
    for source in sources:
        if not isinstance(source, ResolvedSource):
            raise TypeError("account_sources expects ResolvedSource values")
        for asset in source.assets:
            previous = assets.setdefault(asset.key, asset)
            if previous.asset_size_bytes != asset.asset_size_bytes or previous.sha256 != asset.sha256:
                raise ValueError("Conflicting accounting for a reused input asset")
        for cost in source.streaming_costs:
            if costs.setdefault(cost.key, cost) != cost:
                raise ValueError("Conflicting streaming costs for a reused delivery unit")
    unknown = any(cost.additional_streaming_bytes is None for cost in costs.values())
    additional = None if unknown else sum(cost.additional_streaming_bytes for cost in costs.values())
    return {
        "asset_size_bytes": sum(asset.asset_size_bytes for asset in assets.values()),
        "additional_streaming_bytes": additional,
        "amortized_extra_bytes_per_sample": None if additional is None else additional / int(visibility_sample_count),
        "visibility_sample_count": int(visibility_sample_count),
        "unique_asset_count": len(assets),
        "assets": [asdict(asset) for asset in assets.values()],
        "streaming_costs": [asdict(cost) for cost in costs.values()],
        "serialized_input_is_total_media_delivery": False,
    }


class VisibilitySourceResolver:
    """Experiment-local loading cache; reuse one instance across all views.

    Inputs are checked for file-stat changes before cache reuse. Keep resolved
    states immutable while the experiment uses them; rasterizer snapshots are
    owned by the renderer. There is no hidden global asset cache.
    """
    def __init__(self):
        self._cache = {}

    def resolve_source(self, spec: RepresentationSpec) -> ResolvedSource:
        path = _validate_spec(spec)
        key = canonical_json({
            "path": str(path), "source_type": spec.source_type, "frame": int(spec.frame),
            "timestamp": float(spec.timestamp), "quality_id": spec.quality_id,
            "already_available": spec.already_available,
            "streaming_bytes": None if spec.streaming_bytes is None else int(spec.streaming_bytes),
            "provenance": dict(spec.provenance),
        })
        entry = self._cache.get(key)
        if entry is not None:
            resolved, signatures = entry
            current = self._signatures(resolved)
            if current == signatures:
                return resolved
        resolved = resolve_source(spec)
        self._cache[key] = (resolved, self._signatures(resolved))
        return resolved

    @staticmethod
    def _signatures(source):
        return tuple((asset.path, (stat := Path(asset.path).stat()).st_size,
                      stat.st_mtime_ns, stat.st_ctime_ns) for asset in source.assets)

    def resolve_visibility_object(self, object_id: str, spec: RepresentationSpec, object_to_world=None):
        return self.resolve_source(spec).visibility_object(object_id, object_to_world)
