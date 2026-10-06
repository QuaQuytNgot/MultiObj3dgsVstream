"""Deterministic requestable containers, with no implied temporal prediction.

Representation epochs (GoFs), configured refreshes and transport cuts are distinct.
This baseline initializes each frame independently; enhancement dependencies are
only the reconstructed parent of the *same* frame. Container headers use offsets
relative to their data section, so accounting includes actual serialized headers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

MAGIC = b"CPSEG1\n"
SEGMENT_VERSION = "content-preparation.segment.v1"
INDEX_VERSION = "content-preparation.package.v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) == ".":
        raise ValueError(f"Unsafe object-relative path: {value!r}")
    return path.as_posix()


def _safe_id(value: str) -> str:
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError(f"Identifier must contain only letters, digits, '_' or '-': {value!r}")
    return value


def _source(root: Path, relative: str) -> Path:
    path = (root / _relative(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Payload escapes object directory: {relative}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def refresh_interval(config: dict, layer: str, quality: str | None = None,
                     quality_index: int | None = None) -> int:
    """Resolve a common or explicitly per-layer refresh interval."""
    value = config.get("refresh_frames", config.get("refresh_intervals", config.get("gof_frames", 1)))
    if isinstance(value, dict):
        candidates = [layer]
        if quality is not None:
            candidates.append(quality)
        if quality_index is not None:
            candidates.append("Base" if quality_index == 0 else f"E{quality_index}")
        candidates.append("default")
        key = next((candidate for candidate in candidates if candidate in value), None)
        if key is None:
            raise ValueError(f"No refresh interval configured for layer {layer}")
        value = value[key]
    return _positive_int(value, f"refresh_frames[{layer}]")


def _write_segment(path: Path, members: list[dict], root: Path) -> tuple[dict, list[dict]]:
    offset = 0
    descriptors = []
    for payload in members:
        descriptors.append({"id": payload["id"], "offset": offset,
                            "length": payload["bytes"], "sha256": payload["sha256"]})
        offset += payload["bytes"]
    header = {"schema_version": SEGMENT_VERSION, "offset_origin": "data_section",
              "members": descriptors}
    encoded = _json_bytes(header)
    data_offset = len(MAGIC) + 8 + len(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as target:
            target.write(MAGIC)
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for member in members:
                with _source(root, member["path"]).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    container = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                 "sha256": _sha256(path), "header_bytes": data_offset,
                 "payload_bytes": offset, "container_version": SEGMENT_VERSION,
                 "members": [dict(item, offset=data_offset + item["offset"],
                                  offset_origin="file") for item in descriptors]}
    return container, descriptors


def package_object(object_id: str, frames: list[dict], qualities: list,
                   payload_records: list[dict], config: dict, output_dir: Path) -> dict:
    """Package already encoded payloads into actual deterministic segment files.

    ``frames`` defines ordered media samples. Intervals count samples, not absolute
    dataset frame identifiers. All incoming paths are relative to ``output_dir``.
    Encoded records are retained as provenance; clients request segment paths.
    """
    _safe_id(object_id)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    quality_ids = [q["id"] if isinstance(q, dict) else q for q in qualities]
    for quality in quality_ids:
        _safe_id(quality)
    if not frames or not quality_ids or len(set(quality_ids)) != len(quality_ids):
        raise ValueError("Frames and unique ordered qualities are required")
    frame_ids = [f["frame"] for f in frames]
    timestamps = [f["timestamp"] for f in frames]
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frame_ids):
        raise ValueError("Frame identifiers must be integers")
    if any(isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) for time in timestamps):
        raise ValueError("Timestamps must be finite numeric values")
    if len(set(frame_ids)) != len(frame_ids) or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Frame identifiers must be unique and strictly increasing")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    gof_frames = _positive_int(config.get("gof_frames", len(frames)), "gof_frames")
    segment_frames = _positive_int(config.get("segment_frames", 1), "segment_frames")
    frame_positions = {frame: i for i, frame in enumerate(frame_ids)}
    records = copy.deepcopy(payload_records)
    by_key = {}
    for record in records:
        if record.get("temporal_dependencies") or record.get("temporal_prediction") or record.get("codec_access_point") is False:
            raise ValueError("This packaging baseline accepts independently initialized temporal frames only")
        mode = record["mode"]
        if mode not in ("independent", "progressive"):
            raise ValueError(f"Unsupported delivery mode: {mode}")
        quality = record["quality"]
        if quality not in quality_ids or record["frame"] not in frame_positions:
            raise ValueError("Payload has unknown quality or frame")
        record.setdefault("layer", quality if mode == "independent" else
                          ("Base" if quality_ids.index(quality) == 0 else f"E{quality_ids.index(quality)}"))
        _safe_id(record["layer"])
        record.setdefault("id", f"{mode}:{record['layer']}:{record['frame']}")
        record["path"] = _relative(record["path"])
        source = _source(root, record["path"])
        actual_bytes, actual_hash = source.stat().st_size, _sha256(source)
        if record.get("bytes", actual_bytes) != actual_bytes or record.get("sha256", actual_hash) != actual_hash:
            raise ValueError(f"Encoded payload bytes/hash mismatch: {record['id']}")
        record["bytes"], record["sha256"] = actual_bytes, actual_hash
        record.setdefault("dependencies", [])
        key = (mode, quality, record["frame"])
        if key in by_key:
            raise ValueError(f"Duplicate mode/quality/frame payload: {key}")
        by_key[key] = record
    modes = sorted({record["mode"] for record in records})
    if not modes:
        raise ValueError("At least one delivery mode must have payloads")
    for mode in modes:
        for quality_index, quality in enumerate(quality_ids):
            for frame in frame_ids:
                key = (mode, quality, frame)
                if key not in by_key:
                    raise ValueError(f"Missing payload for {key}")
                record = by_key[key]
                if mode == "progressive" and quality_index:
                    parent = by_key[(mode, quality_ids[quality_index - 1], frame)]
                    record.setdefault("parent_layer", parent["layer"])
                    if not record["dependencies"]:
                        record["dependencies"] = [parent["id"]]
                    if record["parent_layer"] != parent["layer"] or record["dependencies"] != [parent["id"]]:
                        raise ValueError("Progressive enhancement must depend on its immediately preceding quality at the same frame")
                    record["refinement_semantics"] = record.get("refinement_semantics", "decoded-parent state replacement")
                else:
                    record.setdefault("parent_layer", None)
                    if record["parent_layer"] is not None:
                        raise ValueError("Self-contained Base/independent payload cannot have parent_layer")
                position = frame_positions[frame]
                gof = position // gof_frames
                within_gof = position % gof_frames
                interval = refresh_interval(config, record["layer"], quality, quality_index)
                record.update({"gof": gof, "epoch": gof, "timestamp": timestamps[position],
                               "temporal_dependencies": [], "codec_access_point": True,
                               "configured_refresh": within_gof % interval == 0,
                               "full_reset": within_gof == 0,
                               "self_contained": mode == "independent" or quality_index == 0})
    # Validation rejects accidental previous-frame references or wrong refinement parents.
    from .validation import validate_dependency_graph
    validate_dependency_graph(records)
    records.sort(key=lambda p: (p["mode"], quality_ids.index(p["quality"]), frame_positions[p["frame"]]))
    segments, representations, epochs = [], [], []
    for start in range(0, len(frames), gof_frames):
        stop = min(start + gof_frames, len(frames))
        epochs.append({"id": start // gof_frames, "frame_range": [frame_ids[start], frame_ids[stop - 1]],
                       "frames": frame_ids[start:stop], "time_range": [timestamps[start], timestamps[stop - 1]],
                       "full_reset_frame": frame_ids[start]})
    for mode in modes:
        for quality in quality_ids:
            selected = [p for p in records if p["mode"] == mode and p["quality"] == quality]
            layers = {p["layer"] for p in selected}
            if len(layers) != 1:
                raise ValueError("A quality must map to one stable layer within a mode")
            layer = selected[0]["layer"]
            interval = refresh_interval(config, layer, quality, quality_ids.index(quality))
            representation_segments = []
            for epoch in epochs:
                gof = epoch["id"]
                start, stop = gof * gof_frames, min((gof + 1) * gof_frames, len(frames))
                boundaries = {start, stop}
                boundaries.update(range(start, stop, interval))
                boundaries.update(range(start, stop, segment_frames))
                ordered = sorted(boundaries)
                for segment_number, (a, b) in enumerate(zip(ordered, ordered[1:])):
                    members = [by_key[(mode, quality, frame)] for frame in frame_ids[a:b]]
                    segment_id = f"{mode}:{layer}:g{gof}:s{segment_number}"
                    path = root / "media" / mode / layer / f"gof_{gof:05d}" / f"segment_{segment_number:05d}.cpseg"
                    container, _ = _write_segment(path, members, root)
                    container.update({"id": segment_id, "mode": mode, "quality": quality,
                                      "layer": layer, "gof": gof, "epoch": gof,
                                      "frame_range": [frame_ids[a], frame_ids[b - 1]],
                                      "frames": frame_ids[a:b], "time_range": [timestamps[a], timestamps[b - 1]],
                                      "access_points": frame_ids[a:b],
                                      "configured_refresh_points": [p["frame"] for p in members if p["configured_refresh"]],
                                      "dependencies": sorted({d for p in members for d in p["dependencies"]}),
                                      "codec": members[0]["codec"]})
                    segments.append(container)
                    representation_segments.append(segment_id)
                    member_index = {m["id"]: m for m in container["members"]}
                    for member in members:
                        member["segment_id"] = segment_id
                        member["segment_path"] = container["path"]
                        member["segment_offset"] = member_index[member["id"]]["offset"]
            representations.append({"id": f"{mode}:{layer}", "mode": mode, "quality": quality,
                                    "nominal_quality": quality_ids.index(quality), "layer": layer,
                                    "parent_layer": selected[0]["parent_layer"],
                                    "self_contained": selected[0]["self_contained"],
                                    "refinement_dependency": selected[0]["parent_layer"],
                                    "refresh_interval_frames": interval,
                                    "refresh_points": [p["frame"] for p in selected if p["configured_refresh"]],
                                    "access_points": frame_ids,
                                    "access_requirements": "same-frame parent closure" if not selected[0]["self_contained"] else "none",
                                    "segments": representation_segments,
                                    "bytes": sum(s["bytes"] for s in segments if s["id"] in representation_segments),
                                    "codec": selected[0]["codec"],
                                    "decoded_state_version": selected[0].get("decoded_state_version", "gaussian-state-v1")})
    index = {"schema_version": INDEX_VERSION, "object_id": object_id, "frames": copy.deepcopy(frames),
             "qualities": quality_ids, "delivery_modes": modes, "epochs": epochs,
             "gof_frames": gof_frames, "segment_frames": segment_frames,
             "temporal_prediction": False, "codec_access": "every frame, with same-frame layer dependency closure",
             "interval_units": "ordered media frame samples; schedules restart at each GoF",
             "boundary_semantics": "union of GoF resets, per-layer configured refresh and network cuts",
             "representations": representations, "payloads": records, "segments": segments,
             "media_bytes": sum(s["bytes"] for s in segments)}
    from .validation import validate_package_index
    validate_package_index(index, root)
    target = root / "package_index.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(_json_bytes(index) + b"\n")
    os.replace(temporary, target)
    return index


def read_segment_header(path: Path) -> tuple[dict, int]:
    """Read only the index; media payloads are never all loaded to compute access."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Invalid segment magic: {path}")
        packed = stream.read(8)
        if len(packed) != 8:
            raise ValueError("Truncated segment header")
        length = struct.unpack("<Q", packed)[0]
        if length > min(path.stat().st_size, 64 * 1024 * 1024):
            raise ValueError("Invalid or oversized segment index")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError("Truncated segment index")
        header = json.loads(encoded)
    if header.get("schema_version") != SEGMENT_VERSION:
        raise ValueError("Unsupported segment version")
    return header, len(MAGIC) + 8 + length


def extract_segment_member(segment_path: Path, payload_id: str) -> bytes:
    """Extract and verify the bytes actually deliverable from a network segment."""
    header, data_offset = read_segment_header(segment_path)
    selected = [m for m in header["members"] if m["id"] == payload_id]
    if len(selected) != 1:
        raise KeyError(payload_id)
    member = selected[0]
    if member["offset"] < 0 or member["length"] < 0:
        raise ValueError("Negative segment member offset/length")
    with Path(segment_path).open("rb") as stream:
        stream.seek(data_offset + member["offset"])
        value = stream.read(member["length"])
    if len(value) != member["length"] or hashlib.sha256(value).hexdigest() != member["sha256"]:
        raise ValueError(f"Segment member bytes/hash mismatch: {payload_id}")
    return value


def cold_access(index: dict, payload_id: str) -> dict:
    """Dependency closure and complete requestable-container bytes for cold access."""
    payloads = {p["id"]: p for p in index["payloads"]}
    if payload_id not in payloads:
        raise KeyError(payload_id)
    ordered, seen, visiting = [], set(), set()
    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError("Cyclic payload dependency")
        if identifier in seen:
            return
        visiting.add(identifier)
        for dependency in payloads[identifier]["dependencies"]:
            visit(dependency)
        visiting.remove(identifier)
        seen.add(identifier)
        ordered.append(identifier)
    visit(payload_id)
    segment_ids = sorted({payloads[p]["segment_id"] for p in ordered})
    segments = {s["id"]: s for s in index["segments"]}
    return {"payload_id": payload_id, "dependencies_in_decode_order": ordered,
            "required_payload_bytes": sum(payloads[p]["bytes"] for p in ordered),
            "segment_ids": segment_ids, "segment_paths": [segments[s]["path"] for s in segment_ids],
            "request_bytes": sum(segments[s]["bytes"] for s in segment_ids),
            "accounting": "whole requestable segments, including headers and unselected members"}
