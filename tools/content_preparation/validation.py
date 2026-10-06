"""Structural, dependency, provenance and actual-byte validation for prepared assets."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re


def safe_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) == ".":
        raise ValueError(f"Unsafe relative asset path: {value!r}")
    return path.as_posix()


def _checksum(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid sha256 for {label}")


def validate_asset(record: dict, root: Path | None = None) -> None:
    safe_relative_path(record["path"])
    if not isinstance(record["bytes"], int) or record["bytes"] < 0:
        raise ValueError("Asset bytes must be nonnegative integer")
    _checksum(record["sha256"], record["path"])
    if root is not None:
        root = Path(root).resolve()
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Asset symlink escapes object directory")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing asset or byte size mismatch: {record['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {record['path']}")


def validate_dependency_graph(payloads: list[dict]) -> None:
    by_id = {p["id"]: p for p in payloads}
    if len(by_id) != len(payloads):
        raise ValueError("Duplicate payload identifiers")
    done, visiting = set(), set()
    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError("Cyclic payload dependency graph")
        if identifier in done:
            return
        if identifier not in by_id:
            raise ValueError(f"Missing dependency: {identifier}")
        visiting.add(identifier)
        payload = by_id[identifier]
        dependencies = payload.get("dependencies", [])
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("Duplicate payload dependencies")
        if payload["mode"] == "independent" and dependencies:
            raise ValueError("Independent representations cannot depend on another representation")
        if payload.get("temporal_dependencies"):
            raise ValueError("Baseline does not implement temporal prediction")
        for parent_id in dependencies:
            if parent_id not in by_id:
                raise ValueError(f"Missing dependency: {parent_id}")
            parent = by_id[parent_id]
            if parent["frame"] != payload["frame"]:
                raise ValueError("Baseline enhancement must depend on the same frame")
            if parent["mode"] != "progressive" or payload["mode"] != "progressive":
                raise ValueError("Dependencies must stay within progressive mode")
            if payload.get("parent_layer") != parent["layer"]:
                raise ValueError("Refinement dependency disagrees with parent_layer")
            visit(parent_id)
        if payload["mode"] == "progressive":
            parent_layer = payload.get("parent_layer")
            if parent_layer is None and dependencies:
                raise ValueError("Base layer cannot have dependencies")
            if parent_layer is not None and len(dependencies) != 1:
                raise ValueError("An enhancement must have exactly one decoded-parent dependency")
        visiting.remove(identifier)
        done.add(identifier)
    for identifier in by_id:
        visit(identifier)


def validate_package_index(index: dict, root: Path | None = None) -> None:
    from .packaging import INDEX_VERSION, read_segment_header
    if index.get("schema_version") != INDEX_VERSION:
        raise ValueError("Unsupported package index version")
    if index.get("temporal_prediction") is not False:
        raise ValueError("Baseline index must explicitly declare no temporal prediction")
    for field in ("gof_frames", "segment_frames"):
        if isinstance(index[field], bool) or not isinstance(index[field], int) or index[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    frames = index["frames"]
    frame_ids = [frame["frame"] for frame in frames]
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Invalid frame mapping")
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frame_ids):
        raise ValueError("Frame identifiers must be integers")
    if any(not isinstance(frame["timestamp"], (int, float)) or not math.isfinite(frame["timestamp"]) for frame in frames):
        raise ValueError("Frame timestamps must be finite numbers")
    if any(b["frame"] <= a["frame"] or b["timestamp"] <= a["timestamp"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Frame/time mapping must strictly increase")
    positions = {frame: position for position, frame in enumerate(frame_ids)}
    expected_epochs = []
    for start in range(0, len(frames), index["gof_frames"]):
        stop = min(start + index["gof_frames"], len(frames))
        expected_epochs.append({"id": start // index["gof_frames"],
                                "frames": frame_ids[start:stop],
                                "frame_range": [frame_ids[start], frame_ids[stop - 1]],
                                "time_range": [frames[start]["timestamp"], frames[stop - 1]["timestamp"]],
                                "full_reset_frame": frame_ids[start]})
    if index["epochs"] != expected_epochs:
        raise ValueError("Representation epoch/reset metadata disagrees with frame mapping")
    validate_dependency_graph(index["payloads"])
    payloads = {p["id"]: p for p in index["payloads"]}
    segments = {s["id"]: s for s in index["segments"]}
    if len(segments) != len(index["segments"]):
        raise ValueError("Duplicate segment identifiers")
    covered = set()
    for segment in index["segments"]:
        validate_asset(segment, root)
        if segment["bytes"] != segment["header_bytes"] + segment["payload_bytes"]:
            raise ValueError("Segment accounting does not include the exact header")
        cursor = segment["header_bytes"]
        member_ids = []
        for member in segment["members"]:
            identifier = member["id"]
            if identifier not in payloads or identifier in covered:
                raise ValueError("Missing or multiply packaged payload")
            payload = payloads[identifier]
            if member["offset"] != cursor or member["length"] != payload["bytes"] or member["sha256"] != payload["sha256"]:
                raise ValueError("Segment member accounting disagrees with encoded payload")
            cursor += member["length"]
            if payload["segment_id"] != segment["id"] or payload["segment_path"] != segment["path"]:
                raise ValueError("Payload belongs to wrong segment")
            if payload["segment_offset"] != member["offset"]:
                raise ValueError("Payload byte offset disagrees with segment index")
            if payload["mode"] != segment["mode"] or payload["layer"] != segment["layer"] or payload["gof"] != segment["gof"]:
                raise ValueError("Segment crosses a mode/layer/GoF boundary")
            member_ids.append(identifier)
            covered.add(identifier)
        if cursor != segment["bytes"]:
            raise ValueError("Segment length mismatch")
        members = [payloads[p] for p in member_ids]
        if segment["frames"] != [p["frame"] for p in members]:
            raise ValueError("Segment frame index mismatch")
        if not members or segment["frame_range"] != [members[0]["frame"], members[-1]["frame"]]:
            raise ValueError("Segment frame range mismatch")
        if segment["time_range"] != [members[0]["timestamp"], members[-1]["timestamp"]]:
            raise ValueError("Segment time range mismatch")
        if segment["access_points"] != segment["frames"]:
            raise ValueError("Baseline codec access points must include every frame")
        expected_refresh = [p["frame"] for p in members if p["configured_refresh"]]
        if segment["configured_refresh_points"] != expected_refresh:
            raise ValueError("Configured refresh points disagree with payload metadata")
        if sorted(segment["dependencies"]) != sorted({d for p in members for d in p["dependencies"]}):
            raise ValueError("Segment dependencies disagree with payload closure")
        if root is not None:
            header, data_offset = read_segment_header(Path(root) / segment["path"])
            expected = [{"id": m["id"], "offset": m["offset"] - data_offset,
                         "length": m["length"], "sha256": m["sha256"]} for m in segment["members"]]
            if header["members"] != expected or data_offset != segment["header_bytes"]:
                raise ValueError("Actual segment index differs from package metadata")
    if covered != set(payloads):
        raise ValueError("Unpackaged payloads")
    if index["media_bytes"] != sum(segment["bytes"] for segment in segments.values()):
        raise ValueError("Total requestable media bytes mismatch")
    representations = index["representations"]
    representation_keys = [(r["mode"], r["quality"]) for r in representations]
    expected_representations = {(mode, quality) for mode in index["delivery_modes"] for quality in index["qualities"]}
    if len(representation_keys) != len(set(representation_keys)) or set(representation_keys) != expected_representations:
        raise ValueError("Representation descriptors are duplicated or incomplete")
    by_sample = {(p["mode"], p["quality"], p["frame"]): p for p in payloads.values()}
    expected_samples = {(mode, quality, frame) for mode in index["delivery_modes"]
                        for quality in index["qualities"] for frame in frame_ids}
    samples = {(p["mode"], p["quality"], p["frame"]) for p in payloads.values()}
    if samples != expected_samples or len(samples) != len(payloads):
        raise ValueError("Mode/quality/frame payload coverage is incomplete")
    for payload in payloads.values():
        validate_asset(payload, root)
        _checksum(payload["decoded_state_hash"], payload["id"])
        codec = payload["codec"]
        if not isinstance(codec, dict) or not codec.get("name") or not codec.get("version"):
            raise ValueError("Each payload must identify codec name and version")
        if payload.get("temporal_prediction"):
            raise ValueError("Baseline payload cannot use temporal prediction")
        position = positions[payload["frame"]]
        if payload["timestamp"] != frames[position]["timestamp"]:
            raise ValueError("Payload timestamp disagrees with media time mapping")
        if payload["gof"] != position // index["gof_frames"]:
            raise ValueError("Payload has incorrect representation epoch")
        representation = next((r for r in representations if r["mode"] == payload["mode"] and r["quality"] == payload["quality"]), None)
        if representation is None:
            raise ValueError("Missing representation description")
        interval = representation["refresh_interval_frames"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("Invalid representation refresh interval")
        if representation["layer"] != payload["layer"] or representation["codec"] != payload["codec"]:
            raise ValueError("Representation layer/codec disagrees with payload")
        qi = index["qualities"].index(payload["quality"])
        if payload["mode"] == "progressive" and qi:
            parent = by_sample[(payload["mode"], index["qualities"][qi - 1], payload["frame"])]
            if payload["parent_layer"] != parent["layer"] or payload["dependencies"] != [parent["id"]]:
                raise ValueError("Progressive dependency must be the immediately preceding quality")
        elif payload["parent_layer"] is not None or payload["dependencies"]:
            raise ValueError("Self-contained payload must have no dependencies")
        self_contained = payload["mode"] == "independent" or qi == 0
        if payload["self_contained"] != self_contained or representation["self_contained"] != self_contained:
            raise ValueError("Incorrect self-contained access flag")
        if representation["parent_layer"] != payload["parent_layer"] or representation["refinement_dependency"] != payload["parent_layer"]:
            raise ValueError("Representation parent refinement graph mismatch")
        refresh = position % index["gof_frames"] % representation["refresh_interval_frames"] == 0
        if payload["configured_refresh"] != refresh or payload["full_reset"] != (position % index["gof_frames"] == 0):
            raise ValueError("Refresh/reset schedule mismatch")
        if not payload["codec_access_point"]:
            raise ValueError("Every baseline frame must be a codec access point")
    for representation in representations:
        selected = [p for p in payloads.values() if p["mode"] == representation["mode"] and p["quality"] == representation["quality"]]
        expected_segment_ids = {p["segment_id"] for p in selected}
        if set(representation["segments"]) != expected_segment_ids:
            raise ValueError("Representation segment coverage mismatch")
        if representation["bytes"] != sum(segments[s]["bytes"] for s in expected_segment_ids):
            raise ValueError("Representation byte accounting mismatch")
        if representation["access_points"] != frame_ids:
            raise ValueError("Representation access schedule mismatch")
        if representation["refresh_points"] != [p["frame"] for p in selected if p["configured_refresh"]]:
            raise ValueError("Representation configured refresh schedule mismatch")
        expected_segments = []
        for epoch in expected_epochs:
            start = epoch["id"] * index["gof_frames"]
            stop = min(start + index["gof_frames"], len(frames))
            boundaries = {start, stop}
            boundaries.update(range(start, stop, representation["refresh_interval_frames"]))
            boundaries.update(range(start, stop, index["segment_frames"]))
            ordered = sorted(boundaries)
            expected_segments.extend(frame_ids[a:b] for a, b in zip(ordered, ordered[1:]))
        actual_segments = [segments[identifier]["frames"] for identifier in representation["segments"]]
        if actual_segments != expected_segments:
            raise ValueError("Segments do not honor GoF, refresh and network boundary union")


def validate_manifest(manifest: dict, root: Path | None = None) -> None:
    from .manifest import MANIFEST_VERSION
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported manifest version")
    obj = manifest["object"]
    if obj["id"] != manifest["package"]["object_id"] or obj["frames"] != manifest["package"]["frames"]:
        raise ValueError("Object metadata disagrees with package")
    if manifest["quality_order"] != manifest["package"]["qualities"]:
        raise ValueError("Manifest quality order disagrees with package")
    if manifest["representations"] != manifest["package"]["representations"] or manifest["delivery_modes"] != manifest["package"]["delivery_modes"]:
        raise ValueError("Manifest representation graph disagrees with package")
    if obj["frame_range"] != [obj["frames"][0]["frame"], obj["frames"][-1]["frame"]] or obj["time_range"] != [obj["frames"][0]["timestamp"], obj["frames"][-1]["timestamp"]]:
        raise ValueError("Manifest object frame/time range mismatch")
    if manifest["accounting"]["media_bytes"] != manifest["package"]["media_bytes"]:
        raise ValueError("Manifest media-byte accounting mismatch")
    validate_package_index(manifest["package"], root)
    for key in ("proxy_path", "quality_profile_path"):
        if obj.get(key):
            safe_relative_path(obj[key])
            if root is not None:
                target = (Path(root) / obj[key]).resolve()
                if not target.is_relative_to(Path(root).resolve()) or not target.is_file():
                    raise ValueError(f"Missing or unsafe manifest {key}")
    transform = obj["canonical_transform"]
    if isinstance(transform, dict) and "scale" in transform:
        if not isinstance(transform["scale"], (int, float)) or not math.isfinite(transform["scale"]) or transform["scale"] <= 0:
            raise ValueError("Canonical scale must be finite and positive")
    matrix = transform.get("matrix") if isinstance(transform, dict) else transform
    if matrix is not None:
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("Canonical transform must be 4x4")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in matrix for v in row):
            raise ValueError("Canonical transform must contain finite values")
    for asset in manifest.get("assets", []):
        validate_asset(asset, root)
