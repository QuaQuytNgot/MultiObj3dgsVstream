"""Describe existing prepared runs without encoding, rendering or CUDA imports.

All deliverable paths are relative to the supplied pipeline output root. Original
checkpoint/raw source locators are provenance, not web request paths. Timing is
read from the journal: an encode task includes loading, encode, verification
decode and writing its CPU cache; it is not an isolated encoder benchmark.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.content_preparation.packaging import cold_access, read_segment_header
from tools.content_preparation.validation import safe_relative_path, validate_package_index

REPORT_VERSION = "content-preparation.run-report.v1"


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else None


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _managed(root, relative):
    safe_relative_path(relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Report asset escapes supplied output root: {relative}")
    return path


def _asset(root, path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Report artifact must be inside the output root")
    record = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
              "sha256": _hash(path)}
    if expected is not None and (record["bytes"] != expected["bytes"] or record["sha256"] != expected["sha256"]):
        raise ValueError(f"Prepared artifact bytes/checksum changed: {record['path']}")
    return record


def _duration(entry):
    if not entry or not entry.get("started_at") or not entry.get("completed_at"):
        return None
    start = datetime.fromisoformat(entry["started_at"].replace("Z", "+00:00"))
    stop = datetime.fromisoformat(entry["completed_at"].replace("Z", "+00:00"))
    if start.tzinfo is None or stop.tzinfo is None:
        raise ValueError("Journal timestamps must specify a timezone")
    duration = (stop - start).total_seconds()
    if duration < 0:
        raise ValueError("Journal completion precedes task start")
    return duration


def _timeline(frames, fps):
    if not frames:
        raise ValueError("Prepared report requires media samples")
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and positive, or None to infer from frame/time mapping")
    if fps is None:
        candidates = [(b["frame"] - a["frame"]) / (b["timestamp"] - a["timestamp"])
                      for a, b in zip(frames, frames[1:])]
        if not candidates or any(not math.isclose(value, candidates[0], rel_tol=1e-6) for value in candidates):
            raise ValueError("Cannot infer a consistent nominal FPS; provide fps explicitly")
        fps = candidates[0]
        fps_source = "inferred from frame identifiers and exact timestamps"
    else:
        fps_source = "explicit report parameter"
    samples = []
    for position, frame in enumerate(frames):
        if "duration_seconds" in frame:
            duration = frame["duration_seconds"]
            source = "explicit frame duration_seconds"
        elif position + 1 < len(frames):
            duration = frames[position + 1]["timestamp"] - frame["timestamp"]
            source = "next exact sample timestamp"
        else:
            duration = 1.0 / fps
            source = "terminal sample: one nominal frame interval"
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Media sample durations must be finite and positive")
        samples.append({"frame": frame["frame"], "timestamp": frame["timestamp"],
                        "duration_seconds": duration, "duration_source": source,
                        "end_timestamp": frame["timestamp"] + duration})
    span = samples[-1]["timestamp"] - samples[0]["timestamp"]
    return {"fps": float(fps), "fps_source": fps_source, "frame_count": len(samples),
            "frame_range": [samples[0]["frame"], samples[-1]["frame"]],
            "first_timestamp": samples[0]["timestamp"], "last_timestamp": samples[-1]["timestamp"],
            "time_span_seconds": span,
            "media_duration_seconds": samples[-1]["end_timestamp"] - samples[0]["timestamp"],
            "definition": "last sample start minus first sample start plus the final sample duration",
            "samples": samples}


def _gaussian_count(path):
    """Read the NPY header only; do not materialize model arrays or import torch."""
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open("ids.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported Gaussian ID NPY header version: {version}")
    if len(shape) != 1 or not np.issubdtype(dtype, np.integer):
        raise ValueError("Decoded Gaussian IDs must be a one-dimensional integer array")
    return int(shape[0])


def _encode_timings(journal, object_id, payloads):
    tasks = journal.get("tasks", {}) if journal else {}
    rows = []
    for payload in payloads:
        key = f"{object_id}/encode/{payload['mode']}/{payload['layer']}/{payload['frame']}"
        entry = tasks.get(key)
        rows.append({"task_key": key, "payload_id": payload["id"], "mode": payload["mode"],
                     "quality": payload["quality"], "layer": payload["layer"], "frame": payload["frame"],
                     "status": entry.get("status") if entry else "unavailable",
                     "started_at": entry.get("started_at") if entry else None,
                     "completed_at": entry.get("completed_at") if entry else None,
                     "wall_seconds": _duration(entry)})
    successes = [row for row in rows if row["status"] == "complete" and row["wall_seconds"] is not None]
    stage = (journal or {}).get("stages", {}).get(f"{object_id}/encode")
    return {"recorded_successful_encode_task_wall_seconds": sum(r["wall_seconds"] for r in successes) if successes else None,
            "measured_successful_payload_count": len(successes), "expected_payload_count": len(payloads),
            "all_payload_timings_available": len(successes) == len(payloads),
            "latest_recorded_encode_stage_wall_seconds": _duration(stage),
            "latest_stage_status": stage.get("status") if stage else "unavailable",
            "tasks": rows,
            "task_scope": "CPU loading + encode + verification decode + encoder-cache write; excludes package/profile",
            "journal_scope": "latest retained task attempt only; older retries/overwrites are not reconstructible",
            "stage_scope": "latest stage invocation, possibly a resume with skipped tasks; never a substitute for original encode runtime",
            "playback_duration_is_not_encoding_runtime": True}


def _object_locations(root):
    manifest = _read(root / "manifest.json")
    if manifest and manifest.get("schema_version") == "content-preparation.server.v1":
        result = []
        for entry in manifest["objects"]:
            path = _managed(root, entry["manifest_path"])
            _asset(root, path, entry)
            result.append((path.parent, _read(path)))
        return result
    if manifest and "object" in manifest:
        return [(root, manifest)]
    if (root / "package.json").is_file() or (root / "package_index.json").is_file():
        return [(root, None)]
    return [(p, _read(p / "manifest.json")) for p in sorted(root.iterdir()) if p.is_dir() and
            ((p / "package.json").is_file() or (p / "package_index.json").is_file())]


def _source_checkpoint(root, record):
    value = {"quality": record.get("level", record.get("quality")), "frame": record["frame"],
             "checkpoint_sha256": record.get("sha256"), "training_argv": record.get("argv", []),
             "source_locator": record.get("path"), "synthetic_fixture": bool(record.get("synthetic_fixture", False))}
    source = Path(record["path"]).resolve() if record.get("path") else None
    if source and source.is_relative_to(root):
        value["checkpoint_path"] = source.relative_to(root).as_posix()
        value["source_locator"] = value["checkpoint_path"]
    return value


def report_prepared_run(output_root, fps=30, variant_id="f32_lossless", raw_provenance=None):
    """Read one pipeline variant and return a web-consumable descriptive report.

    ``fps`` is the nominal source frame rate used for the terminal sample duration;
    pass None to infer it from frame IDs/timestamps. It does not alter timestamps.
    Missing profiles/decoded caches/timings stay explicitly unavailable. Stored
    progressive layer bytes and cumulative delivered-quality bytes are separate.
    """
    root = Path(output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared output directory does not exist: {root}")
    if not isinstance(variant_id, str) or not variant_id.strip():
        raise ValueError("variant_id must be a nonempty descriptive label")
    journal = _read(root / ".preparation" / "journal.json")
    objects = []
    for object_root, manifest in _object_locations(root):
        index = _read(object_root / "package.json") or _read(object_root / "package_index.json") or (manifest or {}).get("package")
        if not index:
            raise FileNotFoundError(f"Missing prepared package for {object_root}")
        validate_package_index(index)
        if manifest and manifest["package"] != index:
            raise ValueError("Manifest package differs from standalone package index")
        obj_id = index["object_id"]
        timeline = _timeline(index["frames"], fps)
        samples = {row["frame"]: row for row in timeline["samples"]}
        segments = {}
        for segment in index["segments"]:
            path = _managed(object_root, segment["path"])
            actual = _asset(root, path, segment)
            header, offset = read_segment_header(path)
            expected_members = [{"id": m["id"], "offset": m["offset"] - offset,
                                 "length": m["length"], "sha256": m["sha256"]} for m in segment["members"]]
            if header["members"] != expected_members or offset != segment["header_bytes"]:
                raise ValueError("Segment header differs from prepared package metadata")
            first, last = samples[segment["frames"][0]], samples[segment["frames"][-1]]
            segments[segment["id"]] = {**actual, "id": segment["id"], "gof": segment["gof"],
                "frames": segment["frames"], "frame_count": len(segment["frames"]),
                "frame_range": segment["frame_range"], "first_timestamp": first["timestamp"],
                "last_timestamp": last["timestamp"], "time_span_seconds": last["timestamp"] - first["timestamp"],
                "media_duration_seconds": last["end_timestamp"] - first["timestamp"],
                "header_bytes": offset, "encoded_payload_bytes": segment["payload_bytes"],
                "access_points": segment["access_points"],
                "configured_refresh_points": segment["configured_refresh_points"], "dependencies": segment["dependencies"]}
        decoding = _read(object_root / "decoding.json") or {}
        decoded = {(r["mode"], r["quality"], r["frame"]): r for r in decoding.get("states", [])}
        timings = _encode_timings(journal, obj_id, index["payloads"])
        profile_path = (manifest or {}).get("object", {}).get("quality_profile_path", "profiles/profile.json")
        profile = _read(_managed(object_root, profile_path)) if profile_path else None
        representations = []
        for representation in index["representations"]:
            payloads = [p for p in index["payloads"] if p["mode"] == representation["mode"] and p["quality"] == representation["quality"]]
            access_rows, counts, cumulative_segment_ids = [], [], set()
            for payload in payloads:
                access = cold_access(index, payload["id"])
                cumulative_segment_ids.update(access["segment_ids"])
                access["segment_paths"] = [segments[s]["path"] for s in access["segment_ids"]]
                access_rows.append({"frame": payload["frame"], "timestamp": payload["timestamp"], **access})
                state_record = decoded.get((payload["mode"], payload["quality"], payload["frame"]))
                state_path = _managed(object_root, state_record["path"]) if state_record else None
                if state_path and state_path.is_file():
                    counts.append({"frame": payload["frame"], "gaussian_count": _gaussian_count(state_path),
                                   "decoded_state_hash": state_record.get("state_hash"), "state_asset": _asset(root, state_path)})
                else:
                    counts.append({"frame": payload["frame"], "gaussian_count": None, "reason": "decoded CPU asset unavailable"})
            selected_segments = [segments[s] for s in representation["segments"]]
            measured = [t for t in timings["tasks"] if t["mode"] == representation["mode"] and t["quality"] == representation["quality"] and t["status"] == "complete" and t["wall_seconds"] is not None]
            hash_checks = [p.get("input_state_hash") == p["decoded_state_hash"] if p.get("input_state_hash") else None for p in payloads]
            representations.append({"id": representation["id"], "quality": representation["quality"],
                "nominal_quality": representation["nominal_quality"], "mode": representation["mode"],
                "layer": representation["layer"], "parent_layer": representation["parent_layer"],
                "self_contained": representation["self_contained"], "codec": representation["codec"],
                "decoded_state_version": representation["decoded_state_version"],
                "stored_layer_segment_bytes": sum(s["bytes"] for s in selected_segments),
                "encoded_payload_bytes": sum(s["encoded_payload_bytes"] for s in selected_segments),
                "segment_header_bytes": sum(s["header_bytes"] for s in selected_segments),
                "cumulative_quality_segment_bytes": sum(segments[s]["bytes"] for s in cumulative_segment_ids),
                "byte_definition": "actual complete segment files; progressive stored-layer bytes exclude parents, cumulative-quality bytes include them once",
                "lossless_relative_to_exported_state_hash": all(hash_checks) if None not in hash_checks else None,
                "gaussian_counts": counts, "segments": selected_segments, "cold_access": access_rows,
                "recorded_successful_encode_task_wall_seconds": sum(t["wall_seconds"] for t in measured) if measured else None,
                "measured_encode_payload_count": len(measured),
                "quality_profile_aggregates": [a for a in (profile or {}).get("aggregates", []) if a.get("mode") == representation["mode"] and a.get("quality") == representation["quality"]]})
        training = _read(object_root / "training.json") or {}
        preprocessing = _read(object_root / "preprocess.json") or {}
        artifact_names = ("manifest.json", "package.json", "package_index.json", "training.json", "encoding.json", "decoding.json", "preprocess.json")
        artifacts = {name: _asset(root, object_root / name) for name in artifact_names if (object_root / name).is_file()}
        if profile:
            artifacts["quality_profile"] = _asset(root, _managed(object_root, profile_path))
        objects.append({"id": obj_id, "variant_id": variant_id, "timeline": timeline,
            "delivery_modes": index["delivery_modes"], "actual_all_modes_segment_bytes": sum(s["bytes"] for s in segments.values()),
            "all_modes_bytes_are_not_a_single_quality_cost": True, "representations": representations,
            "encode_timings": timings, "artifacts": artifacts,
            "quality_profile": {"available": profile is not None,
                "sample_count": len((profile or {}).get("rows", [])), "reference": (profile or {}).get("reference"),
                "metadata": (profile or {}).get("metadata")},
            "provenance": {"manifest": (manifest or {}).get("provenance", {}),
                "training_backend": training.get("backend"), "lineage_verified": training.get("lineage_verified"),
                "checkpoints": [_source_checkpoint(root, row) for row in training.get("commands", [])],
                "preprocessing": preprocessing, "raw": raw_provenance},
            "latest_recorded_stage_timings": {key.removeprefix(obj_id + "/"): {"status": entry.get("status"), "started_at": entry.get("started_at"),
                "completed_at": entry.get("completed_at"), "wall_seconds": _duration(entry)}
                for key, entry in (journal or {}).get("stages", {}).items() if key.startswith(obj_id + "/")}})
    if not objects:
        raise FileNotFoundError("No prepared object package found under output_root")
    return {"schema_version": REPORT_VERSION, "variant_id": variant_id,
            "report_implementation_sha256": _hash(Path(__file__)),
            "path_base": "directory containing run_report.json", "objects": objects,
            "actual_all_objects_all_modes_segment_bytes": sum(o["actual_all_modes_segment_bytes"] for o in objects),
            "journal": _asset(root, root / ".preparation" / "journal.json") if journal else None,
            "measurement_scope": "existing artifacts only; no encoding, model loading, GPU render or metric recomputation",
            "limitations": ["Single-run descriptive accounting is not a rate-distortion, bitrate or encoding-speed claim.",
                "Media playback duration, cumulative retained task wall durations and latest stage wall durations are different quantities.",
                "Journal retains the latest task/stage attempts; resumed stages may only measure hash checking and skipped work.",
                "Raw/checkpoint source locators and argv are provenance; only managed artifact paths are web request paths."]}


def _markdown(report):
    lines = [f"# Prepared run: {report['variant_id']}", "", report["measurement_scope"] + ".", ""]
    for obj in report["objects"]:
        timeline = obj["timeline"]
        timing = obj["encode_timings"]
        lines += [f"## {obj['id']}", "",
                  f"{timeline['frame_count']} frames; nominal FPS {timeline['fps']:g}; timestamp span {timeline['time_span_seconds']:.9g} s; media duration {timeline['media_duration_seconds']:.9g} s.", "",
                  f"Recorded successful encode-task wall sum: {timing['recorded_successful_encode_task_wall_seconds']} s. Latest encode-stage invocation: {timing['latest_recorded_encode_stage_wall_seconds']} s.", "",
                  "| Mode | Quality | Layer | Codec/version | Gaussian count range | Stored segment bytes | Cumulative delivered-quality bytes | Encode-task sum (s) |", "|---|---|---|---|---:|---:|---:|---:|"]
        for rep in obj["representations"]:
            counts = [row["gaussian_count"] for row in rep["gaussian_counts"] if row["gaussian_count"] is not None]
            count_range = f"{min(counts)}–{max(counts)}" if counts else "unavailable"
            codec = rep["codec"]
            lines.append(f"| {rep['mode']} | {rep['quality']} | {rep['layer']} | {codec['name']} / {codec['version']} | {count_range} | {rep['stored_layer_segment_bytes']} | {rep['cumulative_quality_segment_bytes']} | {rep['recorded_successful_encode_task_wall_seconds']} |")
        lines += ["", "| Frame | Exact timestamp (s) | Sample duration (s) |", "|---:|---:|---:|"]
        for frame in timeline["samples"]:
            lines.append(f"| {frame['frame']} | {frame['timestamp']!r} | {frame['duration_seconds']!r} |")
        lines += ["", f"Quality profile samples: {obj['quality_profile']['sample_count']}.", ""]
        for rep in obj["representations"]:
            lines += [f"### {rep['mode']} / {rep['quality']} / {rep['layer']}", "",
                      f"Codec configuration: `{json.dumps(rep['codec'].get('config', {}), sort_keys=True)}`.", "",
                      "| Segment | Frame range | Media duration (s) | Complete bytes | Relative path |",
                      "|---|---|---:|---:|---|"]
            for segment in rep["segments"]:
                lines.append(f"| {segment['id']} | {segment['frame_range']} | {segment['media_duration_seconds']!r} | {segment['bytes']} | {segment['path']} |")
            lines += ["", "| Frame | Gaussians | Cold payload dependency order | Required payload bytes | Complete request bytes |",
                      "|---:|---:|---|---:|---:|"]
            counts = {row["frame"]: row["gaussian_count"] for row in rep["gaussian_counts"]}
            for access in rep["cold_access"]:
                lines.append(f"| {access['frame']} | {counts[access['frame']]} | {', '.join(access['dependencies_in_decode_order'])} | {access['required_payload_bytes']} | {access['request_bytes']} |")
            lines.append("")
        lines += ["### Source provenance", "",
                  f"Training backend: {obj['provenance']['training_backend']}; verified lineage: {obj['provenance']['lineage_verified']}.", "",
                  "| Quality | Frame | Original checkpoint SHA-256 | Source locator |", "|---|---:|---|---|"]
        for source in obj["provenance"]["checkpoints"]:
            lines.append(f"| {source['quality']} | {source['frame']} | {source['checkpoint_sha256']} | {source['source_locator']} |")
        if obj["provenance"]["raw"] is not None:
            lines += ["", "Raw source inventory:", "", "```json",
                      json.dumps(obj["provenance"]["raw"], sort_keys=True, indent=2, allow_nan=False), "```"]
        lines += ["", "Exact hashes, source argv, metric aggregates and timing task records are retained in [run_report.json](run_report.json).", ""]
    lines += ["## Measurement limits", ""] + [f"- {value}" for value in report["limitations"]]
    return "\n".join(lines) + "\n"


def write_prepared_report(output_root, fps=30, variant_id="f32_lossless", raw_provenance=None):
    """Write run_report.json and run_report.md; return the generated report."""
    root = Path(output_root).resolve()
    report = report_prepared_run(root, fps, variant_id, raw_provenance)
    for name, text in (("run_report.json", json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"),
                       ("run_report.md", _markdown(report))):
        path = root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf8")
        os.replace(temporary, path)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--variant-id", default="f32_lossless")
    parser.add_argument("--raw-provenance", type=Path, help="Optional JSON source inventory")
    args = parser.parse_args()
    raw = json.loads(args.raw_provenance.read_text(encoding="utf8")) if args.raw_provenance else None
    result = write_prepared_report(args.output_root, args.fps, args.variant_id, raw)
    print(json.dumps({"variant_id": result["variant_id"], "objects": len(result["objects"])}))
