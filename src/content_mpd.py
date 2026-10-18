"""Pure project MPD model shared by export and metadata consumers.

No filesystem, network, media, renderer, YAML or numerical-library imports.
CPSEG and the Gaussian profile are project extensions, not DASH video playback.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import math
from pathlib import PurePosixPath as Path
import xml.etree.ElementTree as ET

from .content_metadata import content_revision


def safe_relative_path(value):
    from .content_metadata import relative_path
    return relative_path(value)


VERSION = "1.2.0"
NAMESPACE = "urn:mpeg:dash:schema:mpd:2011"
PROFILE = "urn:multiobj3dgsvstream:dash:profile:gaussian:1"
PROPERTY = "urn:multiobj3dgsvstream:content-preparation:"
UINT32_MAX = (1 << 32) - 1
EXTENDED_BANDWIDTH_PROPERTY = "extended-bandwidth-bits-per-second:1"
EXTENDED_BANDWIDTH_SCHEME = PROPERTY + EXTENDED_BANDWIDTH_PROPERTY
INDEX_SCHEMA = "content-preparation.content-index.v1"
DENSITY_PROPERTY = "distortion-density:1"
REVISION_PROPERTY = "content-revision:1"
ET.register_namespace("", NAMESPACE)


def _element(parent, name, **attributes):
    return ET.SubElement(parent, f"{{{NAMESPACE}}}{name}",
                         {key: str(value) for key, value in attributes.items()})


def _descriptor(parent, name, value, essential=False):
    return _element(parent, "EssentialProperty" if essential else "SupplementalProperty",
                    schemeIdUri=PROPERTY + name, value=value)


def _prefix(directory: str, relative: str) -> str:
    safe_relative_path(relative)
    value = (Path(directory) / relative).as_posix()
    safe_relative_path(value)
    return value


def _duration(seconds: float) -> str:
    # Decimal seconds avoid rounding 300/30 to the last frame timestamp.
    return "PT" + format(seconds, ".12f").rstrip("0").rstrip(".") + "S"


def _timing(manifest: dict) -> dict:
    obj = manifest["object"]
    frames = obj["frames"]
    fps = obj.get("fps")
    if fps is None:
        interval = obj.get("sample_duration_seconds")
        if interval is None and len(frames) > 1:
            intervals = [b["timestamp"] - a["timestamp"] for a, b in zip(frames, frames[1:])]
            if any(not math.isclose(v, intervals[0], rel_tol=1e-8, abs_tol=1e-10) for v in intervals):
                raise ValueError("Nonuniform timestamps require explicit object fps")
            interval = intervals[0]
        if interval is None:
            raise ValueError("Single-frame manifest requires fps or sample_duration_seconds")
        fps = 1.0 / interval
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Object fps must be finite and positive")
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    timescale, frame_ticks = rate.numerator, rate.denominator
    ticks = []
    for frame in frames:
        value = frame["timestamp"] * timescale
        if not math.isclose(value, round(value), abs_tol=2e-6):
            raise ValueError("Frame timestamp cannot be represented by the source fps timescale")
        ticks.append(round(value))
    if ticks[0] < 0 or any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError("Media frame ticks must be nonnegative and strictly increasing")
    terminal_tick = ticks[-1] + frame_ticks
    duration = (terminal_tick - ticks[0]) / timescale
    if "duration_seconds" in obj and not math.isclose(obj["duration_seconds"], duration, abs_tol=2e-8):
        raise ValueError("Declared duration must include the terminal sample interval")
    return {"timescale": timescale, "sample_ticks": frame_ticks,
            "frame_ticks": dict(zip((f["frame"] for f in frames), ticks)),
            "frame_positions": {f["frame"]: i for i, f in enumerate(frames)},
            "ordered_ticks": ticks, "terminal_tick": terminal_tick,
            "presentation_time_offset": ticks[0], "duration_seconds": duration,
            "fps": float(rate), "frame_rate": f"{rate.numerator}/{rate.denominator}"}


def _segment_timing(segment: dict, timing: dict) -> dict:
    positions = [timing["frame_positions"][frame] for frame in segment["frames"]]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Segment frames must cover a contiguous part of the media timeline")
    start = timing["ordered_ticks"][positions[0]]
    following = positions[-1] + 1
    end = timing["ordered_ticks"][following] if following < len(timing["ordered_ticks"]) else timing["terminal_tick"]
    return {"start_tick": start, "duration_ticks": end - start,
            "duration_seconds": (end - start) / timing["timescale"]}


def _closure(representation: dict, by_layer: dict) -> list[dict]:
    result, visiting = [], set()

    def visit(current):
        if current["layer"] in visiting:
            raise ValueError("Cyclic representation dependency")
        visiting.add(current["layer"])
        if current.get("parent_layer") is not None:
            if current["parent_layer"] not in by_layer:
                raise ValueError("Missing parent representation")
            visit(by_layer[current["parent_layer"]])
        result.append(current)
        visiting.remove(current["layer"])

    visit(representation)
    return result


def build_mpd(objects: list[dict]) -> tuple[bytes, dict]:
    """Return deterministic MPD XML and JSON sidecar metadata for loaded objects."""
    if not objects:
        raise ValueError("MPD requires at least one object")
    timings = [_timing(record["manifest"]) for record in objects]
    duration = max(t["duration_seconds"] for t in timings)
    xml = ET.Element(f"{{{NAMESPACE}}}MPD", {"type": "static", "profiles": PROFILE,
                     "minBufferTime": "PT0S", "mediaPresentationDuration": _duration(duration)})
    period = _element(xml, "Period", id="content", start="PT0S", duration=_duration(duration))
    index = {"schema_version": INDEX_SCHEMA, "mpd_profile": PROFILE, "writer_version": VERSION,
             "content_revision": content_revision(objects),
             "duration_seconds": duration, "objects": [],
             "interoperability": "experimental CPSEG + Gaussian decoder; not ordinary DASH video playback",
             "bandwidth_policy": "ceil(sum of per-layer peak complete-container bits/s in dependency closure)",
             "bandwidth_signaling": {
                 "standard_attribute_max": UINT32_MAX,
                 "overflow_policy": "saturate bandwidth only with mandatory extended-rate EssentialProperty",
                 "extended_rate_scheme": EXTENDED_BANDWIDTH_SCHEME,
                 "required_rate_field": "required_bandwidth_bits_per_second",
             },
             "access_policy": "every frame with same-frame parent closure; no temporal history",
             "accounting": "actual complete segment files including container and codec headers"}
    for adaptation_id, (record, timing) in enumerate(zip(objects, timings), 1):
        manifest, directory = record["manifest"], record["directory"]
        obj, package = manifest["object"], manifest["package"]
        representations = sorted(manifest["representations"], key=lambda r: r["nominal_quality"])
        by_layer = {r["layer"]: r for r in representations}
        segment_lookup = {s["id"]: s for s in package["segments"]}
        rate_records, peak_rates, boundaries = {}, {}, []
        for rep in representations:
            segments = [segment_lookup[s] for s in rep["segments"]]
            serial = []
            for segment in segments:
                st = _segment_timing(segment, timing)
                serial.append({**copy.deepcopy(segment), **st, "path": _prefix(directory, segment["path"])})
            boundary = [(s["start_tick"], s["duration_ticks"]) for s in serial]
            boundaries.append(boundary)
            # Preserve exact rates when computing the integer required bandwidth.
            # Container bytes and SegmentTimeline ticks are integers; converting
            # through float seconds could round an otherwise exact ceiling.
            own_peak = max(Fraction(s["bytes"] * 8 * timing["timescale"], s["duration_ticks"])
                           for s in serial)
            peak_rates[rep["layer"]] = own_peak
            rate_records[rep["layer"]] = {"segments": serial, "own_bytes": sum(s["bytes"] for s in serial),
                                         "own_peak_bits_per_second": float(own_peak),
                                         "own_mean_bits_per_second": float(Fraction(
                                             rep["bytes"] * 8 * timing["timescale"],
                                             timing["terminal_tick"] - timing["presentation_time_offset"]))}
        aligned = all(boundary == boundaries[0] for boundary in boundaries[1:])
        adaptation = _element(period, "AdaptationSet", id=adaptation_id,
                              mimeType="application/octet-stream", segmentAlignment=str(aligned).lower())
        _descriptor(adaptation, "object-id", obj["id"], essential=True)
        _descriptor(adaptation, "manifest", record["manifest_path"])
        profile_path = _prefix(directory, obj["quality_profile_path"]) if obj.get("quality_profile_path") else None
        proxy_path = _prefix(directory, obj["proxy_path"]) if obj.get("proxy_path") else None
        if profile_path:
            _descriptor(adaptation, "quality-profile", profile_path)
        if proxy_path:
            _descriptor(adaptation, "proxy", proxy_path)
        if record.get("density_profile"):
            _descriptor(adaptation, DENSITY_PROPERTY, record["density_profile"]["path"])
        object_record = {"id": obj["id"], "manifest_path": record["manifest_path"],
                         "manifest_sha256": record["manifest_sha256"], "manifest_bytes": record["manifest_bytes"],
                         "package_index_path": record["package_path"], "package_index_sha256": record["package_sha256"],
                         "package_index_bytes": record["package_bytes"], "quality_profile_path": profile_path,
                         "proxy_path": proxy_path, "fps": timing["fps"], "timescale": timing["timescale"],
                         "duration_seconds": timing["duration_seconds"], "frames": copy.deepcopy(obj["frames"]),
                         "canonical_transform": copy.deepcopy(obj["canonical_transform"]),
                         "provenance": copy.deepcopy(manifest["provenance"]), "epochs": copy.deepcopy(package["epochs"]),
                         "gof_frames": package["gof_frames"], "segment_frames": package["segment_frames"],
                         "temporal_prediction": False, "segment_alignment": aligned, "representations": []}
        for field in ("quality_profile", "density_profile"):
            if record.get(field):
                object_record[field] = copy.deepcopy(record[field])
        for rep in representations:
            closure = _closure(rep, by_layer)
            rep_id = f"{obj['id']}:{rep['layer']}"
            dependencies = [f"{obj['id']}:{r['layer']}" for r in closure[:-1]]
            cumulative_peak = sum((peak_rates[r["layer"]] for r in closure), Fraction())
            required_bandwidth = max(1, math.ceil(cumulative_peak))
            bandwidth_saturated = required_bandwidth > UINT32_MAX
            # The standard MPD bandwidth type is xs:unsignedInt. These
            # experimental uncompressed Gaussian rates can exceed its range.
            # Saturation is signaled explicitly as a required extension, so a
            # project client MUST read the full integer rate before admission;
            # a client lacking this scheme cannot interpret this representation.
            # Normal representations retain their exact standard bandwidth.
            attributes = {"id": rep_id, "bandwidth": min(required_bandwidth, UINT32_MAX),
                          "codingDependency": str(bool(dependencies)).lower()}
            if dependencies:
                attributes["dependencyId"] = " ".join(dependencies)
            representation = _element(adaptation, "Representation", **attributes)
            codec = rep["codec"]
            _descriptor(representation, "codec", f"{codec['name']}/{codec['version']}", essential=True)
            _descriptor(representation, "container", "content-preparation.segment.v1", essential=True)
            if bandwidth_saturated:
                _descriptor(representation, EXTENDED_BANDWIDTH_PROPERTY, required_bandwidth, essential=True)
            _descriptor(representation, "quality", rep["quality"])
            _descriptor(representation, "decoded-state-version", rep["decoded_state_version"])
            _descriptor(representation, "access", rep["access_requirements"])
            _descriptor(representation, "refresh-interval-frames", rep["refresh_interval_frames"])
            listing = _element(representation, "SegmentList", timescale=timing["timescale"],
                               presentationTimeOffset=timing["presentation_time_offset"], startNumber=1)
            timeline = _element(listing, "SegmentTimeline")
            rates = rate_records[rep["layer"]]
            for segment in rates["segments"]:
                _element(timeline, "S", t=segment["start_tick"], d=segment["duration_ticks"])
            for segment in rates["segments"]:
                _element(listing, "SegmentURL", media=segment["path"])
            object_record["representations"].append({**copy.deepcopy(rep), **rates,
                "id": rep_id, "required_representation_ids": dependencies,
                "bandwidth": attributes["bandwidth"],
                "required_bandwidth_bits_per_second": required_bandwidth,
                "bandwidth_saturated": bandwidth_saturated,
                "extended_bandwidth_scheme": EXTENDED_BANDWIDTH_SCHEME if bandwidth_saturated else None,
                "cumulative_bytes": sum(rate_records[r["layer"]]["own_bytes"] for r in closure),
                # Sum integer bytes before conversion: Python 3.12 changed
                # floating sum(), which must not change a publication's bytes.
                "cumulative_mean_bits_per_second": float(Fraction(
                    sum(rate_records[r["layer"]]["own_bytes"] for r in closure) * 8 * timing["timescale"],
                    timing["terminal_tick"] - timing["presentation_time_offset"])),
                "cumulative_peak_bound_bits_per_second": float(cumulative_peak)})
        index["objects"].append(object_record)
    _descriptor(xml, "content-index", "content_index.json")
    _descriptor(xml, REVISION_PROPERTY, index["content_revision"])
    ET.indent(xml, space="  ")
    return ET.tostring(xml, encoding="utf-8", xml_declaration=True) + b"\n", index


