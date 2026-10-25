"""Strict, standalone reader for the measured StreamSTGS 8i metadata bundle.

Only the Python standard library is used. No production/runtime code is imported.
Read and validate all metadata before invoking the timed solver runner.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET

OBJECTS = ("longdress", "soldier", "loot", "redandblack")
CRFS = (20, 28, 36, 40)
GOPS = tuple(range(5))
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[2] / "data_stream" /
                     "streamstgs_8i_quality_metadata")


class MetadataError(ValueError):
    """Missing, incompatible or inconsistent measured metadata."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetadataError(message)


def _safe_path(root: Path, relative: str) -> Path:
    """Reject path traversal in metadata URIs, including through symlinks."""
    result = (root / relative).resolve()
    _require(result.is_relative_to(root.resolve()), f"Unsafe metadata path: {relative}")
    return result


def _finite(value, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"Missing/non-numeric {label}")
    result = float(value)
    _require(math.isfinite(result), f"Non-finite {label}")
    return result


def _positive_integer(value, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"Missing/non-positive integer {label}")
    return value


def _summary_psnr(summary: dict, label: str) -> float:
    _require(isinstance(summary, dict), f"Missing quality summary: {label}")
    _require("mean dB" in summary.get("aggregation", ""),
             f"Expected explicitly declared mean-dB aggregation: {label}")
    return _finite(summary.get("metrics", {}).get("psnr"), f"PSNR {label}")


def _cost_manifest_path(data_root: Path, cost_root: Path | None,
                        obj: str, gop: int, crf: int) -> Path:
    roots = [data_root] if cost_root is None else [cost_root, data_root]
    for root in roots:
        candidates = [root / obj / "content" / f"gop_{gop:03d}_crf_{crf}" / "manifest.json",
                      root / obj / f"gop_{gop:03d}" / "encoded" / f"crf_{crf}" / "manifest.json",
                      root / f"gop_{gop:03d}" / "encoded" / f"crf_{crf}" / "manifest.json"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise MetadataError(
        f"Missing actual per-GOP cost manifest for {obj}/gop={gop}/crf={crf}. "
        "Supply --cost-root pointing to a metadata bundle or extracted publication root. "
        "Representation-average MPD bandwidth cannot replace the missing per-GOP size.")


def load_dataset(data_root: str | Path = DEFAULT_DATA_ROOT,
                 cost_root: str | Path | None = None) -> tuple[list[dict], dict]:
    """Return 80 normalized measured profiles and a source/provenance audit.

    Costs count all files declared by each representation manifest plus that
    manifest itself. The actual 2-second segment duration gives exact integer
    bit/s costs; an incompatible fractional rate fails instead of rounding.
    """
    root = Path(data_root).resolve()
    external_cost_root = Path(cost_root).resolve() if cost_root is not None else None
    _require(root.is_dir(), f"Metadata directory does not exist: {root}")
    checksum_path = root / "SHA256SUMS"
    _require(checksum_path.is_file(), f"Missing checksum inventory: {checksum_path}")
    expected_hashes = {}
    for line in checksum_path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        _require(len(digest) == 64, f"Malformed checksum: {line}")
        relative = relative.lstrip(" *")
        _safe_path(root, relative)
        _require(relative not in expected_hashes, f"Duplicate checksum path: {relative}")
        expected_hashes[relative] = digest
    source_records = {}

    def read(path: Path, expected: str | None = None) -> bytes:
        _require(path.is_file(), f"Missing metadata: {path}")
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if path.is_relative_to(root):
            relative = str(path.relative_to(root))
            inventory_digest = expected_hashes.get(relative)
            _require(inventory_digest is not None, f"No SHA256SUMS entry: {relative}")
            _require(digest == inventory_digest, f"Bundle checksum mismatch: {relative}")
            source_id = relative
        else:
            source_id = str(path)
        if expected is not None:
            _require(digest == expected, f"Cross-reference SHA256 mismatch: {path}")
        source_records[source_id] = {"sha256": digest, "bytes": len(blob)}
        return blob

    def read_json(path: Path, expected: str | None = None) -> dict:
        try:
            result = json.loads(read(path, expected))
        except (json.JSONDecodeError, OSError) as error:
            raise MetadataError(f"Cannot read JSON metadata {path}: {error}") from error
        _require(isinstance(result, dict), f"Expected JSON object: {path}")
        return result

    index = read_json(root / "quality-index.json")
    _require(index.get("schema") == "streamstgs.quality-index.v1",
             f"Unsupported top-level index schema: {index.get('schema')}")
    entries = index.get("profiles", [])
    expected_keys = {(o, g, c) for o in OBJECTS for g in GOPS for c in CRFS}
    keys = [(e.get("object"), e.get("gop"), e.get("crf")) for e in entries]
    _require(len(keys) == len(set(keys)) == 80 and set(keys) == expected_keys,
             "Require exactly 4 objects x 5 GOPs x 4 CRFs; missing or duplicate profile")
    entries_by_key = dict(zip(keys, entries))
    normalized = []
    mpd_checks = []
    schema_inventory = {"quality_index": index["schema"]}

    for obj in OBJECTS:
        publication = read_json(root / obj / "publication.json")
        _require(publication.get("schema") == "streamstgs.publication.v1" and
                 publication.get("object") == obj, f"Publication mismatch: {obj}")
        object_index_path = _safe_path(root / obj, publication["quality_profile"]["uri"])
        object_index = read_json(object_index_path, publication["quality_profile"]["sha256"])
        _require(object_index.get("schema") == "streamstgs.quality-index.v1",
                 f"Unsupported object index schema: {obj}")
        object_entries = {(e.get("gop"), e.get("crf")): e for e in object_index["profiles"]}
        _require(len(object_index["profiles"]) == len(object_entries) == 20,
                 f"Missing/duplicate object profile index entries: {obj}")
        package = read_json(root / obj / "content" / "stream_package_manifest.json",
                            publication["stream_package_manifest_sha256"])
        _require(object_index["package_manifest_sha256"] ==
                 publication["stream_package_manifest_sha256"], f"Package hash mismatch: {obj}")
        _require(package.get("schema") == 1 and package.get("object") == obj and
                 package.get("gop_size") == 60 and package.get("gop_count") == 5 and
                 tuple(package.get("crfs", [])) == CRFS, f"Unexpected package structure: {obj}")
        fps = _positive_integer(package.get("fps"), f"fps {obj}")
        segment_seconds = Fraction(package["gop_size"], fps)
        _require(segment_seconds == 2, f"Expected measured 60 frames / 30fps: {obj}")
        file_list = package.get("files", [])
        files = {e["path"]: e for e in file_list}
        _require(len(file_list) == len(files) == package["file_count"],
                 f"Package file count mismatch: {obj}")
        _require(sum(_positive_integer(e.get("bytes"), e["path"]) for e in file_list) ==
                 package["total_file_bytes"], f"Package byte total mismatch: {obj}")

        xml = ET.fromstring(read(root / obj / publication["mpd"]))
        ns = {"d": "urn:mpeg:dash:schema:mpd:2011"}
        representations = {int(e.attrib["id"].removeprefix("crf_")): e
                           for e in xml.findall(".//d:Representation", ns)}
        _require(set(representations) == set(CRFS), f"Unexpected MPD representations: {obj}")
        for crf in CRFS:
            rep_xml = representations[crf]
            segment_list = rep_xml.find("d:SegmentList", ns)
            _require(segment_list is not None, f"Missing MPD SegmentList: {obj}/{crf}")
            timeline = segment_list.findall("d:SegmentTimeline/d:S", ns)
            urls = segment_list.findall("d:SegmentURL", ns)
            _require(len(timeline) == len(urls) == 5, f"Unexpected MPD timeline: {obj}/{crf}")
            transfer_total = 0
            for gop, (segment, url) in enumerate(zip(timeline, urls)):
                _require(Fraction(int(segment.attrib["d"]),
                                  int(segment_list.attrib["timescale"])) == segment_seconds,
                         f"MPD/package duration mismatch: {obj}/{gop}/{crf}")
                relative_manifest = f"gop_{gop:03d}/encoded/crf_{crf}/manifest.json"
                _require(url.attrib["media"] == relative_manifest,
                         f"Unexpected MPD segment URI: {obj}/{gop}/{crf}")
                entry = entries_by_key[obj, gop, crf]
                local_entry = object_entries.get((gop, crf), {})
                for name in ("sha256", "summary", "representation_manifest_sha256"):
                    _require(entry[name] == local_entry.get(name),
                             f"Global/object quality index disagreement: {obj}/{gop}/{crf}")
                profile_path = _safe_path(root, entry["uri"])
                profile = read_json(profile_path, entry["sha256"])
                _require(profile.get("schema") == "streamstgs.quality-profile.v1",
                         f"Unsupported quality profile schema: {profile_path}")
                metadata = profile.get("metadata", {})
                rep = metadata.get("representation", {})
                _require((rep.get("object"), rep.get("gop"), rep.get("crf")) ==
                         (obj, gop, crf), f"Quality profile identity mismatch: {profile_path}")
                _require(rep.get("manifest_sha256") == entry["representation_manifest_sha256"],
                         f"Profile/index representation hash mismatch: {profile_path}")
                reference = metadata.get("reference", {})
                _require(reference.get("kind") == "original_ply_synthetic_rgb" and
                         reference.get("relative_distortion") is False,
                         f"Expected original synthetic RGB reference: {profile_path}")
                _require(metadata.get("scales") == [1.0] and metadata.get("views") == list(range(24)),
                         f"Unexpected resolution/view coverage: {profile_path}")
                frames = metadata.get("frames", [])
                _require(len(frames) == 60, f"Expected 60 frames: {profile_path}")
                frame_ids = {row["frame"] for row in frames}
                _require(frame_ids == set(range(min(frame_ids), min(frame_ids) + 60)),
                         f"Non-consecutive measured frames: {profile_path}")
                rows = profile.get("rows", [])
                cells = {(row.get("frame"), row.get("view")) for row in rows}
                _require(len(rows) == len(cells) == 1440 and cells ==
                         {(f, v) for f in frame_ids for v in range(24)},
                         f"Missing/duplicate frame x view samples: {profile_path}")
                for row in rows:
                    _require((row.get("object"), row.get("gop"), row.get("crf"), row.get("scale")) ==
                             (obj, gop, crf, 1.0), f"Row identity mismatch: {profile_path}")
                psnr = _summary_psnr(profile.get("summary"), str(profile_path))
                _require(profile["summary"] == entry["summary"] and
                         profile["summary"]["sample_count"] == len(rows),
                         f"Profile/index summary disagreement: {profile_path}")
                recomputed = math.fsum(_finite(row.get("metrics", {}).get("psnr"), "row PSNR")
                                       for row in rows) / len(rows)
                _require(math.isclose(psnr, recomputed, abs_tol=1e-10, rel_tol=1e-12),
                         f"PSNR summary does not equal measured row mean: {profile_path}")

                manifest_path = _cost_manifest_path(root, external_cost_root, obj, gop, crf)
                manifest_record = files.get(relative_manifest)
                _require(manifest_record is not None, f"Manifest absent from package: {relative_manifest}")
                manifest = read_json(manifest_path, rep["manifest_sha256"])
                _require(manifest.get("schema") == 1 and manifest.get("crf") == crf,
                         f"Representation manifest mismatch: {manifest_path}")
                manifest_bytes = _positive_integer(manifest_record.get("bytes"), "manifest bytes")
                _require(manifest_bytes == manifest_path.stat().st_size,
                         f"Manifest byte length mismatch: {manifest_path}")
                components = manifest.get("files", {})
                _require(isinstance(components, dict) and components, f"No cost components: {manifest_path}")
                payload_bytes = sum(_positive_integer(e.get("bytes"), f"component {name}")
                                    for name, e in components.items())
                _require(payload_bytes == manifest.get("actual_payload_bytes") == rep.get("payload_bytes"),
                         f"Per-GOP payload byte mismatch: {manifest_path}")
                prefix = relative_manifest.rsplit("/", 1)[0]
                for name, component in components.items():
                    _require(files.get(f"{prefix}/{name}") == {"path": f"{prefix}/{name}", **component},
                             f"Component/package size or hash disagreement: {manifest_path}/{name}")
                reconstruction = manifest_path.parent / "reconstruction.json"
                read(reconstruction, rep["reconstruction_sha256"])
                _require(rep["reconstruction_sha256"] == components["reconstruction.json"]["sha256"],
                         f"Reconstruction link mismatch: {profile_path}")
                transfer_bytes = payload_bytes + manifest_bytes
                rate = Fraction(8 * transfer_bytes, 1) / segment_seconds
                _require(rate.denominator == 1, "Non-integer measured bit/s rate: use exact scaled integer units")
                transfer_total += transfer_bytes
                normalized.append({
                    "object": obj, "gop": gop, "crf": crf, "psnr_db": psnr,
                    "cost_bps": int(rate), "payload_bytes": payload_bytes,
                    "manifest_bytes": manifest_bytes, "transfer_bytes": transfer_bytes,
                    "duration_seconds": float(segment_seconds), "frames": 60,
                    "views": 24, "sample_count": len(rows),
                    "quality_profile_uri": entry["uri"], "quality_profile_sha256": entry["sha256"],
                    "representation_manifest_sha256": rep["manifest_sha256"],
                })
            measured_average = Fraction(transfer_total * 8, 10)
            published = int(rep_xml.attrib["bandwidth"])
            _require(published == math.ceil(measured_average),
                     f"Actual GOP sizes do not reproduce published MPD bandwidth: {obj}/{crf}")
            mpd_checks.append({"object": obj, "crf": crf, "mpd_average_bps": published,
                               "measured_average_bps": float(measured_average)})

    normalized.sort(key=lambda p: (OBJECTS.index(p["object"]), p["gop"], CRFS.index(p["crf"])))
    audit = {
        "schema": "abr-solver-dataset-audit.v1", "data_root": str(root),
        "cost_root": str(external_cost_root) if external_cost_root else None,
        "profile_count": len(normalized), "objects": list(OBJECTS), "gops_per_object": 5,
        "frames_per_gop": 60, "views_per_frame": 24, "crfs": list(CRFS),
        "total_metric_samples": sum(p["sample_count"] for p in normalized),
        "quality_statistic": "Arithmetic mean of measured PSNR in dB over 60 frames x 24 views at scale 1.0; includes training and held-out views",
        "reference": "original_ply_synthetic_rgb; synthetic point-splat RGB, not captured photographs",
        "cost_definition": "8 * (actual_payload_bytes + representation manifest bytes) / (60 / 30) bit/s; manifest-listed payload files include codec components and auxiliary metadata",
        "cost_arithmetic": "Exact integer bps: GOP duration is exactly 2 seconds, so cost_bps = 4 * transfer_bytes",
        "payloads_read_or_decoded": False,
        "validation": ["80 profile and 80 representation manifest checksums", "80 reconstruction sidecar hashes",
                       "global and object indexes and publication/package cross-links", "all 115200 row identities and frame-view coverage",
                       "PSNR means recomputed from rows", "every component size/hash agrees with package inventory",
                       "all 16 MPD average bandwidths reproduce ceil(sum actual transfer bytes * 8 / 10s)"],
        "validation_limits": "Media payload files and original RGB/PLY contents are not rehashed; their recorded component hashes are cross-checked. Only metadata files used by this experiment are read and hashed.",
        "psnr_db_range": [min(p["psnr_db"] for p in normalized), max(p["psnr_db"] for p in normalized)],
        "cost_bps_range": [min(p["cost_bps"] for p in normalized), max(p["cost_bps"] for p in normalized)],
        "transfer_bytes_range": [min(p["transfer_bytes"] for p in normalized), max(p["transfer_bytes"] for p in normalized)],
        "mpd_average_checks": mpd_checks, "source_files": source_records,
        "checksum_inventory_sha256": hashlib.sha256(checksum_path.read_bytes()).hexdigest(),
    }
    return normalized, audit


def write_audit_report(audit: dict, path: str | Path) -> None:
    lo_q, hi_q = audit["psnr_db_range"]
    lo_r, hi_r = audit["cost_bps_range"]
    lines = ["# Measured metadata audit", "", f"Source: `{audit['data_root']}`.", "",
             "The bundle contains Longdress, Soldier, Loot and RedAndBlack; each has five consecutive "
             "60-frame GOPs at 30 fps, independently encoded as CRF20/28/36/40. All 80 object × GOP × CRF "
             "profiles are present. No quality or bitrate values are synthesized.", "",
             f"PSNR is the profile's arithmetic mean in dB over 60 frames × 24 views (1,440 measurements "
             f"per profile; {audit['total_metric_samples']:,} total). It is not PSNR computed from pooled MSE. "
             "All 24 views, including training and held-out views, are used at scale 1.0. "
             "The reference is original-PLY synthetic RGB rendered by the offline point-splat capture; "
             "it is not captured photography. The observed profile mean range is "
             f"{lo_q:.6f}–{hi_q:.6f} dB.", "",
             "Each quality profile identifies the representation manifest by SHA256 and repeats its "
             "payload byte count. The copied manifests reside under "
             "`<object>/content/gop_XXX_crf_YY/manifest.json`; the original publication paths in the package "
             "inventory are `gop_XXX/encoded/crf_YY/manifest.json`. The loader explicitly reconciles these layouts.", "",
             "The cost counts the sum of all manifest-listed component bytes plus the manifest itself. "
             "This includes compressed media, MLP parameters and auxiliary files listed by the publication. "
             "No cross-object sharing, caching, HTTP headers or startup cost is assumed. The MPD timeline "
             "declares timescale 30 and duration 60: each GOP is exactly two seconds. Therefore "
             "`cost_bps = 4 × transfer_bytes` is an exact integer without bitrate rounding. "
             f"Observed GOP costs range from {lo_r:,} to {hi_r:,} bit/s.", "",
             "All 16 MPD Representation bandwidths equal the ceiling of the measured ten-second "
             "average including component manifests. These averages are checked for provenance; the "
             "solver uses each actual GOP's size, so no GOP variation is replaced by the average.", "",
             "Validation checks the bundle SHA256 inventory for every consumed file, publication → "
             "object quality index → profile → representation/reconstruction hashes, package inventory "
             "file counts/bytes, all frame-view cells, and the mean PSNR recomputed from raw metric rows. "
             "Payload media, PLY and RGB images are not loaded, extracted, decoded or rehashed. Their "
             "recorded hashes and sizes are cross-checked across metadata.", "",
             "Missing or inconsistent metadata causes a hard error. `--cost-root` can point to an "
             "external metadata bundle or extracted object publication if per-GOP cost manifests are "
             "relocated; their hashes and package inventory must still match. There is no fabricated fallback.", "",
             "The complete consumed-file SHA256 list, normalized profile values and MPD checks are in "
             "`inputs/dataset_audit.json` and `inputs/measured_profiles.json`.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
