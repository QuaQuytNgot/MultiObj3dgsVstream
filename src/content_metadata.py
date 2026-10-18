"""Strict, bounded JSON/profile validation shared by MPD export and the client.

Only metadata is inspected. Image paths and recorded hashes are provenance, not
permission to open images, states, codecs, or payloads.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from urllib.parse import unquote, urljoin, urlsplit

MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
PROFILE_SCHEMA = "dynamic-lapisgs.quality-profile.v1"
DENSITY_SCHEMA = "dynamic-lapisgs.error-density.v1"
CAPABILITY_SCHEMA = "content-preparation.quality-capability.v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(data: bytes, *, limit=MAX_DOCUMENT_BYTES):
    if not isinstance(data, bytes) or len(data) > limit:
        raise ValueError("Metadata must be bounded bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            invalid(value)
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=invalid, parse_float=floating)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON metadata") from exc


def number(value, label, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"Invalid {label}")
    return value


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid integer {label}")
    return value


def checksum(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA-256")
    return value


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("Unsafe metadata path")
    decoded = value
    for _ in range(4):
        new = unquote(decoded)
        if new == decoded:
            break
        decoded = new
    # Reject encoded separators too; one canonical spelling per local asset.
    if (decoded != value or any(c.isspace() or ord(c) < 32 for c in value)
            or any(c in value for c in "\\:?#%") or value.startswith("/")
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError(f"Unsafe metadata path: {value!r}")
    return PurePosixPath(value).as_posix()


def absolute_uri(value: str) -> str:
    try:
        p = urlsplit(value)
        if (p.scheme not in ("http", "https", "file") or p.query or p.fragment
                or p.username or p.password or "\\" in value
                or any(c.isspace() or ord(c) < 32 for c in value)
                or (p.scheme != "file" and not p.hostname)
                or (p.scheme == "file" and p.netloc not in ("", "localhost"))):
            raise ValueError("Unsafe absolute metadata URI")
        p.port
        path = p.path
        for _ in range(4):
            path = unquote(path)
        if "\\" in path or any(c in path for c in "\x00\r\n") or ".." in path.split("/"):
            raise ValueError("Unsafe absolute metadata URI path")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsafe metadata URI: {value!r}") from exc
    return value


def resolve_uri(base: str, reference: str) -> str:
    absolute_uri(base)
    if not isinstance(reference, str):
        raise ValueError("Metadata reference must be text")
    parts = urlsplit(reference)
    if parts.scheme:
        absolute_uri(reference)
        if parts.scheme == "file":
            raise ValueError("Absolute file references are not allowed")
    else:
        # A trailing slash is permitted for a directory BaseURL.
        relative_path(reference[:-1] if reference.endswith("/") else reference)
    return absolute_uri(urljoin(base, reference))


def content_revision(records: list[dict]) -> str:
    """Identity of ordered source metadata, independent of deployment URLs."""
    sources = [{"id": r["manifest"]["object"]["id"],
                "manifest": r["manifest_sha256"], "package": r["package_sha256"],
                "mse": r.get("quality_profile", {}).get("sha256"),
                "density": r.get("density_profile", {}).get("sha256")}
               for r in records]
    return digest(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode())


def validate_object_metadata(manifest: dict, package: dict):
    # Existing validators use root=None: no referenced asset is opened.
    from tools.content_preparation.validation import validate_manifest

    if manifest["package"] != package:
        raise ValueError("Object manifest and package_index.json disagree")
    validate_manifest(manifest)
    obj = manifest["object"]
    if re.fullmatch(r"[A-Za-z0-9_-]+", obj["id"]) is None:
        raise ValueError("Unsupported object identity")
    if manifest["delivery_modes"] != ["progressive"] or package["temporal_prediction"] is not False:
        raise ValueError("Only progressive, independently initialized temporal frames are supported")
    qualities = manifest["quality_order"]
    representations = sorted(manifest["representations"], key=lambda r: r["nominal_quality"])
    for quality, rep in enumerate(representations):
        layer = "Base" if quality == 0 else f"E{quality}"
        parent = None if quality == 0 else ("Base" if quality == 1 else f"E{quality-1}")
        if (rep["nominal_quality"] != quality or rep["quality"] != qualities[quality]
                or rep["layer"] != layer or rep["parent_layer"] != parent):
            raise ValueError("Invalid cumulative progressive representation chain")
    for f in obj["frames"]:
        integer(f["frame"], "frame")
        number(f["timestamp"], "timestamp")
    for s in package["segments"]:
        if s.get("container_version") != "content-preparation.segment.v1":
            raise ValueError("Unsupported CPSEG container version")
        integer(s["bytes"], "complete container bytes", 1)
        relative_path(s["path"])
        checksum(s["sha256"])
        for member in s["members"]:
            integer(member["offset"], "member offset")
            integer(member["length"], "member length", 1)
    by_id = {p["id"]: p for p in package["payloads"]}
    by_sample = {(p["frame"], p["quality"]): p for p in by_id.values()}
    for p in by_id.values():
        integer(p["bytes"], "encoded bytes", 1)
        number(p["timestamp"], "payload timestamp")
        if p.get("parent_layer") is not None:
            parent = by_id[p["dependencies"][0]]
            quality = qualities.index(p["quality"])
            if parent != by_sample[(p["frame"], qualities[quality-1])]:
                raise ValueError("Progressive dependency skips its preceding quality")
            if p.get("parent_state_hash") != parent["decoded_state_hash"]:
                raise ValueError("Progressive parent state hash mismatch")


def _row_key(row):
    return (row["object"], row["mode"], row["quality"], row["view_bin"],
            row["scale_bin"], row["time_bin"])


def _context(row):
    return (row["frame"], row["azimuth"], row["elevation"], row["scale"], row["timestamp"])


def validate_profile(document: dict, manifest: dict) -> dict:
    if document.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Unsupported quality profile schema")
    qualities = manifest["quality_order"]
    metadata = document.get("metadata", {})
    if (metadata.get("reference_quality") != qualities[-1]
            or document.get("reference") != "highest_quality_decoded_delivered_representation"):
        raise ValueError("Invalid MSE reference-quality convention")
    frames = {f["frame"]: f["timestamp"] for f in manifest["object"]["frames"]}
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Quality profile requires measured rows")
    seen, contexts = {}, {}
    for row in rows:
        if (row["object"] != manifest["object"]["id"] or row["mode"] != "progressive"
                or row["quality"] not in qualities or row.get("reference_state") != "decoded"
                or row.get("reference_quality") != qualities[-1]):
            raise ValueError("Quality profile identity/reference mismatch")
        integer(row["frame"], "profile frame")
        number(row["timestamp"], "profile timestamp")
        if row["frame"] not in frames or not math.isclose(row["timestamp"], frames[row["frame"]], abs_tol=1e-10):
            raise ValueError("Profile frame/time coverage mismatch")
        for field in ("azimuth", "elevation"):
            number(row[field], field, -360)
        if number(row["scale"], "profile scale") == 0:
            raise ValueError("Profile scale must be positive")
        mse = number(row["metrics"]["mse"], "MSE")
        if row["quality"] == qualities[-1] and mse != 0:
            raise ValueError("Highest-quality self-reference must have MSE zero")
        key = _row_key(row)
        if key in seen:
            raise ValueError("Duplicate quality profile context")
        seen[key] = row
        group = contexts.setdefault(key[:2] + key[3:], {})
        if group and _context(next(iter(group.values()))) != _context(row):
            raise ValueError("Profile bins have inconsistent coordinates")
        group[row["quality"]] = row
    if any(set(group) != set(qualities) for group in contexts.values()):
        raise ValueError("Incomplete quality coverage within profile context")
    return {"schema_version": CAPABILITY_SCHEMA, "metric": "full-image-rgb-mse",
            "profile_schema": PROFILE_SCHEMA, "reference": f"decoded_{qualities[-1]}",
            "normalization": "3*width*height", "pixel_support": "full_image",
            "coverage": {"rows": len(rows), "contexts": len(contexts),
                         "frames": sorted({r["frame"] for r in rows}),
                         "qualities": list(qualities)},
            "renderer_settings": metadata.get("renderer_settings", {})}


def profile_reference(path: str, raw: bytes, document: dict, manifest: dict) -> dict:
    return {"path": relative_path(path), "bytes": len(raw), "sha256": digest(raw),
            **validate_profile(document, manifest)}


def density_reference(path: str, raw: bytes, document: dict, source: dict,
                      source_reference: dict, manifest: dict) -> dict:
    validate_profile(document, manifest)
    meta = document.get("metadata", {}).get("error_density", {})
    if (meta.get("schema") != DENSITY_SCHEMA or meta.get("version") != "1.0.0"
            or meta.get("area_quality") != "Q0" or meta.get("area_reference_state") != "decoded"
            or meta.get("reference_quality") != "Q3" or meta.get("reference_state") != "decoded"
            or meta.get("mse_convention") != "full-image RGB [0,1] sum squared error / (3*width*height); unmasked"):
        raise ValueError("Unsupported MSE-versus-density capability or normalization version")
    if meta.get("source_profile", {}).get("sha256") != source_reference["sha256"]:
        raise ValueError("Density source profile hash mismatch")
    if (meta.get("canonical_transform") != manifest["object"]["canonical_transform"]
            or meta.get("renderer_settings") != source["metadata"]["renderer_settings"]
            or meta.get("source_implementation_sha256") != manifest["provenance"]["implementation_sha256"]):
        raise ValueError("Density rendering/content provenance mismatch")
    epsilon = number(meta.get("epsilon_fraction"), "density area fraction guard")
    if epsilon > 1:
        raise ValueError("Density area fraction guard exceeds image")
    source_rows = {_row_key(r): r for r in source["rows"]}
    areas, dimensions, contexts = {}, set(), {}
    payloads = {(p["frame"], p["quality"]): p for p in manifest["package"]["payloads"]}
    for row in document["rows"]:
        original = source_rows.get(_row_key(row))
        if original is None or any(row.get(k) != v for k, v in original.items()):
            raise ValueError("Density row changes or invents original MSE/context")
        width, height = integer(row["width"], "density width", 1), integer(row["height"], "density height", 1)
        pixels = integer(row["pixel_count"], "density pixel count", 1)
        area = number(row["solo_area"], "Q0 solo area")
        if pixels != width * height or area > pixels * (1 + 1e-5):
            raise ValueError("Invalid density pixel support")
        if (row.get("area_quality") != "Q0" or row.get("area_reference_state") != "decoded"
                or row.get("area_epsilon_fraction") != epsilon
                or row.get("alpha_convention") != "one_minus_transmittance"
                or row.get("solo_area_source", {}).get("decoded_state_hash") != payloads[(row["frame"], "Q0")]["decoded_state_hash"]):
            raise ValueError("Density Q0-area provenance mismatch")
        if not math.isclose(number(row["solo_area_fraction"], "area fraction"), area / pixels, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Density area fraction mismatch")
        if area / pixels <= epsilon:
            if row.get("error_density") is not None or row.get("area_status") == "valid":
                raise ValueError("Near-zero Q0 area must have unavailable density")
        else:
            value = number(row.get("error_density"), "density")
            if row.get("area_status") != "valid" or not math.isclose(value, row["metrics"]["mse"] * pixels / area, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Density value does not match measured MSE/Q0 area")
        context = _context(row)
        if context in areas and areas[context] != area:
            raise ValueError("Quality-specific density denominator")
        areas[context] = area
        contexts[context] = {k: row[k] for k in ("frame", "timestamp", "azimuth", "elevation", "scale")}
        dimensions.add((width, height))
    if len(dimensions) != 1:
        raise ValueError("Density profile requires a common resolution")
    width, height = next(iter(dimensions))
    settings = source["metadata"]["renderer_settings"]
    if (width, height) != (settings.get("width"), settings.get("height")):
        raise ValueError("Density/source resolution mismatch")
    return {"path": relative_path(path), "bytes": len(raw), "sha256": digest(raw),
            "schema_version": CAPABILITY_SCHEMA, "metric": "q0-normalized-distortion-density",
            "profile_schema": DENSITY_SCHEMA, "normalization_version": meta["version"],
            "reference": "decoded_Q3", "normalization": "decoded_Q0_solo_alpha_area",
            "visibility_representation": "decoded_Q0", "pixel_support": "full_image",
            "resolution": [width, height], "area_epsilon_fraction": epsilon,
            "alpha_area_epsilon": epsilon * width * height,
            "source_profile_sha256": source_reference["sha256"],
            "source_implementation_sha256": meta["source_implementation_sha256"],
            "enrichment_implementation_sha256": meta.get("enrichment_implementation_sha256"),
            "producer_parity": "not_verified_by_metadata_export",
            "coverage": {"rows": len(document["rows"]), "contexts": len(contexts),
                         "source_contexts": source_reference["coverage"]["contexts"],
                         "partial": set(source_rows) != {_row_key(r) for r in document["rows"]},
                         "lookup": "exact_sample_only", "samples": list(contexts.values())}}
