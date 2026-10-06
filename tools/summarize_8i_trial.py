#!/usr/bin/env python3
"""Prepare a local web catalog from completed 8i trial variants, using CPU only.

This helper lives outside content_preparation so adding catalog functionality
does not change the preparation implementation fingerprint. It never downloads,
trains, encodes, decodes, renders or recomputes quality measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.run_report import write_prepared_report
from tools.content_preparation.validation import safe_relative_path

CATALOG_VERSION = "content-preparation.8i-trial-catalog.v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe catalog identifier: {value!r}")
    return value


def _safe(root, relative):
    safe_relative_path(relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Catalog path escapes trial root: {relative}")
    return path


def _relative(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Managed catalog artifact is outside trial root: {path}")
    return path.relative_to(root).as_posix()


def _asset(root, path):
    return {"path": _relative(root, path), "bytes": Path(path).stat().st_size, "sha256": _hash(path)}


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf8")
    os.replace(temporary, path)


def _raw_summary(inventory):
    if inventory.get("schema") != "official-8i-download-v1":
        raise ValueError("Expected official-8i-download-v1 raw inventory")
    object_id = _identifier(inventory["object"])
    fps = inventory.get("capture", {}).get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Raw inventory must declare a finite positive capture FPS")
    archive = inventory["archive"]
    archive_frames = archive["frame_ids"]
    if not archive_frames or archive_frames != sorted(set(archive_frames)) or any(isinstance(f, bool) or not isinstance(f, int) for f in archive_frames):
        raise ValueError("Raw archive frame IDs must be nonempty, sorted unique integers")
    if archive.get("frame_count") != len(archive_frames):
        raise ValueError("Raw archive frame count disagrees with its frame IDs")
    frames = inventory.get("frames", [])
    recorded = [f["frame"] for f in frames]
    if len(set(recorded)) != len(recorded) or set(recorded) - set(archive_frames):
        raise ValueError("Raw inventory contains duplicate or foreign frame IDs")
    duration = len(archive_frames) / fps
    span = (archive_frames[-1] - archive_frames[0]) / fps
    capture = inventory.get("capture", {})
    for key, expected in (("samples_duration_seconds", duration), ("timestamp_span_seconds", span)):
        if key in capture and not math.isclose(capture[key], expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Raw capture {key} disagrees with FPS/frame mapping")
    return {"object_id": object_id, "source_url": inventory.get("source_url"),
            "preferred_source_url": inventory.get("preferred_source_url"), "capture_fps": float(fps),
            "fps_basis": capture.get("fps_basis"), "source_archive_frame_count": len(archive_frames),
            "source_archive_frame_range": [archive_frames[0], archive_frames[-1]],
            "source_raw_samples_duration_seconds": duration, "source_raw_timestamp_span_seconds": span,
            "source_raw_media_window_seconds": span + 1 / fps,
            "raw_duration_definition": "sample-count duration = archive samples/FPS; timestamp span and full media window are separate",
            "inventoried_frame_count": len(recorded), "inventoried_frame_ids": sorted(recorded),
            "inventoried_sample_count_duration_seconds": len(recorded) / fps,
            "selection": inventory.get("selection"),
            "recorded_complete_for_selection": inventory.get("complete_for_selection"),
            "archive": {k: archive.get(k) for k in ("bytes", "etag", "last_modified")},
            "transport_provenance": inventory.get("transport_provenance"), "license": inventory.get("license"),
            "raw_integrity_scope": "SHA256/CRC values are retained from the downloader inventory; this catalog does not rehash all raw PLY files"}


def _optional_path(trial, base, relative):
    if not relative:
        return None
    path = _safe(base, relative)
    return _relative(trial, path) if path.is_file() else None


def _renderer_settings(variant_root, obj_id, profile):
    config = _read(variant_root / ".preparation" / "resolved_config.json") if (variant_root / ".preparation" / "resolved_config.json").is_file() else {}
    settings = (profile or {}).get("metadata", {}).get("renderer_settings") or config.get("renderer", {})
    return {key: settings.get(key) for key in ("backend", "width", "height", "fov_degrees", "up_axis", "center", "background")}


def _argv_option(argv, names):
    for index, argument in enumerate(argv):
        if argument in names and index+1 < len(argv):
            return argv[index+1]
        for name in names:
            if argument.startswith(name+"="):
                return argument.split("=", 1)[1]
    return None


def _training_image(command):
    """Read source-image dimensions only; never load a training/render model."""
    from PIL import Image
    argv = command.get("argv", [])
    source = _argv_option(argv, ("-s", "--source_path"))
    if not source:
        return None
    source = Path(source).expanduser()
    source = source.resolve() if source.is_absolute() else (ROOT/source).resolve()
    transforms = source/"transforms_train.json"
    if not transforms.is_file():
        return None
    frames = _read(transforms).get("frames", [])
    if not frames or not frames[0].get("file_path"):
        return None
    relative = Path(frames[0]["file_path"])
    image_path = (source/relative).resolve()
    if not image_path.is_relative_to(source):
        raise ValueError("Training-image provenance escapes its source directory")
    candidates = [image_path] if image_path.suffix else [image_path.with_suffix(extension) for extension in (".png", ".jpg", ".jpeg")]
    image_path = next((path for path in candidates if path.is_file()), None)
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        original = list(image.size)
    # Match the existing utils.camera_utils.loadCam size arithmetic without
    # importing torch/Camera or changing any upstream behavior.
    resolution = float(_argv_option(argv, ("-r", "--resolution")) or -1)
    if not math.isfinite(resolution) or resolution == 0 or resolution < -1:
        raise ValueError("Invalid recorded training camera resolution")
    if resolution in range(1, 9):
        actual = [round(value/resolution) for value in original]
    else:
        scale = max(1., original[0]/1600.) if resolution == -1 else original[0]/resolution
        actual = [int(value/scale) for value in original]
    return {"frame": command.get("frame"), "source_image_size": original,
            "training_image_size": actual, "camera_resolution_argument": resolution,
            "source_image_sha256": _hash(image_path),
            "size_provenance": "source PNG/JPEG header plus recorded upstream camera-resolution argument"}


def _quality_metadata(variant_root, object_id, object_root, qualities):
    config_path = variant_root/".preparation"/"resolved_config.json"
    config = _read(config_path) if config_path.is_file() else {}
    obj = next((item for item in config.get("objects", []) if item.get("id") == object_id), {})
    configured = {item["id"]: item for item in obj.get("qualities", [])}
    training_path = object_root/"training.json"
    commands = (_read(training_path) if training_path.is_file() else {}).get("commands", [])
    result = {}
    for quality in qualities:
        scale = configured.get(quality, {}).get("resolution_scale")
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, int) or scale < 1):
            raise ValueError(f"Invalid quality resolution_scale: {quality}")
        quality_commands = sorted((row for row in commands if row.get("level", row.get("quality")) == quality), key=lambda row: row.get("frame", 0))
        iteration_rows = []
        for command in quality_commands:
            iterations = _argv_option(command.get("argv", []), ("--iterations",))
            if iterations is not None:
                iteration_rows.append({"frame": command.get("frame"), "iterations": int(iterations)})
        source = None
        for command in quality_commands:
            source = _training_image(command)
            if source is not None:
                break
        size = source["training_image_size"] if source else None
        label = quality + (f" · res{scale}" if scale is not None else "")
        if size:
            label += f" · {size[0]}×{size[1]}"
        result[quality] = {"quality": quality, "resolution_scale": scale,
                           "resolution_label": f"res{scale}" if scale is not None else None,
                           "training_image_size": size, "training_image_provenance": source,
                           "training_iterations": {"samples": iteration_rows,
                               "first_frame_iterations": iteration_rows[0]["iterations"] if iteration_rows else None,
                               "later_frame_iteration_values": sorted({row["iterations"] for row in iteration_rows[1:]}),
                               "scope": "recorded training argv; no optimal-convergence claim"},
                           "label": label,
                           "resolution_scale_provenance": "resolved preparation configuration" if scale is not None else "unavailable; no inference from quality ID"}
    return result


def _preview_max_size(value):
    if str(value).lower() == "native":
        return None
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Preview size must be native or a positive pixel limit") from error
    if result < 1:
        raise argparse.ArgumentTypeError("Preview pixel limit must be positive")
    return result


def _thumbnail(trial_root, variant_id, object_id, row, object_root, preview_max_size=None):
    import numpy as np
    from PIL import Image
    source = _safe(object_root, row["image_path"])
    if not source.is_file():
        return None
    mode, quality = _identifier(row["mode"]), _identifier(row["quality"])
    sample_id = _identifier(str(row.get("sample_id", "sample_" + hashlib.sha256(row["image_path"].encode()).hexdigest()[:12])))
    target = _safe(trial_root, f"previews/{variant_id}/{object_id}/{mode}/{quality}/{sample_id}.png")
    with np.load(source, allow_pickle=False) as cache:
        rgb = cache["rgb"]
    if rgb.ndim != 3:
        raise ValueError(f"RGB cache must have three dimensions: {source}")
    if rgb.shape[-1] != 3 and rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError(f"Invalid RGB channels or nonfinite cache values: {source}")
    original_size = [int(rgb.shape[1]), int(rgb.shape[0])]
    if rgb.dtype == np.uint8:
        pixels = rgb
    else:
        if rgb.min() < -1e-6 or rgb.max() > 1 + 1e-6:
            raise ValueError(f"Floating RGB cache must lie in [0,1]: {source}")
        pixels = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    if preview_max_size is not None:
        image.thumbnail((preview_max_size, preview_max_size), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, target)
    return {**_asset(trial_root, target), "kind": "decoded_render_rgb_preview", "mode": mode, "quality": quality,
            "frame": row.get("frame"), "timestamp": row.get("timestamp"),
            "azimuth": row.get("azimuth"), "elevation": row.get("elevation"), "scale": row.get("scale"),
            "sample_id": sample_id, "view_bin": row.get("view_bin"), "scale_bin": row.get("scale_bin"),
            "time_bin": row.get("time_bin"), "distance": row.get("distance"),
            "metrics": row.get("metrics"), "reference_quality": row.get("reference_quality"),
            "source_render_cache": _asset(trial_root, source),
            "original_size": original_size, "preview_size": list(image.size),
            "preview_size_mode": "native" if preview_max_size is None else "bounded",
            "preview_max_size": preview_max_size, "was_downsampled": list(image.size) != original_size,
            "display_conversion": "uint8 PNG RGB; " + ("native cached-render dimensions" if preview_max_size is None else f"at most {preview_max_size}x{preview_max_size}; never upscaled") + "; metrics remain measured from original floating render caches"}


def _make_thumbnails(trial, variant_id, object_id, object_root, profile, limit, preview_max_size=None):
    rows = (profile or {}).get("rows", [])
    grouped = {}
    for row in rows:
        if row.get("image_path"):
            grouped.setdefault((row["mode"], row["quality"]), []).append(row)
    thumbnails = []
    for _, samples in sorted(grouped.items()):
        samples.sort(key=lambda row: (row.get("timestamp", 0), row.get("azimuth", 0), row.get("elevation", 0), row.get("scale", 1), row["image_path"]))
        n = len(samples) if limit is None else min(limit, len(samples))
        positions = [0] if n == 1 else [round(i * (len(samples) - 1) / (n - 1)) for i in range(n)]
        for position in positions:
            result = _thumbnail(trial, variant_id, object_id, samples[position], object_root, preview_max_size)
            if result is not None:
                thumbnails.append(result)
    return thumbnails


def _markdown(catalog):
    raw = catalog["raw_source"]
    lines = [f"# 8i preparation trial: {raw['object_id']}", "",
             f"Official source archive: {raw['source_archive_frame_count']} samples at {raw['capture_fps']:g} FPS; sample-count duration {raw['source_raw_samples_duration_seconds']:.9g} s; timestamp span {raw['source_raw_timestamp_span_seconds']:.9g} s.", "",
             f"Inventoried raw frames: {raw['inventoried_frame_count']}; complete for requested selection: {raw['recorded_complete_for_selection']}.", "",
             f"Source: {raw['source_url']}. [Inventory snapshot]({raw['inventory_snapshot']['path']}).", "",
             "| Variant | Object | Encoded frames | Media duration (s) | Timestamp span (s) | Retained encode-task sum (s) | Latest encode-stage (s) | Render size |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for variant in catalog["variants"]:
        for obj in variant["objects"]:
            time = obj["timeline"]
            timing = obj["encode_timings"]
            resolution = obj["renderer_settings"]
            lines.append(f"| {variant['id']} | {obj['id']} | {time['frame_count']} | {time['media_duration_seconds']:.9g} | {time['time_span_seconds']:.9g} | {timing['recorded_successful_encode_task_wall_seconds']} | {timing['latest_recorded_encode_stage_wall_seconds']} | {resolution['width']}×{resolution['height']} |")
    lines += ["", "| Variant | Mode | Quality / training image resolution | Layer-only complete bytes | Cumulative delivered-quality bytes | Codec/version |", "|---|---|---|---:|---:|---|"]
    for variant in catalog["variants"]:
        for obj in variant["objects"]:
            for rep in obj["representations"]:
                lines.append(f"| {variant['id']} | {rep['mode']} | {rep.get('quality_metadata', {}).get('label', rep['quality'])} | {rep['stored_layer_segment_bytes']} | {rep['cumulative_quality_segment_bytes']} | {rep['codec']['name']} / {rep['codec']['version']} |")
    for variant in catalog["variants"]:
        lines += ["", f"## {variant['id']}", "", f"[Detailed report]({variant['run_report_markdown_path']}) · [JSON report]({variant['run_report_path']})", ""]
        for obj in variant["objects"]:
            for label, key in (("Manifest", "manifest_path"), ("Quality profile", "quality_profile_path"), ("Proxy", "proxy_path")):
                if obj[key]:
                    lines.append(f"- [{label}]({obj[key]})")
            lines.append("")
            for preview in obj["thumbnails"]:
                caption = f"{preview['mode']} {preview['quality']}, frame {preview['frame']}, azimuth {preview['azimuth']}, scale {preview['scale']}"
                lines += [f"![{caption}]({preview['path']})", ""]
    lines += ["## Scope", ""] + [f"- {value}" for value in catalog["limitations"]]
    return "\n".join(lines) + "\n"


def summarize_trial(trial_root, inventory_path, variants=("f32_lossless", "f16"), thumbnails=True,
                    max_thumbnails_per_quality=None, preview_max_size=None):
    """Write per-variant run reports, catalog.json, trial_report.md and CPU PNGs.

    Variant directories can be directly under trial_root or under its variants/
    directory. All web artifact paths are rooted at the catalog directory.
    """
    trial = Path(trial_root).resolve()
    source = Path(inventory_path).resolve()
    if not trial.is_dir():
        raise FileNotFoundError(f"Trial output directory does not exist: {trial}")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Require unique nonempty variant IDs")
    if max_thumbnails_per_quality is not None and (isinstance(max_thumbnails_per_quality, bool) or not isinstance(max_thumbnails_per_quality, int) or max_thumbnails_per_quality < 1):
        raise ValueError("max_thumbnails_per_quality must be None (all samples) or a positive integer")
    if preview_max_size is not None and (isinstance(preview_max_size, bool) or not isinstance(preview_max_size, int) or preview_max_size < 1):
        raise ValueError("preview_max_size must be None (native) or a positive pixel limit")
    # The downloader may atomically refresh this file during a long transfer.
    # Parse and hash the same immutable byte snapshot rather than reading twice.
    source_bytes = source.read_bytes()
    inventory = json.loads(source_bytes.decode("utf8"))
    raw = _raw_summary(inventory)
    # Check all variant locations before writing any report or preview assets.
    locations = []
    for variant_id in variants:
        _identifier(variant_id)
        direct, nested = _safe(trial, variant_id), _safe(trial, f"variants/{variant_id}")
        if direct.is_dir() and nested.is_dir():
            raise ValueError(f"Ambiguous variant directory: {variant_id}")
        location = direct if direct.is_dir() else nested
        if not location.is_dir():
            raise FileNotFoundError(f"Missing variant output: {variant_id}")
        locations.append((variant_id, location))
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    snapshot = _safe(trial, "provenance/raw_inventory.json")
    _atomic_text(snapshot, json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + "\n")
    raw.update({"inventory_snapshot": _asset(trial, snapshot), "original_inventory_sha256": source_hash,
                "source_inventory_locator": str(source),
                "snapshot_path_semantics": "metadata snapshot; original raw/license member paths remain relative to the original inventory directory"})
    result_variants = []
    inventoried = set(raw["inventoried_frame_ids"])
    for variant_id, variant_root in locations:
        report = write_prepared_report(variant_root, fps=raw["capture_fps"], variant_id=variant_id, raw_provenance=inventory)
        objects = []
        for obj in report["objects"]:
            if obj["id"] != raw["object_id"]:
                raise ValueError(f"Variant {variant_id} object does not match raw inventory")
            if set(row["frame"] for row in obj["timeline"]["samples"]) - inventoried:
                raise ValueError(f"Variant {variant_id} encodes frames absent from raw inventory")
            manifest_record = obj["artifacts"].get("manifest.json")
            object_root = _safe(variant_root, manifest_record["path"]).parent if manifest_record else variant_root / obj["id"]
            manifest = _read(_safe(variant_root, manifest_record["path"])) if manifest_record else {}
            metadata = manifest.get("object", {})
            profile_relative = metadata.get("quality_profile_path", "profiles/profile.json")
            profile_path = _safe(object_root, profile_relative) if profile_relative else None
            profile = _read(profile_path) if profile_path and profile_path.is_file() else None
            previews = _make_thumbnails(trial, variant_id, obj["id"], object_root, profile, max_thumbnails_per_quality, preview_max_size) if thumbnails else []
            quality_metadata = _quality_metadata(variant_root, obj["id"], object_root, sorted({row["quality"] for row in obj["representations"]}))
            representations = []
            for rep in obj["representations"]:
                row = copy.deepcopy(rep)
                for segment in row["segments"]:
                    segment["path"] = _relative(trial, _safe(variant_root, segment["path"]))
                for access in row["cold_access"]:
                    access["segment_paths"] = [_relative(trial, _safe(variant_root, path)) for path in access["segment_paths"]]
                for count in row["gaussian_counts"]:
                    if count.get("state_asset"):
                        count["state_asset"]["path"] = _relative(trial, _safe(variant_root, count["state_asset"]["path"]))
                row["thumbnails"] = [p for p in previews if p["mode"] == row["mode"] and p["quality"] == row["quality"]]
                row["quality_metadata"] = quality_metadata[row["quality"]]
                representations.append(row)
            objects.append({"id": obj["id"], "timeline": obj["timeline"], "encode_timings": obj["encode_timings"],
                "renderer_settings": _renderer_settings(variant_root, obj["id"], profile),
                "quality_metadata": quality_metadata,
                "manifest_path": _relative(trial, _safe(variant_root, manifest_record["path"])) if manifest_record else None,
                "quality_profile_path": _optional_path(trial, object_root, profile_relative),
                "proxy_path": _optional_path(trial, object_root, metadata.get("proxy_path", "proxy/index.json")),
                "quality_gain_path": _optional_path(trial, object_root, "profiles/gains.json"),
                "quality_profile_reference": obj["quality_profile"], "representations": representations,
                "sampled_frame_ids": sorted({row["frame"] for row in (profile or {}).get("rows", []) if row.get("frame") is not None}),
                "thumbnails": previews,
                "actual_all_modes_segment_bytes": obj["actual_all_modes_segment_bytes"]})
        result_variants.append({"id": variant_id, "root_path": _relative(trial, variant_root),
            "server_manifest_path": _optional_path(trial, variant_root, "manifest.json"),
            "run_report_path": _relative(trial, variant_root / "run_report.json"),
            "run_report_markdown_path": _relative(trial, variant_root / "run_report.md"), "objects": objects})
    catalog = {"schema_version": CATALOG_VERSION, "path_base": "directory containing catalog.json", "viewer_path": "index.html",
        "raw_source": raw, "variants": result_variants,
        "catalog_implementation_sha256": _hash(Path(__file__)), "preview_generation": "CPU conversion of existing NPZ RGB caches; no renderer invoked",
        "preview_settings": {"size_mode": "native" if preview_max_size is None else "bounded",
                             "max_size": preview_max_size, "upscale": False},
        "limitations": ["Raw archive duration and the encoded checkpoint clip duration describe different frame ranges.",
            "Encode-task sums are retained successful task wall durations, including verification decode/cache writes; playback duration is unrelated.",
            "Latest encode-stage wall duration may describe a resumed invocation that only checked existing artifacts.",
            "Whole-segment bytes include codec/container overhead; progressive cumulative quality costs include parent layers once.",
            "Each variant profile uses its own highest decoded reference; these summaries do not establish cross-variant rate-distortion or perceptual equivalence.",
            "PNG previews are display assets; they are not inputs to objective quality measurements.",
            "This local catalog prepares artifacts for a future web preview; no site is published and no online adaptation is implemented."]}
    _atomic_text(_safe(trial, "catalog.json"), json.dumps(catalog, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _atomic_text(_safe(trial, "trial_report.md"), _markdown(catalog))
    viewer = ROOT / "tools" / "content_trial_viewer" / "index.html"
    if viewer.is_file():
        _atomic_text(_safe(trial, "index.html"), viewer.read_text(encoding="utf8"))
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, default=ROOT / "output/content_prepare_longdress_trial")
    parser.add_argument("--inventory", type=Path, default=ROOT / "output/datasets/8i/longdress/inventory.json")
    parser.add_argument("--variants", nargs="+", default=["f32_lossless", "f16"])
    parser.add_argument("--no-thumbnails", action="store_true")
    parser.add_argument("--max-thumbnails-per-quality", type=int, help="Optional limit per mode/quality; default exports all sampled views")
    parser.add_argument("--preview-max-size", type=_preview_max_size, default=None, metavar="native|PIXELS",
                        help="PNG dimensions: native cached-render size by default, or a maximum edge length; never upscales or rerenders")
    args = parser.parse_args(argv)
    try:
        catalog = summarize_trial(args.trial_root, args.inventory, args.variants, not args.no_thumbnails, args.max_thumbnails_per_quality, args.preview_max_size)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"catalog": str(args.trial_root / "catalog.json"), "variants": [v["id"] for v in catalog["variants"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
