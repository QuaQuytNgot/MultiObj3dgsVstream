#!/usr/bin/env python3
"""CPU-only audit of a completed imported-checkpoint f32/f16 preparation trial.

Does not train, encode, render, import torch or modify preparation artifacts.
An optional report is written outside the pipeline-owned object directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.assets import ATTRIBUTES, load_state, read_ply, state_hash
from tools.content_preparation.codec_adapter import inspect_payload
from tools.content_preparation.manifest import load_manifest
from tools.content_preparation.packaging import extract_segment_member
from tools.content_preparation.upstream import verify_checkpoint_lineage


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def changed_rows(a, b):
    require(a.shape == b.shape, "Attribute shapes differ in row comparison")
    width = int(np.prod(a.shape[1:]))
    left = np.ascontiguousarray(a).reshape(len(a), width).view(np.uint8)
    right = np.ascontiguousarray(b).reshape(len(b), width).view(np.uint8)
    return int(np.any(left != right, axis=1).sum())


def audit(object_root, source_manifest, precision):
    object_root, source_manifest = Path(object_root).resolve(), Path(source_manifest).resolve()
    require(precision in ("f32", "f16"), "This audit supports f32 and f16 baseline trials")
    manifest = load_manifest(object_root / "manifest.json", validate_files=True)
    source = json.loads(source_manifest.read_text())
    require(verify_checkpoint_lineage(source, manifest["quality_order"]), "Source lineage check failed")
    config = json.loads((object_root.parent / ".preparation/resolved_config.json").read_text())
    training = json.loads((object_root / "training.json").read_text())
    decoding = json.loads((object_root / "decoding.json").read_text())["states"]
    exports = json.loads((object_root / "export.json").read_text())["states"]
    package = manifest["package"]
    require(training["backend"] == "existing" and training["lineage_verified"], "Trial is not a verified checkpoint import")
    require(set(manifest["delivery_modes"]) == {"independent", "progressive"}, "Trial must deliver both modes")
    command_map = {(c["frame"], c["level"]): c for c in source["commands"]}
    source_rows = {row["frame"]: row for row in source["frames"]}
    export_map = {(row["quality"], row["frame"]): row for row in exports}
    decoded_map = {(row["mode"], row["quality"], row["frame"]): row for row in decoding}
    payload_map = {(row["mode"], row["quality"], row["frame"]): row for row in package["payloads"]}
    segments = {member["id"]: segment for segment in package["segments"] for member in segment["members"]}
    require(len(decoding) == len(manifest["object"]["frames"]) * len(manifest["quality_order"]) * 2,
            "Decoded-state count differs from the mode/quality/frame grid")
    state_rows, correction_rows = [], []
    for sample in manifest["object"]["frames"]:
        frame = sample["frame"]
        previous_independent = None
        for qi, quality in enumerate(manifest["quality_order"]):
            original_path = Path(source_rows[frame][quality])
            if not original_path.is_absolute():
                original_path = source_manifest.parent / original_path
            command = command_map[(frame, quality)]
            require(sha256(original_path) == command["sha256"], f"Checkpoint hash mismatch {quality}/{frame}")
            original = read_ply(original_path)
            exported = load_state(object_root / export_map[(quality, frame)]["path"])
            require(state_hash(original) == state_hash(exported), f"Export changed numeric state {quality}/{frame}")
            require(exported.metadata.get("stable_ids") and exported.metadata.get("lineage_verified"), "Export lacks verified IDs")
            independent_record = decoded_map[("independent", quality, frame)]
            progressive_record = decoded_map[("progressive", quality, frame)]
            independent = load_state(object_root / independent_record["path"])
            progressive = load_state(object_root / progressive_record["path"])
            require(state_hash(independent) == independent_record["state_hash"], "Independent decoded index hash mismatch")
            require(state_hash(progressive) == progressive_record["state_hash"], "Progressive decoded index hash mismatch")
            require(state_hash(independent) == state_hash(progressive), f"Delivery modes differ {quality}/{frame}")
            require(np.array_equal(independent.ids, original.ids), "Decoded ID ordering changed")
            errors = {}
            for attribute in ATTRIBUTES:
                expected = original.arrays[attribute] if precision == "f32" else original.arrays[attribute].astype("<f2").astype("<f4")
                require(changed_rows(expected, independent.arrays[attribute]) == 0,
                        f"Decoded {attribute} differs from {precision} target {quality}/{frame}")
                errors[attribute] = float(np.max(np.abs(original.arrays[attribute].astype(np.float64) - independent.arrays[attribute])))
            state_rows.append({"frame": frame, "quality": quality, "gaussian_count": independent.count,
                               "sh_degree": independent.sh_degree, "source_sha256": command["sha256"],
                               "export_numeric_exact": True, "decoded_precision_target_exact": True,
                               "delivery_modes_numeric_equal": True, "maximum_attribute_absolute_error": errors,
                               "decoded_state_sha256": state_hash(independent)})
            for mode in ("independent", "progressive"):
                payload = payload_map[(mode, quality, frame)]
                segment = segments[payload["id"]]
                encoded = extract_segment_member(object_root / segment["path"], payload["id"])
                require(len(encoded) == payload["bytes"] and hashlib.sha256(encoded).hexdigest() == payload["sha256"],
                        "Packaged member bytes/hash mismatch")
                header = inspect_payload(object_root / payload["path"])
                require(header["codec"]["config"]["precision"] == precision, "Unexpected codec precision")
                require(header["decoded_state_hash"] == state_hash(independent), "Payload reconstructed-state hash mismatch")
                if mode == "progressive" and qi:
                    parent = previous_independent
                    require(parent is not None, "Missing parent in quality order")
                    require(np.array_equal(independent.ids[:parent.count], parent.ids), "Imported inherited prefix changed IDs")
                    new_count = independent.count - parent.count
                    shared = {a: changed_rows(parent.arrays[a], independent.arrays[a][:parent.count]) for a in ATTRIBUTES}
                    expected_changes = {a: new_count + shared[a] for a in ATTRIBUTES}
                    require(header["changed_rows"] == expected_changes, "Sparse replacements do not cover all changed attributes")
                    require(header["new_ids"] == new_count and header["deleted_ids"] == 0, "Unexpected trial new/deleted ID accounting")
                    require(header["parent_state_hash"] == state_hash(parent), "Refinement parent is not decoded lower quality")
                    correction_rows.append({"frame": frame, "layer": payload["layer"], "new_ids": new_count,
                                            "deleted_ids": 0, "shared_attribute_changed_rows": shared,
                                            "transmitted_attribute_replacement_rows": expected_changes})
            previous_independent = independent

    profile = json.loads((object_root / manifest["object"]["quality_profile_path"]).read_text())
    rows = profile["rows"]
    sampling = config["sampling"]
    sampled_frames = len(manifest["object"]["frames"]) if sampling["times"] == "all" else len(sampling["times"])
    expected_profiles = sampled_frames * len(sampling["azimuth"]) * len(sampling["elevation"]) * len(sampling["scales"]) * len(manifest["quality_order"]) * 2
    require(len(rows) == expected_profiles, "Quality-profile grid is incomplete")
    require(config["renderer"].get("up_axis") == "z", "Prepared 8i trial must explicitly use Z-up cameras")
    image_rows = []
    reference = manifest["quality_order"][-1]
    for row in rows:
        require(row["reference_state"] == "decoded" and row["reference_quality"] == reference,
                "Adaptation profile reference is not the highest decoded quality")
        require(row["metrics"]["lpips"] is not None and np.isfinite(float(row["metrics"]["lpips"])), "LPIPS is absent/nonfinite")
        if row["quality"] == reference:
            require(row["metrics"]["mse"] == 0 and row["metrics"]["psnr"] == "inf" and abs(row["metrics"]["lpips"]) < 1e-6,
                    "Highest decoded quality has nonzero self-distortion")
        with np.load(object_root / row["image_path"], allow_pickle=False) as image:
            rgb, alpha, depth = image["rgb"], image["alpha"], image["depth"]
            require(np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all(), "Nonfinite rendered cache")
            require(float(rgb.max()) > 0 and float(alpha.max()) > 0, "Rendered view is black or uncovered")
            require(float(rgb.min()) >= 0 and float(rgb.max()) <= 1, "Rendered RGB leaves [0,1]")
            image_rows.append({"mode": row["mode"], "quality": row["quality"], "frame": row["frame"],
                               "sample_id": row["sample_id"], "rgb_mean": float(rgb.mean()),
                               "rgb_maximum": float(rgb.max()), "covered_pixels": int(np.count_nonzero(alpha > .01))})
    proxy = json.loads((object_root / manifest["object"]["proxy_path"]).read_text())
    require(proxy["quality_independent"] and len(proxy["frames"]) == len(manifest["object"]["frames"]), "Proxy grid incomplete")
    for record in proxy["frames"]:
        state = load_state(object_root / "proxy" / record["path"])
        require(state.count == proxy["selected_count"] and state.ids.tolist() == proxy["selected_ids"], "Proxy IDs/count changed")
    validation = json.loads((object_root / "validation.json").read_text())
    require(validation["passed"] and validation["decoded_from_segments"], "Pipeline final validation failed")
    guard = validation["gpu_guard"]
    sampled_views = expected_profiles // (len(manifest["quality_order"]) * 2)
    metric_settings = config["metrics"]
    lpips_settings = metric_settings.get("lpips", False)
    lpips_enabled = lpips_settings.get("enabled", True) if isinstance(lpips_settings, dict) else bool(lpips_settings)
    lpips_device = lpips_settings.get("device", metric_settings.get("device", "cpu")) if isinstance(lpips_settings, dict) else metric_settings.get("device", "cpu")
    metric_operations = expected_profiles if lpips_enabled and lpips_device == "cuda" else 0
    expected_gpu_operations = expected_profiles + sampled_views + metric_operations
    require(guard["active"] == 0 and guard["peak_active"] == 1 and guard["views"] == expected_gpu_operations,
            "GPU execution was not one model/view at a time or render count differs")
    require(validation["lpips_batch_size"] == 1 and validation["upstream_unchanged"], "Memory contract/upstream integrity check failed")
    return {"schema": "8i-imported-trial-audit.v1", "passed": True, "precision": precision,
            "source_manifest_sha256": sha256(source_manifest), "source_lineage_verified": True,
            "object": manifest["object"], "quality_order": manifest["quality_order"],
            "state_count": len(state_rows), "profile_sample_count": len(image_rows),
            "manifest_files_hashes_schema_valid": True, "all_rendered_views_nonblack": True,
            "media_bytes": manifest["accounting"]["media_bytes"], "segment_count": len(package["segments"]),
            "proxy_selected_count": proxy["selected_count"], "gpu_guard": guard,
            "gpu_operation_accounting": {"render_views": expected_profiles + sampled_views,
                                         "lpips_batches": metric_operations},
            "render_peak_cuda_allocated_bytes": validation["render_peak_cuda_allocated_bytes"], "states": state_rows,
            "progressive_corrections": correction_rows, "image_checks": image_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pipeline-owned object directory containing manifest.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "output/progressive_gap_real/manifest.json")
    parser.add_argument("--precision", choices=("f32", "f16"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root, args.source_manifest, args.precision)
        if args.report:
            require(not args.report.resolve().is_relative_to(args.root.resolve()), "Write the audit report outside the pipeline-owned object directory")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("object", "states", "progressive_corrections", "image_checks")}, indent=2))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"trial audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
