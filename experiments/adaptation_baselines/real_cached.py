"""Import one frozen, audited GOP without decoding or rendering.

The proposed method's converter owns trace, contribution, SSE and physical-cost
conversion. This adapter only adds the historical original-PLY PSNR proxy,
pooled compression-relative MSE, and full-GOP CRF20 AABB metadata.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from experiments.nava_mse.costs import load_inputs
from experiments.nava_mse.profiles import AverageProfile, DistortionSample
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require, write_json
from experiments.visibility_weighted_sse import real_cached as proposed_cached


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = Path("experiments/nava_mse_audit/gop0_forensic_v1/original_inputs_sha256.json")
NAVA_FREEZE = Path("experiments/nava_mse/configs/gop0.frozen.json")
METHODS = ("NAVA", "VISIBILITY_PSNR", "VISIBILITY_MSE", "SOLO_SSE", "VISIBILITY_WEIGHTED_SSE")


def _bounded_json(path, limit=4 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit, f"Missing or oversized baseline metadata: {path}")
    return read_json(path)


def _verified_file(project, relative, expected, dependencies):
    path = (project / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(project), "Unsafe historical metadata path")
    require(path.is_file() and path.stat().st_size <= 4 * 1024 * 1024,
            f"Missing or oversized historical metadata: {relative}")
    require(file_digest(path) == expected, f"Historical baseline input hash mismatch: {relative}")
    dependencies[str(path)] = expected
    return path


def _audited_json(project, relative, audit, dependencies):
    relative = str(relative)
    require(relative in audit, f"Historical input is not pinned by the NAVA audit: {relative}")
    record = audit[relative]
    path = _verified_file(project, relative, record["sha256"], dependencies)
    require(path.stat().st_size == record["bytes"], f"Historical audited byte length mismatch: {relative}")
    return _bounded_json(path)


def _historical_baselines(source, project=PROJECT):
    """Revalidate saved scalar metadata against the preserved forensic audit."""
    project = Path(project).resolve()
    audit_path = project / AUDIT
    audit = _bounded_json(audit_path)
    dependencies = {str(audit_path): file_digest(audit_path)}
    frozen = _audited_json(project, NAVA_FREEZE, audit, dependencies)
    require(frozen.get("freeze_sha256") == digest({k: v for k, v in frozen.items() if k != "freeze_sha256"}),
            "Historical NAVA freeze hash mismatch")
    config = frozen["config"]
    require(frozen["config_sha256"] == digest(config), "Historical NAVA configuration hash mismatch")
    require(config["gop"] == 0 and config["crfs"] == [20, 28, 36, 40]
            and config["objects"] == ["longdress", "soldier", "loot"]
            and config["reference_crf"] == 20 and config["baseline_crf"] == 40
            and config["alpha_threshold"] == 0.01 and config["rgb_semantics"] == "clip_float_rgb_0_1"
            and config["background"] == [0, 0, 0] and config["gop_duration_s"] == 2,
            "Unsupported historical baseline protocol; this converter imports audited GOP0 only")
    for relative, expected in frozen["inputs"]["input_sha256"].items():
        _verified_file(project, relative, expected, dependencies)
    # Reuse the original metadata-only loader. It verifies each profile's
    # original-PLY reference, arithmetic dB mean, and representation identity.
    actual = load_inputs(config, project)
    for key in ("costs_bytes", "nava_psnr", "representations", "sources", "cost_definition"):
        require(actual[key] == frozen["inputs"][key], f"Historical NAVA metadata differs from freeze: {key}")

    run = Path(config["output_dir"])
    average = _audited_json(project, run / "profiles/average.json", audit, dependencies)
    samples = _audited_json(project, run / "profiles/samples.json", audit, dependencies)
    geometry = _audited_json(project, run / "geometry.json", audit, dependencies)
    require(samples["freeze_sha256"] == geometry["freeze_sha256"] == frozen["freeze_sha256"],
            "Historical MSE/geometry freeze identity mismatch")
    reconstructed = AverageProfile(DistortionSample.from_dict(row) for row in samples["samples"])
    require(reconstructed.dataset_sha256 == samples["dataset_sha256"] and reconstructed.to_dict() == average,
            "Historical AverageMSE does not reproduce the audited pooled scalar samples")
    require(average["aggregation"] == "pooled_sse_over_foreground_rgb_count",
            "Historical MSE aggregation mismatch")
    for row in samples["samples"]:
        provenance = row["provenance"]
        require(provenance["freeze_sha256"] == frozen["freeze_sha256"]
                and provenance["config_sha256"] == frozen["config_sha256"]
                and provenance["alpha_threshold"] == .01 and provenance["background"] == [0, 0, 0]
                and row["metrics"]["normalization"] == "clip"
                and row["metrics"]["channels"] == 3 and row["metrics"]["data_range"] == 1.,
                "Historical MSE reference/normalization mismatch")

    source_freeze = _bounded_json(Path(source) / "freeze.json")
    require(source_freeze["source_freeze_sha256"] == frozen["freeze_sha256"],
            "Weighted SSE and baseline caches refer to different frozen source inputs")
    for rid, rep in source_freeze["content_identity"]["representations"].items():
        require(rid in frozen["inputs"]["representations"] and
                all(rep[key] == frozen["inputs"]["representations"][rid][key] for key in rep),
                "Weighted SSE/baseline representation identity mismatch")

    # The saved AABB covers all 60 presentation frames, never only the four
    # offline image timestamps. Recheck the authoritative per-frame union.
    bounds_path = (project / config["bounds_source"]).resolve()
    bounds = _bounded_json(bounds_path)
    require(bounds["reference_crf"] == 20
            and bounds["coverage"] == "all presentation frames of all five GOPs per content",
            "Historical bounds definition mismatch")
    for obj in config["objects"]:
        record = bounds["contents"][obj]["gops"]["0"]
        rep = frozen["inputs"]["representations"][f"{obj}/0/20"]
        for field, expected in (("manifest", rep["manifest_sha256"]),
                                ("reconstruction", rep["components"]["reconstruction.json"]["sha256"])):
            path = (bounds_path.parent / record[field]).resolve()
            require(path.is_relative_to(bounds_path.parent) and file_digest(path) == expected,
                    "Historical bounds representation mismatch")
        reconstruction = _bounded_json(bounds_path.parent / record["reconstruction"])
        require([(r["frame"], r["seconds"], r["model_time"]) for r in record["frames"]] ==
                [(r["frame"], r["seconds"], r["model_time"]) for r in reconstruction["segment"]["timestamps"]],
                "Historical bounds lack exact full-GOP frame coverage")
        per_frame = np.asarray([row["bounds"] for row in record["frames"]], dtype=np.float64)
        union = np.stack([per_frame[:, 0].min(0), per_frame[:, 1].max(0)])
        require(union.shape == (2, 3) and np.isfinite(union).all() and np.all(union[0] < union[1])
                and np.array_equal(union, record["bounds"])
                and record["bounds"] == geometry["bounds"][obj],
                "Historical full-GOP bound differs from its per-frame union")
        provenance = geometry["bound_provenance"][obj]
        require(provenance["source_sha256"] == file_digest(bounds_path)
                and provenance["manifest_sha256"] == rep["manifest_sha256"]
                and provenance["reconstruction_sha256"] == rep["components"]["reconstruction.json"]["sha256"],
                "Historical full-GOP bounds provenance mismatch")
    return frozen, average, samples, geometry, dependencies


def convert(source, output):
    """Extend the existing proposed-method conversion with audited baselines."""
    source, output = Path(source).resolve(), Path(output).resolve()
    frozen, average, samples, geometry, dependencies = _historical_baselines(source)
    # Keep every proposed-method artifact and its frozen configuration intact.
    result = proposed_cached.convert(source, output / "weighted_sse")
    profile_path = output / "weighted_sse/profiles.json"
    profile = read_json(profile_path)
    source_sse = _bounded_json(source / "prepared/offline_sse.json")
    distribution = source_sse["distribution"]
    config = frozen["config"]
    require(distribution["source_freeze_sha256"] == frozen["freeze_sha256"]
            and distribution["camera_ids"] == [str(c) for c in config["camera_ids"]]
            and distribution["timestamps_s"] == config["timestamps_s"]
            and distribution["resolution"] == [config["width"], config["height"]]
            and distribution["source_scene_sha256"] == {str(s["camera_id"]): digest(s) for s in frozen["scenes"]},
            "Historical MSE and Solo SSE offline camera/time population mismatch")
    expected_population = {(str(camera), timestamp) for camera in config["camera_ids"] for timestamp in config["timestamps_s"]}
    for obj in config["objects"]:
        for crf in config["crfs"]:
            population = [(row["camera_id"], row["timestamp"]) for row in samples["samples"]
                          if row["object_id"] == obj and row["gop"] == 0 and row["crf"] == crf]
            require(len(population) == len(expected_population) and set(population) == expected_population,
                    "Historical MSE profile has incomplete offline camera/time coverage")
    entries = {(row["object_id"], row["gop"], row["crf"]): row for row in average["entries"]}
    objects = []
    for obj in profile["objects"]:
        oid = obj["object_id"]
        obj_index = config["objects"].index(oid)
        qualities = []
        for quality in obj["gops"][0]["offline_sse"]:
            crf = int(quality["quality_id"].removeprefix("CRF"))
            qindex = config["crfs"].index(crf)
            entry = entries[(oid, 0, crf)]
            psnr, mse = frozen["inputs"]["nava_psnr"][obj_index][qindex], entry["metrics"]["mse"]
            require(entry["available"] is True and entry["sample_count"] == 32
                    and math.isfinite(psnr) and math.isfinite(mse) and mse >= 0,
                    "Missing or invalid historical baseline quality")
            qualities.append({"quality_id": quality["quality_id"], "representation_id": quality["representation_id"],
                              "psnr_db": psnr, "mse": mse})
        objects.append({"instance_id": obj["instance_id"], "object_id": oid,
                        "gops": [{"gop_id": "0", "bounds": geometry["bounds"][oid], "qualities": qualities}]})
    metadata = {"schema": "adaptation-baselines.metadata.v1", "input_kind": "validated_cached_artifacts",
        "content_revision": profile["content_revision"], "profile_binding_sha256": file_digest(profile_path),
        "provenance": {"information_role": "OFFLINE_BASELINE_METADATA", "joint_quality_used": False,
            "source_dependencies_sha256": dependencies, "source_freeze_sha256": frozen["freeze_sha256"],
            "source_average_dataset_sha256": average["dataset_sha256"],
            "new_rendering_or_decoding_calls": 0, "RGB_payloads_opened": 0,
            "bounds_definition": geometry["definition"], "source_baseline_values_modified": False},
        "quality_references": {"psnr": {"reference": "original_ply_synthetic_rgb", "aggregation": "arithmetic_mean_db"},
            "mse": {"reference": "own_CRF20_fixed_alpha_gt_0.01_clipped_RGB",
                "aggregation": "pooled_SSE_over_foreground_RGB_scalars", "alpha_threshold": .01,
                "channels": 3, "data_range": 1., "distribution_id": profile["compatibility"]["distribution_id"]}},
        "objects": objects}
    metadata["profile_revision"] = digest(metadata)
    write_json(output / "baseline_metadata.json", metadata)
    common_config = {"schema": "adaptation-baselines.experiment.v1", "experiment_id": "cached_recorded_source_control",
        "input_kind": "validated_cached_artifacts", "prediction_assumption": "ORACLE_CAMERA_TRACE",
        "inputs": {**{key: {"path": f"weighted_sse/{key}.json"} for key in ("trace", "profiles", "visibility")},
                   "baseline_metadata": {"path": "baseline_metadata.json"}},
        "methods": list(METHODS), "solver_limits": {"max_dp_cells": 100_000_000},
        "budgets": [{"budget_id": f"bytes_{value}", "bytes": value} for value in (10_000_000, 20_000_000, 30_000_000)]}
    write_json(output / "config.source.json", common_config)
    from .runner import freeze_config
    freeze_config(output / "config.source.json", output / "config.json")
    return {"status": "ready", "config": str(output / "config.json"), "input_kind": "validated_cached_artifacts",
            "methods": list(METHODS), "source_files_verified": len(set(dependencies) | set(profile["provenance"]["source_dependencies_sha256"])),
            "new_rendering_calls": result["new_rendering_calls"], "RGB_payloads_opened": result["RGB_payloads_opened"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(convert(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
