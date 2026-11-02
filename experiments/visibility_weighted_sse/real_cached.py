"""Bounded conversion of one previously validated, frozen research cache.

Only recorded_source_control is imported. Inputs are geometry, independent
CRF40 alpha contribution scalars, offline Solo SSE metadata and encoded-file
manifests. Joint evaluation scores and RGB payloads are never opened.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import digest, file_digest, read_json, require, write_json
from .trace import COORDINATE_CONVENTION, TraceCameraProvider, source_local_camera, camera_from_dict, camera_to_dict
from .visibility import CachedVisibilityAdapter


def _bounded_json(path, limit=4 * 1024 * 1024):
    require(path.is_file() and path.stat().st_size <= limit, f"Missing or oversized source metadata: {path}")
    return read_json(path)


def convert(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    freeze = _bounded_json(source / "freeze.json")
    require(digest({k: v for k, v in freeze.items() if k != "freeze_sha256"}) == freeze.get("freeze_sha256"),
            "Historical freeze hash mismatch")
    dependencies = {str(source / "freeze.json"): file_digest(source / "freeze.json")}

    def prepared(name):
        path = source / "prepared" / name
        matches = [sha for key, sha in freeze["prepared_files_sha256"].items() if key.endswith("/prepared/" + name)]
        require(len(matches) == 1 and file_digest(path) == matches[0], f"Historical prepared input hash mismatch: {name}")
        dependencies[str(path)] = matches[0]
        return _bounded_json(path)

    sse, denominator, evidence = prepared("offline_sse.json"), prepared("offline_contribution.json"), prepared("offline_provenance.json")
    traces = prepared("traces.json")
    matches = [t for t in traces if t["trace_id"] == "recorded_source_control"]
    require(len(matches) == 1, "Exactly one frozen recorded_source_control trace required")
    old_trace = matches[0]
    require(digest({k: v for k, v in old_trace.items() if k != "trace_sha256"}) == old_trace["trace_sha256"],
            "Historical trace hash mismatch")
    require(old_trace["prediction_assumption"] == "ORACLE_CAMERA_TRACE" and old_trace["predictor_enabled"] is False,
            "Historical trace must declare the oracle assumption")
    require(old_trace["actual_camera_trace"] == old_trace["predicted_camera_trace"], "Historical prediction differs from actual geometry")
    require(digest({k: v for k, v in sse.items() if k != "profile_sha256"}) == sse.get("profile_sha256"),
            "Historical offline Solo SSE hash mismatch")
    require(sse.get("joint_target_or_candidate_used") is False and sse.get("provenance") == ["OFFLINE_AVERAGE_SOLO_SSE"],
            "Historical SSE must be independent offline Solo SSE")
    require(sse["distribution_identity"] == digest(sse["distribution"]) == denominator["distribution_identity"]
            == evidence["distribution_identity"] == freeze["offline_distribution_identity"],
            "Historical offline camera/time distribution mismatch")
    require(sse["source_identity"] == old_trace["source_freeze_sha256"] == freeze["source_freeze_sha256"],
            "Historical offline SSE and trace content source mismatch")
    require(denominator["quality"] == 40 and denominator["provenance"] == ["VISIBILITY_PROXY"]
            and denominator["sample_identities"] == evidence["sample_identities"]
            and evidence["rgb_read_for_contribution"] is False
            and evidence["reference_mask_used_for_contribution"] is False,
            "Historical denominator requires independent validated Q0 alpha provenance")
    require(sse["distribution"]["resolution"] == denominator["resolution"], "Historical offline viewport mismatch")
    require(sse["aggregation"] == "equal_camera_mean_of_four_timestamp_summed_solo_SSE"
            and sse["distribution"]["temporal_weights"] == [.25]*4
            and denominator["temporal_weights"] == [1/32]*32
            and len(denominator["sample_identities"]) == 32,
            "Unsupported historical aggregation; this converter is intentionally bounded")
    require(denominator["convention"] == "CRF40_continuous_alpha_weighted_pixels_with_actual_frustum_clipping",
            "Unsupported historical alpha semantics")
    require(set(denominator["values"]) == set(sse["object_ids"]) and sse["crfs"] == [20,28,36,40],
            "Historical object/representation coverage mismatch")

    content_revision = digest(freeze["content_identity"])
    compatibility = {"resolution": denominator["resolution"], "coordinate_convention": COORDINATE_CONVENTION,
        "contribution_convention": "continuous_alpha_weighted_pixels",
        "time_aggregation": "SSE_camera_mean_timestamp_sum_alpha_camera_time_mean",
        "distribution_id": sse["distribution_identity"], "proxy_representation": "standalone_CRF40",
        "transform_convention": "positive_similarity_object_to_world",
        "alpha_backend": "streamstgs_cuda_contribution_cached"}
    samples = []
    for old in old_trace["samples"]:
        instances = []
        for instance in old["instances"]:
            row = dict(instance)
            row["source_camera_id"] = f'{row["source_camera"]["split"]}:{row["source_camera"]["id"]}'
            row["source_state_id"] = digest({"content_revision": content_revision, "object_id": row["object_id"],
                                           "source_timeline": old["source_timeline"][row["object_id"]]})
            instances.append(row)
        samples.append({"timestamp": old["timestamp_s"], "camera": old["camera"], "instances": instances})
    trace = {"schema": "visibility-weighted-sse.trace.v1", "trace_id": old_trace["trace_id"],
        "coordinate_convention": COORDINATE_CONVENTION, "prediction_assumption": "ORACLE_CAMERA_TRACE",
        "predictor_enabled": False, "gops": [{"gop_id": "0", "start_s": old_trace["gop_start_s"],
            "end_s": old_trace["gop_start_s"]+old_trace["gop_duration_s"], "samples": samples}],
        "source_trace_sha256": old_trace["trace_sha256"]}
    provider = TraceCameraProvider(trace)
    require(list(provider.temporal_weights("0")) == old_trace["temporal_weights"], "Historical GOP integration mismatch")

    paths = sorted((source / "artifacts" / "proxy").glob("*.json"))
    require(0 < len(paths) <= 512, "Missing or excessive historical proxy metadata")
    by_sample = {}
    for path in paths:
        artifact = _bounded_json(path)
        snap = artifact["snapshot_identity"]
        if (snap.get("gop") == old_trace["gop"] and snap.get("instances") == old_trace["instances"]
                and any(snap.get("timestamp_s") == row["timestamp_s"] and snap.get("camera") == row["camera"] for row in old_trace["samples"])):
            require(artifact.get("artifact_kind") == "proxy" and artifact.get("information_role") == "VISIBILITY_PROXY"
                    and digest(snap) == path.stem, "Historical proxy identity/provenance mismatch")
            require(snap["content_identity"] == freeze["content_identity"], "Historical proxy content revision mismatch")
            by_sample.setdefault(snap["timestamp_s"], []).append((path, artifact))
    visibility_samples, package_roots, acquisition_ms, analysis_ms = [], {}, 0., 0.
    for pose, old in zip(provider.get_trajectory("0"), old_trace["samples"]):
        found = by_sample.get(pose.timestamp, [])
        require(len(found) == 1, f"Missing or ambiguous compatible historical visibility: {pose.timestamp}")
        path, artifact = found[0]
        row = artifact["visibility_sample"]
        metadata = row["metadata"]
        require(row["timestamp"] == pose.timestamp and row["quality"] == 40
                and row["resolution"] == compatibility["resolution"]
                and row["proxy_identity"] == path.stem
                and metadata.get("candidate_independent") is True
                and metadata.get("candidate_joint_RGB_or_SSE_used") is False
                and metadata.get("contribution_convention") == denominator["convention"]
                and set(row["provenance"]) == {"ORACLE_CAMERA_TRACE", "VISIBILITY_PROXY"},
                "Historical visibility semantics mismatch")
        payload = path.with_suffix(".npz")
        require(payload.stat().st_size <= 1024 * 1024 and file_digest(payload) == artifact["arrays_sha256"],
                "Historical contribution-only payload hash mismatch")
        with np.load(payload, allow_pickle=False) as archive:
            require(set(archive.files) == {"A_solo", "A_vis", "instance_ids"}, "Contribution payload must contain only scalar alpha data")
            ids = archive["instance_ids"].tolist()
            require(len(ids) == len(set(ids)) == len(row["per_object"]), "Historical proxy object coverage mismatch")
            for index, instance_id in enumerate(ids):
                require(float(archive["A_solo"][index]) == row["per_object"][instance_id]["A_solo"]
                        and float(archive["A_vis"][index]) == row["per_object"][instance_id]["A_vis"],
                        "Historical contribution payload and scalar metadata differ")
        for instance in old["instances"]:
            iid = instance["instance_id"]
            activation = artifact["native_activation_metadata"]["objects"][iid]
            local = camera_to_dict(source_local_camera(camera_from_dict(old["camera"]), instance["object_to_world"]))
            require(activation["object_to_world"] == instance["object_to_world"]
                    and activation["frame_index"] == old["source_timeline"][instance["object_id"]]["frame"]
                    and np.allclose(local["world_to_camera"], activation["source_local_camera"]["world_to_camera"], atol=1e-12, rtol=0),
                    "Historical source-local transform/content timeline mismatch")
            package_roots[instance["object_id"]] = Path(activation["encoded_package"]).resolve().parent
        dependencies[str(path)] = file_digest(path)
        dependencies[str(payload)] = artifact["arrays_sha256"]
        acquisition_ms += metadata["proxy_acquisition_time_ms"]
        analysis_ms += metadata["analysis_time_ms"]
        visibility_samples.append({"timestamp": pose.timestamp, "pose_identity": pose.identity,
            "per_object": {key: {"A_solo": value["A_solo"], "A_visible": value["A_vis"]}
                           for key, value in row["per_object"].items()}})

    resources, objects = {}, []
    for obj_index, obj in enumerate(sse["object_ids"]):
        qualities = []
        for crf in reversed(sse["crfs"]):
            index = sse["crfs"].index(crf)
            representation = f"{obj}/0/{crf}"
            frozen_rep = freeze["content_identity"]["representations"][representation]
            package = package_roots[obj] / f"crf_{crf}"
            manifest_path = package / "manifest.json"
            require(file_digest(manifest_path) == frozen_rep["manifest_sha256"], "Historical representation manifest hash mismatch")
            manifest = _bounded_json(manifest_path)
            require(manifest["files"] == frozen_rep["components"], "Historical component identity mismatch")
            components = dict(manifest["files"])
            components["manifest.json"] = {"bytes": manifest_path.stat().st_size, "sha256": frozen_rep["manifest_sha256"]}
            require(sum(v["bytes"] for v in components.values()) == sse["costs"][obj_index][index],
                    "Historical full representation cost disagrees with physical files")
            resource_ids = []
            for filename, component in sorted(components.items()):
                physical = package / filename
                require(physical.is_file() and physical.stat().st_size == component["bytes"]
                        and file_digest(physical) == component["sha256"], "Historical encoded component hash/size mismatch")
                key = "resource_" + digest({"representation": representation, "filename": filename})
                resources[key] = {**component, "url": str(physical)}
                dependencies[str(physical)] = component["sha256"]
                resource_ids.append(key)
            qualities.append({"quality_id": f"CRF{crf}", "representation_id": representation,
                "average_solo_sse": sse["distortions"][obj_index][index],
                "resource_ids": resource_ids, "dependencies": []})
        objects.append({"object_id": obj, "instance_id": obj, "gops": [{"gop_id": "0", "source_gop_index": 0,
            "offline_mean_solo_contribution": denominator["values"][obj],
            "offline_sample_ids": denominator["sample_identities"], "offline_sample_weights": denominator["temporal_weights"],
            "denominator_metadata": compatibility, "sse_metadata": compatibility, "offline_sse": qualities}]})

    provenance = {"source": str(source), "source_dependencies_sha256": dependencies,
        "information_role": "OFFLINE_AVERAGE_SOLO_SSE", "joint_quality_used": False,
        "original_sse_aggregation": sse["aggregation"], "source_sse_values_modified": False,
        "cost_semantics": "full_selected_standalone_resource_closure_with_src_final_SHA256_cache_deduplication",
        "historical_package_costs_reproduced_before_cross_object_deduplication": True,
        "new_rendering_or_decoding_calls": 0, "RGB_payloads_opened": 0}
    profile = {"schema": "visibility-weighted-sse.profiles.v1", "input_kind": "validated_cached_artifacts",
        "content_revision": content_revision, "profile_revision": sse["profile_sha256"],
        "representation_family": "standalone", "compatibility": compatibility,
        "provenance": provenance, "resources": resources, "objects": objects}
    visibility = {"schema": "visibility-weighted-sse.visibility.v1", "input_kind": "validated_cached_artifacts",
        "trace_id": trace["trace_id"], "content_revision": content_revision, "compatibility": compatibility,
        "provenance": {"source": str(source / "artifacts" / "proxy"), "information_role": "VISIBILITY_PROXY",
            "candidate_independent": True, "joint_quality_used": False, "source_dependencies_sha256": dependencies,
            "historical_proxy_acquisition_ms": acquisition_ms, "historical_visibility_analysis_ms": analysis_ms,
            "native_acquisition_not_measured_in_this_CPU_run": True},
        "gops": [{"gop_id": "0", "samples": visibility_samples}]}
    CachedVisibilityAdapter(visibility, provider, expected_compatibility=compatibility, content_revision=content_revision).aggregate("0")
    from .profiles import profiles_from_dict
    profiles_from_dict(profile)
    write_json(output / "trace.json", trace)
    write_json(output / "profiles.json", profile)
    write_json(output / "visibility.json", visibility)
    config = {"schema": "visibility-weighted-sse.experiment.v1", "experiment_id": "cached_recorded_source_control",
        "input_kind": "validated_cached_artifacts", "prediction_assumption": "ORACLE_CAMERA_TRACE",
        "inputs": {key: {"path": f"{key}.json"} for key in ("trace", "profiles", "visibility")},
        "solver_limits": {"max_dp_cells": 100_000_000},
        "budgets": [{"budget_id": f"bytes_{value}", "bytes": value} for value in (10_000_000,20_000_000,30_000_000)]}
    write_json(output / "config.source.json", config)
    from .runner import freeze_config
    freeze_config(output / "config.source.json", output / "config.json")
    return {"status": "ready", "config": str(output / "config.json"), "input_kind": "validated_cached_artifacts",
            "source_files_verified": len(dependencies), "new_rendering_calls": 0, "RGB_payloads_opened": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(convert(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
