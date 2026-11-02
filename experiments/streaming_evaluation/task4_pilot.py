"""One resumable Task-4 long-line pilot using prior exact native inputs.

This is a bounded two-second pipeline check, not a paper experiment. It
reuses exact GOP0 analyzer snapshots and joint CRF20 references, rendering only
candidate allocations that are absent from the verified cache.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.adaptation_baselines.interface import AdaptationPlanner
from experiments.adaptation_baselines.metadata import load_metadata
from experiments.nava_mse.common import PROJECT, digest, read_json, write_json
from experiments.nava_mse.metrics import foreground_metrics
from experiments.streaming_evaluation.network import NetworkTrace
from experiments.streaming_evaluation.inputs import independent_instance_transfer_profiles
from experiments.streaming_evaluation.simulator import simulate
from experiments.streaming_evaluation.timeline import Timeline
from experiments.visibility_guided_sse.backend import (
    NativeExecutor, _validate_request, load_image, load_proxy, native_cuda_smoke,
)
from experiments.visibility_guided_sse.cache import CacheRegistry
from experiments.visibility_guided_sse.config import job_identity
from experiments.visibility_weighted_sse.common import file_digest
from experiments.visibility_weighted_sse.profiles import load_profiles
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter
from experiments.nava_mse.common import read_json as read_task1_json


ROOT = PROJECT
TASK2_ROOT = ROOT / "experiments/visibility_guided_sse/runs/pilot_v1"
TASK2_FREEZE = read_json(TASK2_ROOT / "freeze.json")
TASK2_TRACES = read_json(TASK2_ROOT / "prepared/traces.json")
SOURCE_FREEZE = read_json(ROOT / TASK2_FREEZE["config"]["source_freeze"])
PROFILE_PATH = ROOT / "experiments/adaptation_baselines/configs/cached_recorded_source_control/weighted_sse/profiles.json"
METADATA_PATH = ROOT / "experiments/adaptation_baselines/configs/cached_recorded_source_control/baseline_metadata.json"
TASK2_REGISTRY = CacheRegistry(TASK2_ROOT)
OUT = ROOT / "experiments/streaming_evaluation/runs/task4_pilot"
METHODS = ["NAVA", "VISIBILITY_PSNR", "VISIBILITY_MSE", "SOLO_SSE", "VISIBILITY_WEIGHTED_SSE"]


def make_trace_and_visibility(trace_row, profiles):
    samples = []
    for sample in trace_row["samples"]:
        instances = []
        for raw in sample["instances"]:
            row = dict(raw)
            row["source_camera_id"] = f'{row["source_camera"]["split"]}:{row["source_camera"]["id"]}'
            row["source_state_id"] = digest({
                "content_revision": profiles.content_revision,
                "object_id": row["object_id"],
                "source_timeline": sample["source_timeline"][row["object_id"]],
            })
            instances.append(row)
        samples.append({"timestamp": sample["timestamp_s"], "camera": sample["camera"],
                        "instances": instances})
    trace = {
        "schema": "visibility-weighted-sse.trace.v1",
        "trace_id": trace_row["trace_id"],
        "coordinate_convention": "opencv_positive_z_column_vectors",
        "prediction_assumption": "ORACLE_CAMERA_TRACE",
        "predictor_enabled": False,
        "gops": [{"gop_id": "0", "start_s": trace_row["gop_start_s"],
                  "end_s": trace_row["gop_start_s"] + trace_row["gop_duration_s"],
                  "samples": samples}],
    }
    provider = TraceCameraProvider(trace)
    observations, lookup_ms, analysis_ms = [], 0.0, 0.0
    for source_sample, pose in zip(trace_row["samples"], provider.get_trajectory("0")):
        allocation = {row["instance_id"]: 40 for row in source_sample["instances"]}
        identity = job_identity(TASK2_FREEZE, source_sample, allocation, proxy=True)
        start = time.perf_counter()
        entry = TASK2_REGISTRY.lookup(identity)
        if entry is None:
            raise FileNotFoundError(f"Missing exact native runtime visibility snapshot at {pose.timestamp}s")
        observed = load_proxy(entry)
        lookup_ms += (time.perf_counter() - start) * 1000
        analysis_ms += float(observed.metadata.get("analysis_time_ms", 0.0))
        observations.append({
            "timestamp": pose.timestamp,
            "pose_identity": pose.identity,
            "per_object": {iid: {"A_solo": value.A_solo, "A_visible": value.A_vis}
                           for iid, value in observed.per_object.items()},
        })
    visibility = {
        "schema": "visibility-weighted-sse.visibility.v1",
        "input_kind": "validated_cached_artifacts",
        "trace_id": trace_row["trace_id"],
        "content_revision": profiles.content_revision,
        "compatibility": dict(profiles.compatibility),
        "provenance": {
            "information_role": "VISIBILITY_PROXY",
            "source": "exact-identity native Visibility Analyzer artifacts from Task-2 pilot",
            "candidate_independent": True,
            "joint_quality_used": False,
        },
        "gops": [{"gop_id": "0", "samples": observations}],
    }
    return trace, provider, visibility, {"cache_lookup_ms": lookup_ms,
        "reused_native_analysis_ms": analysis_ms, "sample_count": len(observations)}


def load_rgb_artifact(identity, executor, remaining_seconds):
    cached = TASK2_REGISTRY.lookup(identity)
    if cached is not None:
        return load_image(cached), {"cache_hit": True, "execution_wall_s": 0.0,
                                   "arrays_sha256": cached["arrays_sha256"]}
    job_id = digest(identity)
    wall_started = time.perf_counter()
    result = executor({"identity": identity, "job_id": job_id}, remaining_seconds=remaining_seconds)
    wall_s = time.perf_counter() - wall_started
    with np.load(result["arrays_path"], allow_pickle=False) as archive:
        image = {key: archive[key].copy() for key in archive.files}
    marker = read_json(result["marker_path"])
    return image, {"cache_hit": False, "execution_wall_s": marker["execution_wall_s"],
                   "wall_s_including_executor": wall_s,
                   "peak_gpu_allocated_bytes": marker.get("peak_gpu_allocated_bytes"),
                   "peak_gpu_reserved_bytes": marker.get("peak_gpu_reserved_bytes"),
                   "arrays_sha256": marker["arrays_sha256"], "marker_path": result["marker_path"]}


class PilotQualityProvider:
    def __init__(self, records):
        self.records = records

    def measure(self, context):
        key = (round(context.content_time_s, 9), tuple(sorted(context.selected_quality.items())))
        record = self.records.get(key)
        if record is None:
            return None
        from experiments.streaming_evaluation.quality import FrameQualityMeasurement
        from experiments.nava_mse.metrics import MetricStats
        return FrameQualityMeasurement(
            context=context,
            metrics=MetricStats.from_dict(record["metrics"]),
            reference_identity=record["reference_identity"],
            mask_mode="reference_foreground",
            provenance=record["provenance"],
        )


def selected_trace(offset_s=18.0):
    source = ROOT / "experiments/abr_network_evaluation/inputs/lumos5g/processed/HV.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = [(float(row.get("time_seconds", row.get("time_s", row.get("at_s", row.get("time"))))),
               float(row.get("mbps", row.get("bandwidth_mbps")))) for row in rows]
    duration = 120.0
    initial_rate = next(rate for timestamp, rate in reversed(points) if timestamp <= offset_s)
    rebased = [(0.0, initial_rate)]
    rebased.extend((timestamp - offset_s, rate) for timestamp, rate in points if timestamp > offset_s)
    return NetworkTrace(rebased, duration_s=duration-offset_s, end_policy="stop"), {
        "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_trace": "HV", "original_start_offset_s": offset_s,
        "available_duration_after_offset_s": duration-offset_s,
        "rebasing": "timestamps only; every measured throughput value is unchanged",
    }


def run():
    started = time.perf_counter()
    trace_rows = {row["trace_id"]: row for row in TASK2_TRACES}
    family_representatives = {
        "camera_rotation_orbit": "orbit_orientation_control",
        "long_line": "long_line_horizontal_pan",
        "random_spatial": "random_seeded_pan_1",
    }
    family_validation = {}
    for family, trace_id in family_representatives.items():
        trace_row = trace_rows[trace_id]
        checks, proxy_analysis_ms = [], []
        for source_sample in trace_row["samples"]:
            identity = job_identity(TASK2_FREEZE, source_sample,
                {row["instance_id"]: 20 for row in source_sample["instances"]})
            _validate_request(identity, SOURCE_FREEZE)
            proxy_identity = job_identity(TASK2_FREEZE, source_sample,
                {row["instance_id"]: 40 for row in source_sample["instances"]}, proxy=True)
            proxy_entry = TASK2_REGISTRY.lookup(proxy_identity)
            if proxy_entry is None:
                raise FileNotFoundError(f"Missing exact native visibility artifact for {trace_id}")
            proxy_analysis_ms.append(float(load_proxy(proxy_entry).metadata.get("analysis_time_ms", 0.0)))
            checks.append(source_sample["timestamp_s"])
        family_validation[family] = {"trace_id": trace_id, "native_source_domain": "valid",
                                     "validated_timestamp_count": len(checks),
                                     "timestamps_s": checks,
                                     "runtime_visibility_cache_samples": len(proxy_analysis_ms),
                                     "reused_native_visibility_analysis_ms_mean":
                                         sum(proxy_analysis_ms) / len(proxy_analysis_ms),
                                     "scope": "frozen GOP0, three existing instances, exact supported trace"}

    source_profiles = load_profiles(PROFILE_PATH)
    metadata = load_metadata(read_task1_json(METADATA_PATH), source_profiles, file_digest(PROFILE_PATH),
                             "validated_cached_artifacts", METHODS)
    profiles = independent_instance_transfer_profiles(source_profiles)
    line = trace_rows["long_line_horizontal_pan"]
    source_trace, provider, visibility_data, visibility_timing = make_trace_and_visibility(line, profiles)
    visibility = CachedVisibilityAdapter(visibility_data, provider,
        expected_compatibility=profiles.compatibility, content_revision=profiles.content_revision)
    planner = AdaptationPlanner(profiles, provider, visibility, baseline_metadata=metadata,
                                expected_object_count=3)
    planner.initialize_solver()

    budget = 18_000_000
    decisions = {method: planner.plan(method, "0", budget) for method in METHODS}
    allocations = {method: {row["instance_id"]: int(row["selected_quality"].removeprefix("CRF"))
                            for row in decision["allocation"]["allocations"]}
                   for method, decision in decisions.items()}
    q20 = [{row["instance_id"]: 20 for row in sample["instances"]} for sample in line["samples"]]
    gpu_smoke = native_cuda_smoke()
    if gpu_smoke["gpu"] != TASK2_FREEZE["renderer_identity"]["gpu"]:
        raise RuntimeError("Host CUDA device differs from the frozen native renderer")

    OUT.mkdir(parents=True, exist_ok=True)
    executor = NativeExecutor(SOURCE_FREEZE, ROOT / TASK2_FREEZE["config"]["source_packages_root"],
        TASK2_FREEZE["renderer_identity"], OUT / "native_artifacts", TASK2_FREEZE["config"]["runtime"])
    images = {}
    render_records = []
    scale_probes = []
    quality_family_validation = {}
    try:
        # Reference Q20 and selected actual joint RGB are keyed by exact pose,
        # allocation and source state. Identical artifacts are reused.
        for sample_index, sample in enumerate(line["samples"]):
            for method in METHODS:
                allocation = allocations[method]
                key = (sample_index, tuple(sorted(allocation.items())))
                if key in images:
                    continue
                identity = job_identity(TASK2_FREEZE, sample, allocation)
                image, record = load_rgb_artifact(identity, executor, 7200.0)
                images[key] = image
                render_records.append({"method_allocation": dict(allocation), "sample_index": sample_index,
                                       "timestamp_s": sample["timestamp_s"], **record})
            ref_key = (sample_index, tuple(sorted(q20[sample_index].items())))
            if ref_key not in images:
                identity = job_identity(TASK2_FREEZE, sample, q20[sample_index])
                image, record = load_rgb_artifact(identity, executor, 7200.0)
                images[ref_key] = image
                render_records.append({"role": "joint_CR20_reference", "sample_index": sample_index,
                                       "timestamp_s": sample["timestamp_s"], **record})
        for family, trace_id in family_representatives.items():
            trace_row = trace_rows[trace_id]
            frame_stats = []
            frame_rows = []
            for sample_index, sample in enumerate(trace_row["samples"]):
                q20_allocation = {row["instance_id"]: 20 for row in sample["instances"]}
                q28_allocation = {row["instance_id"]: 28 for row in sample["instances"]}
                ref_identity = job_identity(TASK2_FREEZE, sample, q20_allocation)
                ref_entry = TASK2_REGISTRY.lookup(ref_identity)
                if ref_entry is None:
                    raise FileNotFoundError(f"Missing native Q20 joint reference for {trace_id}")
                reference = load_image(ref_entry)
                if trace_id == "long_line_horizontal_pan":
                    candidate = images[(sample_index, tuple(sorted(q28_allocation.items()))) ]
                    candidate_record = {"cache_hit": True, "source": "this pilot's verified candidate render"}
                else:
                    candidate_identity = job_identity(TASK2_FREEZE, sample, q28_allocation)
                    candidate, candidate_record = load_rgb_artifact(candidate_identity, executor, 7200.0)
                    render_records.append({"role": "family_quality_validation_CR28", "trace_id": trace_id,
                        "sample_index": sample_index, "timestamp_s": sample["timestamp_s"], **candidate_record})
                stats = foreground_metrics(reference["rgb"], candidate["rgb"], reference["alpha"],
                    alpha_threshold=0.01, normalization="clip", data_range=1.0)
                frame_stats.append(stats)
                frame_rows.append({"timestamp_s": sample["timestamp_s"], **stats.to_dict(),
                                   "candidate_cache_hit": candidate_record["cache_hit"]})
            from experiments.nava_mse.metrics import pool_metrics
            pooled = pool_metrics(frame_stats)
            quality_family_validation[family] = {
                "trace_id": trace_id,
                "candidate": "all three instances CRF28 against same-pose joint CRF20",
                "reference_mask": "joint CRF20 alpha > 0.01",
                "normalization": "clip RGB to [0,1]",
                "frame_count": len(frame_stats),
                "average_frame_psnr_db": sum(stats.psnr for stats in frame_stats) / len(frame_stats),
                "pooled_psnr_db": pooled.psnr,
                "samples": frame_rows,
                "quality_is_actual_joint_rendered_rgb": True,
            }
        # One reference render at each requested N tests renderer capacity and
        # per-frame cost. Replicas retain the source-local camera center; these
        # are load probes only and make no visual-quality claim about overlap.
        original_instances = line["samples"][0]["instances"]
        for count in (5, 7, 8, 10, 12, 15):
            seen = {}
            probe_instances = []
            for index in range(count):
                row = deepcopy(original_instances[index % len(original_instances)])
                object_id = row["object_id"]
                ordinal = seen.get(object_id, 0)
                seen[object_id] = ordinal + 1
                if ordinal:
                    row["instance_id"] = f"{object_id}-replica-{ordinal}"
                probe_instances.append(row)
            probe_sample = {"timestamp_s": line["samples"][0]["timestamp_s"],
                            "camera": line["samples"][0]["camera"], "instances": probe_instances}
            try:
                probe_identity = job_identity(TASK2_FREEZE, probe_sample,
                    {row["instance_id"]: 20 for row in probe_instances})
                _, probe_record = load_rgb_artifact(probe_identity, executor, 7200.0)
                probe_record.update(status=f"N{count}_NATIVE_JOINT_RENDER_PASSED", object_count=count,
                                    purpose="single-frame renderer capacity/timing probe only")
            except Exception as error:
                probe_record = {"status": f"N{count}_NATIVE_JOINT_RENDER_FAILED", "object_count": count,
                                "error_type": type(error).__name__, "error": str(error),
                                "purpose": "single-frame renderer capacity/timing probe only"}
            scale_probes.append(probe_record)
            render_records.append({"role": f"N{count}_capacity_probe", "sample_index": 0,
                                   "timestamp_s": probe_sample["timestamp_s"], **probe_record})
    finally:
        executor.close()

    measurement_records = {}
    reference_identity = digest({
        "kind": "same-pose-joint-CRF20-reference-sequence",
        "trace_id": line["trace_id"],
        "content_revision": profiles.content_revision,
        "gop_id": "0",
        "mask": "reference_alpha_gt_0.01",
        "normalization": "clip_rgb_0_1",
    })
    for sample_index, sample in enumerate(line["samples"]):
        reference = images[(sample_index, tuple(sorted(q20[sample_index].items()))) ]
        for method in METHODS:
            allocation = allocations[method]
            candidate = images[(sample_index, tuple(sorted(allocation.items()))) ]
            stats = foreground_metrics(reference["rgb"], candidate["rgb"], reference["alpha"],
                                       alpha_threshold=0.01, normalization="clip", data_range=1.0)
            metric = stats.to_dict()
            if metric["psnr_db"] is not None:
                metric["psnr_db"] = float(metric["psnr_db"])
            content_time = float(sample["timestamp_s"])
            measurement_records[(round(content_time, 9), tuple(sorted((
                instance, f"CRF{quality}") for instance, quality in allocation.items()))) ] = {
                    "metrics": metric,
                "reference_identity": reference_identity,
                    "provenance": {
                        "reference_frame_id": reference.get("arrays_sha256", "native-joint-CRF20-reference"),
                        "candidate_frame_id": candidate.get("arrays_sha256", "native-joint-candidate"),
                        "measurement_source": "native_joint_viewport_RGB_against_same_pose_CR20_reference",
                    },
                }

    network_trace, network_provenance = selected_trace()
    timeline = Timeline(source_duration_s=10, total_duration_s=2, gop_duration_s=2)
    simulation_config = {
        "experiment_id": "task4-pilot-long-line-gop0",
        "configuration_hash": None,
        "input_kind": "prepared_experiment",
        "adaptation": {"methods": METHODS, "safety_factor": 0.9,
                       "resource_sharing": "per_instance"},
        "network": {"request_latency_s": 0.02,
                    "estimator": {"startup_bps": 80_000_000.0, "window": 5, "stale_s": 30.0}},
        "playback": {"startup_buffer_s": 2.0, "max_buffer_s": 6.0},
        "max_wall_time_s": 102.0,
        "source_gop_ids": ["0", "1", "2", "3", "4"],
        "quality": {"mode": "trace_samples"},
    }
    scenario = {"scene_id": "task4-pilot-long-line-horizontal-pan"}
    quality_provider = PilotQualityProvider(measurement_records)
    summaries, timings, method_runs = [], [], {}
    for method in METHODS:
        sim, performance = simulate(simulation_config, timeline, scenario, network_trace,
            planner, method, quality_provider=quality_provider,
            input_timings={"visibility_analysis_ms": None,
                           "reused_native_visibility_analysis_ms": visibility_timing["reused_native_analysis_ms"],
                           "visibility_cache_lookup_ms": visibility_timing["cache_lookup_ms"],
                           "visibility_analysis_scope": "exact_GOP0_native_cache_reuse"})
        method_runs[method] = {"simulation": sim, "performance": performance}
        summaries.append({"method": method, "status": sim["status"], "metrics": sim["metrics"],
                          "selected_quality": sim["decisions"][0]["selected_quality"],
                          "requested_bytes": sim["metrics"]["bandwidth"]["total_requested_bytes"],
                          "startup_delay_s": sim["metrics"]["playback"]["startup_delay_s"],
                          "stall_count": sim["metrics"]["playback"]["stall_count"],
                          "rebuffering_duration_s": sim["metrics"]["playback"]["total_rebuffering_s"]})
        timings.extend(performance["records"])
        folder = OUT / method
        write_json(folder / "simulation.json", sim)
        write_json(folder / "performance.json", performance)
        write_json(folder / "metrics.json", sim["metrics"])

    result = {
        "schema": "streaming-evaluation.task4-pilot.v1",
        "status": "PILOT_COMPLETE_NOT_FULL_EXPERIMENT",
        "matrix_runs_completed": 0,
        "pilot_method_runs": summaries,
        "scenario_family_native_validity": family_validation,
        "scenario_family_quality_validation": quality_family_validation,
        "object_count_scale_probes": scale_probes,
        "native_cuda_smoke": gpu_smoke,
        "runtime_visibility": {**visibility_timing,
            "source": "exact cached native Visibility Analyzer outputs",
            "gop_id": "0", "trace_id": line["trace_id"], "samples": len(line["samples"]),
            "analysis_reran_for_this_pilot": False},
        "actual_joint_quality": {"sampled_frames_per_method": len(line["samples"]),
            "sampling_mode": "four exact trace samples in one 2-second GOP; not a 30-FPS frame grid",
            "reference": "same-pose joint CRF20 native render, reference alpha mask >0.01",
            "normalization": "clip RGB to [0,1], three channels",
            "available_values_are_actual_rendered_RGB": True},
        "new_candidate_renders": [row for row in render_records if not row["cache_hit"]],
        "native_image_cache_records": render_records,
        "network_trace": network_provenance,
        "resource_accounting": {
            "policy": profiles.provenance["resource_transfer_policy"],
            "transfer_map_sha256": digest(profiles.provenance["resource_transfer_aliases"]),
            "transfer_resource_count": len(profiles.provenance["resource_transfer_aliases"]),
            "map": profiles.provenance["resource_transfer_aliases"],
        },
        "gpu_render_and_simulation_elapsed_s": time.perf_counter() - started,
        "limitations": [
            "The pilot uses only source GOP0 and three unique object instances.",
            "The native quality inputs are four sparse trace samples, not every 30-FPS frame.",
            "The full Task-4 configs have no frozen scenario adaptation/visibility inputs.",
        ],
    }
    write_json(OUT / "pilot.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    run()
