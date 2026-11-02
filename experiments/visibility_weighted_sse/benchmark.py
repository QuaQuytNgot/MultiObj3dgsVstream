"""Small CPU scaling diagnostics built from frozen, cached profile inputs.

Cloned instances are synthetic workloads, not additional scene measurements.
Only utility construction and the existing exact-byte allocation adapter are
timed. Loading cached visibility and constructing the synthetic workloads happen
before those timers, and no visibility generation or image evaluation runs.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import statistics
import time
from types import MappingProxyType

from .common import digest, write_json
from .profiles import build_utilities
from .runner import _gop_inputs, _load

OBJECT_COUNTS = (3, 5, 8, 12)
REPETITIONS = 5


def _clones(profiles, source, count):
    """Keep within-instance physical aliases and isolate bytes across clones."""
    resources, objects = {}, {}
    referenced = {rid for q in source.qualities for rid in q.resource_ids}
    for index in range(count):
        instance_id = f"scaling_instance_{index:02d}"
        resource_ids = {rid: f"scaling_{index:02d}_resource_{resource_index:02d}"
                        for resource_index, rid in enumerate(sorted(referenced))}
        for rid, clone_id in resource_ids.items():
            original = profiles.resources[rid]
            resources[clone_id] = replace(
                original, id=clone_id, url="synthetic-metadata:" + clone_id,
                sha256=digest({"synthetic_clone": index, "source_sha256": original.sha256}))
        qualities = tuple(replace(
            q, representation_id=f"scaling_{index:02d}_{q.representation_id}",
            resource_ids=tuple(resource_ids[rid] for rid in q.resource_ids),
            own_resource_ids=tuple(resource_ids[rid] for rid in q.own_resource_ids))
            for q in source.qualities)
        objects[instance_id, source.gop_id] = replace(
            source, instance_id=instance_id, qualities=qualities)
    return replace(profiles, resources=MappingProxyType(resources),
                   objects=MappingProxyType(objects),
                   provenance=MappingProxyType({"source": "synthetic_scaling_clones",
                                                "empirical_scene_measurement": False}))


def _physical_bytes(profiles, quality):
    return sum({profiles.resources[rid].sha256: profiles.resources[rid].bytes
                for rid in quality.resource_ids}.values())


def _summary(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "minimum_ms": min(samples), "maximum_ms": max(samples)}


def benchmark(config_path, output):
    from .allocation import allocate

    config, _, profiles, visibility, gops, _, _ = _load(config_path)
    gop_id = gops[0]
    aggregated, _, _, _ = _gop_inputs(gop_id, profiles, visibility)
    source = profiles.for_gop(gop_id)[0]
    source_visible = aggregated["per_object"][source.instance_id]["A_visible"]
    records = []
    for count in OBJECT_COUNTS:
        cloned = _clones(profiles, source, count)
        object_profiles = cloned.for_gop(gop_id)
        visible = {p.instance_id: source_visible for p in object_profiles}
        base_bytes = sum(_physical_bytes(cloned, p.qualities[0]) for p in object_profiles)
        maximum_bytes = sum(max(_physical_bytes(cloned, q) for q in p.qualities)
                            for p in object_profiles)
        budget = base_bytes + (maximum_bytes - base_bytes) // 2
        limits = config.get("solver_limits")

        def utilities():
            return {p.instance_id: build_utilities(p, visible[p.instance_id])
                    for p in object_profiles}

        # Warmup also builds/loads the existing native bridge before measurement.
        values = utilities()
        allocate(cloned, gop_id, visible, budget, cached_resource_ids=(),
                 solver_limits=limits, utilities=values)
        utility_samples, allocation_samples, native_samples = [], [], []
        result = None
        for _ in range(REPETITIONS):
            started = time.perf_counter_ns()
            values = utilities()
            utility_samples.append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            result = allocate(cloned, gop_id, visible, budget, cached_resource_ids=(),
                              solver_limits=limits, utilities=values)
            allocation_samples.append((time.perf_counter_ns() - started) / 1e6)
            native_ms = result.get("diagnostics", {}).get("solver_elapsed_ms")
            if native_ms is not None:
                native_samples.append(native_ms)
        result = dict(result)
        result["diagnostics"] = dict(result["diagnostics"])
        for key in ("solver_elapsed_ms", "allocation_adapter_elapsed_ms", "solver_wall_ms", "process_peak_rss_kib"):
            result["diagnostics"].pop(key, None)
        records.append({"object_count": count, "available_budget_bytes": budget,
                        "base_bytes": base_bytes, "maximum_candidate_bytes": maximum_bytes,
                        "utility_construction": _summary(utility_samples),
                        "allocation_adapter": _summary(allocation_samples),
                        "native_solver": _summary(native_samples) if native_samples else None,
                        "allocation": result})
    result = {"schema": "visibility-weighted-sse.scaling.v1",
              "input_kind": "synthetic_scaling_clones", "execution_mode": "cached_input_cpu_only",
              "source_input_kind": config["input_kind"],
              "configuration_hash": config["configuration_hash"],
              "source_instance_id": source.instance_id, "source_gop_id": gop_id,
              "source_visibility_contribution": source_visible,
              "object_counts": list(OBJECT_COUNTS), "warmups_per_count": 1,
              "repetitions_per_count": REPETITIONS,
              "workload_construction": "independent_resource_clones_of_first_frozen_object_GOP",
              "timed_stages": ["weighted_utility_construction", "allocation_adapter", "native_solver"],
              "allocation_adapter_scope": "utility_revalidation_resource_accounting_native_solver_postvalidation",
              "excluded_stages": ["trace_loading", "cached_visibility_acquisition",
                                  "visibility_aggregation", "synthetic_workload_construction",
                                  "native_bridge_compilation_and_warmup"],
              "new_visibility_generated": False, "renderer_invoked": False,
              "quality_evaluation": "unavailable_separate_stage",
              "empirical_paper_measurement": False,
              "supports_realtime_visibility_claim": False, "records": records}
    write_json(Path(output), result)
    return result
