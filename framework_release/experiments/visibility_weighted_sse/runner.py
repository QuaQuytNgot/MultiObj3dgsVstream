"""Frozen cached-input planning and deterministic decision exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from .common import digest, file_digest, integer, read_json, require, write_json

SCHEMA = "visibility-weighted-sse.experiment.v1"
INPUT_NAMES = ("trace", "profiles", "visibility")


def freeze_config(source, destination, *, expected_schema=SCHEMA, input_names=INPUT_NAMES):
    """Bind three local artifacts; freezing never generates missing visibility."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config = read_json(source)
    require(config.get("schema") == expected_schema, "Unsupported experiment configuration")
    for name in input_names:
        record = config["inputs"][name]
        path = (source.parent / record["path"]).resolve()
        require(path.is_file(), f"Missing {name} input: {path}")
        import os
        record.update(path=os.path.relpath(path, destination.parent), sha256=file_digest(path))
    config.pop("configuration_hash", None)
    config["configuration_hash"] = digest(config)
    write_json(destination, config)
    return config


def _load(config_path, *, expected_schema=SCHEMA):
    from .profiles import load_profiles
    from .trace import TraceCameraProvider
    from .visibility import CachedVisibilityAdapter

    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    require(config.get("schema") == expected_schema, "Unsupported experiment configuration")
    require(config.get("prediction_assumption") == "ORACLE_CAMERA_TRACE",
            "Experiment must explicitly declare ORACLE_CAMERA_TRACE")
    require(config.get("input_kind") in ("synthetic_fixture", "validated_cached_artifacts"),
            "Declare synthetic_fixture or validated_cached_artifacts input_kind")
    require(isinstance(config.get("experiment_id"), str) and bool(config["experiment_id"]),
            "Experiment ID is required")
    require(config.get("configuration_hash") == digest({k: v for k, v in config.items()
                                                       if k != "configuration_hash"}),
            "Configuration is not frozen or its hash changed; use freeze first")
    paths = {}
    for name in INPUT_NAMES:
        record = config["inputs"][name]
        path = (config_path.parent / record["path"]).resolve()
        require(path.is_file(), f"Missing {name} input: {path}; no acquisition is performed")
        require(record.get("sha256") == file_digest(path), f"Frozen {name} hash mismatch")
        paths[name] = path

    started = time.perf_counter()
    trace = TraceCameraProvider(read_json(paths["trace"]))
    trace_ms = (time.perf_counter() - started) * 1000
    profiles = load_profiles(paths["profiles"])
    started = time.perf_counter()
    visibility_data = read_json(paths["visibility"])
    require(visibility_data.get("content_revision") == profiles.content_revision,
            "Visibility/profile content revision mismatch")
    # Offline sampling distribution identifies the denominator, whereas the
    # online trace intentionally has another camera distribution.
    online_compatibility = visibility_data.get("compatibility", {})
    for name in (k for k in profiles.compatibility if k != "distribution_id"):
        require(name in online_compatibility and online_compatibility[name] == profiles.compatibility[name],
                f"Visibility/profile compatibility mismatch: {name}")
    require(visibility_data.get("input_kind") == config["input_kind"],
            "Visibility provenance input_kind differs from the experiment")
    require(read_json(paths["profiles"]).get("input_kind") == config["input_kind"],
            "Profile provenance input_kind differs from the experiment")
    visibility = CachedVisibilityAdapter(visibility_data, trace)
    acquisition_ms = (time.perf_counter() - started) * 1000

    gops = [str(row["gop_id"]) for row in trace.gops]
    requested = config.get("gop_ids", gops)
    require(isinstance(requested, list) and all(type(g) in (int, str) for g in requested),
            "GOP IDs must be strings or integers")
    requested = [str(g) for g in requested]
    require(isinstance(requested, list) and requested and len(requested) == len(set(requested))
            and set(requested) <= set(gops), "Unknown, missing or duplicate requested GOP")
    require(requested == [g for g in gops if g in requested], "GOP requests must follow trace order")
    budgets = config.get("budgets", [])
    require(isinstance(budgets, list) and budgets, "At least one byte budget is required")
    ids = [b["budget_id"] for b in budgets]
    require(all(isinstance(b, str) and b for b in ids) and len(set(ids)) == len(ids),
            "Budget IDs must be unique nonempty strings")
    for budget in budgets:
        for gop_id in requested:
            integer(_budget(budget, gop_id), "GOP budget")
    return config, trace, profiles, visibility, requested, trace_ms, acquisition_ms


def _budget(budget, gop_id):
    if "bytes" in budget:
        require("per_gop" not in budget, "Use bytes or per_gop budgets, not both")
        return budget["bytes"]
    require(gop_id in budget.get("per_gop", {}), f"Missing budget for GOP {gop_id}")
    return budget["per_gop"][gop_id]


def _gop_inputs(gop_id, profiles, visibility):
    from .profiles import build_utilities

    started = time.perf_counter()
    aggregated = visibility.aggregate(gop_id)
    aggregation_ms = (time.perf_counter() - started) * 1000
    object_profiles = {iid: p for (iid, gid), p in profiles.objects.items() if gid == gop_id}
    require(set(object_profiles) == set(aggregated["per_object"]),
            f"Profile/trace object coverage mismatch for {gop_id}")
    started = time.perf_counter()
    utilities = {}
    for iid, profile in sorted(object_profiles.items()):
        row = aggregated["per_object"][iid]
        require(row["object_id"] == profile.object_id, f"Object/instance profile mismatch: {iid}")
        utilities[iid] = build_utilities(profile, row["A_visible"])
    utility_ms = (time.perf_counter() - started) * 1000
    return aggregated, utilities, aggregation_ms, utility_ms


def validate(config_path):
    config, trace, profiles, visibility, gops, _, _ = _load(config_path)
    for gop in gops:
        _gop_inputs(gop, profiles, visibility)
    return {"status": "ready", "experiment_id": config["experiment_id"],
            "input_kind": config["input_kind"], "trace_id": trace.trace_id,
            "gop_ids": gops, "configuration_hash": config["configuration_hash"],
            "execution_mode": "cached_input_cpu_only", "renderer_invoked": False,
            "quality_evaluation": "unavailable_separate_stage"}


def run(config_path, output_dir):
    from .allocation import allocate

    total_started = time.perf_counter()
    config, trace, profiles, visibility, gops, trace_ms, acquisition_ms = _load(config_path)
    initialization_ms = (time.perf_counter() - total_started) * 1000
    rows, performance = [], []
    for gop_id in gops:
        gop_started = time.perf_counter()
        aggregated, utilities, aggregation_ms, utility_ms = _gop_inputs(gop_id, profiles, visibility)
        visible = {iid: row["A_visible"] for iid, row in aggregated["per_object"].items()}
        preparation_ms = (time.perf_counter() - gop_started) * 1000
        for budget in config["budgets"]:
            started = time.perf_counter()
            allocation = allocate(profiles, gop_id, visible, _budget(budget, gop_id),
                                  cached_resource_ids=config.get("cached_resource_ids", ()),
                                  solver_limits=config.get("solver_limits"), utilities=utilities)
            allocation_ms = (time.perf_counter() - started) * 1000
            # Measured timings live separately: identical frozen inputs retain
            # byte-identical policy artifacts while runtime measurements vary.
            diagnostics = allocation.get("diagnostics", {})
            solver_ms = diagnostics.pop("solver_elapsed_ms", None)
            adapter_ms = diagnostics.pop("allocation_adapter_elapsed_ms", None)
            solver_wall_ms = diagnostics.pop("solver_wall_ms", None)
            peak_rss_kib = diagnostics.pop("process_peak_rss_kib", None)
            performance.append({"gop_id": gop_id, "budget_id": budget["budget_id"],
                                "trace_loading_ms": trace_ms,
                                "cached_visibility_acquisition_ms": acquisition_ms,
                                "visibility_aggregation_ms": aggregation_ms,
                                "utility_construction_ms": utility_ms,
                                "allocation_ms": allocation_ms, "native_solver_ms": solver_ms,
                                "allocation_adapter_ms": adapter_ms,
                                "solver_wall_ms": solver_wall_ms,
                                "process_peak_rss_kib": peak_rss_kib,
                                "input_initialization_ms": initialization_ms,
                                "gop_preparation_ms": preparation_ms,
                                "measured_stage_sum_ms": trace_ms + acquisition_ms + aggregation_ms + utility_ms + allocation_ms,
                                "total_decision_ms": initialization_ms + preparation_ms + allocation_ms,
                                "total_decision_scope": "shared_input_initialization_plus_GOP_preparation_plus_one_budget_allocation_excludes_export",
                                "loading_scope": "whole_selected_trace_and_cache_file_once_per_run",
                                "shared_gop_stages_reused_across_budgets": True})
            rows.append({"experiment_id": config["experiment_id"], "trace_id": trace.trace_id,
                         "gop_id": gop_id, "budget_id": budget["budget_id"],
                         "content_revision": profiles.content_revision,
                         "profile_revision": profiles.profile_revision,
                         "configuration_hash": config["configuration_hash"],
                         "perfect_viewport_prediction": True, "prediction_assumption": "ORACLE_CAMERA_TRACE",
                         "visibility": aggregated, "utilities": utilities, "allocation": allocation,
                         "performance_reference": {"file": "performance.json", "record": len(performance) - 1}})
    decisions = {"schema": "visibility-weighted-sse.decisions.v1", "policy": "VISIBILITY_WEIGHTED_SSE",
                 "input_kind": config["input_kind"], "execution_mode": "cached_input_cpu_only",
                 "configuration_hash": config["configuration_hash"],
                 "inputs": config["inputs"], "quality_evaluation": "unavailable_separate_stage",
                 "utility_interpretation": "visibility_scale_adjusted_approximate_solo_SSE_reduction",
                 "cache_semantics": "fixed_declared_resident_snapshot_each_GOP_no_delivery_simulation",
                 "results": rows}
    destination = Path(output_dir)
    write_json(destination / "decisions.json", decisions)
    _export_csv(destination, decisions)
    write_json(destination / "performance.json", {
        "schema": "visibility-weighted-sse.performance.v1", "execution_mode": "cached_input_cpu_only",
        "configuration_hash": config["configuration_hash"], "records": performance,
        "total_run_ms": (time.perf_counter() - total_started) * 1000,
        "timings_excluded_from_deterministic_decisions": True,
        "includes_new_visibility_generation": False, "supports_realtime_visibility_claim": False})
    return decisions


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(",", ":"))
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _export_csv(destination, decisions):
    # Lossless flattened records keep nested allocation contracts reviewable
    # without coupling this exporter to a solver-specific selection format.
    allocations, object_allocations, visibility, utilities = [], [], [], []
    for row in decisions["results"]:
        identity = {k: row[k] for k in ("experiment_id", "trace_id", "gop_id", "budget_id",
                                        "configuration_hash", "content_revision", "profile_revision")}
        allocations.append({**identity, "allocation": row["allocation"]})
        allocation = row["allocation"]
        selected = {item["instance_id"]: item for item in allocation["allocations"]}
        for iid, item in row["utilities"].items():
            choice = selected.get(iid, {})
            object_allocations.append({**identity, "object_id": item["object_id"], "instance_id": iid,
                "selected_quality": choice.get("selected_quality"),
                "representation_id": choice.get("representation_id"),
                "dependencies": choice.get("dependencies", []),
                "required_resource_ids": choice.get("required_resource_ids", []),
                "physical_byte_cost": choice.get("physical_byte_cost"),
                "total_selected_bytes": allocation["total_selected_bytes"],
                "base_bytes": allocation["base_bytes"], "enhancement_bytes": allocation["enhancement_bytes"],
                "budget_bytes": allocation["available_budget_bytes"],
                "solver_status": allocation["solver_status"], "budget_feasible": allocation["budget_feasible"],
                "offline_mean_solo_contribution": item["offline_mean_solo_contribution"], "weight": item["weight"]})
        for iid, item in row["visibility"]["per_object"].items():
            visibility.append({**identity, "instance_id": iid, "visibility": item})
        for iid, item in row["utilities"].items():
            for quality in item["qualities"]:
                utilities.append({**identity, "object_id": item["object_id"], "instance_id": iid,
                                  "weight": item["weight"], **quality})
    _csv(destination / "decisions.csv", allocations)
    _csv(destination / "allocations.csv", object_allocations)
    _csv(destination / "visibility.csv", visibility)
    _csv(destination / "utilities.csv", utilities)
