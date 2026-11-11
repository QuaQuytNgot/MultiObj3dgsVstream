"""Reproducible adaptation comparison over the existing frozen-input pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from experiments.visibility_weighted_sse.common import digest, file_digest, integer, read_json, require, write_json
from experiments.visibility_weighted_sse import runner as weighted_runner
from .interface import AdaptationPlanner
from .metadata import load_metadata
from .methods import SUPPORTED_METHODS, SOLO_SSE, VISIBILITY_WEIGHTED_SSE, construct_utilities

SCHEMA = "adaptation-baselines.experiment.v1"


def _methods(config):
    methods = config.get("methods")
    require(isinstance(methods, list) and methods and all(isinstance(m, str) and m in SUPPORTED_METHODS for m in methods)
            and len(set(methods)) == len(methods), "Select unique supported method IDs in methods")
    return methods


def freeze_config(source, destination):
    config = read_json(source)
    methods = _methods(config)
    names = list(weighted_runner.INPUT_NAMES)
    if set(methods) - {SOLO_SSE, VISIBILITY_WEIGHTED_SSE} or "baseline_metadata" in config["inputs"]:
        names.append("baseline_metadata")
    return weighted_runner.freeze_config(source, destination, expected_schema=SCHEMA, input_names=names)


def load_experiment(path):
    began = time.perf_counter()
    loaded = weighted_runner._load(path, expected_schema=SCHEMA)
    config, trace, profiles, visibility, gops, trace_ms, visibility_ms = loaded
    methods = _methods(config)
    path = Path(path).resolve()
    metadata = None
    needs_metadata = bool(set(methods) - {SOLO_SSE, VISIBILITY_WEIGHTED_SSE})
    record = config["inputs"].get("baseline_metadata")
    if record is not None:
        source = (path.parent / record["path"]).resolve()
        require(source.is_file(), f"Missing baseline metadata: {source}")
        require(record.get("sha256") == file_digest(source), "Frozen baseline metadata hash mismatch")
        metadata = load_metadata(read_json(source), profiles, config["inputs"]["profiles"]["sha256"],
                                 config["input_kind"], methods)
    require(not needs_metadata or metadata is not None, "Selected baselines require independent baseline metadata")
    count = config.get("expected_object_count")
    if count is not None:
        integer(count, "expected_object_count", 1)
    quality_levels = config.get("quality_levels")
    if quality_levels is not None:
        require(isinstance(quality_levels, dict), "quality_levels must map each instance to eligible quality IDs")
        for gid in gops:
            rows = profiles.for_gop(gid)
            require(set(quality_levels) == {row.instance_id for row in rows}, "Quality-level instance coverage mismatch")
            for row in rows:
                eligible = quality_levels[row.instance_id]
                require(isinstance(eligible, list) and eligible and all(isinstance(q, str) for q in eligible)
                        and len(eligible) == len(set(eligible))
                        and set(eligible) <= {q.quality_id for q in row.qualities}, "Unknown or duplicate eligible quality")
                require(row.qualities[0].quality_id in eligible, "Eligible qualities must include mandatory base")
    planner = AdaptationPlanner(profiles, trace, visibility, baseline_metadata=metadata,
        cached_resource_ids=config.get("cached_resource_ids", ()), allowed_quality_ids=quality_levels,
        solver_limits=config.get("solver_limits"), expected_object_count=count)
    return {"config": config, "planner": planner, "methods": methods, "gops": gops,
            "timings": {"input_initialization_ms": (time.perf_counter() - began)*1000,
                        "trace_loading_ms": trace_ms, "cached_visibility_acquisition_ms": visibility_ms}}


def validate(path):
    experiment = load_experiment(path)
    planner = experiment["planner"]
    for gid in experiment["gops"]:
        trajectory, aggregate, _ = planner.prepare(gid)
        for method in experiment["methods"]:
            construct_utilities(method, planner.profiles, gid, trajectory, aggregate, planner.metadata)
    return {"status": "ready", "methods": experiment["methods"], "gop_ids": experiment["gops"],
            "configuration_hash": experiment["config"]["configuration_hash"],
            "input_kind": experiment["config"]["input_kind"], "renderer_invoked": False,
            "quality_evaluation": "unavailable_separate_stage"}


def run(path, output):
    began = time.perf_counter()
    experiment = load_experiment(path)
    config, planner = experiment["config"], experiment["planner"]
    initialization_ms = planner.initialize_solver()
    decisions, timings, common_gops = [], [], {}
    for gid in experiment["gops"]:
        trajectory, aggregate, shared_ms = planner.prepare(gid)
        common_gops[gid] = {"trace_id": planner.trace.trace_id, "visibility": aggregate,
                           "instance_ids": [p.instance_id for p in planner.profiles.for_gop(gid)]}
        for budget in config["budgets"]:
            amount = weighted_runner._budget(budget, gid)
            for method in experiment["methods"]:
                decision = planner.plan(method, gid, amount)
                runtime = decision.pop("timings")
                runtime.update(gop_id=gid, budget_id=budget["budget_id"], method=method,
                               **experiment["timings"])
                runtime["shared_solver_initialization_ms"] = initialization_ms
                runtime["total_decision_including_shared_inputs_ms"] = (
                    experiment["timings"]["input_initialization_ms"] + initialization_ms + shared_ms + runtime["decision_call_ms"])
                timings.append(runtime)
                decision.update(experiment_id=config["experiment_id"], trace_id=planner.trace.trace_id,
                                budget_id=budget["budget_id"], configuration_hash=config["configuration_hash"],
                                perfect_viewport_prediction=True,
                                performance_reference={"file": "performance.json", "record": len(timings)-1})
                decisions.append(decision)
    result = {"schema": "adaptation-baselines.decisions.v1", "experiment_id": config["experiment_id"],
              "configuration_hash": config["configuration_hash"], "methods": experiment["methods"],
              "inputs": config["inputs"], "input_kind": config["input_kind"],
              "content_revision": planner.profiles.content_revision,
              "sse_profile_revision": planner.profiles.profile_revision,
              "baseline_profile_revision": planner.metadata["profile_revision"] if planner.metadata else None,
              "representation_family": planner.profiles.representation_family,
              "quality_levels": config.get("quality_levels", "all_declared_candidates"),
              "execution_mode": "cached_input_CPU_only", "prediction_assumption": "ORACLE_CAMERA_TRACE",
              "shared_solver": "src_final.ProposedDPSolver_exact_integer_byte_adapter",
              "cost_semantics": "full_selected_dependency_resource_union_excluding_declared_resident_SHA256",
              "quality_evaluation": "unavailable_separate_stage",
              "utility_cross_method_comparison": "different_units_do_not_compare_as_measured_quality",
              "gops": common_gops, "results": decisions}
    output = Path(output)
    write_json(output / "decisions.json", result)
    export_csv(output, result)
    write_json(output / "performance.json", {"schema": "adaptation-baselines.performance.v1",
        "configuration_hash": config["configuration_hash"], "records": timings,
        "total_run_ms": (time.perf_counter()-began)*1000,
        "execution_mode": "cached_input_CPU_only", "includes_visibility_generation": False,
        "shared_input_stages_reused_across_methods_and_budgets": True,
        "total_decision_scope": "shared_initialization_and_GOP_preparation_plus_one_method_call_excludes_export",
        "native_bridge_warmup": "shared_initialization_separate_from_all_compared_method_calls_no_candidate_solve_warmup"})
    return result


def _csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def export_csv(output, result):
    decisions, selections, candidates = [], [], []
    for row in result["results"]:
        identity = {key: row[key] for key in ("experiment_id", "trace_id", "gop_id", "budget_id", "method")}
        solved = row["allocation"]
        decisions.append({**identity, "budget_bytes": row["budget_bytes"], "status": solved["status"],
            "total_required_bytes": row["total_required_bytes"], "base_bytes": solved["base_bytes"],
            "estimated_total_utility": row["estimated_total_utility"], "selected_quality": row["selected_quality"],
            "utility_units": row["method_diagnostics"]["utility_units"], "budget_feasible": solved["feasible"]})
        by_id = {item["instance_id"]: item for item in solved["allocations"]}
        for iid, utility in row["utilities"].items():
            selected = by_id.get(iid, {})
            selections.append({**identity, "object_id": utility["object_id"], "instance_id": iid,
                "quality_id": selected.get("selected_quality"), "representation_id": selected.get("representation_id"),
                "estimated_utility": selected.get("weighted_utility"), "physical_byte_cost": selected.get("physical_byte_cost"),
                "dependencies": selected.get("dependencies", []), "status": solved["status"]})
            for quality in utility["qualities"]:
                eligible = result["quality_levels"]
                candidates.append({**identity, "object_id": utility["object_id"], "instance_id": iid,
                    "quality_id": quality["quality_id"], "representation_id": quality["representation_id"],
                    "eligible": eligible == "all_declared_candidates" or quality["quality_id"] in eligible[iid],
                    "estimated_utility": quality["weighted_utility"], "weight": utility["weight"],
                    "utility_units": row["method_diagnostics"]["utility_units"]})
    _csv(output / "decisions.csv", decisions)
    _csv(output / "selections.csv", selections)
    _csv(output / "utilities.csv", candidates)
