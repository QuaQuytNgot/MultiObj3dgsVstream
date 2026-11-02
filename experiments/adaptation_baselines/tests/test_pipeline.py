"""Frozen-input comparison contracts, independent of rendering and transport."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.adaptation_baselines.methods import METHODS, SOLO_SSE, VISIBILITY_WEIGHTED_SSE
from experiments.adaptation_baselines.interface import AdaptationPlanner
from experiments.adaptation_baselines.runner import freeze_config, load_experiment, run, validate
from experiments.visibility_weighted_sse.allocation import allocate
from experiments.visibility_weighted_sse.common import file_digest, read_json, write_json
from experiments.visibility_weighted_sse.trace import TraceCameraProvider

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]


def frozen_fixture(destination, *, mutate_config=None, mutate_inputs=None):
    """Copy small synthetic JSON artifacts only; never touch historical caches."""
    source = EXPERIMENT / "configs" / "fixture.json"
    config = read_json(source)
    # Byte-scale synthetic fixtures certify the exact reference; the production
    # bitrate grid is exercised with realistic byte sizes in test_discretization.
    config.setdefault("solver_limits", {})["discretization"] = "exact_bytes"
    data = {name: read_json((source.parent / record["path"]).resolve())
            for name, record in config["inputs"].items()}
    data["baseline_metadata"].pop("profile_binding_sha256", None)
    if mutate_inputs:
        mutate_inputs(data)
    for name in ("trace", "profiles", "visibility"):
        write_json(destination / f"{name}.json", data[name])
    data["baseline_metadata"].setdefault("profile_binding_sha256", file_digest(destination / "profiles.json"))
    write_json(destination / "baseline_metadata.json", data["baseline_metadata"])
    config["inputs"] = {name: {"path": f"{name}.json"} for name in data}
    if mutate_config:
        mutate_config(config)
    source_path, frozen = destination / "source.json", destination / "frozen.json"
    write_json(source_path, config)
    freeze_config(source_path, frozen)
    return frozen


def test_all_methods_share_budgets_input_profiles_and_exact_resource_costs(tmp_path):
    config = frozen_fixture(tmp_path)
    assert validate(config)["methods"] == list(METHODS)
    experiment = load_experiment(config)
    profiles = experiment["planner"].profiles
    result = run(config, tmp_path / "run")
    assert len(result["results"]) == 40
    assert result["methods"] == list(METHODS)
    assert result["quality_evaluation"] == "unavailable_separate_stage"
    for record in result["results"]:
        solved = record["allocation"]
        assert solved["base_bytes"] == 30
        assert solved["available_budget_bytes"] == record["budget_bytes"]
        assert solved["diagnostics"]["algorithm"] == "src_final.ProposedDPSolver"
        assert record["perfect_viewport_prediction"] is True
        if record["budget_bytes"] < 30:
            assert solved["status"] == "infeasible_base"
            assert record["selected_quality"] is None
            assert record["estimated_utility_per_object"] is None
            assert solved["allocations"] == []
            continue
        assert solved["status"] == "optimal"
        assert set(record["selected_quality"]) == {"instance-0", "instance-1", "instance-2"}
        chosen = [next(q for q in profiles.objects[item["instance_id"], record["gop_id"]].qualities
                       if q.quality_id == item["selected_quality"]) for item in solved["allocations"]]
        resources = {profiles.resources[key].sha256: profiles.resources[key].bytes
                     for quality in chosen for key in quality.resource_ids}
        assert record["total_required_bytes"] == sum(resources.values())
        assert sum(item["physical_byte_cost"] for item in solved["allocations"]) == sum(resources.values())
        assert record["total_required_bytes"] <= record["budget_bytes"]
        assert record["estimated_total_utility"] == sum(record["estimated_utility_per_object"].values())
        assert solved["selected_resource_sha256"] == sorted(resources)
    assert result["gops"]["g1"]["visibility"]["temporal_weights"] == [.125, .375, .5]


def test_proposed_decisions_match_existing_direct_allocator_for_every_gop_budget(tmp_path):
    config = frozen_fixture(tmp_path)
    experiment = load_experiment(config)
    planner = experiment["planner"]
    for gid in experiment["gops"]:
        _, aggregated, _ = planner.prepare(gid)
        visible = {iid: row["A_visible"] for iid, row in aggregated["per_object"].items()}
        for budget in experiment["config"]["budgets"]:
            amount = budget["bytes"]
            direct = allocate(planner.profiles, gid, visible, amount, solver_limits=planner.solver_limits)
            direct["diagnostics"].pop("solver_elapsed_ms")
            direct["diagnostics"].pop("allocation_adapter_elapsed_ms")
            direct["diagnostics"].pop("solver_wall_ms")
            direct["diagnostics"].pop("process_peak_rss_kib")
            shared = planner.plan(VISIBILITY_WEIGHTED_SSE, gid, amount)
            assert shared["allocation"] == direct
            if direct["feasible"]:
                assert shared["selected_quality"]["instance-1"] == "Q0"


def test_json_and_csv_exports_are_reproducible_with_runtime_separate(tmp_path):
    config = frozen_fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    run(config, first)
    run(config, second)
    for name in ("decisions.json", "decisions.csv", "selections.csv", "utilities.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    result = read_json(first / "decisions.json")
    for row in result["results"]:
        assert "timings" not in row
        assert "solver_elapsed_ms" not in row["allocation"]["diagnostics"]
        assert "allocation_adapter_elapsed_ms" not in row["allocation"]["diagnostics"]
    expected_counts = {"decisions.csv": 40, "selections.csv": 120, "utilities.csv": 480}
    for name, expected in expected_counts.items():
        with (first / name).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == expected
    performance = read_json(first / "performance.json")
    assert performance["includes_visibility_generation"] is False
    assert len(performance["records"]) == 40
    for row in performance["records"]:
        assert all(row[field] >= 0 for field in ("utility_construction_ms", "allocation_ms",
            "native_solver_ms", "decision_call_ms", "total_decision_including_shared_inputs_ms"))


@pytest.mark.parametrize("method,reference", [("NAVA", "psnr"), ("VISIBILITY_PSNR", "psnr"),
                                               ("VISIBILITY_MSE", "mse")])
def test_direct_python_planner_rejects_quality_reference_tampering(tmp_path, method, reference):
    planner = load_experiment(frozen_fixture(tmp_path))["planner"]
    # Direct callers can hold and alter metadata after loading. The public
    # planning boundary still enforces each recovered baseline's reference.
    planner.metadata["quality_references"][reference]["reference"] = "incompatible-reference"
    with pytest.raises(ValueError, match="PSNR requires|MSE requires"):
        planner.plan(method, "g0", 150)


@pytest.mark.parametrize("provenance", [
    {"information_role": "EVALUATION_ONLY_JOINT_SSE", "joint_quality_used": False},
    {"information_role": "OFFLINE_BASELINE_METADATA", "joint_quality_used": True}])
def test_direct_python_planner_rejects_quality_provenance_tampering(tmp_path, provenance):
    planner = load_experiment(frozen_fixture(tmp_path))["planner"]
    planner.metadata["provenance"] = provenance
    with pytest.raises(ValueError, match="Independent offline baseline provenance"):
        planner.plan("NAVA", "g0", 150)


def test_native_initialization_is_shared_before_all_compared_method_timing_scopes(tmp_path, monkeypatch):
    events = []
    initialize = AdaptationPlanner.initialize_solver
    plan = AdaptationPlanner.plan
    shared_initialization_ms = 123.456

    def initialize_once(self):
        measured = initialize(self)
        assert measured >= 0
        events.append("initialize")
        # A known duration lets this test check accounting without any sleeps
        # or assumptions about machine speed or native bridge cache warmth.
        return shared_initialization_ms

    def measured_plan(self, method, gop_id, budget_bytes):
        assert events and events[0] == "initialize"
        events.append(method)
        return plan(self, method, gop_id, budget_bytes)

    monkeypatch.setattr(AdaptationPlanner, "initialize_solver", initialize_once)
    monkeypatch.setattr(AdaptationPlanner, "plan", measured_plan)
    output = tmp_path / "output"
    result = run(frozen_fixture(tmp_path), output)
    assert events.count("initialize") == 1
    assert len(events) == 41
    performance = read_json(output / "performance.json")
    assert performance["native_bridge_warmup"] == (
        "shared_initialization_separate_from_all_compared_method_calls_no_candidate_solve_warmup")
    assert "shared_solver_initialization_ms" not in json.dumps(result)
    for row in performance["records"]:
        assert row["shared_solver_initialization_ms"] == shared_initialization_ms
        expected = (row["input_initialization_ms"] + shared_initialization_ms
                    + row["shared_GOP_preparation_ms"] + row["decision_call_ms"])
        assert row["total_decision_including_shared_inputs_ms"] == pytest.approx(expected)


@pytest.mark.parametrize("methods", [[SOLO_SSE], [VISIBILITY_WEIGHTED_SSE], [SOLO_SSE, VISIBILITY_WEIGHTED_SSE]])
def test_sse_method_subsets_do_not_require_unrelated_psnr_mse_metadata(tmp_path, methods):
    def select(config):
        config["methods"] = methods
        config["inputs"].pop("baseline_metadata")
        config["gop_ids"] = ["g1"]
        config["budgets"] = [{"budget_id": "per-gop", "per_gop": {"g1": 58}}]
    config = frozen_fixture(tmp_path, mutate_config=select)
    assert validate(config)["methods"] == methods
    result = run(config, tmp_path / "out")
    assert result["baseline_profile_revision"] is None
    assert [row["method"] for row in result["results"]] == methods
    assert all(row["budget_bytes"] == 58 and row["gop_id"] == "g1" for row in result["results"])


def test_non_sse_baselines_require_bound_independent_metadata(tmp_path):
    def omit(config):
        config["inputs"].pop("baseline_metadata")
    with pytest.raises(KeyError, match="baseline_metadata"):
        frozen_fixture(tmp_path, mutate_config=omit)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_configured_quality_counts_preserve_full_dependency_closures(tmp_path, count):
    selected = {1: ["Q0"], 2: ["Q0", "Q2"], 3: ["Q0", "Q1", "Q2"],
                4: ["Q0", "Q1", "Q2", "Q3"]}[count]
    def configure(config):
        config["quality_levels"] = {f"instance-{i}": selected for i in range(3)}
        config["gop_ids"] = ["g0"]
        config["budgets"] = [{"budget_id": "ample", "bytes": 150}]
    config = frozen_fixture(tmp_path, mutate_config=configure)
    result = run(config, tmp_path / "out")
    assert len(result["results"]) == 5
    for row in result["results"]:
        assert set(row["selected_quality"].values()) <= set(selected)
        assert len(row["utilities"]["instance-0"]["qualities"]) == 4
        if count == 1:
            assert row["total_required_bytes"] == 30
    if count == 2:
        solo = next(row for row in result["results"] if row["method"] == SOLO_SSE)
        assert set(solo["selected_quality"].values()) == {"Q2"}
        assert solo["total_required_bytes"] == 90
        for choice in solo["allocation"]["allocations"]:
            assert choice["dependencies"] == ["Q0", "Q1"]
            assert choice["required_representations"] == ["layer-0", "layer-1", "layer-2"]
        with (tmp_path / "out" / "utilities.csv").open(newline="") as stream:
            candidates = list(csv.DictReader(stream))
        assert all(row["eligible"] == str(row["quality_id"] in selected) for row in candidates)


def resized_scene(data, count):
    """Synthetic symbols only: construct a new full scene and its new cache."""
    original = deepcopy(data)
    revision = f"synthetic-resized-{count}-content-v1"
    data["profiles"]["content_revision"] = data["visibility"]["content_revision"] = revision
    data["baseline_metadata"]["content_revision"] = revision
    data["profiles"]["objects"], data["profiles"]["resources"] = [], {}
    data["baseline_metadata"]["objects"] = []
    for i in range(count):
        source = i % 3
        obj = deepcopy(original["profiles"]["objects"][source])
        meta = deepcopy(original["baseline_metadata"]["objects"][source])
        for row in (obj, meta):
            row.update(object_id=f"object-{i}", instance_id=f"instance-{i}")
        for gop in obj["gops"]:
            for quality in gop["offline_sse"]:
                keys = []
                for old in quality["resource_ids"]:
                    key = old.replace(f"object-{source}:", f"object-{i}:")
                    resource = deepcopy(original["profiles"]["resources"][old])
                    resource["sha256"] = hashlib.sha256(f"resized-{count}-{i}-{old}".encode()).hexdigest()
                    data["profiles"]["resources"][key] = resource
                    keys.append(key)
                quality["resource_ids"] = keys
        data["profiles"]["objects"].append(obj)
        data["baseline_metadata"]["objects"].append(meta)
    for gop, original_gop in zip(data["trace"]["gops"], original["trace"]["gops"]):
        for sample, old_sample in zip(gop["samples"], original_gop["samples"]):
            sample["instances"] = []
            for i in range(count):
                instance = deepcopy(old_sample["instances"][i % 3])
                instance.update(instance_id=f"instance-{i}", object_id=f"object-{i}",
                                source_state_id=f"synthetic-resized/{gop['gop_id']}/{sample['timestamp']}/{i}")
                sample["instances"].append(instance)
    provider = TraceCameraProvider(data["trace"])
    for gop, old_gop in zip(data["visibility"]["gops"], original["visibility"]["gops"]):
        for sample, old_sample, pose in zip(gop["samples"], old_gop["samples"], provider.get_trajectory(gop["gop_id"])):
            sample["pose_identity"] = pose.identity
            sample["per_object"] = {f"instance-{i}": deepcopy(old_sample["per_object"][f"instance-{i % 3}"])
                                    for i in range(count)}


@pytest.mark.parametrize("count", [1, 5])
def test_object_count_is_configurable_through_a_complete_new_synthetic_scene(tmp_path, count):
    config = frozen_fixture(tmp_path, mutate_inputs=lambda data: resized_scene(data, count),
        mutate_config=lambda c: c.update(expected_object_count=count, gop_ids=["g0"],
                                         budgets=[{"budget_id": "ample", "bytes": 1000}]))
    assert validate(config)["status"] == "ready"
    result = run(config, tmp_path / "out")
    assert len(result["results"]) == 5
    assert all(len(row["selected_quality"]) == count for row in result["results"])
    assert all(row["allocation"]["base_bytes"] == count * 10 for row in result["results"])


def test_expected_object_count_is_an_assertion_not_silent_scene_truncation(tmp_path):
    config = frozen_fixture(tmp_path, mutate_config=lambda c: c.update(expected_object_count=1))
    with pytest.raises(ValueError, match="object count"):
        validate(config)


@pytest.mark.parametrize("name", ["trace", "profiles", "visibility", "baseline_metadata"])
def test_changed_frozen_input_is_rejected(tmp_path, name):
    config = frozen_fixture(tmp_path)
    with (tmp_path / f"{name}.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(config)


@pytest.mark.parametrize("issue", ["joint", "psnr_reference", "mse_reference", "mse_distribution", "profile_binding",
                                    "representation", "missing_gop", "visibility_joint", "trace_missing"])
def test_refrozen_semantic_provenance_reference_and_coverage_errors_fail_closed(tmp_path, issue):
    def mutate(data):
        baseline = data["baseline_metadata"]
        if issue == "joint":
            baseline["provenance"]["joint_quality_used"] = True
        elif issue == "psnr_reference":
            baseline["quality_references"]["psnr"]["reference"] = "own_CRF20"
        elif issue == "mse_reference":
            baseline["quality_references"]["mse"]["reference"] = "original_ply_synthetic_rgb"
        elif issue == "mse_distribution":
            baseline["quality_references"]["mse"]["distribution_id"] = "other-offline-sampling"
        elif issue == "profile_binding":
            baseline["profile_binding_sha256"] = "0" * 64
        elif issue == "representation":
            baseline["objects"][0]["gops"][0]["qualities"][0]["representation_id"] = "wrong"
        elif issue == "missing_gop":
            baseline["objects"][0]["gops"].pop()
        elif issue == "visibility_joint":
            data["visibility"]["provenance"]["joint_quality_used"] = True
        else:
            data["visibility"]["gops"][0]["samples"].pop()
    config = frozen_fixture(tmp_path, mutate_inputs=mutate)
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("methods", [[], ["INVENTED"], [SOLO_SSE, SOLO_SSE]])
def test_unknown_empty_and_duplicate_method_selections_are_rejected(tmp_path, methods):
    with pytest.raises(ValueError, match="supported method"):
        frozen_fixture(tmp_path, mutate_config=lambda c: c.update(methods=methods))


def test_configuration_hash_changes_are_rejected(tmp_path):
    config = frozen_fixture(tmp_path)
    data = read_json(config)
    data["budgets"][0]["bytes"] += 1
    write_json(config, data)
    with pytest.raises(ValueError, match="hash changed"):
        validate(config)


def test_cli_runs_all_five_methods_with_gpu_renderer_and_network_imports_forbidden(tmp_path):
    config = frozen_fixture(tmp_path)
    code = r'''
import importlib.abc
import runpy
import sys
class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "cupy", "gsplat", "diff_gaussian_rasterization", "requests", "httpx"} or \
           fullname.startswith("src_final.multiobj3dgs.renderer"):
            raise AssertionError("CPU comparison imported forbidden module: " + fullname)
sys.meta_path.insert(0, ForbiddenImports())
sys.argv = ["experiments.adaptation_baselines", "run", "--config", sys.argv[1], "--output", sys.argv[2]]
try:
    runpy.run_module("experiments.adaptation_baselines", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0, exc.code
assert "torch" not in sys.modules
assert "diff_gaussian_rasterization" not in sys.modules
'''
    output = tmp_path / "cli"
    completed = subprocess.run([sys.executable, "-c", code, str(config), str(output)],
                               cwd=REPOSITORY, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert summary["decisions"] == 40
    assert summary["methods"] == list(METHODS)


def test_cli_missing_input_returns_machine_readable_failure(tmp_path):
    config = frozen_fixture(tmp_path)
    (tmp_path / "baseline_metadata.json").unlink()
    completed = subprocess.run([sys.executable, "-m", "experiments.adaptation_baselines", "validate",
                                "--config", str(config)], cwd=REPOSITORY, capture_output=True,
                               text=True, timeout=60)
    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["status"] == "blocked_invalid_or_missing_input"
    assert failure["renderer_invoked"] is False


def test_historical_aabb_mse_is_available_through_frozen_common_interface(tmp_path):
    from experiments.adaptation_baselines.methods import AABB_MSE_HISTORICAL
    path = frozen_fixture(tmp_path, mutate_config=lambda c: c.update(methods=[AABB_MSE_HISTORICAL]))
    planner = load_experiment(path)["planner"]
    result = planner.plan(AABB_MSE_HISTORICAL, "g0", 1000)
    assert result["allocation"]["feasible"]
    assert result["method_diagnostics"]["visibility_source"] == "normalized_projected_AABB_paper_discard"


def test_alpha_mse_only_config_does_not_require_aabb_bounds(tmp_path):
    def remove_bounds(data):
        for obj in data["baseline_metadata"]["objects"]:
            for gop in obj["gops"]:
                gop.pop("bounds")
    path = frozen_fixture(tmp_path, mutate_config=lambda c: c.update(methods=["VISIBILITY_MSE"]),
                          mutate_inputs=remove_bounds)
    result = load_experiment(path)["planner"].plan("VISIBILITY_MSE", "g0", 1000)
    assert result["allocation"]["feasible"]
    assert result["method_diagnostics"]["geometry_method"] is None
