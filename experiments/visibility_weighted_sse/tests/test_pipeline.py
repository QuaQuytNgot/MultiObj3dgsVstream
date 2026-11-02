"""CPU integration checks; all scene measurements below are synthetic fixtures."""
from __future__ import annotations

import csv
from pathlib import Path
import json
import subprocess
import sys

import pytest

from experiments.visibility_weighted_sse.benchmark import benchmark
from experiments.visibility_weighted_sse.common import read_json, write_json
from experiments.visibility_weighted_sse.runner import freeze_config, run, validate

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]


def frozen_fixture(destination, *, mutate_config=None, mutate_inputs=None):
    config = read_json(EXPERIMENT / "configs" / "fixture.json")
    config.setdefault("solver_limits", {})["discretization"] = "exact_bytes"
    data = {name: read_json(EXPERIMENT / "fixtures" / f"{name}.json")
            for name in ("trace", "profiles", "visibility")}
    if mutate_inputs:
        mutate_inputs(data)
    for name, value in data.items():
        write_json(destination / f"{name}.json", value)
        config["inputs"][name] = {"path": f"{name}.json"}
    if mutate_config:
        mutate_config(config)
    source, frozen = destination / "source.json", destination / "frozen.json"
    write_json(source, config)
    freeze_config(source, frozen)
    return frozen


def test_complete_pipeline_multiple_gops_budgets_and_physical_dependencies(tmp_path):
    config = frozen_fixture(tmp_path)
    status = validate(config)
    assert status["status"] == "ready"
    assert status["gop_ids"] == ["g0", "g1"]
    assert status["input_kind"] == "synthetic_fixture"
    assert status["renderer_invoked"] is False
    output = run(config, tmp_path / "results")
    assert output["quality_evaluation"] == "unavailable_separate_stage"
    assert output["policy"] == "VISIBILITY_WEIGHTED_SSE"
    assert [(r["gop_id"], r["budget_id"]) for r in output["results"]] == [
        (gop, budget) for gop in ("g0", "g1")
        for budget in ("infeasible", "base-only", "limited", "ample")]
    for row in output["results"]:
        assert row["perfect_viewport_prediction"] is True
        assert row["prediction_assumption"] == "ORACLE_CAMERA_TRACE"
        allocation = row["allocation"]
        assert allocation["base_bytes"] == 30
        if row["budget_id"] == "infeasible":
            assert allocation["solver_status"] == "infeasible_base"
            assert allocation["budget_feasible"] is False
            assert allocation["allocations"] == []
            continue
        assert allocation["solver_status"] == "optimal"
        assert allocation["budget_feasible"] is True
        selected = {r["instance_id"]: r for r in allocation["allocations"]}
        assert selected["instance-1"]["selected_quality"] == "Q0"
        assert allocation["total_selected_bytes"] <= allocation["available_budget_bytes"]
        assert sum(r["physical_byte_cost"] for r in selected.values()) == allocation["total_selected_bytes"]
        assert allocation["enhancement_bytes"] == allocation["total_selected_bytes"] - 30
        for item in selected.values():
            assert len(item["required_representations"]) == len(item["dependencies"]) + 1
            if item["selected_quality"] != "Q0":
                assert "Q0" in item["dependencies"]
        if row["budget_id"] == "base-only":
            assert {r["selected_quality"] for r in selected.values()} == {"Q0"}
        elif row["budget_id"] == "limited":
            assert selected["instance-0"]["selected_quality"] == "Q2"
            assert selected["instance-2"]["selected_quality"] == "Q1"
            assert allocation["total_selected_bytes"] == 58
        elif row["budget_id"] == "ample":
            assert selected["instance-0"]["selected_quality"] == "Q3"
            assert selected["instance-2"]["selected_quality"] == "Q3"
            assert allocation["total_selected_bytes"] == 110
    g0, g1 = output["results"][0], output["results"][4]
    assert g0["visibility"]["timestamps"] == [0., .5, 1., 1.5]
    assert g1["visibility"]["timestamps"] == [2., 2.25, 3.]
    assert g0["utilities"]["instance-0"]["weight"] == pytest.approx(2.3)
    assert g1["utilities"]["instance-0"]["weight"] == pytest.approx(2.275)
    assert g1["visibility"]["temporal_weights"] == [.125, .375, .5]
    for row in output["results"]:
        assert row["utilities"]["instance-1"]["weight"] == 0
        assert all(q["weighted_utility"] == 0 for q in row["utilities"]["instance-1"]["qualities"])


def test_deterministic_json_and_all_csv_exports_exclude_runtime_measurements(tmp_path):
    config = frozen_fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    run(config, first)
    run(config, second)
    for name in ("decisions.json", "decisions.csv", "allocations.csv", "visibility.csv", "utilities.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    decisions = read_json(first / "decisions.json")
    for row in decisions["results"]:
        diagnostics = row["allocation"]["diagnostics"]
        assert "solver_elapsed_ms" not in diagnostics
        assert "allocation_adapter_elapsed_ms" not in diagnostics
    with (first / "decisions.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    performance = read_json(first / "performance.json")
    assert performance["timings_excluded_from_deterministic_decisions"] is True
    assert performance["includes_new_visibility_generation"] is False
    assert performance["supports_realtime_visibility_claim"] is False
    for row in performance["records"]:
        for name in ("trace_loading_ms", "cached_visibility_acquisition_ms",
                     "visibility_aggregation_ms", "utility_construction_ms", "allocation_ms"):
            assert row[name] >= 0


@pytest.mark.parametrize("name", ["trace", "profiles", "visibility"])
def test_frozen_input_hash_tampering_is_rejected(tmp_path, name):
    config = frozen_fixture(tmp_path)
    with (tmp_path / f"{name}.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match=f"Frozen {name} hash mismatch"):
        validate(config)


def test_configuration_hash_tampering_is_rejected(tmp_path):
    config = frozen_fixture(tmp_path)
    value = read_json(config)
    value["budgets"][0]["bytes"] = 100000
    write_json(config, value)
    with pytest.raises(ValueError, match="hash changed"):
        validate(config)


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda d: d["visibility"].update(content_revision="other-content"), "revision"),
    (lambda d: d["visibility"]["compatibility"].update(proxy_representation="different-proxy"), "compatibility"),
    (lambda d: d["visibility"]["compatibility"].update(transform_convention="different-transform"), "compatibility"),
    (lambda d: d["profiles"]["objects"][0]["gops"][0]["denominator_metadata"].update(resolution=[1, 1]), "denominator_metadata"),
    (lambda d: d["visibility"]["provenance"].update(joint_quality_used=True), "quality evaluation"),
    (lambda d: d["visibility"]["gops"].pop(), "Missing cached visibility GOP"),
    (lambda d: d["profiles"]["objects"][0]["gops"].pop(), "coverage mismatch"),
    (lambda d: d["visibility"]["gops"][0]["samples"].pop(), "complete upcoming trace"),
])
def test_refrozen_but_incompatible_or_missing_metadata_is_rejected(tmp_path, mutate, message):
    config = frozen_fixture(tmp_path, mutate_inputs=mutate)
    with pytest.raises(ValueError, match=message):
        validate(config)


def test_requested_gops_must_follow_trace_order(tmp_path):
    config = frozen_fixture(tmp_path, mutate_config=lambda c: c.update(gop_ids=["g1", "g0"]))
    with pytest.raises(ValueError, match="trace order"):
        validate(config)


@pytest.mark.parametrize("field", ["proxy_representation", "transform_convention"])
def test_omitted_offline_identity_is_rejected(tmp_path, field):
    def omit(data):
        data["profiles"]["compatibility"].pop(field)
    config = frozen_fixture(tmp_path, mutate_inputs=omit)
    with pytest.raises(ValueError, match="proxy representation|transform convention"):
        validate(config)


def test_integer_trace_gops_match_string_profile_ids(tmp_path):
    def numeric_trace(data):
        from experiments.visibility_weighted_sse.trace import TraceCameraProvider
        for index, gop in enumerate(data["trace"]["gops"]):
            gop["gop_id"] = index
        for obj in data["profiles"]["objects"]:
            for index, gop in enumerate(obj["gops"]):
                gop["gop_id"] = str(index)
        provider = TraceCameraProvider(data["trace"])
        for index, gop in enumerate(data["visibility"]["gops"]):
            gop["gop_id"] = index
            for sample, pose in zip(gop["samples"], provider.get_trajectory(index)):
                sample["pose_identity"] = pose.identity
    config = frozen_fixture(tmp_path, mutate_inputs=numeric_trace,
                            mutate_config=lambda c: c.update(gop_ids=[0, 1]))
    output = run(config, tmp_path / "output")
    assert [r["gop_id"] for r in output["results"]] == ["0"]*4 + ["1"]*4


def test_single_gop_and_per_gop_budget_configuration(tmp_path):
    config = frozen_fixture(tmp_path, mutate_config=lambda c: c.update(
        gop_ids=["g1"], budgets=[{"budget_id": "per-gop", "per_gop": {"g1": 58}}]))
    output = run(config, tmp_path / "output")
    assert len(output["results"]) == 1
    row = output["results"][0]
    assert row["gop_id"] == "g1"
    assert row["allocation"]["available_budget_bytes"] == 58
    assert row["allocation"]["total_selected_bytes"] == 58


def test_cli_runs_with_gpu_renderer_and_network_imports_forbidden(tmp_path):
    config = frozen_fixture(tmp_path)
    output = tmp_path / "cli"
    code = r'''
import importlib.abc
import runpy
import sys
class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "cupy", "gsplat", "diff_gaussian_rasterization", "requests", "httpx"} or \
           fullname.startswith("src_final.multiobj3dgs.renderer"):
            raise AssertionError("CPU research planner imported forbidden module: " + fullname)
sys.meta_path.insert(0, ForbiddenImports())
sys.argv = ["experiments.visibility_weighted_sse", "run", "--config", sys.argv[1], "--output", sys.argv[2]]
try:
    runpy.run_module("experiments.visibility_weighted_sse", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0, exc.code
assert "torch" not in sys.modules
assert "diff_gaussian_rasterization" not in sys.modules
'''
    completed = subprocess.run([sys.executable, "-c", code, str(config), str(output)],
                               cwd=REPOSITORY, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert summary["decisions"] == 8
    assert (output / "decisions.json").is_file()


def test_cli_reports_missing_input_without_invoking_renderer(tmp_path):
    config = frozen_fixture(tmp_path)
    (tmp_path / "visibility.json").unlink()
    completed = subprocess.run([sys.executable, "-m", "experiments.visibility_weighted_sse",
                                "validate", "--config", str(config)],
                               cwd=REPOSITORY, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["status"] == "blocked_invalid_or_missing_input"
    assert failure["renderer_invoked"] is False
    assert "Missing visibility input" in failure["error"]


def test_cpu_scaling_diagnostics_are_explicitly_synthetic_and_separate_stages(tmp_path):
    config = frozen_fixture(tmp_path)
    output = tmp_path / "scaling.json"
    result = benchmark(config, output)
    assert read_json(output) == result
    assert result["object_counts"] == [3, 5, 8, 12]
    assert result["input_kind"] == "synthetic_scaling_clones"
    assert result["renderer_invoked"] is False
    assert result["new_visibility_generated"] is False
    assert result["empirical_paper_measurement"] is False
    assert result["supports_realtime_visibility_claim"] is False
    assert "cached_visibility_acquisition" in result["excluded_stages"]
    for row in result["records"]:
        assert row["allocation"]["solver_status"] == "optimal"
        assert len(row["allocation"]["allocations"]) == row["object_count"]
        assert row["allocation"]["total_selected_bytes"] <= row["available_budget_bytes"]
        for stage in ("utility_construction", "allocation_adapter", "native_solver"):
            assert len(row[stage]["samples_ms"]) == result["repetitions_per_count"]
            assert row[stage]["median_ms"] >= 0
