"""Tiny synthetic end-to-end contracts; never experimental measurements.

Source data are Task 1's hand-authored three-object, two-GOP fixture. All files
and exports created here live in pytest temporary directories. The actual common
planner and native solver run, but no media, measured trace, or renderer is used.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.adaptation_baselines.methods import METHODS, SOLO_SSE, VISIBILITY_WEIGHTED_SSE
from experiments.adaptation_baselines.runner import load_experiment
from experiments.adaptation_baselines.tests.test_pipeline import frozen_fixture
from experiments.streaming_evaluation.config import freeze_config, load_config
from experiments.streaming_evaluation.inputs import build_planner
from experiments.streaming_evaluation.scenarios import generate_scenario
from experiments.streaming_evaluation.simulator import run, simulate, validate
from experiments.streaming_evaluation.timeline import Timeline
from experiments.visibility_weighted_sse.common import digest, read_json, write_json
from experiments.visibility_weighted_sse.trace import TraceCameraProvider


def synthetic_config(directory, *, modify=None, freeze=True):
    """Rebind frozen synthetic profiles to a small global looping scene."""
    directory.mkdir(parents=True, exist_ok=True)
    source_path = frozen_fixture(directory / "source")
    source_planner = load_experiment(source_path)["planner"]
    config = {
        "schema": "streaming-evaluation.config.v1", "experiment_id": "synthetic-correctness-only",
        "input_kind": "synthetic_fixture", "prediction_assumption": "ORACLE_CAMERA_TRACE",
        "timeline": {"source_duration_s": 4., "total_duration_s": 12., "gop_duration_s": 2.},
        "source_gop_ids": ["g0", "g1"],
        "scenario": {"kind": "long_line", "object_count": 3, "samples_per_gop": 2,
            "source_objects": [{"object_id": f"object-{i}", "source_camera_id": "synthetic-camera"}
                               for i in range(3)],
            "camera": {"width": 64, "height": 48,
                       "K": [[50., 0., 31.5], [0., 50., 23.5], [0., 0., 1.]]}},
        "network": {"points": [[0., .002]], "duration_s": 100., "end_policy": "stop",
                    "estimator": {"startup_bps": 2_000., "window": 5, "stale_s": 30.}},
        "adaptation": {"methods": list(METHODS), "safety_factor": 1.},
        "playback": {"startup_buffer_s": 2., "max_buffer_s": 4.},
        "quality": {"mode": "trace_samples"}, "max_wall_time_s": 100.,
        "inputs": {"adaptation_config": {"path": "source/frozen.json"},
                   "visibility_cache": {"path": "global_visibility.json"}},
    }
    if modify:
        modify(config)
    timeline = Timeline(**config["timeline"])
    scenario = generate_scenario(config["scenario"], timeline)
    provider = TraceCameraProvider(scenario["trace"])
    visibility = deepcopy(source_planner.visibility.data)
    visibility.update(input_kind="synthetic_fixture", trace_id=provider.trace_id, gops=[])
    visibility["provenance"] = {"information_role": "VISIBILITY_PROXY", "candidate_independent": True,
                                "joint_quality_used": False,
                                "source": "hand-authored synthetic correctness fixture, no image measurements"}
    for segment in timeline:
        samples = []
        for pose in provider.get_trajectory(segment.gop_id):
            samples.append({"timestamp": pose.timestamp, "pose_identity": pose.identity,
                "per_object": {row["instance_id"]: {"A_solo": 100., "A_visible": 10.+index*20.}
                               for index, row in enumerate(pose.instances)}})
        visibility["gops"].append({"gop_id": segment.gop_id, "samples": samples})
    write_json(directory / "global_visibility.json", visibility)
    draft, frozen = directory / "draft.json", directory / "frozen.json"
    write_json(draft, config)
    if freeze:
        freeze_config(draft, frozen)
        return frozen
    return draft


def execute_fixture(path, method=SOLO_SSE):
    config, timeline, scenario, trace = load_config(path)
    planner, timing = build_planner(config, path, timeline, scenario)
    planner.initialize_solver()
    return simulate(config, timeline, scenario, trace, planner, method, input_timings=timing)


@pytest.mark.parametrize("method", METHODS)
def test_all_five_methods_use_real_common_planner_and_identical_loop_inputs(tmp_path, method):
    path = synthetic_config(tmp_path)
    simulation, performance = execute_fixture(path, method)
    assert simulation["status"] == "completed"
    assert simulation["method"] == method
    assert simulation["future_bandwidth_available_to_policy"] is False
    assert simulation["metrics"]["adaptation"]["decision_count"] == 6
    assert simulation["metrics"]["adaptation"]["infeasible_decision_count"] == 0
    assert simulation["metrics"]["adaptation"]["budget_violation_count"] == 0
    assert [d["gop_id"] for d in simulation["decisions"]] == [str(i) for i in range(6)]
    assert [d["source_gop_id"] for d in simulation["decisions"]] == ["g0", "g1"]*3
    assert [d["loop_index"] for d in simulation["decisions"]] == [0, 0, 1, 1, 2, 2]
    for decision in simulation["decisions"]:
        assert decision["allocation"]["diagnostics"]["algorithm"] == "src_final.ProposedDPSolver"
        assert set(decision["selected_quality"]) == {"object-0000", "object-0001", "object-0002"}
        assert decision["total_required_bytes"] <= decision["budget_bytes"]
        assert decision["allocation"]["status"] == "optimal"
        assert "timings" not in decision
    assert len(performance["timings"]) == 6
    assert all(row["decision_call_ms"] >= 0 and row["native_solver_ms"] >= 0 for row in performance["timings"])
    assert all(row["visibility_analysis_ms"] is None for row in performance["timings"])
    assert performance["input_timings"]["cached_visibility_acquisition_ms"] >= 0
    assert performance["input_timings"]["visibility_analysis_ms"] is None
    assert performance["input_timings"]["visibility_analysis_scope"] == "not_executed_cached_outputs_only"
    assert len(performance["records"]) == 6
    assert all(row["cached_visibility_acquisition_ms"] >= 0 for row in performance["records"])
    quality = simulation["metrics"]["quality"]
    assert quality["average_frame_psnr_db"] is None
    assert quality["pooled_psnr_db"] is None
    assert quality["measured_frame_count"] == 0
    assert quality["unavailable_frame_count"] == quality["frame_count"] == 12


def test_repeated_resources_pay_again_and_estimator_and_playback_never_reset(tmp_path):
    simulation, _ = execute_fixture(synthetic_config(tmp_path), VISIBILITY_WEIGHTED_SSE)
    requests = simulation["requests"]
    assert len({row["request_id"] for row in requests}) == len(requests)
    for source_gop in ("g0", "g1"):
        groups = [{row["resource_sha256"] for row in requests
                   if row["source_gop_id"] == source_gop and row["loop_index"] == loop} for loop in range(3)]
        assert groups[0] and groups[0] == groups[1] == groups[2]
    bandwidth = simulation["metrics"]["bandwidth"]
    expected = sum(decision["total_required_bytes"] for decision in simulation["decisions"])
    assert bandwidth["total_requested_bytes"] == bandwidth["total_delivered_bytes"] == expected
    assert expected > sum(d["total_required_bytes"] for d in simulation["decisions"][:2])
    assert bandwidth["total_delivered_bytes"] <= simulation["available_capacity_bytes"]
    assert bandwidth["physical_base_delivered_bytes"] + bandwidth["physical_enhancement_delivered_bytes"] == expected
    assert bandwidth["base_requested_bytes"] + bandwidth["enhancement_requested_bytes"] == expected
    decisions = simulation["decisions"]
    assert [row["bandwidth_estimate"]["sample_count"] for row in decisions] == [0, 1, 2, 3, 4, 5]
    assert len(simulation["bandwidth_estimator_samples"]) == 6
    assert all(right["decision_wall_s"] > left["decision_wall_s"] for left, right in zip(decisions, decisions[1:]))
    assert all(right["wall_time_s"] >= left["wall_time_s"]
               and right["media_time_s"] >= left["media_time_s"]
               for left, right in zip(simulation["buffer_samples"], simulation["buffer_samples"][1:]))
    assert sum(event["kind"] == "startup_complete" for event in simulation["playback_events"]) == 1
    playback = simulation["metrics"]["playback"]
    assert playback["played_duration_s"] == 12
    assert playback["completion_wall_time_s"] == pytest.approx(12+playback["startup_delay_s"]+playback["total_rebuffering_s"])
    assert playback["final_buffer_s"] == 0
    assert max(row["buffer_s"] for row in simulation["buffer_samples"]) <= 4.+1e-9


def test_independent_repeated_instances_do_not_share_transfer_sha(tmp_path):
    def clone_one(config):
        config["scenario"]["object_count"] = 4
        config["adaptation"]["resource_sharing"] = "per_instance"

    path = synthetic_config(tmp_path, modify=clone_one)
    config, timeline, scenario, trace = load_config(path)
    planner, timing = build_planner(config, path, timeline, scenario)
    planner.initialize_solver()
    simulation, _ = simulate(config, timeline, scenario, trace, planner, SOLO_SSE,
                             input_timings=timing)

    rows = {row.instance_id: row for row in planner.profiles.for_gop("0")}
    original, clone = rows["object-0000"], rows["object-0003"]
    aliases = planner.profiles.provenance["resource_transfer_aliases"]
    source_hashes = lambda iid: {entry["source_sha256"] for entry in aliases.values()
                                 if entry["instance_id"] == iid}
    transfer_hashes = lambda iid: {transfer_sha for transfer_sha, entry in aliases.items()
                                   if entry["instance_id"] == iid}
    assert source_hashes(original.instance_id) & source_hashes(clone.instance_id)
    assert not transfer_hashes(original.instance_id) & transfer_hashes(clone.instance_id)
    assert simulation["resource_sharing_policy"] == "independent_instance_no_inter_instance_payload_sharing"
    assert simulation["transfer_resource_count"] > 0
    assert simulation["transfer_map_sha256"]
    assert simulation["metrics"]["bandwidth"]["total_requested_bytes"] == sum(
        row["total_required_bytes"] for row in simulation["decisions"])
    assert len({row["request_id"] for row in simulation["requests"]}) == len(simulation["requests"])


def test_future_trace_changes_cannot_change_earlier_decisions(tmp_path):
    def tail(rate):
        def modify(config):
            config["network"]["points"] = [[0., .002], [3., rate]]
        return modify
    first, _ = execute_fixture(synthetic_config(tmp_path / "first", modify=tail(.001)))
    second, _ = execute_fixture(synthetic_config(tmp_path / "second", modify=tail(.004)))
    prefix_first = [row for row in first["decisions"] if row["decision_wall_s"] < 1.]
    prefix_second = [row for row in second["decisions"] if row["decision_wall_s"] < 1.]
    assert len(prefix_first) >= 2
    assert prefix_first == prefix_second
    assert prefix_first[0]["estimated_bandwidth_bps"] == 2000
    assert prefix_first[0]["bandwidth_estimate"]["sample_count"] == 0


def test_stalls_follow_continuous_wall_clock_and_satisfy_playback_identity(tmp_path):
    def change(config):
        config["network"]["points"] = [[0., .002], [.7, 0.], [8., .002]]
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=change))
    playback = simulation["metrics"]["playback"]
    assert simulation["status"] == "completed"
    assert playback["stall_count"] >= 1
    assert playback["total_rebuffering_s"] > 0
    assert playback["completion_wall_time_s"] == pytest.approx(12+playback["startup_delay_s"]+playback["total_rebuffering_s"])
    assert sum(row["duration_s"] for row in simulation["stalls"]) == pytest.approx(playback["total_rebuffering_s"])
    assert simulation["metrics"]["bandwidth"]["bandwidth_utilization"] <= 1.


def test_short_final_interval_still_downloads_whole_physical_source_segment(tmp_path):
    def trim(config):
        config["timeline"]["total_duration_s"] = 4.1
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=trim))
    assert simulation["status"] == "completed"
    assert len(simulation["decisions"]) == 3
    final = simulation["segment_downloads"][-1]
    first = simulation["segment_downloads"][0]
    assert final["duration_s"] == .1
    assert final["full_source_segment_duration_s"] == 2.
    assert final["requested_bytes"] == first["requested_bytes"] > 0
    assert simulation["playback"]["played_duration_s"] == 4.1
    assert simulation["decisions"][-1]["budget_bytes"] > 30


@pytest.mark.parametrize("method", METHODS)
def test_base_infeasible_uses_actual_q0_delivery_and_stalls_across_loops(tmp_path, method):
    def insufficient(config):
        config["network"]["estimator"]["startup_bps"] = 1.
        config["network"]["points"] = [[0., .00008]]  # 10 bytes/s, 20-byte estimated GOP budget
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=insufficient), method)
    assert simulation["status"] == "completed"
    assert len(simulation["decisions"]) == 6
    for decision in simulation["decisions"]:
        assert decision["allocation"]["status"] == "infeasible_base"
        assert not decision["allocation"]["feasible"]
        assert decision["allocation"]["allocations"] == []
        assert decision["delivery_policy"] == "q0_fallback"
        assert not decision["delivery_allocation"]["budget_feasible"]
        assert not decision["delivery_allocation"]["feasible"]
        assert decision["delivery_allocation"]["solver_status"] == "not_a_solver_solution"
        assert set(decision["selected_quality"].values()) == {"Q0"}
        assert decision["total_required_bytes"] == 30
        assert decision["budget_overrun_bytes"] == 30-decision["budget_bytes"] > 0
    requests = simulation["requests"]
    assert len(requests) == len({r["request_id"] for r in requests}) == 18
    assert {r["kind"] for r in requests} == {"base"}
    assert all(r["requested_bytes"] == r["delivered_bytes"] == 10 for r in requests)
    metrics = simulation["metrics"]
    assert metrics["bandwidth"]["total_requested_bytes"] == 180
    assert metrics["bandwidth"]["total_delivered_bytes"] == 180
    assert metrics["playback"]["startup_delay_s"] == pytest.approx(3.)
    assert metrics["playback"]["stall_count"] > 0
    assert metrics["playback"]["total_rebuffering_s"] > 0
    assert metrics["playback"]["played_duration_s"] == 12
    assert metrics["adaptation"]["infeasible_decision_count"] == 6
    assert metrics["adaptation"]["q0_fallback_count"] == 6
    assert metrics["adaptation"]["budget_violation_count"] == 6
    assert not metrics["adaptation"]["all_selected_budgets_feasible"]
    assert metrics["adaptation"]["total_budget_overrun_bytes"] == sum(
        d["budget_overrun_bytes"] for d in simulation["decisions"])
    assert metrics["quality"]["average_frame_psnr_db"] is None


def test_q0_fallback_low_estimate_recovers_from_completed_downloads(tmp_path):
    def low_start(config):
        config["network"]["estimator"]["startup_bps"] = 100.
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=low_start))
    assert simulation["status"] == "completed"
    assert simulation["decisions"][0]["budget_overrun_bytes"] == 5
    assert simulation["metrics"]["adaptation"]["q0_fallback_count"] == 1
    assert all(d["delivery_policy"] == "solver_selection" for d in simulation["decisions"][1:])


def test_q0_fallback_partial_network_delivery_does_not_become_playable(tmp_path):
    def exhausted(config):
        config["network"]["estimator"]["startup_bps"] = 1.
        config["network"]["duration_s"] = .03
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=exhausted))
    assert simulation["status"] == "network_trace_exhausted"
    assert simulation["metrics"]["adaptation"]["q0_fallback_count"] == 1
    assert simulation["metrics"]["bandwidth"]["total_requested_bytes"] == 30
    assert simulation["metrics"]["bandwidth"]["total_delivered_bytes"] == pytest.approx(7.5)
    assert simulation["playback"]["played_duration_s"] == 0
    assert simulation["bandwidth_estimator_samples"] == []


def test_partial_trace_queues_full_request_cost_and_never_observes_incomplete_gop(tmp_path):
    def exhausted(config):
        config["network"]["duration_s"] = .03
        config["network"]["max_concurrent_streams"] = 1  # exercise unadmitted cancellation
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=exhausted))
    assert simulation["status"] == "network_trace_exhausted"
    assert simulation["bandwidth_estimator_samples"] == []
    bandwidth = simulation["metrics"]["bandwidth"]
    assert bandwidth["total_requested_bytes"] == simulation["decisions"][0]["total_required_bytes"]
    assert bandwidth["total_delivered_bytes"] == pytest.approx(7.5)
    assert bandwidth["total_delivered_bytes"] < bandwidth["total_requested_bytes"]
    assert bandwidth["bandwidth_utilization"] == pytest.approx(1.)
    assert any(row["status"] == "cancelled_before_service" for row in simulation["requests"])
    assert simulation["metrics"]["adaptation"]["selected_quality"][0]["download_completed"] is False
    assert simulation["metrics"]["playback"]["completion_wall_time_s"] is None


def test_all_policy_exports_are_deterministic_and_runtime_is_separate(tmp_path):
    path = synthetic_config(tmp_path / "inputs")
    first, second = tmp_path / "first", tmp_path / "second"
    first_results, second_results = run(path, first), run(path, second)
    assert first_results == second_results
    assert (first / "comparison.json").read_bytes() == (second / "comparison.json").read_bytes()
    for method in METHODS:
        for name in ("simulation.json", "metrics.json", "requests.csv", "buffer.csv", "decisions.csv", "frames.csv"):
            assert (first / method / name).read_bytes() == (second / method / name).read_bytes()
        simulation = read_json(first / method / "simulation.json")
        performance = read_json(first / method / "performance.json")
        assert performance["information_role"] == "OBSERVED_RUNTIME_ONLY_NOT_SIMULATED_TIME"
        assert performance["shared_solver_initialization_ms"] >= 0
        assert len(performance["timings"]) == len(simulation["decisions"]) == 6
        assert "decision_call_ms" not in json.dumps(simulation)
        assert "native_solver_ms" not in json.dumps(simulation)


@pytest.mark.parametrize("kind,count", [("rotation_orbit", 1), ("long_line", 4), ("random_spatial", 5)])
def test_scenario_family_object_count_and_quality_masks_are_configurable(tmp_path, kind, count):
    def configure(config):
        config["scenario"].update(kind=kind, object_count=count, seed=7)
        config["timeline"]["total_duration_s"] = 2.
        config["adaptation"]["quality_levels"] = {f"object-{i:04d}": ["Q0", "Q2"] for i in range(count)}
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=configure))
    assert simulation["status"] == "completed"
    decision = simulation["decisions"][0]
    assert len(decision["selected_quality"]) == count
    assert set(decision["selected_quality"].values()) <= {"Q0", "Q2"}


def test_validate_and_cli_use_only_synthetic_metadata_and_no_gpu_or_network_modules(tmp_path):
    path = synthetic_config(tmp_path)
    ready = validate(path)
    assert ready["status"] == "ready_for_future_execution"
    assert ready["experimental_measurements_executed"] is False
    assert ready["methods"] == list(METHODS)
    script = """
import sys
from experiments.streaming_evaluation.__main__ import main
code = main(['validate', '--config', sys.argv[1]])
assert code == 0
assert 'torch' not in sys.modules
assert 'gsplat' not in sys.modules
assert 'requests' not in sys.modules
assert 'socket' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready_for_future_execution"


def test_missing_inputs_report_preparation_status_without_execution(tmp_path):
    draft = synthetic_config(tmp_path, freeze=False)
    data = read_json(draft)
    data["inputs"].pop("visibility_cache")
    write_json(draft, data)
    result = validate(draft)
    assert result["status"] == "geometry_ready_inputs_missing"
    assert result["missing_inputs"] == ["visibility_cache"]
    assert result["experimental_measurements_executed"] is False


@pytest.mark.parametrize("mutation,match", [
    ("configuration", "Configuration is not frozen"),
    ("visibility", "Frozen input hash mismatch"),
    ("source_config", "Frozen input hash mismatch"),
    ("missing_file", "Missing input visibility_cache"),
    ("source_profile", "hash mismatch"),
])
def test_frozen_artifacts_are_checked_before_any_simulation(tmp_path, mutation, match):
    path = synthetic_config(tmp_path)
    if mutation == "configuration":
        config = read_json(path)
        config["playback"]["max_buffer_s"] = 6.
        write_json(path, config)
    elif mutation == "visibility":
        cache = read_json(tmp_path / "global_visibility.json")
        cache["gops"][0]["samples"][0]["per_object"]["object-0000"]["A_visible"] += 1.
        write_json(tmp_path / "global_visibility.json", cache)
    elif mutation == "source_config":
        source = read_json(tmp_path / "source/frozen.json")
        source["experiment_id"] = "changed"
        write_json(tmp_path / "source/frozen.json", source)
    elif mutation == "missing_file":
        (tmp_path / "global_visibility.json").unlink()
    else:
        profiles = read_json(tmp_path / "source/profiles.json")
        profiles["content_revision"] = "tampered"
        write_json(tmp_path / "source/profiles.json", profiles)
    with pytest.raises(ValueError, match=match):
        # Validation reports missing preparations as a status; the execution
        # boundary must reject the same absent file before a simulation starts.
        (load_config if mutation == "missing_file" else validate)(path)


@pytest.mark.parametrize("mutation,match", [
    ("pose", "identity mismatch"),
    ("candidate_quality", "Candidate joint RGB"),
    ("wrong_kind", "input kind mismatch"),
    ("missing_gop", "Missing cached visibility GOP"),
])
def test_rehashed_but_invalid_visibility_is_rejected_by_existing_adapter(tmp_path, mutation, match):
    path = synthetic_config(tmp_path)
    cache_path = tmp_path / "global_visibility.json"
    cache = read_json(cache_path)
    if mutation == "pose":
        cache["gops"][0]["samples"][0]["pose_identity"] = "incorrect-pose"
    elif mutation == "candidate_quality":
        cache["provenance"]["joint_quality_used"] = True
    elif mutation == "wrong_kind":
        cache["input_kind"] = "validated_cached_artifacts"
    else:
        cache["gops"].pop()
    write_json(cache_path, cache)
    freeze_config(tmp_path / "draft.json", path)
    with pytest.raises(ValueError, match=match):
        validate(path)


def test_rehashed_cached_video_credit_is_forbidden(tmp_path):
    path = synthetic_config(tmp_path)
    config = read_json(path)
    config["adaptation"]["cached_resource_ids"] = ["not-a-permitted-cache"]
    config["configuration_hash"] = digest({key: value for key, value in config.items() if key != "configuration_hash"})
    write_json(path, config)
    with pytest.raises(ValueError, match="caching is disabled"):
        load_config(path)


def test_fractional_startup_threshold_requires_whole_gop_prefix_capacity(tmp_path):
    def impossible_startup(config):
        # Every 2s GOP fits, but reaching a 3s startup threshold needs two GOPs.
        config["playback"] = {"startup_buffer_s": 3., "max_buffer_s": 3.}
    with pytest.raises(ValueError, match="Whole-GOP startup prefix"):
        synthetic_config(tmp_path, modify=impossible_startup)


def test_network_trace_file_alias_is_canonicalized_rebased_and_hash_bound(tmp_path):
    def file_trace(config):
        path = tmp_path / "synthetic_network.csv"
        path.write_text("time_seconds,bandwidth_mbps\n0,0.002\n1,0.003\n")
        config["network"].pop("points")
        config["network"]["trace_file"] = path.name
    original = synthetic_config(tmp_path, modify=file_trace)
    assert "trace_file" not in read_json(original)["network"]
    second = tmp_path / "elsewhere/frozen.json"
    freeze_config(tmp_path / "draft.json", second)
    frozen = read_json(second)
    assert frozen["network"]["path"] == "../synthetic_network.csv"
    assert len(frozen["network"]["sha256"]) == 64
    _, _, _, trace = load_config(second)
    assert trace.rate_bps(0) == 2000
    assert trace.rate_bps(1) == 3000
    (tmp_path / "synthetic_network.csv").write_text("time_seconds,bandwidth_mbps\n0,0.004\n")
    with pytest.raises(ValueError, match="Frozen network trace hash mismatch"):
        load_config(second)


@pytest.mark.parametrize("update,match", [
    ({"request_latency_s": -1.}, "request_latency_s"),
    ({"estimator": {"startup_bps": 0.}}, "startup_bps"),
    ({"estimator": {"window": 0}}, "window"),
    ({"estimator": {"stale_s": 0.}}, "stale_s"),
])
def test_invalid_network_rules_rejected_before_execution(tmp_path, update, match):
    def invalid(config):
        config["network"].update(update)
    with pytest.raises(ValueError, match=match):
        synthetic_config(tmp_path, modify=invalid)


@pytest.mark.parametrize("quality,match", [
    ({"mode": "utility_as_psnr"}, "quality sampling mode"),
    ({"mode": "frame_grid", "fps": 0.}, "quality fps"),
    ({"mode": "frame_grid", "fps": -1.}, "quality fps"),
])
def test_invalid_quality_sampling_rejected_before_execution(tmp_path, quality, match):
    with pytest.raises(ValueError, match=match):
        synthetic_config(tmp_path, modify=lambda config: config.update(quality=quality))


@pytest.mark.parametrize("mask,match", [
    (["Q0"], "must map scenario instance"),
    ({"object-0000": ["Q0"]}, "instance coverage mismatch"),
    ({f"object-{i:04d}": ["Q0", "unknown"] for i in range(3)}, "Unknown or duplicate"),
    ({f"object-{i:04d}": ["Q0", "Q0"] for i in range(3)}, "Unknown or duplicate"),
    ({f"object-{i:04d}": ["Q2"] for i in range(3)}, "mandatory base"),
])
def test_quality_masks_are_validated_for_every_scenario_instance(tmp_path, mask, match):
    def invalid(config):
        config["adaptation"]["quality_levels"] = mask
    path = synthetic_config(tmp_path, modify=invalid)
    with pytest.raises(ValueError, match=match):
        validate(path)


def test_prepared_experiment_cannot_use_synthetic_source_metadata(tmp_path):
    path = synthetic_config(tmp_path, modify=lambda config: config.update(input_kind="prepared_experiment"))
    with pytest.raises(ValueError, match="Source adaptation metadata input kind mismatch"):
        validate(path)


@pytest.mark.parametrize("fps,expected", [(1., "ready_for_future_execution"), (2., "missing_pose")])
def test_validation_checks_full_frame_grid_pose_coverage(tmp_path, fps, expected):
    path = synthetic_config(tmp_path, modify=lambda config: config.update(quality={"mode": "frame_grid", "fps": fps}))
    if expected == "missing_pose":
        with pytest.raises(ValueError, match="Missing exact camera pose"):
            validate(path)
    else:
        assert validate(path)["status"] == expected


@pytest.mark.parametrize("valid", [False, True])
def test_validation_parses_optional_quality_cache_without_measuring(tmp_path, valid):
    path = synthetic_config(tmp_path)
    quality_path = tmp_path / "quality_cache.json"
    document = {"schema_version": 1, "information_role": "ACTUAL_VIEWPORT_FRAME_MEASUREMENTS",
                "measurement_kind": "joint_viewport_rgb", "reference_identity": "synthetic-reference",
                "protocol": {"mask_mode": "full_viewport", "channels": 3, "data_range": 1., "normalization": "raw"},
                "frames": []}
    if not valid:
        document["information_role"] = "OFFLINE_BASELINE_METADATA"
    write_json(quality_path, document)
    draft = read_json(tmp_path / "draft.json")
    draft["inputs"]["quality_measurements"] = {"path": quality_path.name}
    write_json(tmp_path / "draft.json", draft)
    freeze_config(tmp_path / "draft.json", path)
    if valid:
        assert validate(path)["experimental_measurements_executed"] is False
    else:
        with pytest.raises(ValueError, match="actual joint viewport"):
            validate(path)


def test_run_rejects_missing_frame_poses_before_any_virtual_transfer(tmp_path, monkeypatch):
    path = synthetic_config(tmp_path, modify=lambda config: config.update(
        quality={"mode": "frame_grid", "fps": 2.}))
    from experiments.streaming_evaluation.network import NetworkLink
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid frame coverage reached virtual network execution")
    monkeypatch.setattr(NetworkLink, "download", forbidden)
    with pytest.raises(ValueError, match="Missing exact camera pose"):
        run(path, tmp_path / "output", method=SOLO_SSE)


@pytest.mark.parametrize("family", ["standalone", "progressive"])
def test_q0_fallback_reuses_physical_union_with_shared_base_and_quality_masks(family):
    from types import SimpleNamespace
    from experiments.adaptation_baselines.tests.test_shared_allocator import payload, utilities
    from experiments.visibility_weighted_sse.profiles import profiles_from_dict
    from experiments.visibility_weighted_sse.allocation import allocate_utilities
    from experiments.streaming_evaluation.simulator import _q0_fallback
    data = payload(family)
    data["resources"]["r1_0"] = deepcopy(data["resources"]["r0_0"])
    profiles = profiles_from_dict(data)
    values = utilities(profiles, ((17, 20, 30, 40), (18, 20, 30, 40)))
    original = allocate_utilities(profiles, "g0", 9, values)
    before = deepcopy(original)
    planner = SimpleNamespace(profiles=profiles, allowed_quality_ids={"i0": ["Q0", "Q2"], "i1": ["Q0"]})
    decision = {"allocation": original, "utilities": values, "budget_bytes": 9, "timings": {}}
    delivery = _q0_fallback(decision, planner, "g0")
    assert original == before
    assert original["status"] == "infeasible_base" and not original["feasible"]
    assert decision["total_required_bytes"] == delivery["base_bytes"] == 10
    assert len(delivery["selected_resource_sha256"]) == 1
    assert sum(row["physical_byte_cost"] for row in delivery["allocations"]) == 10
    assert delivery["enhancement_bytes"] == 0
    assert decision["selected_quality"] == {"i0": "Q0", "i1": "Q0"}
    assert decision["estimated_total_utility"] == 35
    assert decision["budget_overrun_bytes"] == 1
    assert not delivery["feasible"] and not delivery["budget_feasible"]
    assert not any("elapsed" in key for key in delivery["diagnostics"])


def test_historical_aabb_mse_can_be_selected_explicitly_in_streaming_config(tmp_path):
    from experiments.adaptation_baselines.methods import AABB_MSE_HISTORICAL
    path = synthetic_config(tmp_path, modify=lambda c: c["adaptation"].update(methods=[AABB_MSE_HISTORICAL]))
    simulation, _ = execute_fixture(path, AABB_MSE_HISTORICAL)
    assert simulation["status"] == "completed"
    assert all(d["method_diagnostics"]["geometry_method"] == "paper_discard" for d in simulation["decisions"])
