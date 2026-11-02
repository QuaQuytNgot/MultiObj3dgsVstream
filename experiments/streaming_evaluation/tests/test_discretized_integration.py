"""CPU integration of actual GOP grids, bounded subsets and read-only inputs."""
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from experiments.adaptation_baselines.interface import AdaptationPlanner
from experiments.adaptation_baselines.methods import SOLO_SSE, VISIBILITY_WEIGHTED_SSE
from experiments.adaptation_baselines.tests.test_shared_allocator import payload
from experiments.streaming_evaluation import runtime_matrix, runtime_results
from experiments.streaming_evaluation.network import NetworkTrace
from experiments.streaming_evaluation.runtime_adapter import ReadOnlyVisibilityEngine
from experiments.streaming_evaluation.runtime_native import commit, physical_inputs
from experiments.streaming_evaluation.runtime_scenes import make_scene, make_trace
from experiments.streaming_evaluation.scenarios import generate_scenario
from experiments.streaming_evaluation.simulator import simulate
from experiments.streaming_evaluation.tests.test_runtime import cameras
from experiments.streaming_evaluation.timeline import Timeline
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json
from experiments.visibility_weighted_sse.profiles import profiles_from_dict
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter


def one_second_planner(timeline):
    data = payload(counts=(4,))
    for q, cost in enumerate((100000, 131250, 200000, 250000)):
        data["resources"][f"r0_{q}"]["bytes"] = cost
    source = profiles_from_dict(data)
    scenario = generate_scenario({"kind": "long_line", "object_count": 1,
        "samples_per_gop": 1,
        "source_objects": [{"object_id": "object0", "source_camera_id": "synthetic-camera"}],
        "camera": {"width": 100, "height": 100,
            "K": [[50., 0., 49.5], [0., 50., 49.5], [0., 0., 1.]]}}, timeline)
    trace = TraceCameraProvider(scenario["trace"])
    instance = scenario["instances"][0]["instance_id"]
    original = source.for_gop("g0")[0]
    profiles = replace(source, objects=MappingProxyType({
        (instance, segment.gop_id): replace(original, instance_id=instance, gop_id=segment.gop_id)
        for segment in timeline}))
    visibility = CachedVisibilityAdapter({
        "schema": "visibility-weighted-sse.visibility.v1", "trace_id": trace.trace_id,
        "content_revision": source.content_revision, "compatibility": dict(source.compatibility),
        "provenance": {"information_role": "VISIBILITY_PROXY", "candidate_independent": True,
            "joint_quality_used": False, "source": "synthetic CPU duration boundary fixture"},
        "gops": [{"gop_id": segment.gop_id, "samples": [
            {"timestamp": pose.timestamp, "pose_identity": pose.identity,
             "per_object": {instance: {"A_solo": 100., "A_visible": 100.}}}
            for pose in trace.get_trajectory(segment.gop_id)]} for segment in timeline]}, trace)
    # A stale configured duration tests that simulate uses actual source GOPs.
    planner = AdaptationPlanner(profiles, trace, visibility,
        solver_limits={"discretization": "bitrate", "delta_bps": 250000,
                       "gop_duration_seconds": 2.})
    return planner, scenario, instance


@pytest.mark.parametrize("source_duration", [1., 1.5])
def test_simulator_uses_actual_source_gop_duration_for_native_grid_and_fallback(source_duration):
    timeline = Timeline(source_duration, source_duration, 1.)
    planner, scenario, instance = one_second_planner(timeline)
    config = dict(experiment_id="synthetic-duration-correctness", input_kind="synthetic_fixture",
        adaptation=dict(methods=[SOLO_SSE], safety_factor=1.),
        network=dict(estimator=dict(startup_bps=1050000., window=5, stale_s=30.)),
        playback=dict(startup_buffer_s=1., max_buffer_s=3.), max_wall_time_s=10.,
        source_gop_ids=["g0"]*len(timeline), quality=dict(mode="trace_samples"))
    sim, performance = simulate(config, timeline, scenario,
        NetworkTrace([(0., 1.05)], duration_s=10.), planner, SOLO_SSE)
    assert sim["status"] == "completed"
    first = sim["decisions"][0]
    assert first["selected_quality"] == {instance: "Q1"}
    assert first["total_required_bytes"] == first["budget_bytes"] == 131250
    assert first["allocation"]["diagnostics"]["gop_duration_seconds"] == 1.
    assert first["allocation"]["diagnostics"]["native_budget_states_max"] == 2
    assert performance["solver_resources"][0]["exact_byte_feasible"] is True
    assert sim["playback"]["played_duration_s"] == source_duration
    if source_duration == 1.5:
        last = sim["decisions"][1]
        assert last["allocation"]["diagnostics"]["gop_duration_seconds"] == .5
        assert last["delivery_policy"] == "q0_fallback"
        assert last["selected_quality"] == {instance: "Q0"}
        assert last["delivery_allocation"]["diagnostics"]["gop_duration_seconds"] == .5
        assert last["delivery_allocation"]["diagnostics"]["delta_bps"] == 250000


@pytest.fixture
def readonly_visibility(tmp_path):
    source = tmp_path/"historical"
    commit(source/"manifest.json", {"schema": "synthetic-native-source", "fixture": True})
    scene = make_scene("camera_rotation", 2, cameras(), width=16, height=12)
    trace = make_trace(scene, Timeline(6, 8, 2), "synthetic-content")
    pose = trace.get_trajectory("0")[0]
    fingerprint = digest(read_json(source/"manifest.json"))
    identity = dict(schema="streaming-native-job.v1", fingerprint=fingerprint, kind="proxy",
        **physical_inputs(pose, 0), allocation={row["instance_id"]: 40 for row in pose.instances})
    values = {row["instance_id"]: {"A_solo": 20., "A_visible": 10.} for row in pose.instances}
    path = source/"cache/visibility"/f"{digest(identity)}.json"
    commit(path, dict(identity=identity, payload=values, payload_sha256=digest(values),
                      analysis_ms=3.5, native_wall_ms=4.5))
    engine = ReadOnlyVisibilityEngine(source, SimpleNamespace(check=lambda: None))
    return source, path, trace, engine


def test_readonly_visibility_reuses_only_exact_physical_loop_identity_without_writes(readonly_visibility):
    source, _, trace, engine = readonly_visibility
    before = {str(path.relative_to(source)): file_digest(path) for path in source.rglob("*") if path.is_file()}
    first, first_timing = engine.visibility(trace.get_trajectory("0")[0], 0)
    repeated, repeated_timing = engine.visibility(trace.get_trajectory("3")[0], 0)
    assert first == repeated and engine.stats["visibility_cache_hits"] == 2
    assert first_timing["cache_hit"] is True and repeated_timing["cache_hit"] is True
    assert first_timing["analysis_ms"] == first_timing["native_wall_ms"] == 0.
    assert first_timing["original_analysis_ms"] == 3.5
    assert first_timing["original_native_wall_ms"] == 4.5
    assert before == {str(path.relative_to(source)): file_digest(path) for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("corrupt", ["payload", "identity"])
def test_readonly_visibility_rejects_corrupt_record_without_rendering(readonly_visibility, corrupt):
    _, path, trace, engine = readonly_visibility
    record = read_json(path)
    if corrupt == "payload":
        next(iter(record["payload"].values()))["A_visible"] += 1
    else:
        record["identity"]["allocation"][next(iter(record["identity"]["allocation"]))] = 28
    commit(path, record)
    with pytest.raises(ValueError, match="identity/payload mismatch"):
        engine.visibility(trace.get_trajectory("0")[0], 0)
    assert engine.stats["visibility_cache_hits"] == 0


def test_readonly_visibility_rejects_changed_transform_or_source_frame(readonly_visibility):
    _, _, trace, engine = readonly_visibility
    pose = trace.get_trajectory("0")[0]
    changed = [deepcopy(dict(instance)) for instance in pose.instances]
    changed[0]["object_to_world"][0][3] += .25
    with pytest.raises(ValueError, match="Exact read-only visibility input missing"):
        engine.visibility(replace(pose, instances=tuple(changed)), 0)
    with pytest.raises(ValueError, match="Exact read-only visibility input missing"):
        engine.visibility(pose, 1)
    assert engine.stats["visibility_cache_hits"] == 0


def test_runtime_matrix_honors_configured_method_subset_in_counts_smoke_and_runs(tmp_path, monkeypatch):
    cfg = read_json(runtime_matrix.DEFAULT_CONFIG)
    cfg.update(methods=[SOLO_SSE, VISIBILITY_WEIGHTED_SSE], object_counts=[5],
               scenarios=["long_line"], playback_durations_s=[10])
    path = tmp_path/"subset.json"
    commit(path, cfg)
    loaded = runtime_matrix.load_settings(path)
    runner = runtime_matrix.MatrixRunner(tmp_path/"result", loaded)
    assert runner.state["total_runs"] == 2
    calls = []
    monkeypatch.setattr(runtime_matrix, "validate_inputs", lambda cfg: (None, None, {}, cameras(), {}, None, {}))
    monkeypatch.setattr(runtime_matrix, "resolve_execution", lambda *args: ("synthetic", "synthetic"))
    monkeypatch.setattr(runtime_matrix, "NativeEngine", lambda *args, **kwargs: SimpleNamespace(stats={}, close=lambda: None))
    monkeypatch.setattr(runtime_results, "make_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "progress", lambda **kwargs: None)
    monkeypatch.setattr(runner, "execute", lambda p, m, s, d, method, n, **kwargs:
        calls.append((method, kwargs.get("smoke", False))))
    assert runner.run() == 0
    assert calls == [(SOLO_SSE, True), (VISIBILITY_WEIGHTED_SSE, True),
                     (SOLO_SSE, False), (VISIBILITY_WEIGHTED_SSE, False)]
