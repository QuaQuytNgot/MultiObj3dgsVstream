"""Small synthetic correctness fixtures: no renderer, traces, or measurements."""
from copy import deepcopy
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from experiments.streaming_evaluation.scenarios import generate_scenario, SCENARIO_KINDS
from experiments.streaming_evaluation.timeline import Timeline, build_timeline
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from src_final.multiobj3dgs.shared.geometry import validate_transform
from src_final.multiobj3dgs.client.scene.timeline import media_time


def fixture_config(kind="rotation_orbit", count=3):
    return {"kind": kind, "object_count": count,
            "source_objects": [{"object_id": "a", "source_camera_id": "recorded-camera-a"},
                               {"object_id": "b", "source_camera_id": "recorded-camera-b"}],
            "samples_per_gop": 3, "seed": 52,
            "timeline": {"source_duration_s": 2.5, "total_duration_s": 6., "gop_duration_s": 1.}}


def test_loop_mapping_matches_existing_production_timeline():
    from types import SimpleNamespace
    timeline = Timeline(10, 30, 2)
    instance = SimpleNamespace(scene_start=0., active_duration=None, media_offset=0., loop=True)
    content = SimpleNamespace(duration=10., fps=30)
    for timestamp, expected in [(0, 0), (9.5, 9.5), (10, 0), (11, 1), (12, 2), (20, 0), (21, 1), (30, 0)]:
        assert timeline.content_time(timestamp) == expected
        assert timeline.content_time(timestamp) == media_time(instance, content, timestamp)


def test_source_gop_and_final_boundaries_split_without_cost_discount():
    timeline = Timeline(2.5, 6.2, 1)
    assert [s.global_start_s for s in timeline] == [0., 1., 2., 2.5, 3.5, 4.5, 5., 6.]
    assert [s.duration_s for s in timeline] == [1., 1., .5, 1., 1., .5, 1., .2]
    assert [s.source_gop_index for s in timeline] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert [s.loop_index for s in timeline] == [0, 0, 0, 1, 1, 1, 2, 2]
    assert timeline.segments[-1].source_segment_duration_s == 1.
    assert timeline.segments[-1].duration_s == .2
    assert timeline.segments[-1].content_start_s == 1.
    assert timeline.segments[-1].content_end_s == 1.2


def test_decimal_boundaries_do_not_introduce_extra_gops_or_drift():
    timeline = Timeline("0.3", "1.2", "0.1")
    assert len(timeline) == 12
    assert [row.global_start_s for row in timeline] == [i/10 for i in range(12)]
    assert [row.source_gop_index for row in timeline] == [0, 1, 2]*4
    assert [row.content_start_s for row in timeline] == [0., .1, .2]*4
    assert timeline.content_time_exact("0.9") == 0
    assert timeline.sample_times_exact(timeline.segments[3], 3) == (Fraction(3, 10), Fraction(1, 3), Fraction(11, 30))


def test_each_loop_uses_unique_request_and_global_adaptation_identity():
    timeline = Timeline(2, 6, 1)
    assert [row.gop_id for row in timeline] == [str(i) for i in range(6)]
    assert [row.source_gop_id for row in timeline] == ["0", "1"]*3
    assert len({row.request_id for row in timeline}) == 6
    assert timeline.segments[0].request_id != timeline.segments[2].request_id
    assert timeline.to_dict()["segment_cache_enabled"] is False


def test_half_open_lookup_and_content_endpoint():
    timeline = Timeline(2.5, 6, 1)
    assert timeline.segment_at(2.499).loop_index == 0
    assert timeline.segment_at(2.5).loop_index == 1
    assert timeline.segment_at("3.5").source_gop_index == 1
    assert timeline.content_time(6) == 1
    for timestamp in (-1, 6, 7):
        with pytest.raises(ValueError):
            timeline.segment_at(timestamp)
    with pytest.raises(ValueError):
        timeline.content_time(7)


@pytest.mark.parametrize("value", [0, -1, True, "nan", float("inf"), None])
@pytest.mark.parametrize("field", ["source_duration_s", "total_duration_s", "gop_duration_s"])
def test_rejects_invalid_durations(field, value):
    row = {"source_duration_s": 2., "total_duration_s": 4., "gop_duration_s": 1.}
    row[field] = value
    with pytest.raises(ValueError):
        build_timeline(row)


def test_gop_larger_than_source_and_source_larger_than_experiment():
    timeline = Timeline(1, 2.5, 3)
    assert [row.duration_s for row in timeline] == [1., 1., .5]
    assert [row.source_segment_duration_s for row in timeline] == [1., 1., 1.]
    short = Timeline(10, .2, 2)
    assert len(short) == 1
    assert short.segments[0].source_segment_duration_s == 2


@pytest.mark.parametrize("kind", SCENARIO_KINDS)
def test_three_scenarios_are_deterministic_and_trace_provider_compatible(kind):
    config = fixture_config(kind)
    original = deepcopy(config)
    scenario = generate_scenario(config)
    assert config == original
    assert scenario == generate_scenario(config)
    assert json.loads(json.dumps(scenario, allow_nan=False)) == scenario
    assert scenario["object_count"] == 3
    assert scenario["validity"]["primary_quality_eligible"] is False
    provider = TraceCameraProvider(scenario["trace"])
    for gop in provider.gops:
        assert len(provider.get_trajectory(gop["gop_id"])) == 3
        assert math.isclose(sum(provider.temporal_weights(gop["gop_id"])), 1.)
        for pose in provider.get_trajectory(gop["gop_id"]):
            assert {row["object_id"] for row in pose.instances} == {"a", "b"}
            assert len({row["instance_id"] for row in pose.instances}) == 3
            for instance in pose.instances:
                validate_transform(instance["object_to_world"])
                assert instance["source_state_binding"] == "UNBOUND_CONTENT_TIME_DESCRIPTOR"


def test_random_spatial_uses_local_seed_and_changes_with_seed():
    config = fixture_config("random_spatial", 5)
    np.random.seed(201)
    expected = np.random.random(4)
    np.random.seed(201)
    first = generate_scenario(config)
    assert np.array_equal(np.random.random(4), expected)
    second = generate_scenario({**config, "seed": config["seed"]+1})
    assert first["instances"] != second["instances"]
    assert first["scene_id"] != second["scene_id"]
    for instance in first["instances"]:
        position = np.array(instance["object_to_world"])[:3, 3]
        assert np.all(position >= [-4., 0., -4.])
        assert np.all(position < [4., 3., 4.])


def test_line_layout_uses_configured_origin_spacing_and_direction():
    config = fixture_config("long_line", 4)
    config["layout"] = {"origin": [1., 2., 3.], "direction": [0., 0., 2.], "spacing": 3.}
    scenario = generate_scenario(config)
    positions = np.array([np.array(row["object_to_world"])[:3, 3] for row in scenario["instances"]])
    assert np.allclose(positions, [[1, 2, -1.5], [1, 2, 1.5], [1, 2, 4.5], [1, 2, 7.5]])


def test_explicit_positions_and_positive_similarity_transforms():
    config = fixture_config(count=2)
    config["instances"] = [{"instance_id": "x", "object_id": "a", "position": [1, 2, 3], "scale": 2.,
                            "rotation": [math.sqrt(.5), 0, math.sqrt(.5), 0]},
                           {"instance_id": "y", "object_id": "b", "object_to_world": np.eye(4).tolist()}]
    scenario = generate_scenario(config)
    first = np.array(scenario["instances"][0]["object_to_world"])
    assert np.allclose(first[:3, 3], [1, 2, 3])
    assert np.allclose(first[:3, :3].T@first[:3, :3], np.eye(3)*4)
    config = fixture_config(count=2)
    config["positions"] = [[-1., 2., 3.], [4., 5., 6.]]
    scenario = generate_scenario(config)
    assert [np.array(row["object_to_world"])[:3, 3].tolist() for row in scenario["instances"]] == config["positions"]


def test_camera_orbit_is_global_and_does_not_reset_on_source_loop():
    config = fixture_config(count=1)
    config["camera"] = {"trajectory": {"kind": "orbit", "radius": 9., "height": 3., "period_s": 100., "target": [1., 0., 2.]}}
    provider = TraceCameraProvider(generate_scenario(config)["trace"])
    target = np.array([1., 0., 2.])
    centers, intrinsics = [], []
    for gop in provider.gops:
        for pose in provider.get_trajectory(gop["gop_id"]):
            center = pose.camera.center
            assert np.isclose(np.linalg.norm((center-target)[[0, 2]]), 9.)
            assert np.isclose(center[1]-target[1], 3.)
            assert np.allclose(pose.camera.world_to_camera[:3, :3]@(target-center), [0., 0., math.sqrt(90.)])
            centers.append(center)
            intrinsics.append(pose.camera.K)
    assert all(np.array_equal(intrinsics[0], item) for item in intrinsics)
    assert not np.allclose(provider.get_trajectory("0")[0].camera.center,
                           provider.get_trajectory("3")[0].camera.center)


def test_pan_has_fixed_center_and_line_motion_fixed_view_direction():
    config = fixture_config()
    config["camera"] = {"trajectory": {"kind": "pan", "center": [2., 3., -4.]}}
    provider = TraceCameraProvider(generate_scenario(config)["trace"])
    for gop in provider.gops:
        assert all(np.allclose(p.camera.center, [2., 3., -4.]) for p in provider.get_trajectory(gop["gop_id"]))
    config = fixture_config("long_line")
    provider = TraceCameraProvider(generate_scenario(config)["trace"])
    rotations = [pose.camera.world_to_camera[:3, :3] for g in provider.gops for pose in provider.get_trajectory(g["gop_id"])]
    assert all(np.array_equal(rotations[0], rotation) for rotation in rotations)


def test_explicit_source_states_require_exact_content_sample_coverage():
    config = fixture_config(count=1)
    config["source_objects"] = [{"object_id": "a", "source_camera_id": "cam", "source_states": {"0": "frame0"}}]
    config["timeline"] = {"source_duration_s": 1., "total_duration_s": 2., "gop_duration_s": 1.}
    config["samples_per_gop"] = 1
    scenario = generate_scenario(config)
    first, repeated = [gop["samples"][0]["instances"][0] for gop in scenario["trace"]["gops"]]
    assert first["source_state_id"] == repeated["source_state_id"] == "frame0"
    assert first["source_state_binding"] == "EXPLICIT_SOURCE_STATE_ID"
    assert scenario["trace"]["gops"][0]["request_id"] != scenario["trace"]["gops"][1]["request_id"]
    config["samples_per_gop"] = 2
    with pytest.raises(ValueError, match="No exact source_state_id"):
        generate_scenario(config)


def test_recorded_ground_truth_trajectory_uses_exact_samples():
    config = fixture_config(count=1)
    config["timeline"] = {"source_duration_s": 1., "total_duration_s": 2., "gop_duration_s": 1.}
    base = generate_scenario(config)
    camera = base["trace"]["gops"][0]["samples"][0]["camera"]
    config["camera"] = {key: camera[key] for key in ("width", "height", "K", "near", "far")}
    config["camera"]["trajectory"] = {"kind": "trace", "samples": [
        {"timestamp_s": timestamp, "camera": camera} for timestamp in [0, .2, .8, 1., 1.4]]}
    trace = generate_scenario(config)["trace"]
    assert [[s["timestamp"] for s in g["samples"]] for g in trace["gops"]] == [[0., .2, .8], [1., 1.4]]
    config["camera"]["trajectory"]["samples"] = config["camera"]["trajectory"]["samples"][1:]
    with pytest.raises(ValueError, match="exact GOP start"):
        generate_scenario(config)


@pytest.mark.parametrize("update,error", [
    ({"kind": "zoom"}, "scenario.kind"),
    ({"object_count": 0}, "object_count"),
    ({"object_count": True}, "object_count"),
    ({"samples_per_gop": 0}, "samples_per_gop"),
    ({"positions": [[0, 0, 0]]}, "positions"),
    ({"camera": {"trajectory": {"kind": "zoom"}}}, "zoom"),
    ({"camera": {"trajectory": {"kind": "linear", "end": [1., 1., 10.]}}}, "lateral"),
    ({"camera": {"horizontal_fov_degrees": 180}}, "< 180"),
    ({"camera": {"trajectory": {"kind": "orbit", "radius": -1.}}}, "positive"),
    ({"instances": [{"object_id": "a", "instance_id": "same"}]*3}, "unique"),
    ({"source_objects": ["a", "a"]}, "duplicate"),
])
def test_invalid_scenario_configuration_is_rejected(update, error):
    config = fixture_config()
    config.update(update)
    with pytest.raises(ValueError, match=error):
        generate_scenario(config)


@pytest.mark.parametrize("matrix", [np.diag([-1., 1., 1., 1.]), np.diag([1., 2., 1., 1.])])
def test_invalid_non_similarity_instance_transform_is_rejected(matrix):
    config = fixture_config(count=1)
    config["instances"] = [{"object_id": "a", "object_to_world": matrix.tolist()}]
    with pytest.raises(ValueError, match="positive uniform scale"):
        generate_scenario(config)


def test_recorded_camera_cannot_change_intrinsics():
    config = fixture_config(count=1)
    base = generate_scenario(config)["trace"]["gops"][0]["samples"][0]["camera"]
    different = deepcopy(base)
    different["K"][0][0] *= 1.2
    config["camera"] = {key: base[key] for key in ("width", "height", "K", "near", "far")}
    config["camera"]["trajectory"] = {"kind": "trace", "samples": [
        {"timestamp_s": 0., "camera": base}, {"timestamp_s": .5, "camera": different}]}
    with pytest.raises(ValueError, match="fixed calibration"):
        generate_scenario(config)
