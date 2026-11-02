from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from experiments.visibility_weighted_sse.trace import (
    COORDINATE_CONVENTION, TraceCameraProvider, source_local_camera)
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter
from src_final.multiobj3dgs.shared.geometry import Camera, similarity_matrix


def inputs(times=(0., .5, 1., 1.5), values=None):
    camera = {"width": 10, "height": 10, "K": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
              "world_to_camera": np.eye(4).tolist()}
    instance = {"object_id": "source", "instance_id": "instance", "source_camera_id": "camera_0",
                "source_state_id": "frame_0", "object_to_world": np.eye(4).tolist()}
    trace = {"schema": "visibility-weighted-sse.trace.v1", "trace_id": "test_trace",
             "coordinate_convention": COORDINATE_CONVENTION,
             "prediction_assumption": "ORACLE_CAMERA_TRACE", "predictor_enabled": False,
             "gops": [{"gop_id": 0, "start_s": 0., "end_s": 2., "samples": [
                 {"timestamp": t, "camera": deepcopy(camera), "instances": [deepcopy(instance)]} for t in times]}]}
    provider = TraceCameraProvider(trace)
    masses = values or [(20., 10.)] * len(times)
    cache = {"schema": "visibility-weighted-sse.visibility.v1", "trace_id": "test_trace",
             "content_revision": "content_v1", "compatibility": {"resolution": [10, 10],
                 "coordinate_convention": COORDINATE_CONVENTION,
                 "contribution_convention": "continuous_alpha_weighted_pixels",
                 "time_aggregation": "normalized_camera_time_mean",
                 "proxy_representation": "Q0",
                 "transform_convention": "positive_similarity_object_to_world"},
             "provenance": {"information_role": "VISIBILITY_PROXY", "candidate_independent": True,
                            "joint_quality_used": False, "source": "synthetic_unit_fixture"},
             "gops": [{"gop_id": 0, "samples": [
                 {"timestamp": pose.timestamp, "pose_identity": pose.identity,
                  "per_object": {"instance": {"A_solo": solo, "A_visible": vis}}}
                 for pose, (solo, vis) in zip(provider.get_trajectory(0), masses)]}]}
    return trace, provider, cache


def test_full_oracle_trace_and_uniform_gop_integration():
    _, provider, cache = inputs()
    poses = provider.get_trajectory(0)
    assert len(poses) == 4
    assert [pose.prediction.target_at for pose in poses] == [0., .5, 1., 1.5]
    assert all(p.prediction.evidence == "ORACLE_CAMERA_TRACE" for p in poses)
    result = CachedVisibilityAdapter(cache, provider).aggregate(0)
    assert result["temporal_weights"] == [.25] * 4
    assert result["per_object"]["instance"]["A_solo"] == 20.
    assert result["per_object"]["instance"]["A_visible"] == 10.
    assert result["per_object"]["instance"]["occlusion_ratio"] == .5


def test_nonuniform_left_hold_includes_final_interval():
    _, provider, cache = inputs((0., .2, 1.7), [(10., 10.), (20., 5.), (0., 0.)])
    result = CachedVisibilityAdapter(cache, provider).aggregate(0)
    assert result["temporal_weights"] == pytest.approx([.1, .75, .15])
    assert result["per_object"]["instance"]["A_visible"] == pytest.approx(4.75)


def test_enter_leave_projected_contribution_and_occlusion():
    _, provider, cache = inputs(values=[(0., 0.), (20., 20.), (40., 10.), (0., 0.)])
    result = CachedVisibilityAdapter(cache, provider).aggregate(0)["per_object"]["instance"]
    assert result["A_solo"] == 15.
    assert result["A_visible"] == 7.5
    assert [r["occlusion_ratio"] for r in result["time_series"]] == [None, 0., .75, None]


@pytest.mark.parametrize("values", [[(20., 20.)]*4, [(20., 0.)]*4, [(0., 0.)]*4])
def test_unoccluded_fully_occluded_and_outside_fov(values):
    _, provider, cache = inputs(values=values)
    result = CachedVisibilityAdapter(cache, provider).aggregate(0)["per_object"]["instance"]
    assert result["A_solo"] == values[0][0]
    assert result["A_visible"] == values[0][1]


def test_source_local_camera_similarity_preserves_pixels_and_center():
    camera = Camera(100, 80, np.eye(4), np.array([[70., 0., 50.], [0., 70., 40.], [0., 0., 1.]]))
    transform = similarity_matrix((2, 1, 3), (1, 0, 0, 0), 2)
    local = source_local_camera(camera, transform)
    point = np.array([1., 2., 4., 1.])
    world_projected = camera.K @ (camera.world_to_camera @ transform @ point)[:3]
    local_projected = local.K @ (local.world_to_camera @ point)[:3]
    assert local_projected/local_projected[-1] == pytest.approx(world_projected/world_projected[-1])
    assert local.center == pytest.approx((np.linalg.inv(transform) @ np.r_[camera.center, 1.])[:3])
    assert local.near == camera.near/2
    assert local.far == camera.far/2


@pytest.mark.parametrize("mutation", [
    lambda t: t.update(prediction_assumption="current_pose_hold"),
    lambda t: t.update(predictor_enabled=True),
    lambda t: t.update(coordinate_convention="unknown"),
    lambda t: t["gops"][0]["samples"][0].update(timestamp=.1),
    lambda t: t["gops"][0]["samples"][1].update(timestamp=0.),
    lambda t: t["gops"][0]["samples"][-1].update(timestamp=2.),
    lambda t: t["gops"][0]["samples"][0]["camera"].update(width=True),
    lambda t: t["gops"][0]["samples"][0]["camera"].update(K=[[0,0,0],[0,1,0],[0,0,1]]),
    lambda t: t["gops"][0]["samples"][0]["instances"][0].pop("source_state_id"),
])
def test_invalid_trace_is_rejected(mutation):
    trace, _, _ = inputs()
    mutation(trace)
    with pytest.raises((ValueError, KeyError)):
        TraceCameraProvider(trace)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(trace_id="another"),
    lambda c: c["provenance"].update(joint_quality_used=True),
    lambda c: c["provenance"].update(information_role="JOINT_IMAGE_GROUND_TRUTH"),
    lambda c: c["provenance"].update(candidate_independent=False),
    lambda c: c["compatibility"].update(resolution=[20,10]),
    lambda c: c["gops"][0]["samples"].pop(),
    lambda c: c["gops"][0]["samples"][1].update(pose_identity="different_transform"),
    lambda c: c["gops"][0]["samples"][1]["per_object"]["instance"].update(A_visible=99),
    lambda c: c["gops"][0]["samples"][1]["per_object"]["instance"].update(A_solo=101),
])
def test_invalid_cache_is_rejected(mutation):
    _, provider, cache = inputs()
    mutation(cache)
    with pytest.raises(ValueError):
        CachedVisibilityAdapter(cache, provider)


def test_revision_compatibility_and_missing_gop_are_explicit_errors():
    _, provider, cache = inputs()
    with pytest.raises(ValueError, match="revision"):
        CachedVisibilityAdapter(cache, provider, content_revision="v2")
    mismatch = dict(cache["compatibility"], profile_distribution="different")
    with pytest.raises(ValueError, match="compatibility"):
        CachedVisibilityAdapter(cache, provider, expected_compatibility=mismatch)
    with pytest.raises(ValueError, match="Missing GOP"):
        provider.get_trajectory(1)
    with pytest.raises(ValueError, match="Missing cached"):
        CachedVisibilityAdapter(cache, provider).get_results(1)


def test_multiple_gops_determinism_and_temporal_order():
    trace, _, _ = inputs()
    second = deepcopy(trace["gops"][0])
    second.update(gop_id=1, start_s=2., end_s=4.)
    for sample in second["samples"]:
        sample["timestamp"] += 2
    trace["gops"].append(second)
    a, b = TraceCameraProvider(trace), TraceCameraProvider(deepcopy(trace))
    assert [g["gop_id"] for g in a.gops] == [0, 1]
    assert [p.identity for p in a.get_trajectory(1)] == [p.identity for p in b.get_trajectory(1)]
    trace["gops"].reverse()
    with pytest.raises(ValueError, match="chronologically"):
        TraceCameraProvider(trace)


def test_imports_do_not_load_torch_renderer_or_network():
    code = "import sys; import experiments.visibility_weighted_sse.visibility; assert 'torch' not in sys.modules; assert 'diff_gaussian_rasterization' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
