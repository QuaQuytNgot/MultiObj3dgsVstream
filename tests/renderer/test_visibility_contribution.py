"""Contribution correctness against the actual, unchanged CUDA rasterizer.

Run explicitly with VISIBILITY_CUDA_TESTS=1 and, if needed, configure
VISIBILITY_EXTENSION_PATH. Opted-in tests fail when CUDA/the extension is
unavailable; they never substitute a CPU rasterizer.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from src.client.visibility import analyze_visibility
from src.renderer.analysis import VisibilityObject
from src.renderer.upstream import prepare_scene, validate_and_snapshot
from tools.content_preparation.assets import GaussianState
from tools.content_preparation.renderer_adapter import GPUExecutionGuard, RendererUnavailable, cleanup_gpu
from tools.content_preparation.spatial_rendering import project_points, render_state


_ATOL = 2e-6
_RTOL = 2e-5
_TIMESTAMP = 1.25
_SH_C0 = 0.28209479177387814


def _camera():
    return {"id": "visibility_fixture", "world_view": np.eye(4).tolist(),
            "width": 65, "height": 65, "fovx": math.pi / 2,
            "fovy": math.pi / 2, "near": .01, "far": 100.}


def _state(xyz, *, alpha=.6, scale=.1, rotation=None, color=(.2, .4, .7), ids=None):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    count = len(xyz)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (count,))
    scales = np.broadcast_to(np.asarray(scale, dtype=np.float64), (count, 3))
    rotations = (np.tile([1., 0, 0, 0], (count, 1)) if rotation is None
                 else np.broadcast_to(rotation, (count, 4)).copy())
    rgb = np.broadcast_to(np.asarray(color, dtype=np.float64), (count, 3))
    return GaussianState({"xyz": xyz, "rotation": rotations,
                          "scale": np.log(scales),
                          "opacity": (np.log(alpha) - np.log1p(-alpha))[:, None],
                          "sh": ((rgb - .5) / _SH_C0)[:, :, None]},
                         {"fixture": "visibility"},
                         np.arange(count) if ids is None else np.asarray(ids))


def _screen_state(pixels, depths, *, sigma=3., alpha=.6, **kwargs):
    """Choose world-space scales for a prescribed on-axis pixel sigma.

    The .3 term is the original rasterizer's screen covariance low-pass term.
    Off-axis covariance can be larger, so overlap assertions use broad bounds.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    depths = np.broadcast_to(np.asarray(depths, dtype=np.float64), (len(pixels),))
    xy = (pixels - 32.) * depths[:, None] / 32.5
    scales = depths / 32.5 * np.sqrt(np.asarray(sigma) ** 2 - .3)
    return _state(np.column_stack((xy, depths)), scale=scales[:, None],
                  alpha=alpha, **kwargs)


def _object(object_id, state, transform=None):
    return VisibilityObject(object_id=object_id, proxy=state, timestamp=_TIMESTAMP,
                            object_to_world=np.eye(4) if transform is None else transform)


@pytest.fixture
def cuda_settings():
    if os.environ.get("VISIBILITY_CUDA_TESTS") != "1":
        pytest.skip("Set VISIBILITY_CUDA_TESTS=1 to verify the real CUDA backend")
    # Import/backend errors must fail after explicit opt-in.
    import torch
    assert torch.cuda.is_available(), "VISIBILITY_CUDA_TESTS=1 requires accessible CUDA"
    settings = {}
    if os.environ.get("VISIBILITY_EXTENSION_PATH"):
        settings["extension_path"] = os.environ["VISIBILITY_EXTENSION_PATH"]
    return settings


def _analyze(objects, settings, *, diagnostics=True, camera=None):
    return analyze_visibility(objects, _camera() if camera is None else camera, _TIMESTAMP,
                              diagnostics=diagnostics, renderer_settings=settings)


def _raw_render(objects, settings, *, colors=None, background=(0., 0., 0.), camera=None):
    """Independent normal/override call; never reimplement alpha compositing."""
    import torch
    snapshots, camera, validated_settings = validate_and_snapshot(
        objects, _camera() if camera is None else camera, _TIMESTAMP, settings)
    scene = rendered = color_tensor = background_tensor = None
    with GPUExecutionGuard():
        try:
            with torch.no_grad():
                scene = prepare_scene(snapshots, camera, validated_settings)
                background_tensor = torch.tensor(background, dtype=torch.float32, device="cuda")
                if colors == "white":
                    color_tensor = torch.ones((len(scene.object_slots), 3), device="cuda")
                elif colors == "first_group":
                    color_tensor = torch.zeros((len(scene.object_slots), 3), device="cuda")
                    for slot in range(min(3, len(objects))):
                        color_tensor[scene.object_slots == slot, slot] = 1.
                elif colors is not None:
                    raise ValueError(f"Unsupported test color override: {colors}")
                rendered = scene.render(scene.joint_model, colors=color_tensor,
                                        background=background_tensor)
                return {"rgb": rendered["render"].detach().cpu().numpy().copy(),
                        "radii": rendered["radii"].detach().cpu().numpy().copy()}
        finally:
            scene = rendered = color_tensor = background_tensor = None
            cleanup_gpu()


def _assert_zero_occlusion(measurement):
    assert measurement.A_solo > 1e-6
    assert not measurement.out_of_view
    assert measurement.occlusion_ratio == pytest.approx(0., abs=2e-5)
    assert measurement.A_visible == pytest.approx(measurement.A_solo, rel=2e-5, abs=1e-6)


def test_snapshot_applies_similarity_transform_without_mutating_inputs():
    # A noncommuting local X rotation checks world/local quaternion order.
    state = _state([[.2, .1, 2.]], scale=[.1, .2, .3], rotation=[math.sqrt(3), 1., 0, 0])
    original = {key: value.copy() for key, value in state.arrays.items()}
    transform = np.array([[0., -2., 0., 1.], [2., 0., 0., 2.],
                          [0., 0., 2., 3.], [0., 0., 0., 1.]])
    snapshots, camera, settings = validate_and_snapshot(
        [_object("moving", state, transform)], _camera(), _TIMESTAMP)
    snapshot = snapshots[0]
    assert snapshot.object_id == "moving"
    assert snapshot.timestamp == _TIMESTAMP
    np.testing.assert_allclose(snapshot.proxy.arrays["xyz"], [[.8, 2.4, 7.]], atol=1e-6)
    np.testing.assert_allclose(snapshot.proxy.arrays["scale"],
                               original["scale"] + math.log(2), atol=1e-7)
    np.testing.assert_allclose(snapshot.proxy.arrays["rotation"],
                               [[math.sqrt(1.5), math.sqrt(.5), math.sqrt(.5), math.sqrt(1.5)]],
                               atol=1e-6)
    for key in ("opacity", "sh"):
        np.testing.assert_array_equal(snapshot.proxy.arrays[key], original[key])
    for key, value in original.items():
        np.testing.assert_array_equal(state.arrays[key], value)
        assert not np.shares_memory(snapshot.proxy.arrays[key], state.arrays[key])
    assert camera["width"] == 65
    assert settings["scaling_modifier"] == 1.


def test_identity_snapshot_preserves_raw_attributes_and_local_ids():
    state = _state([[.2, -.3, 2.], [0, 0, 4.]], rotation=[2., .5, 0, 0], ids=[42, 43])
    snapshots, _, _ = validate_and_snapshot([_object("a", state), _object("b", state)],
                                            _camera(), _TIMESTAMP)
    for snapshot in snapshots:
        for key, value in state.arrays.items():
            np.testing.assert_array_equal(snapshot.proxy.arrays[key], value)
            assert not np.shares_memory(snapshot.proxy.arrays[key], value)
        np.testing.assert_array_equal(snapshot.proxy.ids, [42, 43])


@pytest.mark.parametrize("linear", [
    [[2, 0, 0], [0, 1, 0], [0, 0, 1]],  # nonuniform scale
    [[1, .2, 0], [0, 1, 0], [0, 0, 1]],  # shear
    [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],  # reflection
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # singular scale
])
def test_snapshot_rejects_unsupported_transform(linear):
    transform = np.eye(4)
    transform[:3, :3] = linear
    with pytest.raises(ValueError):
        validate_and_snapshot([_object("a", _state([[0, 0, 2]]), transform)],
                              _camera(), _TIMESTAMP)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf"), True, None])
def test_snapshot_rejects_invalid_media_timestamp(timestamp):
    with pytest.raises(ValueError):
        validate_and_snapshot([], _camera(), timestamp)


def test_snapshot_rejects_unresolved_timestamp():
    obj = VisibilityObject("a", _state([[0, 0, 2]]), _TIMESTAMP + 1e-4, np.eye(4))
    with pytest.raises(ValueError, match="timestamp"):
        validate_and_snapshot([obj], _camera(), _TIMESTAMP)


def test_snapshot_accepts_timestamp_agreement_within_tolerance():
    obj = VisibilityObject("a", _state([[0, 0, 2]]), _TIMESTAMP + 5e-10, np.eye(4))
    snapshots, _, _ = validate_and_snapshot([obj], _camera(), _TIMESTAMP)
    assert len(snapshots) == 1


def test_snapshot_rejects_duplicate_object_identity():
    state = _state([[0, 0, 2]])
    with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
        validate_and_snapshot([_object("a", state), _object("a", state)],
                              _camera(), _TIMESTAMP)


@pytest.mark.parametrize("object_id", ["", 0, None])
def test_snapshot_rejects_invalid_object_identity(object_id):
    with pytest.raises(ValueError):
        validate_and_snapshot([_object(object_id, _state([[0, 0, 2]]))],
                              _camera(), _TIMESTAMP)


def test_snapshot_revalidates_mutable_proxy_attributes():
    state = _state([[0, 0, 2]])
    state.arrays["xyz"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_and_snapshot([_object("a", state)], _camera(), _TIMESTAMP)


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("width", True), ("width", 64.5), ("height", 0),
    ("fovx", 0.), ("fovy", math.pi), ("near", 0.), ("far", .01),
    ("near", float("nan")), ("far", float("inf")),
])
def test_snapshot_rejects_invalid_camera(field, value):
    camera = _camera()
    camera[field] = value
    with pytest.raises(ValueError):
        validate_and_snapshot([], camera, _TIMESTAMP)


@pytest.mark.parametrize("modification", ["nonrigid", "nonaffine", "reflection"])
def test_snapshot_rejects_invalid_camera_matrix(modification):
    camera = _camera()
    if modification == "nonrigid":
        camera["world_view"][0][0] = 2.
    elif modification == "nonaffine":
        camera["world_view"][3][0] = 1.
    else:
        camera["world_view"][0][0] = -1.
    with pytest.raises(ValueError):
        validate_and_snapshot([], camera, _TIMESTAMP)


@pytest.mark.parametrize("settings", [
    {"backend": "cpu_fixture"}, {"scaling_modifier": 0.},
    {"scaling_modifier": -1.}, {"scaling_modifier": float("nan")},
    {"scaling_modifier": float("inf")}, {"scaling_modifier": True},
])
def test_snapshot_rejects_unsupported_renderer_settings(settings):
    with pytest.raises(ValueError):
        validate_and_snapshot([], _camera(), _TIMESTAMP, settings)


@pytest.mark.parametrize("objects", [[], [_object("empty", _state([]))]])
def test_empty_geometry_never_loads_cuda(objects, monkeypatch):
    # Prevent a backend import from succeeding even on a GPU-equipped host.
    monkeypatch.setitem(sys.modules, "torch", None)
    result = _analyze(objects, {}, diagnostics=True)
    assert result.timestamp == _TIMESTAMP
    assert result.resolution == (65, 65)
    assert set(result.per_object) == {obj.object_id for obj in objects}
    if objects:
        assert result.per_object["empty"].A_visible == 0.
        assert result.per_object["empty"].A_solo == 0.
        assert result.per_object["empty"].out_of_view
        assert result.per_object["empty"].occlusion_ratio is None
        assert not result.diagnostics.joint_maps["empty"].any()
        assert not result.diagnostics.solo_maps["empty"].any()


def test_nonempty_geometry_reports_missing_backend_without_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RendererUnavailable):
        _analyze([_object("a", _state([[0, 0, 2]]))], {})


def test_one_object_includes_self_occlusion(cuda_settings):
    objects = [_object("a", _screen_state([[32, 32], [32, 32]], [2, 4], alpha=[.6, .4]))]
    result = _analyze(objects, cuda_settings)
    _assert_zero_occlusion(result.per_object["a"])
    joint = result.diagnostics.joint_maps["a"]
    solo = result.diagnostics.solo_maps["a"]
    np.testing.assert_allclose(joint, solo, atol=_ATOL, rtol=_RTOL)
    assert joint[32, 32] == pytest.approx(.6 + (1 - .6) * .4, abs=_ATOL)
    assert joint.dtype == np.float32 and joint.shape == (65, 65)
    assert result.per_object["a"].A_visible == pytest.approx(
        float(joint.sum(dtype=np.float64)), rel=1e-12, abs=1e-9)


def test_two_nonoverlapping_objects(cuda_settings):
    objects = [_object("a", _screen_state([[10, 32]], 2, sigma=1.3)),
               _object("b", _screen_state([[54, 32]], 3, sigma=1.3))]
    result = _analyze(objects, cuda_settings)
    for value in result.per_object.values():
        _assert_zero_occlusion(value)
    assert not np.any((result.diagnostics.joint_maps["a"] > 0)
                      & (result.diagnostics.joint_maps["b"] > 0))


def test_rear_object_is_highly_occluded_but_still_has_solo_contribution(cuda_settings):
    objects = [_object("front", _screen_state([[32, 32]], 2, sigma=30, alpha=.99)),
               _object("rear", _screen_state([[32, 32]], 4, sigma=3, alpha=.5))]
    result = _analyze(objects, cuda_settings)
    _assert_zero_occlusion(result.per_object["front"])
    rear = result.per_object["rear"]
    assert rear.A_solo > 1e-6 and not rear.out_of_view
    assert rear.occlusion_ratio > .95
    assert result.diagnostics.joint_maps["rear"][32, 32] == pytest.approx(.005, abs=_ATOL)


def test_partial_overlap_produces_partial_occlusion(cuda_settings):
    objects = [_object("front", _screen_state([[29, 32]], 2, alpha=.9)),
               _object("rear", _screen_state([[35, 32]], 4, alpha=.6))]
    result = _analyze(objects, cuda_settings)
    _assert_zero_occlusion(result.per_object["front"])
    assert .05 < result.per_object["rear"].occlusion_ratio < .9


@pytest.mark.parametrize("state", [
    _screen_state([[500, 32]], 2, sigma=1),
    _state([[0, 0, -2]]),
    _state([[0, 0, 2]], alpha=1e-6),
    _state([]),
], ids=["outside_image", "behind_camera", "no_alpha_contribution", "empty"])
def test_no_solo_contribution_has_undefined_occlusion(state, cuda_settings):
    result = _analyze([_object("a", state)], cuda_settings)
    value = result.per_object["a"]
    assert value.A_solo <= 1e-6
    assert value.A_visible == 0.
    assert value.out_of_view
    assert value.occlusion_ratio is None
    assert not result.diagnostics.solo_maps["a"].any()


def test_swapping_depth_order_swaps_occlusion(cuda_settings):
    near = _screen_state([[32, 32]], 2, alpha=.8)
    far = _screen_state([[32, 32]], 4, alpha=.8)
    first = _analyze([_object("a", near), _object("b", far)], cuda_settings)
    swapped = _analyze([_object("a", far), _object("b", near)], cuda_settings)
    _assert_zero_occlusion(first.per_object["a"])
    _assert_zero_occlusion(swapped.per_object["b"])
    assert first.per_object["b"].occlusion_ratio > .3
    assert swapped.per_object["a"].occlusion_ratio > .3
    assert first.per_object["a"].A_visible > swapped.per_object["a"].A_visible
    assert first.per_object["b"].A_visible < swapped.per_object["b"].A_visible
    np.testing.assert_allclose(first.diagnostics.joint_maps["a"],
                               swapped.diagnostics.joint_maps["b"], atol=_ATOL, rtol=_RTOL)


def test_off_group_object_still_occludes_measured_objects(cuda_settings):
    # Slot 3 is black in the first joint group, but must retain its opacity.
    objects = [_object("rear", _screen_state([[32, 32]], 4, alpha=.5)),
               _object("left", _screen_state([[8, 8]], 3, sigma=1)),
               _object("right", _screen_state([[56, 8]], 3, sigma=1)),
               _object("front", _screen_state([[32, 32]], 2, sigma=30, alpha=.99))]
    result = _analyze(objects, cuda_settings)
    assert result.per_object["rear"].occlusion_ratio > .95
    assert result.diagnostics.joint_maps["rear"][32, 32] == pytest.approx(.005, abs=_ATOL)
    _assert_zero_occlusion(result.per_object["front"])
    assert set(result.diagnostics.joint_maps) == {obj.object_id for obj in objects}


def test_splats_are_sorted_by_depth_across_objects(cuda_settings):
    # Input object order is B,A; neither object can be composited as one layer.
    a = _screen_state([[32, 32], [32, 32]], [2, 4], alpha=[.4, .5], ids=[42, 43])
    b = _screen_state([[32, 32], [32, 32]], [3, 5], alpha=[.6, .3], ids=[42, 43])
    result = _analyze([_object("b", b), _object("a", a)], cuda_settings)
    maps = result.diagnostics.joint_maps
    assert maps["a"][32, 32] == pytest.approx(.4 + .6 * .4 * .5, abs=_ATOL)
    assert maps["b"][32, 32] == pytest.approx(.6 * .6 + .6 * .4 * .5 * .3, abs=_ATOL)
    assert result.diagnostics.solo_maps["a"][32, 32] == pytest.approx(.7, abs=_ATOL)
    assert result.diagnostics.solo_maps["b"][32, 32] == pytest.approx(.72, abs=_ATOL)


@pytest.mark.parametrize("last_alpha,last_weight", [(.6, 0.), (.49, .000098)])
def test_alpha_early_exit_matches_upstream_discard_semantics(last_alpha, last_weight,
                                                            cuda_settings):
    objects = [_object("a", _screen_state([[32, 32]], 2, alpha=.99)),
               _object("b", _screen_state([[32, 32]], 3, alpha=.98)),
               _object("c", _screen_state([[32, 32]], 4, alpha=last_alpha))]
    result = _analyze(objects, cuda_settings)
    maps = result.diagnostics.joint_maps
    assert maps["a"][32, 32] == pytest.approx(.99, abs=_ATOL)
    assert maps["b"][32, 32] == pytest.approx(.0098, abs=_ATOL)
    assert maps["c"][32, 32] == pytest.approx(last_weight, abs=2e-8)


def test_alpha_is_clamped_and_small_alpha_is_rejected(cuda_settings):
    clamped = _analyze([_object("a", _screen_state([[32, 32]], 2, alpha=.999999))],
                       cuda_settings)
    assert clamped.diagnostics.joint_maps["a"][32, 32] == pytest.approx(.99, abs=_ATOL)
    rejected = _analyze([_object("a", _screen_state([[32, 32]], 2, alpha=1 / 300))],
                        cuda_settings)
    assert rejected.per_object["a"].out_of_view
    assert not rejected.diagnostics.joint_maps["a"].any()


def test_rotated_anisotropic_transform_matches_explicit_world_state(cuda_settings):
    local = _state([[0, 0, 2]], scale=[.3, .1, .05])
    transform = np.array([[0., -1.5, 0., .2], [1.5, 0., 0., -.1],
                          [0., 0., 1.5, -1.], [0., 0., 0., 1.]])
    expected = _state([[.2, -.1, 2]], scale=[.45, .15, .075],
                      rotation=[math.sqrt(.5), 0, 0, math.sqrt(.5)])
    transformed = _analyze([_object("a", local, transform)], cuda_settings)
    explicit = render_state(expected, _camera(), cuda_settings, alpha=True)
    np.testing.assert_allclose(transformed.diagnostics.joint_maps["a"], explicit["alpha"],
                               atol=_ATOL, rtol=_RTOL)
    y, x = np.indices((65, 65), dtype=np.float64)
    weights = transformed.diagnostics.joint_maps["a"].astype(np.float64)
    mean_x, mean_y = (weights * x).sum() / weights.sum(), (weights * y).sum() / weights.sum()
    var_x = (weights * (x - mean_x) ** 2).sum() / weights.sum()
    var_y = (weights * (y - mean_y) ** 2).sum() / weights.sum()
    assert var_y > 4 * var_x, "90-degree object rotation must rotate the anisotropic footprint"
    _assert_zero_occlusion(transformed.per_object["a"])


def test_identity_camera_footprint_and_radii_match_existing_bridge(cuda_settings):
    state = _state([[.1, -.15, 2], [-.2, .1, 4]], alpha=[.6, .4], scale=[.15, .1, .08])
    objects = [_object("a", state)]
    visibility = _analyze(objects, cuda_settings)
    normal = _raw_render(objects, cuda_settings)
    override = _raw_render(objects, cuda_settings, colors="first_group")
    bridge = render_state(state, _camera(), cuda_settings, alpha=True)
    np.testing.assert_array_equal(normal["radii"], override["radii"])
    np.testing.assert_array_equal(normal["radii"], bridge["radii"])
    np.testing.assert_allclose(normal["rgb"].transpose(1, 2, 0), bridge["rgb"],
                               atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(visibility.diagnostics.joint_maps["a"], bridge["alpha"],
                               atol=_ATOL, rtol=_RTOL)


def test_nonidentity_camera_and_asymmetric_projection_match_existing_bridge(cuda_settings):
    camera = _camera()
    angle = .3
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)],
                         [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, [.2, -.15, .5]
    camera.update(world_view=matrix.tolist(), width=91, height=47, fovx=1.1, fovy=.8)
    camera_point = np.array([[.2, .1, 3.]])
    world_point = (camera_point - matrix[:3, 3]) @ rotation
    state = _state(world_point, scale=.08)
    objects = [_object("a", state)]
    visibility = _analyze(objects, cuda_settings, camera=camera)
    normal = _raw_render(objects, cuda_settings, camera=camera)
    bridge = render_state(state, camera, cuda_settings, alpha=True)
    contribution = visibility.diagnostics.joint_maps["a"]
    assert contribution.shape == (47, 91)
    np.testing.assert_array_equal(normal["radii"], bridge["radii"])
    np.testing.assert_allclose(normal["rgb"].transpose(1, 2, 0), bridge["rgb"],
                               atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(contribution, bridge["alpha"], atol=_ATOL, rtol=_RTOL)
    expected_pixels, depths = project_points(state.arrays["xyz"], camera)
    peak_y, peak_x = np.unravel_index(contribution.argmax(), contribution.shape)
    np.testing.assert_allclose([peak_x, peak_y], expected_pixels[0], atol=.6)
    assert depths[0] == pytest.approx(3., abs=1e-6)
    assert peak_x > 45 and peak_y > 23  # Positive camera Y projects downward.


@pytest.mark.parametrize("scaling_modifier", [1., 1.3])
def test_contribution_maps_reconstruct_normal_rgb_with_background(scaling_modifier,
                                                                  cuda_settings):
    colors = np.array([[.8, .1, .3], [.1, .9, .2], [.2, .3, .9], [.5, .7, .1]], dtype=np.float32)
    background = np.array([.15, .25, .35], dtype=np.float32)
    objects = [_object(str(index), _screen_state([[29 + index * 2, 30 + index]],
                                               2 + index, alpha=.65, color=colors[index]))
               for index in range(4)]
    settings = {**cuda_settings, "scaling_modifier": scaling_modifier}
    result = _analyze(objects, settings)
    normal = _raw_render(objects, settings, background=background)
    white = _raw_render(objects, settings, colors="white")
    group = _raw_render(objects, settings, colors="first_group")
    total = sum(result.diagnostics.joint_maps.values())
    reconstructed = background[:, None, None] * (1. - total)[None]
    for index, color in enumerate(colors):
        reconstructed += color[:, None, None] * result.diagnostics.joint_maps[str(index)][None]
    np.testing.assert_allclose(reconstructed, normal["rgb"], atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(total, white["rgb"][0], atol=_ATOL, rtol=_RTOL)
    np.testing.assert_array_equal(normal["radii"], white["radii"])
    np.testing.assert_array_equal(normal["radii"], group["radii"])


def test_normal_online_execution_returns_scalars_without_retaining_maps(cuda_settings):
    objects = [_object("a", _screen_state([[32, 32]], 2))]
    result = _analyze(objects, cuda_settings, diagnostics=False)
    assert result.diagnostics is None
    _assert_zero_occlusion(result.per_object["a"])
