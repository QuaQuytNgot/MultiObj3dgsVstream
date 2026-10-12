"""Client contract tests using scalar renderer outputs, without rasterization.

Actual compositing and composite parity are exercised separately by CUDA tests.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.client import visibility
from src.renderer.analysis import (
    ContributionDiagnostics,
    ContributionMeasurements,
    ObjectContribution,
    VisibilityObject,
)


@pytest.fixture
def renderer_measurements(monkeypatch):
    calls = []

    def install(areas, *, diagnostics=None, metadata=None):
        measured = ContributionMeasurements(
            timestamp=2.5,
            resolution=(65, 49),
            per_object={
                object_id: ObjectContribution(object_id, visible, solo)
                for object_id, visible, solo in areas
            },
            diagnostics=diagnostics,
            metadata={"backend": "test scalar measurements"} if metadata is None else metadata,
        )

        def measure(objects, camera, timestamp, **kwargs):
            calls.append((objects, camera, timestamp, kwargs))
            return measured

        monkeypatch.setattr(visibility, "measure_contributions", measure)
        return measured

    return install, calls


def test_unoccluded_partial_and_fully_suppressed_objects(renderer_measurements):
    install, _ = renderer_measurements
    install([("clear", 40.0, 40.0), ("partial", 6.0, 24.0), ("hidden", 0.0, 10.0)])

    result = visibility.analyze_visibility([], {}, 2.5)

    assert result.per_object["clear"].occlusion_ratio == 0.0
    assert result.per_object["partial"].occlusion_ratio == pytest.approx(0.75)
    hidden = result.per_object["hidden"]
    assert hidden.occlusion_ratio == 1.0
    assert not hidden.out_of_view
    assert all(not item.out_of_view for item in result.per_object.values())
    assert result.per_object["partial"].A_visible == 6.0
    assert result.per_object["partial"].A_solo == 24.0
    assert result.diagnostics is None


@pytest.mark.parametrize("solo", [0.0, 0.5e-6, 1e-6])
def test_no_contribution_and_epsilon_boundary_are_out_of_view(renderer_measurements, solo):
    install, _ = renderer_measurements
    install([("outside", 0.0, solo)])

    item = visibility.analyze_visibility([], {}, 2.5).per_object["outside"]

    assert item.out_of_view
    assert item.occlusion_ratio is None
    assert item.A_solo == solo


def test_solo_just_above_epsilon_has_defined_occlusion(renderer_measurements):
    install, _ = renderer_measurements
    solo = np.nextafter(1e-6, np.inf)
    install([("tiny", solo, solo)])

    item = visibility.analyze_visibility([], {}, 2.5).per_object["tiny"]

    assert not item.out_of_view
    assert item.occlusion_ratio == 0.0


def test_zero_epsilon_and_user_threshold(renderer_measurements):
    install, _ = renderer_measurements
    install([("small", 0.001, 0.002)])
    assert not visibility.analyze_visibility([], {}, 2.5, epsilon=0).per_object["small"].out_of_view
    item = visibility.analyze_visibility([], {}, 2.5, epsilon=0.01).per_object["small"]
    assert item.out_of_view and item.occlusion_ratio is None


@pytest.mark.parametrize("epsilon", [
    -1.0, float("nan"), float("inf"), -float("inf"), True, None, "1e-6",
    np.bool_(True), np.bool_(False), np.array([1e-6]), 10 ** 1000,
])
def test_invalid_epsilon_fails_before_renderer(renderer_measurements, epsilon):
    install, calls = renderer_measurements
    install([])
    with pytest.raises(ValueError, match="epsilon"):
        visibility.analyze_visibility([], {}, 2.5, epsilon=epsilon)
    assert calls == []


@pytest.mark.parametrize("epsilon", [np.float32(1e-6), np.float64(1e-6), np.int64(0)])
def test_numpy_real_scalar_epsilon_is_accepted(renderer_measurements, epsilon):
    install, _ = renderer_measurements
    install([])
    result = visibility.analyze_visibility([], {}, 2.5, epsilon=epsilon)
    assert type(result.epsilon) is float
    assert result.epsilon == float(epsilon)


@pytest.mark.parametrize("visible,solo,message", [
    (float("nan"), 1.0, "finite"),
    (1.0, float("nan"), "finite"),
    (float("inf"), 1.0, "finite"),
    (0.0, -float("inf"), "finite"),
    (-1e-3, 1.0, "nonnegative"),
    (0.0, -1e-3, "nonnegative"),
    (1.01, 1.0, "A_visible exceeds A_solo"),
    (100.0 + 1.1 * (1e-6 + 1e-5 * 100.0), 100.0, "A_visible exceeds A_solo"),
    (1e-5, 0.0, "A_visible exceeds A_solo"),
])
def test_invalid_measurements_raise_instead_of_masking_errors(
    renderer_measurements, visible, solo, message,
):
    install, _ = renderer_measurements
    install([("broken", visible, solo)])
    with pytest.raises(ValueError, match=f"broken:.*{message}"):
        visibility.analyze_visibility([], {}, 2.5)


def test_numerical_excursions_only_are_clamped(renderer_measurements):
    install, _ = renderer_measurements
    tolerance = 1e-6 + 1e-5 * 100.0
    original = install([
        ("roundoff-high", 100.0 + tolerance / 2, 100.0),
        ("roundoff-negative", -5e-7, 1.0),
        ("roundoff-empty", -5e-7, -5e-7),
        ("tiny-real-occlusion", 100.0 - tolerance / 2, 100.0),
    ])

    result = visibility.analyze_visibility([], {}, 2.5)

    high = result.per_object["roundoff-high"]
    assert high.A_visible == high.A_solo == 100.0
    assert high.occlusion_ratio == 0.0
    negative = result.per_object["roundoff-negative"]
    assert negative.A_visible == 0.0 and negative.occlusion_ratio == 1.0
    empty = result.per_object["roundoff-empty"]
    assert empty.A_visible == empty.A_solo == 0.0
    assert empty.out_of_view and empty.occlusion_ratio is None
    assert result.per_object["tiny-real-occlusion"].occlusion_ratio > 0.0
    assert original.per_object["roundoff-high"].A_visible > 100.0


def test_forwards_resolved_scene_and_preserves_result_context(renderer_measurements):
    install, calls = renderer_measurements
    joint = np.ones((2, 3), dtype=np.float32)
    solo = np.ones((2, 3), dtype=np.float32)
    diagnostics = ContributionDiagnostics({"object": joint}, {"object": solo})
    measured = install(
        [("object", 6.0, 6.0)], diagnostics=diagnostics,
        metadata={"backend": "CUDA", "joint_passes": 1, "solo_passes": 1},
    )
    objects, camera = [], {"width": 65, "height": 49}
    settings = {"debug": True, "scaling_modifier": 1.5}

    result = visibility.analyze_visibility(
        objects, camera, 2.5, diagnostics=True, renderer_settings=settings,
    )

    assert len(calls) == 1
    assert calls[0][0] is objects and calls[0][1] is camera
    assert calls[0][2] == 2.5
    assert calls[0][3] == {"diagnostics": True, "renderer_settings": settings}
    assert calls[0][3]["renderer_settings"] is settings
    assert result.timestamp == 2.5 and result.resolution == (65, 49)
    assert result.epsilon == 1e-6
    assert result.diagnostics is diagnostics
    assert result.diagnostics.joint_maps["object"] is joint
    assert result.metadata == measured.metadata
    assert result.metadata is not measured.metadata
    assert result.per_object["object"].object_id == "object"
    with pytest.raises(FrozenInstanceError):
        result.per_object["object"].out_of_view = True


def test_empty_measurements_return_empty_visibility(renderer_measurements):
    install, _ = renderer_measurements
    install([])
    result = visibility.analyze_visibility([], {}, 2.5)
    assert result.per_object == {}
    assert result.diagnostics is None


def test_renderer_errors_propagate_without_substitute_measurements(monkeypatch):
    failure = RuntimeError("CUDA backend unavailable")

    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(visibility, "measure_contributions", unavailable)
    with pytest.raises(RuntimeError, match="CUDA backend unavailable") as raised:
        visibility.analyze_visibility([], {}, 2.5)
    assert raised.value is failure


def test_inconsistent_renderer_object_identity_is_rejected(renderer_measurements):
    install, _ = renderer_measurements
    measured = install([("expected", 1.0, 1.0)])
    measured.per_object["expected"] = ObjectContribution("different", 1.0, 1.0)
    with pytest.raises(ValueError, match="object_id does not match"):
        visibility.analyze_visibility([], {}, 2.5)


def test_public_snapshot_type_is_renderer_snapshot_type():
    assert visibility.VisibilityObject is VisibilityObject
