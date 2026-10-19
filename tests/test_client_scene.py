"""CPU-only contracts for independent object state publication and cameras."""
import math

import numpy as np
import pytest

from src.client.scene import SceneManager
from src.renderer.backend import Camera, RenderObject, create_renderer
from tools.content_preparation.assets import synthetic_state
from tools.content_preparation.renderer_adapter import camera_parameters


def test_progressive_update_preserves_other_object_and_revisions():
    scene = SceneManager()
    scene.add_object("a")
    scene.add_object("b")
    first, second = synthetic_state(), synthetic_state(seed=23)
    scene.update_object("a", first, 0, quality=0, frame_index=0)
    scene.update_object("b", second, 0, quality=1, frame_index=0)
    before = scene.snapshot(0)
    improved = synthetic_state(quality_count=32)
    scene.update_object("a", improved, 0, quality=3)
    after = scene.snapshot(0)
    assert after[0].revision == before[0].revision + 1
    assert after[1].revision == before[1].revision
    assert after[1].gaussian_state is second
    assert scene.objects["a"].quality == 3
    scene.update_object("a", improved, 0, quality=3)
    assert scene.snapshot(0)[0].revision == after[0].revision


def test_media_synchronization_pause_seek_and_loop():
    scene = SceneManager()
    for name in ("a", "b"):
        scene.add_object(name)
        scene.update_object(name, synthetic_state(), 0)
    scene.update_object("a", synthetic_state(frame=1), 1 / 30, frame_index=1)
    with pytest.raises(ValueError, match="timestamp"):
        scene.snapshot(1 / 30)
    scene.set_playing("b", False)
    snapshot = scene.snapshot(1 / 30)
    assert snapshot[1].timestamp == 0
    with pytest.raises(ValueError, match="Out-of-order"):
        scene.update_object("a", synthetic_state(), 0)
    scene.update_object("a", synthetic_state(), 0, allow_rewind=True)
    assert scene.snapshot(0)[0].timestamp == 0


def test_transform_lifecycle_and_shared_sequence_instances():
    scene = SceneManager()
    state = synthetic_state()
    left, right = np.eye(4), np.eye(4)
    left[0, 3], right[0, 3] = -1, 1
    for name, transform in (("left", left), ("right", right)):
        scene.add_object(name, transform)
        scene.update_object(name, state, 0)
    snapshot = scene.snapshot(0)
    assert snapshot[0].gaussian_state is snapshot[1].gaussian_state
    np.testing.assert_equal(snapshot[0].object_to_world[0, 3], -1)
    revision = snapshot[0].revision
    scene.set_transform("left", left)
    assert scene.snapshot(0)[0].revision == revision
    left[0, 3] = -2
    scene.set_transform("left", left)
    assert scene.snapshot(0)[0].revision == revision + 1
    assert snapshot[0].object_to_world[0, 3] == -1
    scene.remove_object("right")
    assert len(scene.snapshot(0)) == 1
    scene.close()
    assert not scene.snapshot(0)


@pytest.mark.parametrize("linear", [np.diag([2, 1, 1]), np.diag([-1, 1, 1]),
                                    np.zeros((3, 3)), [[1, .1, 0], [0, 1, 0], [0, 0, 1]]])
def test_scene_rejects_unsupported_transform(linear):
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    with pytest.raises(ValueError):
        SceneManager().add_object("invalid", matrix)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), True])
def test_scene_rejects_invalid_media_time(timestamp):
    scene = SceneManager()
    scene.add_object("a")
    with pytest.raises(ValueError):
        scene.update_object("a", synthetic_state(), timestamp)


def test_scene_rejects_duplicate_identity_and_invalid_quality():
    scene = SceneManager()
    scene.add_object("a")
    with pytest.raises(ValueError, match="Duplicate"):
        scene.add_object("a")
    for quality in (-1, 4, True, 1.5):
        with pytest.raises(ValueError, match="quality"):
            scene.update_object("a", synthetic_state(), 0, quality=quality)


def test_camera_matches_reference_orbit_exactly():
    sample = {"azimuth": 35, "elevation": 11, "distance": 4, "target": [.1, -.2, .3]}
    settings = {"width": 97, "height": 65, "fov_degrees": 53, "up_axis": "z"}
    expected = camera_parameters(sample, settings)
    camera = Camera.orbit(sample, settings)
    np.testing.assert_array_equal(camera.world_view, expected.world_view)
    np.testing.assert_allclose(camera.eye, expected.eye, atol=1e-6)
    assert camera.fovx == expected.fovx
    assert camera.fovy == expected.fovy
    assert camera.intrinsics[0, 0] == pytest.approx(97 / (2 * math.tan(camera.fovx / 2)))
    assert camera.intrinsics[0, 2] == 48.5


def test_camera_and_object_matrix_snapshots_are_read_only():
    camera = Camera.orbit()
    obj = RenderObject("a", synthetic_state(), 0)
    with pytest.raises(ValueError):
        camera.world_view[0, 0] = 0
    with pytest.raises(ValueError):
        obj.object_to_world[0, 0] = 0


def test_backend_selection_does_not_import_unrequested_backends():
    with pytest.raises(ValueError, match="Unsupported"):
        create_renderer("spark")


def test_scene_delegates_one_joint_render_call():
    scene = SceneManager()
    for name in ("a", "b"):
        scene.add_object(name)
        scene.update_object(name, synthetic_state(), 2)

    class RecordingBackend:
        def render(self, objects, camera, timestamp):
            assert [obj.object_id for obj in objects] == ["a", "b"]
            assert timestamp == 2
            return "joint-result"

    assert scene.render(RecordingBackend(), Camera.orbit(), 2) == "joint-result"
