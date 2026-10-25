"""Controls are testable without a display; GLFW/interop smoke needs target hardware."""
import numpy as np
from multiobj3dgs.client.viewer import CameraController


def test_camera_controls_preserve_intrinsics_and_rigid_view():
    controller=CameraController(800,600)
    original=controller.camera()
    controller.orbit(50,10);controller.pan(5,7);controller.zoom(2)
    camera=controller.camera()
    assert np.array_equal(camera.K,original.K)
    assert np.allclose(camera.world_to_camera[:3,:3]@camera.world_to_camera[:3,:3].T,np.eye(3))
    assert not np.allclose(camera.center,original.center)
    controller.resize(1600,1200)
    assert controller.camera().K[0,0]==2*original.K[0,0]


def test_free_camera_motion_translates_camera_and_target():
    controller=CameraController()
    before=controller.camera().center.copy()
    controller.move(1,0,0,.1)
    assert not np.allclose(controller.camera().center,before)
