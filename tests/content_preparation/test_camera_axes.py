"""CPU-only orbit conventions for prepared Y-up and Z-up objects."""
import math
import unittest

import numpy as np

from tools.content_preparation.assets import synthetic_state
from tools.content_preparation.config import validate_config
from tools.content_preparation.renderer_adapter import camera_parameters, render


def axis_config():
    return {"version": 1, "output": "unused", "objects": [{"id": "axes",
            "dataset": {"kind": "synthetic"}, "frames": [0], "qualities": [{"id": "Q0"}]}],
            "training": {"backend": "synthetic_fixture"},
            "renderer": {"backend": "cpu_fixture", "width": 32, "height": 32}}


class CameraAxesTests(unittest.TestCase):
    def test_default_y_axis_preserves_existing_orbit(self):
        sample = {"azimuth": 30, "elevation": 20, "distance": 4, "scale": 2}
        settings = {"center": [0.1, 0.2, 0.3]}
        camera = camera_parameters(sample, settings)
        az, el = math.radians(30), math.radians(20)
        expected = np.array(settings["center"]) + 2 * np.array([math.sin(az)*math.cos(el), math.sin(el), math.cos(az)*math.cos(el)])
        np.testing.assert_array_equal(camera.eye, expected)
        explicit = camera_parameters(sample, {**settings, "up_axis": "y"})
        np.testing.assert_array_equal(camera.world_view, explicit.world_view)
        self.assertEqual(camera.up_axis, "y")
        np.testing.assert_allclose(camera_parameters({}, {}).eye, [0, 0, 3])

    def test_z_axis_azimuth_and_elevation(self):
        settings = {"up_axis": "z"}
        np.testing.assert_allclose(camera_parameters({"azimuth": 0}, settings).eye, [0, 3, 0], atol=1e-12)
        np.testing.assert_allclose(camera_parameters({"azimuth": 90}, settings).eye, [3, 0, 0], atol=1e-12)
        elevated = camera_parameters({"azimuth": 0, "elevation": 30}, settings)
        np.testing.assert_allclose(elevated.eye, [0, 3*math.cos(math.pi/6), 1.5], atol=1e-12)
        np.testing.assert_allclose(camera_parameters({}, settings).rotation[1], [0, 0, -1], atol=1e-12)

    def test_orthonormal_world_view_and_pole_alternate_up(self):
        target = np.array([0.2, -0.1, 0.4])
        for axis in ("y", "z"):
            for azimuth in (0, 45, 90, 180):
                for elevation in (-90, -89.99, -20, 0, 20, 89.99, 90):
                    with self.subTest(axis=axis, azimuth=azimuth, elevation=elevation):
                        camera = camera_parameters({"azimuth": azimuth, "elevation": elevation},
                                                   {"up_axis": axis, "center": target})
                        self.assertTrue(np.isfinite(camera.world_view).all())
                        np.testing.assert_allclose(camera.rotation @ camera.rotation.T, np.eye(3), atol=1e-12)
                        self.assertAlmostEqual(np.linalg.det(camera.rotation), 1, places=12)
                        np.testing.assert_allclose(camera.world_view @ np.r_[camera.eye, 1], [0, 0, 0, 1], atol=2e-7)
                        np.testing.assert_allclose(camera.world_view @ np.r_[target, 1], [0, 0, 3, 1], atol=2e-7)

    def test_configuration_and_render_axis_metadata(self):
        cfg = validate_config(axis_config())
        self.assertEqual(cfg["renderer"]["up_axis"], "y")
        raw = axis_config()
        raw["renderer"]["up_axis"] = "z"
        self.assertEqual(validate_config(raw)["renderer"]["up_axis"], "z")
        image = render(synthetic_state(0, 2), {}, {"backend": "cpu_fixture", "width": 32, "height": 32, "up_axis": "z"})
        self.assertEqual(image["metadata"]["up_axis"], "z")
        for bad in ("x", "auto", None, 1, ["y"]):
            with self.subTest(bad=bad):
                raw = axis_config()
                raw["renderer"]["up_axis"] = bad
                with self.assertRaisesRegex(ValueError, "renderer.up_axis"):
                    validate_config(raw)
                with self.assertRaisesRegex(ValueError, "renderer.up_axis"):
                    camera_parameters({}, {"up_axis": bad})
        raw = axis_config()
        raw["renderer"]["backend"] = "auto"
        with self.assertRaisesRegex(ValueError, "renderer backend"):
            validate_config(raw)


if __name__ == "__main__":
    unittest.main()
