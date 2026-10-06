"""Exercise preparation compatibility without importing Open3D or using a GPU."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from plyfile import PlyData
from PIL import Image

from tools.content_preparation.compatibility import (
    compatible_nerf_loader, configure_training_compatibility, render_2d_image, store_point_cloud,
)


class CompatibilityTests(unittest.TestCase):
    def fake_nerf_readers(self):
        readers = ModuleType("scene.dataset_readers")
        readers.__dict__.update(np=np, Image=Image)
        exec("""
def readCamerasFromTransforms(path, transformsfile, white_background):
    arr = np.array([[[.75, .5, 1.]]])
    return Image.fromarray(np.array(arr * 255., dtype=np.byte), "RGB")

def readNerfSyntheticInfo(path, white_background, eval):
    return readCamerasFromTransforms(path, "transforms_train.json", white_background)
""", readers.__dict__)
        return readers

    def test_original_nerf_body_uses_unsigned_rgb_without_mutating_its_namespace(self):
        readers = self.fake_nerf_readers()
        original = readers.readCamerasFromTransforms
        loader = compatible_nerf_loader(readers)
        image = loader("input", False, True)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (191, 127, 255))
        self.assertIs(readers.readCamerasFromTransforms, original)
        self.assertIs(original.__globals__["np"], np)
        self.assertIs(original.__globals__["Image"], Image)
        self.assertIs(np.byte, np.int8)

    def test_training_bootstrap_changes_only_process_local_blender_callback(self):
        readers = self.fake_nerf_readers()
        scene = ModuleType("scene")
        unchanged = object()
        before = {"Blender": readers.readNerfSyntheticInfo, "Colmap": unchanged}
        scene.sceneLoadTypeCallbacks = before
        scene.GaussianModel = unchanged
        with patch.dict(sys.modules, {"scene": scene, "scene.dataset_readers": readers}):
            configure_training_compatibility()
        self.assertIsNot(scene.sceneLoadTypeCallbacks, before)
        self.assertIs(before["Blender"], readers.readNerfSyntheticInfo)
        self.assertIs(scene.sceneLoadTypeCallbacks["Colmap"], unchanged)
        self.assertIs(scene.GaussianModel, unchanged)
        self.assertEqual(scene.sceneLoadTypeCallbacks["Blender"]("input", False, True).getpixel((0, 0)), (191, 127, 255))

    def test_point_cloud_initialization_round_trips_unsigned_rgb(self):
        xyz = np.array([[.5, -.25, 1.], [2., 3., 4.]], dtype=np.float32)
        rgb = np.array([[127, 128, 255], [200, 254, 0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/points3d.ply"
            store_point_cloud(path, xyz, rgb)
            vertices = PlyData.read(path)["vertex"].data
            self.assertEqual(vertices.dtype.names, ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"))
            for index, name in enumerate(("x", "y", "z")):
                np.testing.assert_array_equal(vertices[name], xyz[:, index])
            for index, name in enumerate(("red", "green", "blue")):
                self.assertEqual(vertices.dtype[name], np.dtype("uint8"))
                np.testing.assert_array_equal(vertices[name], rgb[:, index].astype(np.uint8))
            for name in ("nx", "ny", "nz"):
                np.testing.assert_array_equal(vertices[name], np.zeros(2))

    def test_invalid_point_cloud_data_is_rejected_before_file_creation(self):
        valid_xyz, valid_rgb = np.zeros((2, 3)), np.ones((2, 3)) * 200
        invalid = [
            (np.zeros((2, 2)), valid_rgb), (valid_xyz, np.zeros((1, 3))),
            (np.full((2, 3), np.nan), valid_rgb), (valid_xyz, np.full((2, 3), np.inf)),
            (valid_xyz, np.full((2, 3), -1)), (valid_xyz, np.full((2, 3), 256)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points3d.ply"
            for xyz, rgb in invalid:
                with self.subTest(xyz=xyz.tolist(), rgb=rgb.tolist()), self.assertRaises(ValueError):
                    store_point_cloud(path, xyz, rgb)
                self.assertFalse(path.exists())

    def test_original_preprocessing_body_runs_with_only_legacy_renderer_option_removed(self):
        module = ModuleType("dataset_prepare")
        device, material = object(), object()
        renderer = Mock(return_value=device)
        rendering = SimpleNamespace(OffscreenRenderer=renderer, MaterialRecord=Mock(return_value=material))
        module.__dict__.update(rendering=rendering, events=[])
        exec("""
def render_2d_image(pc_path, render_dir, pose_json_path, pt_size=1, width=600, height=600):
    material = rendering.MaterialRecord()
    events.append((pc_path, render_dir, pose_json_path, material))
    device = rendering.OffscreenRenderer(width, height, headless=False, requested_setting=pt_size)
    return device
""", module.__dict__)
        original = module.render_2d_image
        globals_before = dict(module.__dict__)
        with patch.dict(sys.modules, {"dataset_prepare": module}):
            result = render_2d_image("input.ply", "images", "poses.json", pt_size=2, width=64, height=32)
        self.assertIs(result, device)
        renderer.assert_called_once_with(64, 32, requested_setting=2)
        rendering.MaterialRecord.assert_called_once_with()
        self.assertEqual(module.events, [("input.ply", "images", "poses.json", material)])
        self.assertIs(module.rendering, rendering)
        self.assertIs(module.render_2d_image, original)
        self.assertIs(original.__globals__["rendering"], rendering)
        self.assertEqual(set(module.__dict__), set(globals_before))
        for name, value in globals_before.items():
            self.assertIs(module.__dict__[name], value)


if __name__ == "__main__":
    unittest.main()
