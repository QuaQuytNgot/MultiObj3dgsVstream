"""CPU self-tests; the CLI smoke additionally exercises the original CUDA renderer."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.content_preparation.assets import GaussianState, state_hash
from tools.content_preparation.metrics import MetricEvaluator, MetricResult, DistortionAggregator, aggregate
from tools.content_preparation.quality_profile import QualityProfile
from tools.content_preparation.renderer_adapter import render, camera_parameters, GPUExecutionGuard, RendererUnavailable
from tools.content_preparation.proxy import generate_proxy, load_proxy, render_proxy, stable_subset, validate_proxy_render


def fixture_state(frame=0):
    arrays = {"xyz": np.array([[-.2, 0, 0], [.2, 0, .1]], dtype=np.float32),
              "rotation": np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
              "scale": np.full((2, 3), math.log(.12), dtype=np.float32),
              "opacity": np.array([[1.5], [2]], dtype=np.float32),
              "sh": ((np.array([[.8, .2, .1], [.1, .4, .9]])-.5)/.28209479177387814)[:, :, None].astype(np.float32)}
    arrays["xyz"][:, 1] += frame*.01
    return GaussianState(arrays, {"frame": frame, "stable_ids": True}, np.array([12, 37]))


class RenderProfileTests(unittest.TestCase):
    def test_decoded_state_cpu_fixture_can_render(self):
        state = fixture_state()
        image = render(state, {"azimuth": 0, "elevation": 0, "distance": 3}, {"backend": "cpu_fixture", "width": 32, "height": 32})
        self.assertEqual(image["rgb"].shape, (32, 32, 3))
        self.assertGreater(image["alpha"].max(), .5)
        self.assertTrue(np.isfinite(image["depth"]).all())
        self.assertAlmostEqual(float(image["depth"][image["alpha"] > .1].mean()), 2.94, delta=.08)
        self.assertEqual(image["metadata"]["backend"], "cpu_fixture")
        changed = fixture_state()
        changed.arrays["sh"] = np.tile(changed.arrays["sh"], (1, 1, 4))
        with self.assertRaisesRegex(ValueError, "degree zero"):
            render(changed, {}, {"backend": "cpu_fixture"})

    def test_gpu_unavailable_is_explicit_not_fallback(self):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            with self.assertRaises(RendererUnavailable):
                render(fixture_state(), {}, {"backend": "upstream_cuda"})
        with self.assertRaisesRegex(ValueError, "Unknown renderer backend"):
            render(fixture_state(), {}, {"backend": "auto"})

    def test_camera_quality_independent_scale_and_wrap(self):
        settings = {"width": 64, "height": 32, "center": [0, 0, 0]}
        a = camera_parameters({"azimuth": -180, "scale": 2, "distance": 3}, settings)
        b = camera_parameters({"azimuth": 180, "scale": 2, "distance": 3}, settings)
        np.testing.assert_allclose(a.eye, b.eye, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(a.eye), 1.5)
        self.assertGreater(a.fovx, a.fovy)
        with self.assertRaises(ValueError):
            camera_parameters({"scale": 0}, settings)

    def test_lpips_same_image_real_pretrained_weights(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch unavailable; full smoke requires project environment.")
        import torch
        torch.set_num_threads(2)
        cache = Path(torch.hub.get_dir())/"checkpoints"
        if not (cache/"vgg.pth").exists() or not (cache/"vgg16-397923af.pth").exists():
            self.skipTest("Official pretrained LPIPS-VGG weights missing; production initialization fails with diagnostic.")
        with MetricEvaluator({"lpips": {"net": "vgg", "device": "cpu"}}) as evaluator:
            image = render(fixture_state(), {}, {"backend": "cpu_fixture", "width": 32, "height": 32})["rgb"]
            same = evaluator.compare(image, image)
            self.assertEqual(same.mse, 0)
            self.assertTrue(math.isinf(same.psnr))
            self.assertAlmostEqual(same.lpips, 0, places=7)
            different = evaluator.compare(image, np.clip(image+.05, 0, 1))
            self.assertGreater(different.mse, 0)
            self.assertGreater(different.lpips, 0)
            json.dumps(same.to_dict(), allow_nan=False)
            self.assertEqual(MetricResult.from_dict(same.to_dict()), same)
            self.assertEqual(evaluator.metadata()["lpips_input_range"], [-1, 1])

    def test_metric_validation_and_batch_one(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            MetricEvaluator({"lpips": {"batch_size": 2}})
        with MetricEvaluator({"lpips": False}) as evaluator:
            same = evaluator.compare(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))
            self.assertEqual(same.to_dict()["psnr"], "inf")
            self.assertIsNone(same.lpips)
            with self.assertRaises(ValueError):
                evaluator.compare(np.ones((4, 4, 3))*2, np.ones((4, 4, 3)))
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            DistortionAggregator("composite")
        self.assertAlmostEqual(DistortionAggregator()([MetricResult(.2, 1., .1), MetricResult(.4, 1., .1)]), .3)

    def test_all_aggregation_helpers(self):
        values = [1, 2, 3, 4]
        self.assertEqual(aggregate(values), 2.5)
        self.assertEqual(aggregate(values, "weighted_mean", [0, 0, 0, 1]), 4)
        self.assertEqual(aggregate(values, "median"), 2.5)
        self.assertAlmostEqual(aggregate(values, "p95"), 3.85)
        self.assertEqual(aggregate(values, "worst"), 4)
        self.assertTrue(math.isinf(aggregate([math.inf]*2, "p95")))
        with self.assertRaises(ValueError):
            aggregate(values, "weighted_mean", [-1, 1, 1, 1])

    def test_profile_indexing_roundtrip_and_gains(self):
        profile = QualityProfile(metadata={"object_id": "fixture"})
        for view, azimuth in enumerate([-45, 45]):
            for scale_bin, scale in enumerate([1, 2]):
                for frame in [0, 1]:
                    for quality in ["Q0", "Q1"]:
                        mse = .1 if quality == "Q0" else 0
                        profile.add({"quality": quality, "mode": "progressive", "view_bin": view, "azimuth": azimuth,
                                     "elevation": 0, "scale_bin": scale_bin, "scale": scale, "time_bin": frame, "frame": frame,
                                     "timestamp": frame*.1, "metrics": {"mse": mse, "psnr": "inf" if mse == 0 else 10., "lpips": mse}})
        self.assertEqual(profile.indexed("fixture", "Q0", 0, 0, 0, "progressive")["metrics"]["mse"], .1)
        self.assertEqual(len(profile.rows), 16)
        self.assertEqual(len(profile.transition_gains(["Q0", "Q1"])), 8)
        self.assertEqual(profile.transition_gains(["Q0", "Q1"])[0]["distortion_reduction"]["mse"], .1)
        query = profile.query("Q0", 0, 0, 1.5, .05, method="multilinear")
        self.assertEqual(len(query["source_samples"]), 8)
        self.assertAlmostEqual(query["metrics"]["mse"], .1)
        self.assertAlmostEqual(profile.query("Q0", 0, 0, 1.5, .05, method="inverse_distance")["metrics"]["mse"], .1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"profile.json"
            profile.save(path, ["Q0", "Q1"])
            loaded = QualityProfile.load(path)
            self.assertEqual(profile.rows, loaded.rows)
            json.dumps(loaded.to_dict(), allow_nan=False)
        with self.assertRaises(ValueError):
            profile.add({**profile.rows[0], "metrics": {"mse": 1}})

    def test_profile_azimuth_wrap_and_sparse_grid_diagnostic(self):
        rows = [{"quality": "Q0", "azimuth": a, "elevation": 0, "scale": 1, "timestamp": 0,
                 "metrics": {"mse": v, "psnr": 1., "lpips": .1}} for a, v in [(-170, 1.), (170, 3.)]]
        profile = QualityProfile(rows)
        self.assertAlmostEqual(profile.query("Q0", 180, 0, 1, 0, method="multilinear")["metrics"]["mse"], 2.)
        self.assertEqual(profile.query("Q0", 190, 0, 1, 0)["metrics"]["mse"], 1.)
        sparse = QualityProfile([{**rows[0], "scale": 2}, rows[1]])
        with self.assertRaisesRegex(ValueError, "complete local"):
            sparse.query("Q0", 180, 0, 1.5, 0, method="multilinear")
        self.assertGreater(sparse.query("Q0", 180, 0, 1.5, 0, method="inverse_distance")["metrics"]["mse"], 1.)

    def test_proxy_frozen_subset_load_render_validation(self):
        settings = {"backend": "cpu_fixture", "width": 32, "height": 32}
        with tempfile.TemporaryDirectory() as directory:
            generate_proxy(fixture_state(), {"count": 2}, directory, 0)
            record = generate_proxy(fixture_state(1), {"count": 2}, directory, 1)
            state = load_proxy(Path(directory)/"proxy.json", 1)
            self.assertEqual(state.ids.tolist(), [12, 37])
            image = render_proxy(Path(directory)/"proxy.json", {}, settings, 1)
            reference = render(fixture_state(1), {}, settings)
            comparison = validate_proxy_render(image, reference)
            self.assertEqual(comparison["silhouette_iou"], 1)
            self.assertEqual(comparison["alpha_mae"], 0)
            self.assertEqual(comparison["depth_mae"], 0)
            self.assertTrue(record["quality_independent"])
            one = stable_subset(fixture_state(), [37])
            self.assertEqual(one.ids.tolist(), [37])
            with self.assertRaisesRegex(ValueError, "disappeared"):
                stable_subset(one, [12])
            filename = Path(directory)/"frame_000001.npz"
            filename.write_bytes(filename.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_proxy(Path(directory)/"proxy.json", 1)

    def test_sequential_gpu_guard_rejects_overlap(self):
        before = GPUExecutionGuard.statistics()["views"]
        with GPUExecutionGuard():
            self.assertEqual(GPUExecutionGuard.statistics()["active"], 1)
            with self.assertRaisesRegex(RuntimeError, "Concurrent GPU work"):
                with GPUExecutionGuard():
                    pass
        with GPUExecutionGuard():
            pass
        self.assertEqual(GPUExecutionGuard.statistics()["peak_active"], 1)
        self.assertEqual(GPUExecutionGuard.statistics()["active"], 0)
        self.assertEqual(GPUExecutionGuard.statistics()["views"], before+2)


if __name__ == "__main__":
    unittest.main()
