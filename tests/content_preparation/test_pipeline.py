"""One-frame CPU integration: delivered segments, actual LPIPS and resumption."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import yaml

from tools.content_preparation.assets import load_state, state_hash, synthetic_state, write_ply
from tools.content_preparation.checkpoint import sha256
from tools.content_preparation.config import load_config
from tools.content_preparation.manifest import load_manifest
from tools.content_preparation.pipeline import Pipeline
from tools.content_preparation.prepare_content import main as cli_main


def mini_config(output):
    return {
        "version": 1, "output": str(output),
        "delivery_modes": ["independent", "progressive"],
        "objects": [{"id": "cpu_mini", "dataset": {"kind": "synthetic"},
                     "frames": [0], "fps": 30,
                     "qualities": [{"id": "Q0", "resolution_scale": 2},
                                   {"id": "Q1", "resolution_scale": 1}]}],
        "training": {"backend": "synthetic_fixture", "sh_degree": 0},
        "encoding": {"precision": "f16", "compression": "zlib", "level": 6},
        "packaging": {"gof_frames": 1, "segment_frames": 1,
                      "refresh_frames": {"Base": 1, "E1": 1}},
        "sampling": {"azimuth": [0], "elevation": [0], "scales": [1.0],
                     "distance": 3.0, "times": "all"},
        "renderer": {"backend": "cpu_fixture", "width": 32, "height": 32},
        "metrics": {"lpips": True, "lpips_net": "vgg", "device": "cpu",
                    "batch_size": 1, "distortion": "mse"},
        "proxy": {"backend": "fixed_gaussian_subset", "max_gaussians": 4},
        "runtime": {"seed": 7, "gpu_batch_size": 1},
    }


def deliverable_hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".preparation" not in p.relative_to(root).parts}


class PipelineTests(unittest.TestCase):
    def test_progressive_import_requires_original_hash_and_argv_before_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            source.mkdir()
            for q, count in (("Q0", 8), ("Q1", 12)):
                write_ply(source / f"{q}.ply", synthetic_state(0, count))
            for missing in ("sha256", "argv"):
                with self.subTest(missing=missing):
                    commands = [{"frame": 0, "level": q, "path": str(source / f"{q}.ply"),
                                 "sha256": sha256(source / f"{q}.ply"), "argv": ["train.py"]}
                                for q in ("Q0", "Q1")]
                    commands[0].pop(missing)
                    manifest_path = temporary / f"import_{missing}.json"
                    manifest_path.write_text(json.dumps({"frames": [{"frame": 0,
                        "Q0": str(source / "Q0.ply"), "Q1": str(source / "Q1.ply")}],
                        "commands": commands}))
                    config_path = temporary / f"config_{missing}.yaml"
                    cfg = mini_config(temporary / f"output_{missing}")
                    cfg["objects"][0]["dataset"] = {"kind": "checkpoints", "checkpoint_manifest": str(manifest_path)}
                    cfg["training"]["backend"] = "existing"
                    cfg["renderer"]["backend"] = "upstream_cuda"
                    config_path.write_text(yaml.safe_dump(cfg))
                    pipeline = Pipeline(load_config(config_path), config_path)
                    with redirect_stdout(io.StringIO()):
                        pipeline.run("preprocess")
                        with self.assertRaisesRegex(ValueError, "ORIGINAL training argv and sha256"):
                            pipeline.run("train")
                    root = pipeline.output / "cpu_mini"
                    self.assertFalse((root / "checkpoints/Q0/0.json").exists())
                    self.assertFalse((root / "training.json").exists())
                    journal = json.loads((pipeline.output / ".preparation/journal.json").read_text())
                    self.assertNotIn("cpu_mini/train/Q0/0", journal["tasks"])

    def test_prepared_sources_copied_and_initialized_without_mutation(self):
        try:
            from plyfile import PlyData
        except ImportError:
            self.skipTest("Prepared-source initialization requires the project plyfile dependency")
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "prepared_sources"
            for q in ("Q0", "Q1"):
                source = source_root / q
                source.mkdir(parents=True)
                for split in ("train", "test"):
                    (source / f"transforms_{split}.json").write_text(json.dumps({"camera_angle_x": 1.0, "frames": []}))
                (source / "source_marker.bin").write_bytes(b"immutable prepared-source fixture")
            before = deliverable_hashes(source_root)
            config_path = temporary / "prepared.yaml"
            cfg = mini_config(temporary / "output")
            cfg["objects"][0]["dataset"] = {"kind": "prepared", "source_template": str(source_root / "{quality}")}
            cfg["training"]["backend"] = "upstream"
            cfg["renderer"]["backend"] = "upstream_cuda"
            config_path.write_text(yaml.safe_dump(cfg))
            pipeline = Pipeline(load_config(config_path), config_path)
            with redirect_stdout(io.StringIO()):
                result = pipeline.run("preprocess")
            self.assertEqual(result["tasks_executed"], 2)
            self.assertEqual(before, deliverable_hashes(source_root))
            prepared = json.loads((pipeline.output / "cpu_mini/preprocess.json").read_text())
            copies = prepared["frames"][0]["sources"]
            hashes = []
            for q, owned in copies.items():
                owned = Path(owned)
                self.assertTrue(owned.is_relative_to(pipeline.output))
                self.assertNotEqual(owned, source_root / q)
                points = PlyData.read(owned / "points3d.ply")["vertex"].data
                self.assertEqual(len(points), 100000)
                expected = (np.random.RandomState(7).random_sample((100000, 3)) * 2.6 - 1.3)[0].astype(np.float32)
                np.testing.assert_array_equal(np.array([points["x"][0], points["y"][0], points["z"][0]]), expected)
                hashes.append(sha256(owned / "points3d.ply"))
                self.assertEqual((owned / "source_marker.bin").read_bytes(), b"immutable prepared-source fixture")
                self.assertFalse((source_root / q / "points3d.ply").exists())
            self.assertEqual(hashes[0], hashes[1])
            pipeline.resume = True
            with redirect_stdout(io.StringIO()):
                resumed = pipeline.run("preprocess")
            self.assertEqual(resumed["tasks_executed"], 0)
            self.assertEqual(before, deliverable_hashes(source_root))

    def test_cpu_end_to_end_resume_protection_and_payload_only_decode(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Actual LPIPS integration requires the project's PyTorch environment")
        cache = Path(torch.hub.get_dir()) / "checkpoints"
        if not all((cache / name).is_file() for name in ("vgg.pth", "vgg16-397923af.pth")):
            self.skipTest("Offline LPIPS integration requires cached official VGG backbone and linear weights")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                temporary = Path(temporary)
                output = temporary / "output"
                config_path = temporary / "mini.yaml"
                config_path.write_text(yaml.safe_dump(mini_config(output)))
                config = load_config(config_path)
                pipeline = Pipeline(config, config_path)
                with redirect_stdout(io.StringIO()):
                    result = pipeline.run("all")
                self.assertGreater(result["tasks_executed"], 0)
                self.assertEqual(result["stages"], ["preprocess", "train", "export", "encode",
                    "package", "decode", "profile", "proxy", "manifest"])
                root = output / "cpu_mini"
                manifest = load_manifest(root / "manifest.json", validate_files=True)
                self.assertEqual(set(manifest["delivery_modes"]), {"independent", "progressive"})
                self.assertEqual(manifest["quality_reference"], "highest-quality decoded deliverable representation")
                self.assertEqual(manifest["planner_distortion"], "mse")
                profile = json.loads((root / "profiles/profile.json").read_text())
                rows = profile.get("rows", profile.get("samples", []))
                self.assertEqual(len(rows), 4)
                for row in rows:
                    self.assertIsInstance(row["metrics"]["lpips"], (int, float))
                    self.assertEqual(row["reference_state"], "decoded")
                    if row["quality"] == "Q1":
                        self.assertEqual(row["metrics"]["mse"], 0)
                        self.assertEqual(row["metrics"]["psnr"], "inf")
                        self.assertLess(abs(row["metrics"]["lpips"]), 1e-7)
                proxy = json.loads((root / "proxy/index.json").read_text())
                self.assertTrue(proxy["quality_independent"])
                self.assertEqual(proxy["selected_count"], 4)
                self.assertTrue(json.loads((root / "validation.json").read_text())["passed"])
                self.assertTrue(json.loads((output / "validation_upstream.json").read_text())["unchanged"])
                independent = {q: state_hash(load_state(root / "decoded/independent" / q / "0.npz"))
                               for q in ("Q0", "Q1")}
                for q, layer in (("Q0", "Base"), ("Q1", "E1")):
                    self.assertEqual(independent[q], state_hash(load_state(root / "decoded/progressive" / layer / "0.npz")))

                before = deliverable_hashes(output)
                pipeline.resume = True
                with redirect_stdout(io.StringIO()):
                    resumed = pipeline.run("all")
                self.assertEqual(resumed["tasks_executed"], 0)
                self.assertGreater(resumed["tasks_skipped"], 0)
                self.assertEqual(before, deliverable_hashes(output))
                pipeline.resume = False
                with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                    pipeline.run("all")
                self.assertEqual(before, deliverable_hashes(output))

                # The delivered container is authoritative. Remove checkpoint,
                # exported state, standalone codec files and decoded caches.
                for directory in ("input", "checkpoints", "qualities", "encoded", "decoded", ".work"):
                    shutil.rmtree(root / directory, ignore_errors=True)
                captured = io.StringIO()
                with redirect_stdout(captured):
                    status = cli_main(["--config", str(config_path), "--stage", "decode", "--overwrite"])
                self.assertEqual(status, 0)
                cli_result = json.loads("{" + captured.getvalue().split("\n{", 1)[1])
                self.assertEqual(cli_result["tasks_executed"], 5)
                for q, layer in (("Q0", "Base"), ("Q1", "E1")):
                    self.assertEqual(independent[q], state_hash(load_state(root / "decoded/independent" / q / "0.npz")))
                    self.assertEqual(independent[q], state_hash(load_state(root / "decoded/progressive" / layer / "0.npz")))
                self.assertEqual(json.loads((root / "decoding.json").read_text())["source"],
                                 "requestable segment containers only")
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
