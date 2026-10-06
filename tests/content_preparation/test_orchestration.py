"""Fast orchestration checks: no real training, dataset run or CUDA allocation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from tools.content_preparation.assets import GaussianState
from tools.content_preparation.checkpoint import Journal, file_lock, sha256
from tools.content_preparation.config import ROOT, STAGES, config_hash, frame_records, load_config, validate_config
from tools.content_preparation.pipeline import Pipeline
from tools.content_preparation.upstream import build_training_command, source_snapshot, verify_checkpoint_lineage


def tiny_config(output="unused", object_id="fixture"):
    return {"version": 1, "output": str(output), "objects": [
        {"id": object_id, "frames": [3, 5], "fps": 2,
         "dataset": {"kind": "synthetic"},
         "qualities": [{"id": "Q0"}, {"id": "Q1"}]}],
        "training": {"backend": "synthetic_fixture"},
        "renderer": {"backend": "cpu_fixture", "width": 32, "height": 32},
        "sampling": {"azimuth": [0, 90], "elevation": [0], "scales": [1]},
        "packaging": {"gof_frames": 2, "segment_frames": 1,
                      "refresh_frames": {"Base": 2, "E1": 1}}}


class ConfigurationTests(unittest.TestCase):
    def test_yaml_defaults_hashes_and_explicit_timestamps(self):
        raw = tiny_config()
        raw["objects"][0]["timestamps"] = [.25, 1.75]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(yaml.safe_dump(raw))
            cfg = load_config(path)
        self.assertEqual(cfg["metrics"]["batch_size"], 1)
        self.assertEqual(cfg["runtime"]["gpu_batch_size"], 1)
        self.assertEqual(cfg["metrics"]["distortion"], "mse")
        self.assertEqual(frame_records(cfg["objects"][0]), [{"frame": 3, "timestamp": .25}, {"frame": 5, "timestamp": 1.75}])
        self.assertEqual(config_hash(cfg), config_hash(dict(reversed(list(cfg.items())))))
        self.assertEqual(raw["renderer"], {"backend": "cpu_fixture", "width": 32, "height": 32})

    def test_invalid_media_mapping_has_clear_errors(self):
        changes = [{"frames": []}, {"fps": 0}, {"frames": [3, 3]},
                   {"frames": {"start": 5, "end": 3}},
                   {"timestamps": [0]}, {"timestamps": [0, 0]},
                   {"timestamps": [0, float("nan")]},
                   {"qualities": [{"id": "Q0", "resolution_scale": 1.5}]}]
        for change in changes:
            cfg = tiny_config()
            cfg["objects"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(cfg)

    def test_memory_contract_and_methodology_are_validated(self):
        edits = [lambda c: c["renderer"].update({"width": 16}),
                 lambda c: c.update({"metrics": {"batch_size": 2}}),
                 lambda c: c.update({"runtime": {"gpu_batch_size": 2}}),
                 lambda c: c.update({"metrics": {"lpips": False}}),
                 lambda c: c.update({"metrics": {"distortion": "psnr"}}),
                 lambda c: c["sampling"].update({"times": [999]}),
                 lambda c: c["objects"][0].update({"dataset": {"kind": "raw"}}),
                 lambda c: c["packaging"].update({"refresh_frames": {"Base": 2}}),
                 lambda c: c.update({"unsupported": 1})]
        for edit in edits:
            cfg = tiny_config()
            edit(cfg)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_config(cfg)


class TrainingCommandTests(unittest.TestCase):
    def setUp(self):
        # Command construction tests never open a socket in restricted CI.
        self.socket_patch = patch("tools.content_preparation.upstream.socket.socket")
        socket = self.socket_patch.start()
        socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 6009)
        self.addCleanup(self.socket_patch.stop)

    def training(self):
        return validate_config(tiny_config())["training"]

    def test_first_quality_foundation_and_temporal_orchestration(self):
        training = self.training()
        q = {"id": "Q1", "training_args": ["--position_lr_init", ".0001"]}
        first = build_training_command("data", "model", q, training, 25, foundation="lower.ply")
        self.assertIn("--dynamic_opacity", first)
        self.assertEqual(first[first.index("--foundation_gs_path") + 1], "lower.ply")
        self.assertEqual(first[first.index("--data_device") + 1], "cpu")
        dynamic = build_training_command("data", "model", q, training, 25, previous="previous.ply")
        self.assertIn("--dynamic_lapis", dynamic)
        self.assertNotIn("--foundation_gs_path", dynamic)
        self.assertEqual(dynamic[dynamic.index("--densify_until_iter") + 1], "0")
        self.assertGreater(int(dynamic[dynamic.index("--opacity_reset_interval") + 1]), 25)
        self.assertEqual(dynamic[dynamic.index("--save_iterations") + 1], "25")

    def test_protected_args_cannot_override_lineage_or_memory(self):
        extras = [["--initial_gs_path", "foreign.ply"], ["--initial_gs_path=foreign.ply"],
                  ["--initial_gs_pat", "foreign.ply"], ["--iteration=10"],
                  ["--start_checkpoint", "foreign.pth"], ["-sforeign"], ["-mforeign"],
                  ["--source", "foreign"], ["--data_device", "cuda"],
                  ["--densify_until_iter", "10"]]
        for args in extras:
            with self.subTest(args=args), self.assertRaises(ValueError):
                build_training_command("data", "model", {"id": "Q0", "training_args": args}, self.training(), 25, previous="previous.ply")

    def test_lineage_checks_hashes_prefix_and_temporal_static_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {(frame, q): root / f"{frame}_{q}.ply" for frame in (1, 2) for q in ("Q0", "Q1")}
            for key, path in paths.items():
                path.write_bytes(str(key).encode())
            def state(n, motion=0):
                xyz = np.zeros((n, 3), dtype="float32")
                xyz[:, 0] = motion
                return GaussianState({"xyz": xyz, "rotation": np.tile([1, 0, 0, 0], (n, 1)),
                                      "scale": np.zeros((n, 3)), "opacity": np.zeros((n, 1)),
                                      "sh": np.zeros((n, 3, 1))})
            states = {str(paths[(1, "Q0")]): state(2), str(paths[(1, "Q1")]): state(3),
                      str(paths[(2, "Q0")]): state(2, .1), str(paths[(2, "Q1")]): state(3, .2)}
            rows = [{"frame": frame, **{q: str(paths[(frame, q)]) for q in ("Q0", "Q1")}} for frame in (1, 2)]
            commands = []
            for frame in (1, 2):
                for q in ("Q0", "Q1"):
                    argv = []
                    if frame == 1 and q == "Q1":
                        argv = ["--foundation_gs_path", str(paths[(1, "Q0")]), "--dynamic_opacity"]
                    if frame == 2:
                        argv = ["--initial_gs_path", str(paths[(1, q)]), "--dynamic_lapis", "--iterations", "25",
                                "--densify_until_iter", "0", "--opacity_reset_interval", "26"]
                    commands.append({"frame": frame, "level": q, "argv": argv, "sha256": sha256(paths[(frame, q)])})
            manifest = {"frames": rows, "commands": commands}
            with patch("tools.content_preparation.assets.read_ply", side_effect=lambda p: states[str(p)]):
                self.assertTrue(verify_checkpoint_lineage(manifest, ["Q0", "Q1"]))
                states[str(paths[(1, "Q1")])].arrays["xyz"][0, 0] = 99
                with self.assertRaisesRegex(ValueError, "prefix"):
                    verify_checkpoint_lineage(manifest, ["Q0", "Q1"])
                states[str(paths[(1, "Q1")])].arrays["xyz"][0, 0] = 0
                states[str(paths[(2, "Q1")])].arrays["opacity"][0, 0] = 1
                with self.assertRaisesRegex(ValueError, "static state"):
                    verify_checkpoint_lineage(manifest, ["Q0", "Q1"])
            paths[(1, "Q0")].write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_checkpoint_lineage(manifest, ["Q0", "Q1"])


class ResumeTests(unittest.TestCase):
    def test_resume_completed_tasks_retry_failed_view_and_protect_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.write_text("input")
            first, second = root / "view1.json", root / "view2.json"
            counts = {"first": 0, "second": 0}
            def action_first():
                counts["first"] += 1
                first.write_text('{"view":1}')
                return 1, [first]
            def crash_second():
                counts["second"] += 1
                second.write_text("partial")
                raise RuntimeError("simulated crash at next view")
            journal = Journal(root)
            journal.run("profile/view1", {"resolution": 32}, [source], action_first)
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                journal.run("profile/view2", {"resolution": 32}, [source], crash_second)
            self.assertEqual(journal.data["tasks"]["profile/view2"]["status"], "failed")
            resumed = Journal(root, resume=True)
            self.assertEqual(resumed.run("profile/view1", {"resolution": 32}, [source], action_first), 1)
            def recover_second():
                counts["second"] += 1
                second.write_text('{"view":2}')
                return 2, [second]
            self.assertEqual(resumed.run("profile/view2", {"resolution": 32}, [source], recover_second), 2)
            self.assertEqual(counts, {"first": 1, "second": 2})
            self.assertEqual((resumed.skipped, resumed.executed), (1, 1))
            first.write_text("changed")
            with self.assertRaises(FileExistsError):
                Journal(root, resume=True).run("profile/view1", {"resolution": 32}, [source], action_first)
            Journal(root, resume=True, overwrite=True).run("profile/view1", {"resolution": 32}, [source], action_first)
            self.assertEqual(counts["first"], 2)

    def test_fingerprint_changes_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "result"
            source.write_text("input")
            def action():
                output.write_text("result")
                return {"ok": True}, [output]
            Journal(root).run("encode/Q0", {"precision": "f32"}, [source], action)
            with self.assertRaises(FileExistsError):
                Journal(root, resume=True).run("encode/Q0", {"precision": "f16"}, [source], action)
            source.write_text("changed input")
            with self.assertRaises(FileExistsError):
                Journal(root, resume=True).run("encode/Q0", {"precision": "f32"}, [source], action)
            self.assertEqual(Journal(root).data["tasks"]["encode/Q0"]["outputs"][0]["path"], "result")

    def test_process_lock_blocks_second_gpu_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gpu.lock"
            with file_lock(path):
                with self.assertRaisesRegex(RuntimeError, "sequential"):
                    with file_lock(path):
                        self.fail("A second owner acquired an active GPU lock")


class PipelineTests(unittest.TestCase):
    def test_dry_run_does_not_create_output_and_covers_requested_stage(self):
        before = source_snapshot()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "absent"
            cfg = validate_config(tiny_config(output))
            result = Pipeline(cfg, Path(folder) / "config.yaml").run("all", dry_run=True)
            self.assertFalse(output.exists())
            self.assertEqual(result["stages"], list(STAGES))
            self.assertEqual(result["objects"][0]["sampled_views_per_frame"], 2)
            self.assertEqual(Pipeline(cfg, "unused").run("encode", dry_run=True)["stages"], ["encode"])
        self.assertEqual(source_snapshot(), before)

    def test_object_quality_frame_order_and_no_upstream_changes(self):
        before = source_snapshot()
        with tempfile.TemporaryDirectory() as folder:
            raw = tiny_config(Path(folder) / "prepared")
            obj = copy.deepcopy(raw["objects"][0])
            obj["id"] = "second"
            raw["objects"].append(obj)
            cfg = validate_config(raw)
            pipeline = Pipeline(cfg, "unused")
            seen = []
            # Spy preserves real asset serialization and records exact execution order.
            from tools.content_preparation.assets import save_state
            def spy(path, state):
                path = Path(path)
                seen.append((path.parents[2].name, path.parent.name, state.metadata["frame"]))
                return save_state(path, state)
            pipeline.run("preprocess")
            pipeline = Pipeline(cfg, "unused", resume=True)
            with patch("tools.content_preparation.assets.save_state", side_effect=spy):
                pipeline.run("train")
            self.assertEqual(seen, [(o, q, f) for o in ("fixture", "second") for q in ("Q0", "Q1") for f in (3, 5)])
            journal = json.loads((Path(raw["output"]) / ".preparation" / "journal.json").read_text())
            self.assertTrue(all(entry["status"] == "complete" for entry in journal["tasks"].values()))
            validation = json.loads((Path(raw["output"]) / "validation_upstream.json").read_text())
            self.assertTrue(validation["unchanged"])
            self.assertEqual(source_snapshot(), before)

    def test_unowned_output_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            important = root / "existing.txt"
            important.write_text("retain")
            cfg = validate_config(tiny_config(root))
            with self.assertRaisesRegex(FileExistsError, "not owned"):
                Pipeline(cfg, "unused", overwrite=True).run("preprocess")
            self.assertEqual(important.read_text(), "retain")

    def test_prepared_source_mutation_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = tiny_config(root / "prepared")
            raw["training"]["backend"] = "upstream"
            raw["renderer"]["backend"] = "upstream_cuda"
            raw["objects"][0]["dataset"] = {"kind": "prepared", "source_template": str(root / "source" / "{quality}" / "{frame}")}
            sources = []
            for quality in ("Q0", "Q1"):
                for frame in (3, 5):
                    path = root / "source" / quality / str(frame) / "train" / "image.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"tiny input fixture; preprocessing imports without rendering")
                    for split in ("train", "test"):
                        (path.parent.parent / f"transforms_{split}.json").write_text('{"frames": []}')
                    # Existing initialization avoids generating upstream's 100k
                    # random points in this metadata-only mutation regression.
                    (path.parent.parent / "points3d.ply").write_bytes(b"unused existing initialization")
                    sources.append(path)
            cfg = validate_config(raw)
            Pipeline(cfg, "unused").run("preprocess")
            matched = Pipeline(cfg, "unused", resume=True).run("preprocess")
            self.assertEqual(matched["tasks_executed"], 0)
            sources[0].write_bytes(b"changed prepared training image")
            with self.assertRaises(FileExistsError):
                Pipeline(cfg, "unused", resume=True).run("preprocess")
            rebuilt = Pipeline(cfg, "unused", resume=True, overwrite=True).run("preprocess")
            self.assertGreater(rebuilt["tasks_executed"], 0)

    def test_progressive_checkpoint_import_requires_original_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {q: root / f"{q}.ply" for q in ("Q0", "Q1")}
            for q, path in paths.items():
                path.write_bytes(f"checkpoint fixture {q}".encode())
            imported = root / "imported.json"
            imported.write_text(json.dumps({"frames": [{"frame": 3, **{q: str(p) for q, p in paths.items()}}],
                                            "commands": [{"frame": 3, "level": q, "argv": ["train.py"]} for q in paths]}))
            raw = tiny_config(root / "prepared")
            raw["delivery_modes"] = ["progressive"]
            raw["training"]["backend"] = "existing"
            raw["renderer"]["backend"] = "upstream_cuda"
            raw["objects"][0]["frames"] = [3]
            raw["objects"][0]["dataset"] = {"kind": "checkpoints", "checkpoint_manifest": str(imported)}
            cfg = validate_config(raw)
            Pipeline(cfg, "unused").run("preprocess")
            with self.assertRaisesRegex(ValueError, "(?i)(hash|provenance)"):
                Pipeline(cfg, "unused", resume=True).run("train")


if __name__ == "__main__":
    unittest.main()
