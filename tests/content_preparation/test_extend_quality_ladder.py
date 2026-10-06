"""CPU-only extension checks; original trainer subprocesses are replaced."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData
import yaml

from tools.extend_quality_ladder import LadderExtension
from tools.content_preparation.assets import GaussianState, read_ply, synthetic_state, write_ply
from tools.content_preparation.checkpoint import sha256, write_json
from tools.content_preparation.config import load_config
from tools.content_preparation.upstream import verify_checkpoint_lineage


def refined(parent):
    extra = synthetic_state(0, 1, seed=23)
    arrays = {k: np.concatenate([v.copy(), extra.arrays[k]], axis=0)
              for k, v in parent.arrays.items()}
    arrays["opacity"][:len(parent.ids)] += np.float32(.1)
    return GaussianState(arrays, {"stable_ids": True})


def moved(parent):
    arrays = {k: v.copy() for k, v in parent.arrays.items()}
    arrays["xyz"][:, 0] += np.float32(.02)
    return GaussianState(arrays, {"stable_ids": True})


class ExtensionTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        poses, images, existing = root / "poses", root / "native", root / "existing"
        poses.mkdir(); existing.mkdir()
        for split in ("train", "test"):
            write_json(poses / f"transforms_{split}.json", {"camera_angle_x": .7,
                "frames": [{"file_path": f"./{split}/r_0", "transform_matrix": np.eye(4).tolist()}]})
        rows, commands = [], []
        first = {"Q0": synthetic_state(0, 2)}
        first["Q1"] = refined(first["Q0"])
        previous = {}
        for frame in (1051, 1052):
            row = {"frame": frame}
            for split in ("train", "test"):
                directory = images / str(frame) / split
                directory.mkdir(parents=True)
                Image.new("RGBA", (1024, 1024), (30, 80, 140, 255)).save(directory / "r_0.png")
            for q in ("Q0", "Q1"):
                state = first[q] if frame == 1051 else moved(read_ply(previous[q]))
                path = existing / f"{q}_{frame}.ply"
                write_ply(path, state)
                argv = ["python", "train.py", "--iterations", "4"]
                if frame == 1051 and q == "Q1":
                    argv += ["--dynamic_opacity", "--foundation_gs_path", row["Q0"]]
                elif frame == 1052:
                    argv += ["--dynamic_lapis", "--initial_gs_path", str(previous[q]),
                             "--densify_until_iter", "0", "--opacity_reset_interval", "5"]
                row[q] = str(path)
                commands.append({"frame": frame, "level": q, "path": str(path),
                                 "sha256": sha256(path), "argv": argv})
                previous[q] = path
            rows.append(row)
        source_manifest = root / "original.json"
        spec = root / "original_data_spec.json"
        write_json(spec, {"raw": "preserved fixture provenance"})
        write_json(source_manifest, {"frames": rows, "commands": commands,
            "data_spec": str(spec), "data_spec_sha256": sha256(spec)})
        work = root / "work"
        cfg_path = root / "four.yaml"
        cfg_path.write_text(yaml.safe_dump({"version": 1, "output": str(root / "encoded"),
            "objects": [{"id": "fixture", "frames": [1051, 1052], "qualities": [
                {"id": q, "resolution_scale": scale} for q, scale in
                zip(("Q0", "Q1", "Q2", "Q3"), (8, 4, 2, 1))],
                "dataset": {"kind": "checkpoints", "checkpoint_manifest": str(work / "manifest.json")}}],
            "training": {"backend": "existing", "initial_iterations": 10,
                         "dynamic_iterations": 4, "extra_args": ["-r", "1"]}}))
        return dict(config=load_config(cfg_path), config_path=cfg_path, source_manifest=source_manifest,
                    work_root=work, fullres_template=str(images / "{frame}"), poses_root=poses)

    def test_native_prepare_foundation_temporal_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(temporary)
            original_hash = sha256(args["source_manifest"])
            native_hashes = {str(p): sha256(p) for p in Path(temporary, "native").rglob("*.png")}
            calls = []
            def train(command, log, config):
                calls.append(command)
                def value(flag): return command[command.index(flag) + 1]
                source, model = Path(value("-s")), Path(value("-m"))
                steps = int(value("--iterations"))
                with Image.open(source / "train/r_0.png") as image:
                    self.assertEqual(image.width, 512 if source.parent.name == "Q2" else 1024)
                self.assertEqual(value("--data_device"), "cpu")
                self.assertEqual(value("-r"), "1")
                self.assertEqual(len(PlyData.read(source / "points3d.ply")["vertex"]), 100000)
                if "--foundation_gs_path" in command:
                    self.assertIn("--dynamic_opacity", command)
                    self.assertEqual(steps, 10)
                    state = refined(read_ply(value("--foundation_gs_path")))
                else:
                    self.assertIn("--dynamic_lapis", command)
                    self.assertEqual(steps, 4)
                    self.assertEqual(value("--densify_until_iter"), "0")
                    self.assertGreater(int(value("--opacity_reset_interval")), steps)
                    state = moved(read_ply(value("--initial_gs_path")))
                write_ply(model / "point_cloud" / f"iteration_{steps}" / "point_cloud.ply", state)
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                Path(log).write_text("mock original trainer completed\n")
            with patch("tools.content_preparation.upstream.socket.socket") as socket:
                socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 1234)
                with patch("tools.extend_quality_ladder.run_command", side_effect=train), redirect_stdout(io.StringIO()):
                    result = LadderExtension(**args).run()
            self.assertEqual(len(calls), 4)
            self.assertEqual([Path(c[c.index("-m") + 1]).parent.name for c in calls], ["Q2", "Q2", "Q3", "Q3"])
            work = args["work_root"]
            self.assertFalse((work / "source/Q0").exists())
            self.assertFalse((work / "source/Q1").exists())
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertTrue(verify_checkpoint_lineage(manifest, ["Q0", "Q1", "Q2", "Q3"]))
            self.assertEqual(manifest["data_spec_sha256"], json.loads(args["source_manifest"].read_text())["data_spec_sha256"])
            self.assertEqual(manifest["extension_prepared_index_sha256"], sha256(work / "prepared.json"))
            self.assertEqual(len(manifest["extension_prepared_sources"]), 4)
            self.assertTrue(all(len(s["content_sha256"]) == 64 for s in manifest["extension_prepared_sources"]))
            self.assertEqual([c["reused"] for c in manifest["commands"]], [True] * 4 + [False] * 4)
            self.assertTrue(result["upstream_unchanged"])
            self.assertEqual(original_hash, sha256(args["source_manifest"]))
            self.assertEqual(native_hashes, {str(p): sha256(p) for p in Path(temporary, "native").rglob("*.png")})
            before = sha256(work / "manifest.json")
            with patch("tools.content_preparation.upstream.socket.socket") as socket:
                socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 4321)
                with patch("tools.extend_quality_ladder.run_command", side_effect=AssertionError("resume must not train")), redirect_stdout(io.StringIO()):
                    resumed = LadderExtension(**args, resume=True).run()
            self.assertEqual(resumed["tasks_executed"], 0)
            self.assertEqual(resumed["tasks_skipped"], result["tasks_executed"])
            self.assertEqual(before, sha256(work / "manifest.json"))
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                LadderExtension(**args).run()

    def test_dry_run_and_path_ownership_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(temporary)
            with patch("tools.extend_quality_ladder.run_command", side_effect=AssertionError("dry run must not train")):
                plan = LadderExtension(**args).run(dry_run=True)
            self.assertEqual([q["training_resolution"] for q in plan["new_qualities"]], [512, 1024])
            self.assertFalse(args["work_root"].exists())
            args["work_root"].mkdir()
            marker = args["work_root"] / "valuable.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "unowned"):
                LadderExtension(**args, overwrite=True).run()
            self.assertEqual(marker.read_text(), "keep")
            unsafe = deepcopy(args)
            unsafe["config"]["objects"][0]["dataset"]["checkpoint_manifest"] = str(Path(temporary) / "outside.json")
            with self.assertRaisesRegex(ValueError, "inside work-root"):
                LadderExtension(**unsafe)
            overlapping = deepcopy(args)
            overlapping["work_root"] = Path(temporary) / "native"
            overlapping["config"]["objects"][0]["dataset"]["checkpoint_manifest"] = str(overlapping["work_root"] / "manifest.json")
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                LadderExtension(**overlapping)

    def test_original_provenance_and_native_image_size_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(temporary)
            source = json.loads(args["source_manifest"].read_text())
            del source["commands"][0]["sha256"]
            write_json(args["source_manifest"], source)
            with self.assertRaisesRegex(ValueError, "Original hash/training argv"):
                LadderExtension(**args)
            self.assertFalse(args["work_root"].exists())
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(temporary)
            Image.new("RGBA", (512, 512)).save(Path(temporary) / "native/1051/train/r_0.png")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "1024x1024 RGBA"):
                LadderExtension(**args).run("prepare")


if __name__ == "__main__":
    unittest.main()
