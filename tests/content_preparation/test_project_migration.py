"""Verify content-preparation portability without data, CUDA, builds or Internet."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.content_preparation.checkpoint import sha256
from tools.content_preparation.config import ROOT, load_config, resolve_path, validate_config
from tools.content_preparation.paths import PROJECT_ROOT, UPSTREAM_ROOT
from tools.content_preparation.upstream import build_training_command, source_snapshot


def small_configuration(output):
    return {
        "version": 1,
        "output": str(output),
        "objects": [{
            "id": "portable_fixture", "frames": [0],
            "dataset": {"kind": "synthetic"},
            "qualities": [{"id": "Q0"}],
        }],
        "training": {"backend": "synthetic_fixture"},
        "renderer": {"backend": "cpu_fixture", "width": 32, "height": 32},
    }


class ProjectMigrationTests(unittest.TestCase):
    def test_project_paths_and_provenance_use_pinned_backend(self):
        project = Path(__file__).resolve().parents[2]
        self.assertEqual(PROJECT_ROOT, project)
        self.assertEqual(ROOT, project)
        self.assertEqual(UPSTREAM_ROOT, project / "third_party/dynamic-lapis-gs")
        snapshot = source_snapshot()
        self.assertEqual(snapshot["train.py"], sha256(UPSTREAM_ROOT / "train.py"))
        self.assertEqual(snapshot["dataset_prepare.py"], sha256(UPSTREAM_ROOT / "dataset_prepare.py"))
        for path in ("submodules/diff-gaussian-rasterization/setup.py", "submodules/simple-knn/setup.py"):
            self.assertEqual(snapshot[path], sha256(UPSTREAM_ROOT / path))
        self.assertFalse((project / "train.py").exists())

    def test_training_command_uses_upstream_while_artifacts_use_project(self):
        config = validate_config(small_configuration("output/portable_fixture"))
        self.assertFalse(config["runtime"]["extension_path"])
        self.assertEqual(resolve_path(config["output"]), PROJECT_ROOT / "output/portable_fixture")
        with patch("tools.content_preparation.upstream.socket.socket") as socket:
            socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 6009)
            command = build_training_command(
                "prepared/source", "output/model", {"id": "Q0"}, config["training"], 1,
            )
        self.assertEqual(Path(command[1]), PROJECT_ROOT / "tools/content_preparation/backend.py")
        self.assertEqual(command[2], "--entry")
        self.assertEqual(Path(command[3]), UPSTREAM_ROOT / "train.py")
        self.assertIn("output/model", command)

    def test_migrated_configurations_are_portable_project_artifacts(self):
        configurations = sorted((PROJECT_ROOT / "configs").glob("content_prepare*.yaml"))
        self.assertGreaterEqual(len(configurations), 6)
        for path in configurations:
            with self.subTest(config=path.name):
                raw = yaml.safe_load(path.read_text())
                self.assertNotIn("/home/fil/", path.read_text())
                config = load_config(path)
                self.assertFalse(Path(raw["output"]).is_absolute())
                self.assertTrue(resolve_path(config["output"]).is_relative_to(PROJECT_ROOT))
                extension = config["runtime"].get("extension_path")
                self.assertFalse(extension and Path(extension).is_absolute())
                for obj in config["objects"]:
                    for name in ("checkpoint_manifest", "source_template", "raw_template", "poses_root", "fullres_template"):
                        value = obj["dataset"].get(name)
                        self.assertFalse(value and Path(value).is_absolute())

    def test_cli_dry_run_from_an_unrelated_directory_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            config_path = temporary / "fixture.yaml"
            output = temporary / "artifacts"
            config_path.write_text(yaml.safe_dump(small_configuration(output)))
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "tools/content_preparation/prepare_content.py"),
                 "--config", str(config_path), "--dry-run"],
                cwd=temporary, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertTrue(plan["dry_run"])
            self.assertFalse(output.exists())
            self.assertEqual(sorted(p.name for p in temporary.iterdir()), ["fixture.yaml"])

    def test_backend_import_setup_preserves_project_owned_tools(self):
        # Run in isolation to verify import precedence, independent of the modules
        # that pytest/unittest have already loaded in this process.
        program = """
import importlib.util
import json
from tools.content_preparation.paths import configure_upstream_imports
configure_upstream_imports()
import tools.prepare_progressive_real as prepare
import tools.check_progressive_composability as composition
import tools.content_preparation.pipeline as pipeline
print(json.dumps({
    "prepare": prepare.__file__, "composition": composition.__file__,
    "pipeline": pipeline.__file__,
    "dataset": importlib.util.find_spec("dataset_prepare").origin,
    "renderer": importlib.util.find_spec("gaussian_renderer").origin,
}))
"""
        environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", program], cwd=temporary, env=environment,
                text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        locations = json.loads(result.stdout)
        for name in ("prepare", "composition", "pipeline"):
            self.assertTrue(Path(locations[name]).is_relative_to(PROJECT_ROOT / "tools"), locations[name])
        for name in ("dataset", "renderer"):
            self.assertTrue(Path(locations[name]).is_relative_to(UPSTREAM_ROOT), locations[name])


if __name__ == "__main__":
    unittest.main()
