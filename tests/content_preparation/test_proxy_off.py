"""Proxy exclusion is executable policy, including manifest and overwrite paths."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.content_preparation.config import load_config, validate_config
from tools.content_preparation.pipeline import Pipeline
from tests.content_preparation.test_pipeline import mini_config


class ProxyOffTests(unittest.TestCase):
    def test_disabled_proxy_stage_rejected_before_output_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = mini_config(root / "run")
            cfg["proxy"] = {"enabled": False}
            pipeline = Pipeline(validate_config(cfg), root / "config.yaml")
            self.assertNotIn("proxy", pipeline.plan()["stages"])
            with self.assertRaisesRegex(ValueError, "disabled"):
                pipeline.run("proxy")
            self.assertFalse((root / "run").exists())

    def test_external_descriptor_requires_disabled_proxy_and_relative_path(self):
        for proxy in ({"enabled": True, "external_descriptor": "existing.json"},
                      {"enabled": False, "external_descriptor": "../other.json"},
                      {"enabled": "false"}):
            cfg = mini_config("unused")
            cfg["proxy"] = proxy
            with self.subTest(proxy=proxy), self.assertRaises(ValueError):
                validate_config(cfg)

    def test_all_and_external_reference_never_generate_or_validate_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            cfg = mini_config(root / "run")
            cfg["proxy"] = {"enabled": False}
            cfg["delivery_modes"] = ["progressive"]
            config_path.write_text(yaml.safe_dump(cfg))
            pipeline = Pipeline(load_config(config_path), config_path)
            with patch.object(Pipeline, "stage_proxy", side_effect=AssertionError("proxy work is out of scope")), redirect_stdout(io.StringIO()):
                result = pipeline.run()
            self.assertNotIn("proxy", result["stages"])
            obj_root = pipeline.output / "cpu_mini"
            manifest = json.loads((obj_root / "manifest.json").read_text())
            self.assertEqual(manifest["object"]["proxy_preparation"], "disabled")
            self.assertIsNone(manifest["object"]["proxy_path"])
            self.assertFalse((obj_root / "proxy").exists())
            # Existing descriptor is referenced as an opaque prepared asset.
            # No proxy generation or proxy-quality audit is attempted.
            descriptor = obj_root / "external/proxy.json"
            descriptor.parent.mkdir()
            descriptor.write_text('{"prepared_elsewhere": true}\n')
            pipeline.cfg["proxy"]["external_descriptor"] = "external/proxy.json"
            pipeline.overwrite = True
            with patch.object(Pipeline, "stage_proxy", side_effect=AssertionError("unexpected proxy work")), redirect_stdout(io.StringIO()):
                pipeline.run("manifest")
            manifest = json.loads((obj_root / "manifest.json").read_text())
            self.assertEqual(manifest["object"]["proxy_preparation"], "external")
            self.assertEqual(manifest["object"]["proxy_path"], "external/proxy.json")
            self.assertFalse((obj_root / "proxy").exists())


if __name__ == "__main__":
    unittest.main()
