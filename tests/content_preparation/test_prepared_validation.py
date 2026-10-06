"""Independent audits detect altered media, reconstructed states and metrics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml

from tools.content_preparation.assets import load_state, save_state
from tools.content_preparation.config import load_config
from tools.content_preparation.pipeline import Pipeline
from tools.content_preparation.prepared_validation import PreparedValidator
from tools.content_preparation.mpd import export_mpd
from tools.export_decoded_viewer_assets import export_viewer
from tools.validate_prepared_content import main as validate_cli
from tests.content_preparation.test_pipeline import mini_config


class PreparedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.config_path = cls.root / "config.yaml"
        raw = mini_config(cls.root / "run")
        raw["objects"][0]["frames"] = [0, 1]
        raw["delivery_modes"] = ["progressive"]
        raw["sampling"]["azimuth"] = [0, 120]
        raw["proxy"] = {"enabled": False}
        cls.config_path.write_text(yaml.safe_dump(raw))
        cls.config = load_config(cls.config_path)
        cls.validator = PreparedValidator(cls.config)
        with redirect_stdout(io.StringIO()):
            Pipeline(cls.config, cls.config_path).run()
        cls.object_root = cls.root / "run/cpu_mini"

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_each_core_artifact_stage_is_auditable(self):
        for stage in ("dataset", "preprocess", "checkpoints", "export", "encoding", "package", "decode", "profile"):
            with self.subTest(stage=stage):
                self.assertTrue(self.validator.validate(stage)["passed"])

    @unittest.skipUnless(shutil.which("xmllint"), "Pinned MPD XSD validation requires xmllint")
    def test_final_gate_requires_real_browser_evidence_after_export(self):
        """Export auditing is real; browser readiness is never simulated."""
        prepared_root = self.object_root.parent
        original_media = json.loads((self.object_root / "package.json").read_text())["media_bytes"]
        mpd = export_mpd(prepared_root, resume=True)
        viewer = export_viewer(prepared_root, resume=True)
        self.assertTrue(mpd["schema_validated"])
        self.assertEqual(viewer["asset_count"], 4)
        with self.assertRaisesRegex(FileNotFoundError, "viewer_validation"):
            self.validator.validate("final")
        manifest = json.loads((self.object_root / "manifest.json").read_text())
        self.assertEqual(manifest["object"]["proxy_preparation"], "disabled")
        self.assertFalse((self.object_root / "proxy").exists())
        sidecar = json.loads((prepared_root / "content_index.json").read_text())
        self.assertEqual(sum(r["own_bytes"] for r in sidecar["objects"][0]["representations"]), original_media)
        self.assertFalse(json.loads((prepared_root / "viewer_assets/index.json").read_text())["included_in_encoded_media_bytes"])
        self.assertEqual(export_mpd(prepared_root, resume=True)["tasks_executed"], 0)
        self.assertEqual(export_viewer(prepared_root, resume=True)["executed"], 0)
        asset = json.loads((prepared_root / "viewer_assets/index.json").read_text())["assets"][0]
        asset_path = prepared_root / asset["path"]
        original = asset_path.read_bytes()
        try:
            asset_path.write_bytes(original + b"damaged viewer state")
            with self.assertRaises(ValueError):
                self.validator.validate("final")
        finally:
            asset_path.write_bytes(original)

    def test_decode_uses_segments_when_standalone_payloads_unavailable(self):
        encoded = self.object_root / "encoded"
        archived = self.root / "archived_encoded"
        encoded.rename(archived)
        try:
            result = self.validator.validate("decode")
            self.assertTrue(result["passed"])
            self.assertEqual(result["checks"]["decode"]["cpu_mini"]["states"], 4)
        finally:
            archived.rename(encoded)

    def test_corrupt_delivered_segment_is_rejected(self):
        segment = next((self.object_root / "media").rglob("*.cpseg"))
        original = segment.read_bytes()
        try:
            altered = bytearray(original)
            altered[-1] ^= 1
            segment.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "checksum|hash"):
                self.validator.validate("decode")
        finally:
            segment.write_bytes(original)

    def test_altered_decoded_cache_cannot_pass_independent_decode(self):
        path = self.object_root / "decoded/progressive/Base/0.npz"
        original = path.read_bytes()
        try:
            state = load_state(path)
            state.arrays["opacity"][0, 0] += .5
            save_state(path, state)
            with self.assertRaisesRegex(ValueError, "differs from actual delivered"):
                self.validator.validate("decode")
        finally:
            path.write_bytes(original)

    def test_wrong_metric_and_missing_condition_are_rejected(self):
        path = self.object_root / "profiles/profile.json"
        original = path.read_bytes()
        try:
            document = json.loads(original)
            document["rows"][0]["metrics"]["psnr"] = 123
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "PSNR"):
                self.validator.validate("profile")
            document = json.loads(original)
            document["rows"].pop()
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "coverage"):
                self.validator.validate("profile")
        finally:
            path.write_bytes(original)

    def test_validator_cli_resume_and_input_hash_protection(self):
        arguments = ["--config", str(self.config_path), "--stage", "encoding", "--resume"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(validate_cli(arguments), 0)
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(validate_cli(arguments), 0)
        self.assertEqual(json.loads(captured.getvalue())["tasks_executed"], 0)
        journal = json.loads((self.root / "run/.preparation/journal.json").read_text())
        entry = journal["tasks"]["validation/encoding/checks/encoding.json"]
        self.assertEqual(entry["status"], "complete")
        self.assertTrue(entry["input_hashes"])
        self.assertTrue(entry["config_hash"])


if __name__ == "__main__":
    unittest.main()
