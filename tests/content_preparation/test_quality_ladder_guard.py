"""Provenance guard policy tests; no original trainer/GPU calls are made."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.extend_quality_ladder import LadderExtension
from tools.extend_quality_ladder_guard import guard_path, run_guarded
from tools.content_preparation.checkpoint import sha256, write_json
try:
    from . import test_extend_quality_ladder as extension_fixtures
except ImportError:  # unittest discovery imports these files as top-level modules.
    import test_extend_quality_ladder as extension_fixtures


class QualityLadderGuardTests(unittest.TestCase):
    def extension(self, root):
        args = extension_fixtures.ExtensionTests().fixture(root)
        extension = LadderExtension(**args)
        extension.run = Mock(return_value={"tasks_executed": 0, "tasks_skipped": 12})
        return extension

    @staticmethod
    def runtime(binary):
        return {"python": "3.11", "torch": "2.3", "initialization_rng_seed": 0,
                "extension_artifacts": {"vendor/_C.so": sha256(binary)},
                "gpu": "visible-or-hidden", "cuda_runtime": "11.8"}

    def completed(self, extension, root, runtime):
        manifest = {"upstream_sources": extension.snapshot,
                    "extension_wrapper_sha256": extension.wrapper_sha256,
                    "lineage_verified": True}
        write_json(extension.manifest_path, manifest)
        write_json(extension.work/".preparation/journal.json", {
            "stages": {"extension/train": {"status": "complete"}},
            "tasks": {"train/manifest": {"status": "complete", "outputs": [
                {"path": str(extension.manifest_path.relative_to(extension.work)),
                 "sha256": sha256(extension.manifest_path)}]}}})
        proof = Path(root)/"prior_runtime_manifest.json"
        write_json(proof, {"provenance": {"runtime": runtime}})
        return proof

    def test_fresh_guard_is_outside_work_and_preserves_original_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = self.extension(root)
            binary = root/"extension.so"; binary.write_bytes(b"verified extension")
            original = sha256(Path(__file__).resolve().parents[2]/"tools/extend_quality_ladder.py")
            with patch("tools.extend_quality_ladder_guard.runtime_provenance", return_value=self.runtime(binary)):
                result = run_guarded(extension, stage="prepare")
                extension.run.assert_called_once_with("prepare", dry_run=False)
                self.assertFalse(extension.work.exists())
                sidecar = guard_path(extension)
                self.assertEqual(sidecar.parent, extension.work.parent)
                self.assertTrue(sidecar.is_file())
                self.assertFalse(result["environment_guard"]["adopted_existing_work"])
                recorded = json.loads(sidecar.read_text())["environment"]
                self.assertNotIn("gpu", recorded["runtime"])
                self.assertNotIn("cuda_runtime", recorded["runtime"])
                before = sha256(sidecar)
                run_guarded(extension)
                self.assertEqual(before, sha256(sidecar))
            self.assertEqual(original, sha256(Path(__file__).resolve().parents[2]/"tools/extend_quality_ladder.py"))

    def test_changed_binary_source_runtime_or_wrapper_rejected_before_forwarding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = self.extension(root)
            binary = root/"extension.so"; binary.write_bytes(b"original")
            with patch("tools.extend_quality_ladder_guard.runtime_provenance", side_effect=lambda cfg: self.runtime(binary)):
                run_guarded(extension)
                extension.run.reset_mock()
                binary.write_bytes(b"different binary")
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    run_guarded(extension)
                binary.write_bytes(b"original")
                saved = extension.snapshot
                extension.snapshot = dict(saved, **{"train.py": "changed"})
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    run_guarded(extension)
                extension.snapshot = saved
                with patch("tools.extend_quality_ladder_guard.runtime_provenance", return_value={**self.runtime(binary), "python": "new version"}):
                    with self.assertRaisesRegex(RuntimeError, "environment changed"):
                        run_guarded(extension)
                extension.wrapper_sha256 = "changed wrapper"
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    run_guarded(extension)
                extension.run.assert_not_called()

    def test_dry_run_writes_neither_sidecar_nor_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = self.extension(root)
            binary = root/"extension.so"; binary.write_bytes(b"original")
            with patch("tools.extend_quality_ladder_guard.runtime_provenance", return_value=self.runtime(binary)):
                result = run_guarded(extension, stage="train", dry_run=True)
            extension.run.assert_called_once_with("train", dry_run=True)
            self.assertTrue(result["environment_guard"]["dry_run"])
            self.assertFalse(extension.work.exists())
            self.assertFalse(guard_path(extension).exists())
            self.assertFalse(guard_path(extension).with_suffix(".lock").exists())

    def test_adoption_requires_completed_work_and_matching_explicit_runtime_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = self.extension(root)
            binary = root/"extension.so"; binary.write_bytes(b"original")
            runtime = self.runtime(binary)
            extension.work.mkdir()
            (extension.work/"important.txt").write_text("preserve")
            with patch("tools.extend_quality_ladder_guard.runtime_provenance", return_value=runtime):
                with self.assertRaisesRegex(FileExistsError, "adopt-runtime-manifest"):
                    run_guarded(extension)
                missing = root/"nonexistent.json"
                with self.assertRaisesRegex(FileExistsError, "unfinished"):
                    run_guarded(extension, adopt_runtime_manifest=missing)
                proof = self.completed(extension, root, runtime)
                wrong = deepcopy(runtime)
                wrong["extension_artifacts"] = {"vendor/_C.so": "different"}
                write_json(proof, {"provenance": {"runtime": wrong}})
                with self.assertRaisesRegex(ValueError, "identical CUDA extension"):
                    run_guarded(extension, adopt_runtime_manifest=proof)
                self.assertFalse(guard_path(extension).exists())
                extension.run.assert_not_called()
                write_json(proof, {"provenance": {"runtime": runtime}})
                result = run_guarded(extension, adopt_runtime_manifest=proof)
                self.assertTrue(result["environment_guard"]["adopted_existing_work"])
                sidecar = json.loads(guard_path(extension).read_text())
                self.assertEqual(sidecar["adoption"]["runtime_manifest_sha256"], sha256(proof))
            self.assertEqual((extension.work/"important.txt").read_text(), "preserve")

    def test_adoption_rejects_changed_completed_manifest_and_unfinished_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = self.extension(root)
            binary = root/"extension.so"; binary.write_bytes(b"original")
            runtime = self.runtime(binary)
            proof = self.completed(extension, root, runtime)
            journal_path = extension.work/".preparation/journal.json"
            journal = json.loads(journal_path.read_text())
            journal["stages"]["extension/train"]["status"] = "running"
            write_json(journal_path, journal)
            with patch("tools.extend_quality_ladder_guard.runtime_provenance", return_value=runtime):
                with self.assertRaisesRegex(ValueError, "unfinished"):
                    run_guarded(extension, adopt_runtime_manifest=proof)
                journal["stages"]["extension/train"]["status"] = "complete"
                write_json(journal_path, journal)
                manifest = json.loads(extension.manifest_path.read_text())
                manifest["extra"] = "tampered"
                write_json(extension.manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "journal checksum"):
                    run_guarded(extension, adopt_runtime_manifest=proof)
                extension.run.assert_not_called()
            self.assertFalse(guard_path(extension).exists())


if __name__ == "__main__":
    unittest.main()
