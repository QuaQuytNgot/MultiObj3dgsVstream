"""Test command forwarding/provenance without importing CUDA or executing training."""
import sys
import unittest
from unittest.mock import patch

from tools.content_preparation import backend
from tools.content_preparation.paths import UPSTREAM_ROOT


class BackendLauncherTests(unittest.TestCase):
    def test_forwarding_keeps_pinned_entry_and_upstream_arguments(self):
        original = sys.argv

        def check(path, run_name):
            self.assertEqual(path, str(UPSTREAM_ROOT / "train.py"))
            self.assertEqual(run_name, "__main__")
            self.assertEqual(sys.argv, [path, "-s", "data", "-m", "model", "--iterations", "10"])

        with patch.object(backend, "configure_training_compatibility") as compatibility, \
             patch.object(backend.runpy, "run_path", side_effect=check):
            self.assertEqual(backend.main(["--entry", "train.py", "-s", "data", "-m", "model", "--iterations", "10"]), 0)
        compatibility.assert_called_once_with()
        self.assertIs(sys.argv, original)

    def test_entry_is_restricted_and_exception_restores_argv(self):
        original = sys.argv
        with patch.object(backend, "configure_training_compatibility") as compatibility:
            with self.assertRaises(SystemExit):
                backend.main(["--entry", "../../other.py"])
            compatibility.assert_not_called()
        with patch.object(backend, "configure_training_compatibility"), \
             patch.object(backend.runpy, "run_path", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                backend.main(["--entry", str(UPSTREAM_ROOT / "train.py")])
        self.assertIs(sys.argv, original)
