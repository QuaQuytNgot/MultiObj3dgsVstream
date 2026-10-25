"""Standalone metadata/workload validation; tiny invented fixtures are tests only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from metadata import (CRFS, DEFAULT_DATA_ROOT, MetadataError, _cost_manifest_path,
                      _safe_path, _summary_psnr, load_dataset)
from workloads import generate_workloads


class MetadataFailureTests(unittest.TestCase):
    def test_missing_metadata_directory_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MetadataError, "does not exist"):
                load_dataset(Path(tmp) / "missing")

    def test_missing_cost_has_configurable_source_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MetadataError, "--cost-root"):
                _cost_manifest_path(Path(tmp), None, "longdress", 0, 20)

    def test_missing_nonfinite_psnr_never_imputed(self):
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(MetadataError):
                _summary_psnr({"aggregation": "mean dB", "metrics": {"psnr": value}}, "fixture")

    def test_pooled_psnr_not_substituted(self):
        with self.assertRaisesRegex(MetadataError, "mean-dB"):
            _summary_psnr({"aggregation": "pooled", "metrics": {"psnr": 30}}, "fixture")

    def test_uri_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MetadataError, "Unsafe metadata path"):
                _safe_path(Path(tmp), "../outside.json")

    def test_missing_or_duplicate_profile_coverage_rejected(self):
        # Fail at coverage validation, before loading any sidecars. These dummy
        # entries are a negative-test fixture, never benchmark inputs.
        for entries in ([], [{"object": "longdress", "gop": 0, "crf": 20}] * 80):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                blob = json.dumps({"schema": "streamstgs.quality-index.v1", "profiles": entries}).encode()
                (root / "quality-index.json").write_bytes(blob)
                (root / "SHA256SUMS").write_text(hashlib.sha256(blob).hexdigest() + "  quality-index.json\n")
                with self.assertRaisesRegex(MetadataError, "missing or duplicate profile"):
                    load_dataset(root)

    def test_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "quality-index.json").write_text("{}")
            (root / "SHA256SUMS").write_text("0" * 64 + "  quality-index.json\n")
            with self.assertRaisesRegex(MetadataError, "checksum mismatch"):
                load_dataset(root)


@unittest.skipUnless(DEFAULT_DATA_ROOT.is_dir(), "External measured metadata bundle is not installed")
class MeasuredDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles, cls.audit = load_dataset()
        cls.cases = generate_workloads(cls.profiles)

    def test_all_measured_profiles_and_mpd_averages(self):
        self.assertEqual(len(self.profiles), 80)
        self.assertEqual(self.audit["total_metric_samples"], 115200)
        self.assertEqual(len(self.audit["mpd_average_checks"]), 16)
        for check in self.audit["mpd_average_checks"]:
            self.assertEqual(check["mpd_average_bps"], math.ceil(check["measured_average_bps"]))
        for profile in self.profiles:
            self.assertEqual(profile["cost_bps"], 4 * profile["transfer_bytes"])
            self.assertEqual(profile["transfer_bytes"], profile["payload_bytes"] + profile["manifest_bytes"])

    def test_every_workload_value_is_exact_measured_row_and_weighted_psnr(self):
        measured = {(p["object"], p["gop"], p["crf"]): p for p in self.profiles}
        self.assertEqual(len(self.cases), 72)
        for case in self.cases:
            self.assertAlmostEqual(math.fsum(case["weights"]), 1.0, places=14)
            self.assertGreaterEqual(case["budget_bps"], sum(min(row) for row in case["costs_bps"]))
            for index, instance in enumerate(case["instances"]):
                for k, crf in enumerate(CRFS):
                    original = measured[instance["object"], instance["gop"], crf]
                    self.assertEqual(case["costs_bps"][index][k], original["cost_bps"])
                    self.assertEqual(case["psnr_db"][index][k], original["psnr_db"])
                    self.assertEqual(case["utilities"][index][k],
                                     case["weights"][index] * original["psnr_db"])
            counts = Counter(i["object"] for i in case["instances"])
            if case["n"] >= 4:
                self.assertEqual(len(set(counts.values())), 1)

    def test_all_solvers_budget_regimes_share_identical_prepared_matrices(self):
        groups = {}
        for case in self.cases:
            key = case["n"], case["seed"]
            snapshot = (case["costs_bps"], case["utilities"], case["weights"],
                        case["psnr_db"], case["matrix_sha256"])
            if key in groups:
                self.assertEqual(snapshot, groups[key])
            groups[key] = snapshot

    def test_visibility_seed_reproducible(self):
        self.assertEqual(self.cases, generate_workloads(self.profiles))
        other = generate_workloads(self.profiles, sizes=(4,), seeds=(3001,))
        self.assertNotEqual(self.cases[4]["weights"], other[0]["weights"])

    def test_missing_measured_profile_never_imputed(self):
        with self.assertRaisesRegex(MetadataError, "80 unique measured profiles"):
            generate_workloads(self.profiles[:-1])

    def test_missing_quality_sidecar_fails_before_solver_initialization(self):
        original_is_file = Path.is_file
        missing = DEFAULT_DATA_ROOT / "longdress/quality/profiles/gop_000_crf_20.json"
        with patch.object(Path, "is_file", autospec=True,
                          side_effect=lambda p: False if p == missing else original_is_file(p)):
            with self.assertRaisesRegex(MetadataError, "Missing metadata"):
                load_dataset()


if __name__ == "__main__":
    unittest.main()
