"""Audit isolation, timing boundaries, and measured-input serialization.

Run ``python -m unittest discover -s tests`` from the experiment directory
after ``python workloads.py``. These checks use no streaming/runtime modules.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import unittest


EXPERIMENT = Path(__file__).resolve().parents[1]


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class SourceIsolationTests(unittest.TestCase):
    def test_python_does_not_import_production_modules(self):
        files = list(EXPERIMENT.glob("*.py")) + list((EXPERIMENT / "tests").glob("*.py"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=str(path.relative_to(EXPERIMENT))):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    for name in names:
                        self.assertFalse({"src", "src_final"}.intersection(name.split(".")), name)

    def test_quoted_cpp_includes_stay_within_experiment(self):
        paths = list((EXPERIMENT / "cpp").glob("*.cpp")) + list((EXPERIMENT / "cpp").glob("*.hpp"))
        paths += list((EXPERIMENT / "tests").glob("*.cpp"))
        self.assertTrue(paths)
        for path in paths:
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), re.MULTILINE):
                candidates = [(path.parent / include).resolve(), (EXPERIMENT / "cpp" / include).resolve()]
                self.assertTrue(any(p.is_file() and p.is_relative_to(EXPERIMENT) for p in candidates),
                                f"External/local missing C++ include: {path}: {include}")

    def test_timed_interval_contains_only_the_solve_call(self):
        source = (EXPERIMENT / "cpp" / "benchmark.cpp").read_text()
        begin = source.index("const auto start=std::chrono::steady_clock::now();")
        end = source.index("const auto end=std::chrono::steady_clock::now();", begin)
        timed = source[begin + len("const auto start=std::chrono::steady_clock::now();"):end]
        self.assertEqual(" ".join(timed.split()), "Result result=job.solver(a,job.opts);")
        self.assertLess(source.index("std::ifstream in("), begin)
        self.assertLess(source.index("std::shuffle("), begin)
        self.assertGreater(source.index("Result verified=result;evaluate(a,verified)"), end)
        self.assertGreater(source.index("samples.push_back("), end)
        self.assertGreater(source.index("std::ofstream out("), end)
        self.assertIn("for(const auto& a:instances)", source)
        self.assertIn("using Solver=Result(*)(const Instance&,const Options&);", source)


class PreparedDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inputs = EXPERIMENT / "inputs"
        for name in ("prepared_cases.json", "measured_profiles.json", "runner_input.txt"):
            if not (inputs / name).is_file():
                raise RuntimeError(f"Missing {name}; run python workloads.py before the complete validation suite")
        cls.prepared = json.loads((inputs / "prepared_cases.json").read_text())
        cls.cases = cls.prepared["cases"]
        cls.profiles = json.loads((inputs / "measured_profiles.json").read_text())
        cls.wire = (inputs / "runner_input.txt").read_text().split()

    def test_cases_are_measured_matrices_with_visibility_only_synthesized(self):
        profiles = {(p["object"], p["gop"], p["crf"]): p for p in self.profiles}
        self.assertEqual(len(profiles), 80)
        self.assertEqual(self.prepared["case_count"], len(self.cases))
        self.assertEqual(self.prepared["cases_sha256"], stable_hash(self.cases))
        groups = {}
        for case in self.cases:
            with self.subTest(case=case["id"]):
                n = case["n"]
                self.assertEqual(case["k"], 4)
                self.assertEqual(case["crfs"], [20, 28, 36, 40])
                self.assertEqual(len(case["instances"]), n)
                self.assertEqual(len(case["weights"]), n)
                self.assertTrue(all(math.isfinite(w) and w >= 0 for w in case["weights"]))
                self.assertAlmostEqual(math.fsum(case["weights"]), 1.0, places=14)
                self.assertEqual(len(case["costs_bps"]), n)
                self.assertEqual(len(case["psnr_db"]), n)
                self.assertEqual(len(case["utilities"]), n)
                for i, obj in enumerate(case["instances"]):
                    self.assertEqual(len(case["costs_bps"][i]), 4)
                    self.assertEqual(len(case["psnr_db"][i]), 4)
                    self.assertEqual(len(case["utilities"][i]), 4)
                    for k, crf in enumerate(case["crfs"]):
                        profile = profiles[obj["object"], obj["gop"], crf]
                        self.assertEqual(case["costs_bps"][i][k], profile["cost_bps"])
                        self.assertEqual(case["psnr_db"][i][k], profile["psnr_db"])
                        self.assertEqual(case["utilities"][i][k], case["weights"][i] * profile["psnr_db"])
                        self.assertEqual(obj["profile_sha256_by_crf"][str(crf)], profile["quality_profile_sha256"])
                minimum = sum(min(row) for row in case["costs_bps"])
                maximum = sum(max(row) for row in case["costs_bps"])
                self.assertEqual(case["minimum_feasible_bps"], minimum)
                self.assertEqual(case["maximum_selection_bps"], maximum)
                self.assertGreaterEqual(case["budget_bps"], minimum)
                self.assertEqual(case["budget_bps"], minimum + math.floor(
                    case["budget_fraction_of_upgrade_span"] * (maximum - minimum)))
                matrix = {"costs_bps": case["costs_bps"], "utilities": case["utilities"]}
                self.assertEqual(case["matrix_sha256"], stable_hash(matrix))
                self.assertEqual(case["decision_sha256"], stable_hash({
                    "matrix_sha256": case["matrix_sha256"], "budget_bps": case["budget_bps"]}))
                # All budget regimes share identical input matrices. All four
                # solver calls consume these same immutable arrays in C++.
                group_key = case["n"], case["seed"]
                if group_key in groups:
                    self.assertEqual(groups[group_key], matrix)
                groups[group_key] = matrix

    def test_wire_format_round_trips_identical_integer_costs_and_double_utilities(self):
        iterator = iter(self.wire)
        self.assertEqual(int(next(iterator)), len(self.cases))
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(next(iterator), case["id"])
                self.assertEqual(int(next(iterator)), case["n"])
                self.assertEqual(int(next(iterator)), case["budget_bps"])
                for costs, utilities in zip(case["costs_bps"], case["utilities"]):
                    self.assertEqual([int(next(iterator)) for _ in range(4)], costs)
                    self.assertEqual([float(next(iterator)) for _ in range(4)], utilities)
        self.assertIsNone(next(iterator, None), "Unexpected trailing solver input")


if __name__ == "__main__":
    unittest.main()
