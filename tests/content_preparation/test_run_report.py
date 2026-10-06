"""Descriptive reports from actual tiny encoded/container/decoded CPU assets."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.content_preparation.assets import GaussianState, save_state, state_hash
from tools.content_preparation.codec_adapter import CodecAdapter
from tools.content_preparation.manifest import build_manifest, write_manifest
from tools.content_preparation.packaging import package_object
from tools.content_preparation.run_report import report_prepared_run, write_prepared_report


class PreparedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "variant"
        self.obj = self.root / "longdress"
        self.obj.mkdir(parents=True)
        self.frames = [{"frame": 1051, "timestamp": 10.0}, {"frame": 1054, "timestamp": 10.1}]
        records, decoded, tasks, commands = [], [], {}, []
        reconstructed = {}
        codec = CodecAdapter()
        for mode in ("independent", "progressive"):
            for qi, quality in enumerate(("Q0", "Q1")):
                layer = quality if mode == "independent" else ("Base" if qi == 0 else "E1")
                for row in self.frames:
                    frame = row["frame"]
                    n = qi + 2
                    xyz = np.zeros((n, 3), dtype=np.float32)
                    xyz[:, 0] = qi * .01 + (frame - 1051) * .03
                    state = GaussianState({"xyz": xyz, "rotation": np.tile([1, 0, 0, 0], (n, 1)),
                                           "scale": np.zeros((n, 3)), "opacity": np.zeros((n, 1)),
                                           "sh": np.zeros((n, 3, 1))}, {"stable_ids": True}, np.arange(n))
                    parent_layer = "Base" if mode == "progressive" and qi else None
                    parent = reconstructed[(mode, parent_layer, frame)] if parent_layer else None
                    path = self.obj / "encoded" / mode / layer / f"{frame}.cpgs"
                    meta = codec.encode(state, path, {"precision": "f32", "compression": "zlib"}, parent)
                    target = codec.decode(path, parent)
                    reconstructed[(mode, layer, frame)] = target
                    identifier = f"{mode}:{layer}:{frame}"
                    records.append(dict(meta, id=identifier, mode=mode, quality=quality, layer=layer,
                                        frame=frame, path=path.relative_to(self.obj).as_posix(),
                                        parent_layer=parent_layer,
                                        dependencies=[f"{mode}:{parent_layer}:{frame}"] if parent_layer else []))
                    state_path = self.obj / "decoded" / mode / layer / f"{frame}.npz"
                    save_state(state_path, target)
                    decoded.append({"frame": frame, "quality": quality, "mode": mode,
                                    "layer": layer, "path": state_path.relative_to(self.obj).as_posix(),
                                    "state_hash": state_hash(target)})
                    key = f"longdress/encode/{mode}/{layer}/{frame}"
                    tasks[key] = {"status": "complete", "started_at": "2026-10-05T00:00:00Z",
                                  "completed_at": "2026-10-05T00:00:01Z"}
                    if mode == "independent":
                        checkpoint = self.obj / "checkpoints" / quality / f"{frame}.npz"
                        save_state(checkpoint, state)
                        commands.append({"frame": frame, "level": quality, "path": str(checkpoint),
                                         "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "argv": ["train.py"]})
        index = package_object("longdress", self.frames, ["Q0", "Q1"], records,
                               {"gof_frames": 2, "segment_frames": 2, "refresh_frames": {"Base": 2, "E1": 1}}, self.obj)
        (self.obj / "package.json").write_text(json.dumps(index))
        (self.obj / "encoding.json").write_text(json.dumps({"payloads": records}))
        (self.obj / "decoding.json").write_text(json.dumps({"states": decoded}))
        (self.obj / "training.json").write_text(json.dumps({"backend": "existing", "lineage_verified": True, "commands": commands}))
        (self.obj / "preprocess.json").write_text(json.dumps({"kind": "checkpoints", "source_sha256": "a" * 64, "frames": self.frames}))
        profiles = self.obj / "profiles" / "profile.json"
        profiles.parent.mkdir()
        profiles.write_text(json.dumps({"reference": "highest decoded", "metadata": {"renderer_backend": "upstream_cuda"},
                                        "rows": [{"quality": "Q0", "azimuth": 0, "metrics": {"mse": .1, "psnr": 10, "lpips": .2}}],
                                        "aggregates": [{"mode": "independent", "quality": "Q0", "metrics": {"mse": {"uniform_mean": .1}}}]}))
        manifest = build_manifest("longdress", self.frames, ["Q0", "Q1"], index,
                                  provenance={"config_sha256": "b" * 64},
                                  quality_profile_path="profiles/profile.json", output_dir=self.obj)
        write_manifest(self.obj / "manifest.json", manifest)
        manifest_data = (self.obj / "manifest.json").read_bytes()
        (self.root / "manifest.json").write_text(json.dumps({"schema_version": "content-preparation.server.v1", "objects": [
            {"id": "longdress", "manifest_path": "longdress/manifest.json", "bytes": len(manifest_data), "sha256": hashlib.sha256(manifest_data).hexdigest()}]}))
        # Encode-index metadata writes are deliberately excluded from payload-task timing.
        tasks["longdress/encode/index"] = {"status": "complete", "started_at": "2026-10-05T00:00:00Z", "completed_at": "2026-10-05T00:10:00Z"}
        journal = {"tasks": tasks, "stages": {"longdress/encode": {"status": "complete", "started_at": "2026-10-05T01:00:00Z", "completed_at": "2026-10-05T01:00:00.010000Z"}}}
        (self.root / ".preparation").mkdir()
        (self.root / ".preparation" / "journal.json").write_text(json.dumps(journal))

    def test_exact_timeline_terminal_duration_and_not_encode_duration(self):
        report = report_prepared_run(self.root, fps=30, variant_id="f32_lossless")
        obj = report["objects"][0]
        timeline = obj["timeline"]
        self.assertEqual(timeline["fps"], 30)
        self.assertEqual(timeline["frame_count"], 2)
        self.assertEqual([r["timestamp"] for r in timeline["samples"]], [10.0, 10.1])
        self.assertAlmostEqual(timeline["time_span_seconds"], .1)
        self.assertAlmostEqual(timeline["media_duration_seconds"], .1 + 1 / 30)
        self.assertAlmostEqual(timeline["samples"][-1]["duration_seconds"], 1 / 30)
        timing = obj["encode_timings"]
        self.assertEqual(timing["recorded_successful_encode_task_wall_seconds"], 8)
        self.assertEqual(timing["measured_successful_payload_count"], 8)
        self.assertEqual(timing["latest_recorded_encode_stage_wall_seconds"], .01)
        inferred = report_prepared_run(self.root, fps=None)["objects"][0]["timeline"]
        self.assertAlmostEqual(inferred["fps"], 30)
        self.assertIn("inferred", inferred["fps_source"])

    def test_whole_segment_bytes_progressive_parents_counts_and_paths(self):
        report = report_prepared_run(self.root)
        obj = report["objects"][0]
        self.assertEqual(obj["actual_all_modes_segment_bytes"], sum(p.stat().st_size for p in self.obj.rglob("*.cpseg")))
        reps = {r["id"]: r for r in obj["representations"]}
        base, enhancement = reps["progressive:Base"], reps["progressive:E1"]
        self.assertEqual(enhancement["cumulative_quality_segment_bytes"], base["stored_layer_segment_bytes"] + enhancement["stored_layer_segment_bytes"])
        self.assertEqual(base["stored_layer_segment_bytes"], base["encoded_payload_bytes"] + base["segment_header_bytes"])
        self.assertTrue(enhancement["lossless_relative_to_exported_state_hash"])
        self.assertEqual([r["gaussian_count"] for r in enhancement["gaussian_counts"]], [3, 3])
        access = enhancement["cold_access"][0]
        self.assertEqual(access["dependencies_in_decode_order"], ["progressive:Base:1051", "progressive:E1:1051"])
        self.assertGreater(access["request_bytes"], access["required_payload_bytes"])
        self.assertTrue(all(path.startswith("longdress/media/") for path in access["segment_paths"]))
        self.assertAlmostEqual(base["segments"][0]["media_duration_seconds"], .1 + 1 / 30)
        self.assertAlmostEqual(enhancement["segments"][1]["media_duration_seconds"], 1 / 30)
        self.assertEqual(reps["independent:Q0"]["quality_profile_aggregates"][0]["metrics"]["mse"]["uniform_mean"], .1)

    def test_read_only_no_torch_import_and_json_markdown_roundtrip(self):
        before = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        importer = builtins.__import__
        def no_gpu(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch.") or name.startswith("gaussian_renderer"):
                self.fail("A report imported a GPU/model dependency")
            return importer(name, *args, **kwargs)
        raw = {"source_url": "https://example.invalid/8i", "raw_sha256": "c" * 64}
        with patch("builtins.__import__", side_effect=no_gpu):
            report = report_prepared_run(self.root, raw_provenance=raw)
        self.assertEqual(before, {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(report["objects"][0]["provenance"]["raw"], raw)
        written = write_prepared_report(self.root, raw_provenance=raw)
        self.assertEqual(json.loads((self.root / "run_report.json").read_text()), written)
        self.assertIn("Exact timestamp", (self.root / "run_report.md").read_text())
        self.assertIn("f32_lossless", (self.root / "run_report.md").read_text())
        self.assertIn("gaussian_attribute_zlib / 1.0.0", (self.root / "run_report.md").read_text())
        self.assertIn("Cold payload dependency order", (self.root / "run_report.md").read_text())
        self.assertIn("Media duration (s)", (self.root / "run_report.md").read_text())
        self.assertIn("https://example.invalid/8i", (self.root / "run_report.md").read_text())
        self.assertEqual(write_prepared_report(self.root, raw_provenance=raw), written)

    def test_optional_decoded_profiles_timings_report_unavailable(self):
        (self.root / ".preparation" / "journal.json").unlink()
        (self.obj / "profiles" / "profile.json").unlink()
        first = self.obj / "decoded" / "independent" / "Q0" / "1051.npz"
        first.unlink()
        report = report_prepared_run(self.root)
        obj = report["objects"][0]
        self.assertIsNone(obj["encode_timings"]["recorded_successful_encode_task_wall_seconds"])
        self.assertFalse(obj["encode_timings"]["all_payload_timings_available"])
        self.assertFalse(obj["quality_profile"]["available"])
        self.assertIsNone(obj["representations"][0]["gaussian_counts"][0]["gaussian_count"])

    def test_failed_or_missing_task_does_not_claim_complete_encode_measurement(self):
        path = self.root / ".preparation" / "journal.json"
        journal = json.loads(path.read_text())
        journal["tasks"]["longdress/encode/independent/Q0/1051"]["status"] = "failed"
        del journal["tasks"]["longdress/encode/independent/Q0/1054"]
        path.write_text(json.dumps(journal))
        timing = report_prepared_run(self.root)["objects"][0]["encode_timings"]
        self.assertEqual(timing["recorded_successful_encode_task_wall_seconds"], 6)
        self.assertFalse(timing["all_payload_timings_available"])

    def test_actual_media_corruption_and_manifest_paths_are_rejected(self):
        segment = next(self.obj.rglob("*.cpseg"))
        with segment.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        with self.assertRaisesRegex(ValueError, "checksum"):
            report_prepared_run(self.root)
        server = json.loads((self.root / "manifest.json").read_text())
        server["objects"][0]["manifest_path"] = "../outside.json"
        (self.root / "manifest.json").write_text(json.dumps(server))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            report_prepared_run(self.root)


if __name__ == "__main__":
    unittest.main()
