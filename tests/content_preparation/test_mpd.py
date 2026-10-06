"""MPD contracts, dependency closures, timing, byte rates and ownership."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tools.content_preparation.checkpoint import sha256, write_json
from tools.content_preparation.manifest import build_manifest, write_manifest
from tools.content_preparation.mpd import (
    EXTENDED_BANDWIDTH_SCHEME, NAMESPACE, PROFILE, UINT32_MAX,
    build_mpd, export_mpd, load_prepared, validate_mpd,
    validate_mpd_schema,
)
from tools.content_preparation.packaging import package_object


class MPDTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.server_records = []

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, object_id="longdress", frame_count=9, fps=30, qualities=3,
                refresh=None, single_timestamp=0):
        object_root = self.root / object_id
        object_root.mkdir()
        frames = [{"frame": 1051 + i, "timestamp": single_timestamp + i / fps} for i in range(frame_count)]
        quality_ids = [f"Q{i}" for i in range(qualities)]
        records = []
        for qi, quality in enumerate(quality_ids):
            layer = "Base" if qi == 0 else f"E{qi}"
            for frame in frames:
                data = f"{layer}-{frame['frame']}".encode() * (qi + 1)
                relative = f"encoded/{layer}/{frame['frame']}.bin"
                path = object_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                records.append({"id": f"progressive:{layer}:{frame['frame']}", "mode": "progressive",
                    "quality": quality, "layer": layer, "frame": frame["frame"], "path": relative,
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "codec": {"name": "test-lossless", "version": "1.0.0"},
                    "decoded_state_hash": hashlib.sha256(data).hexdigest()})
        config = {"gof_frames": 6, "segment_frames": 4,
                  "refresh_frames": refresh if refresh is not None else {"Base": 6, "default": 2}}
        index = package_object(object_id, frames, quality_ids, records, config, object_root)
        profile = object_root / "profiles/profile.json"
        write_json(profile, {"rows": []})
        manifest = build_manifest(object_id, frames, quality_ids, index,
            provenance={"config_sha256": "a" * 64, "upstream_sources": {"train.py": "b" * 64}},
            quality_profile_path="profiles/profile.json", output_dir=object_root)
        manifest["object"].update(fps=fps, sample_duration_seconds=1 / fps,
                                  duration_seconds=frame_count / fps)
        manifest_path = object_root / "manifest.json"
        write_manifest(manifest_path, manifest)
        self.server_records.append({"id": object_id, "manifest_path": f"{object_id}/manifest.json",
                                    "bytes": manifest_path.stat().st_size, "sha256": sha256(manifest_path)})
        write_json(self.root / "manifest.json", {"schema_version": "content-preparation.server.v1",
                                                 "objects": self.server_records})
        return manifest

    def parse(self, encoded):
        return ET.fromstring(encoded), {"d": NAMESPACE}

    def test_nonuniform_segment_alignment_timing_dependency_and_actual_rates(self):
        self.fixture(qualities=5)
        objects, inputs = load_prepared(self.root)
        encoded, sidecar = build_mpd(objects)
        xml, ns = self.parse(encoded)
        self.assertEqual(xml.attrib["profiles"], PROFILE)
        self.assertEqual(xml.attrib["mediaPresentationDuration"], "PT0.3S")
        adaptation = xml.find("d:Period/d:AdaptationSet", ns)
        self.assertEqual(adaptation.attrib["segmentAlignment"], "false")
        reps = adaptation.findall("d:Representation", ns)
        self.assertEqual(len(reps), 5)
        self.assertNotIn("dependencyId", reps[0].attrib)
        self.assertEqual(reps[-1].attrib["dependencyId"], "longdress:Base longdress:E1 longdress:E2 longdress:E3")
        self.assertTrue(all("startWithSAP" not in r.attrib and "codecs" not in r.attrib for r in reps))
        timelines = [[int(s.attrib["d"]) for s in r.findall("d:SegmentList/d:SegmentTimeline/d:S", ns)] for r in reps]
        self.assertEqual(timelines[0], [4, 2, 3])
        self.assertEqual(timelines[1], [2, 2, 2, 2, 1])
        for rep, record in zip(reps, sidecar["objects"][0]["representations"]):
            urls = rep.findall("d:SegmentList/d:SegmentURL", ns)
            self.assertEqual(record["own_bytes"], sum((self.root / url.attrib["media"]).stat().st_size for url in urls))
            self.assertEqual(float(record["own_mean_bits_per_second"]), record["own_bytes"] * 8 / 0.3)
            self.assertGreaterEqual(int(rep.attrib["bandwidth"]), record["cumulative_peak_bound_bits_per_second"])
            self.assertEqual(int(rep.attrib["bandwidth"]), record["required_bandwidth_bits_per_second"])
            self.assertFalse(record["bandwidth_saturated"])
            self.assertFalse(any(p.attrib["schemeIdUri"] == EXTENDED_BANDWIDTH_SCHEME
                                 for p in rep.findall("d:EssentialProperty", ns)))
            self.assertTrue(all(not url.attrib["media"].startswith("/") and ".." not in url.attrib["media"] for url in urls))
        self.assertIsNone(sidecar["objects"][0]["proxy_path"])
        self.assertEqual(sidecar["objects"][0]["quality_profile_path"], "longdress/profiles/profile.json")
        self.assertTrue(inputs)

    def test_multiobject_max_duration_and_aligned_boundaries(self):
        self.fixture("longdress", frame_count=9, qualities=2, refresh=6)
        self.fixture("soldier", frame_count=6, fps=60, qualities=2, refresh=6)
        encoded, sidecar = build_mpd(load_prepared(self.root)[0])
        xml, ns = self.parse(encoded)
        adaptations = xml.findall("d:Period/d:AdaptationSet", ns)
        self.assertEqual(len(adaptations), 2)
        self.assertEqual([a.attrib["segmentAlignment"] for a in adaptations], ["true", "true"])
        self.assertEqual(sidecar["duration_seconds"], 0.3)
        self.assertEqual([o["timescale"] for o in sidecar["objects"]], [30, 60])

    def test_single_frame_terminal_interval_and_presentation_offset(self):
        self.fixture(frame_count=1, qualities=1, single_timestamp=2.0)
        encoded, _ = build_mpd(load_prepared(self.root)[0])
        xml, ns = self.parse(encoded)
        listing = xml.find("d:Period/d:AdaptationSet/d:Representation/d:SegmentList", ns)
        self.assertEqual(listing.attrib["presentationTimeOffset"], "60")
        self.assertEqual(listing.find("d:SegmentTimeline/d:S", ns).attrib, {"t": "60", "d": "1"})
        self.assertEqual(xml.attrib["mediaPresentationDuration"], "PT0.033333333333S")

    @unittest.skipUnless(shutil.which("xmllint"), "xmllint is required for pinned offline XSD check")
    def test_schema_export_resume_and_deterministic_xml(self):
        self.fixture()
        first = export_mpd(self.root)
        self.assertEqual((first["tasks_executed"], first["tasks_skipped"]), (1, 0))
        before = (self.root / "manifest.mpd").read_bytes()
        self.assertTrue(validate_mpd(self.root / "manifest.mpd", self.root)["validated"])
        second = export_mpd(self.root, resume=True)
        self.assertEqual((second["tasks_executed"], second["tasks_skipped"]), (0, 1))
        self.assertEqual(before, (self.root / "manifest.mpd").read_bytes())
        self.assertEqual(json.loads((self.root / ".preparation/mpd/journal.json").read_text())["tasks"]["mpd/export"]["status"], "complete")
        self.assertFalse((self.root / ".preparation/journal.json").exists())

    @unittest.skipUnless(shutil.which("xmllint"), "xmllint required")
    def test_changed_manifest_requires_overwrite_and_tamper_is_rejected(self):
        self.fixture()
        export_mpd(self.root)
        path = self.root / "longdress/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["provenance"]["project_commit"] = "c" * 40
        write_manifest(path, manifest)
        self.server_records[0].update(bytes=path.stat().st_size, sha256=sha256(path))
        write_json(self.root / "manifest.json", {"schema_version": "content-preparation.server.v1", "objects": self.server_records})
        with self.assertRaises(FileExistsError):
            export_mpd(self.root, resume=True)
        export_mpd(self.root, resume=True, overwrite=True)
        self.assertTrue(validate_mpd(self.root / "manifest.mpd")["passed"])
        content = json.loads((self.root / "content_index.json").read_text())
        content["objects"][0]["representations"][0]["own_bytes"] += 1
        write_json(self.root / "content_index.json", content)
        with self.assertRaisesRegex(ValueError, "Content index"):
            validate_mpd(self.root / "manifest.mpd")
        export_mpd(self.root, resume=True, overwrite=True)
        mpd = self.root / "manifest.mpd"
        mpd.write_text(mpd.read_text().replace('bandwidth="', 'bandwidth="1', 1))
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            validate_mpd(mpd)

    def test_owner_paths_and_actual_file_integrity(self):
        manifest = self.fixture()
        (self.root / "manifest.mpd").write_text("unowned")
        with self.assertRaisesRegex(FileExistsError, "unowned"):
            export_mpd(self.root, overwrite=True)
        with self.assertRaisesRegex(ValueError, "direct child"):
            export_mpd(self.root, self.root / "nested/manifest.mpd")
        with self.assertRaisesRegex(ValueError, "collides"):
            export_mpd(self.root, self.root / "manifest.json")
        segment = self.root / "longdress" / manifest["package"]["segments"][0]["path"]
        segment.write_bytes(segment.read_bytes() + b"damage")
        with self.assertRaises(ValueError):
            load_prepared(self.root)

    def test_invalid_duration_and_timestamp_grid(self):
        self.fixture()
        objects, _ = load_prepared(self.root)
        damaged = copy.deepcopy(objects)
        damaged[0]["manifest"]["object"]["duration_seconds"] -= 1 / 30
        with self.assertRaisesRegex(ValueError, "terminal"):
            build_mpd(damaged)
        damaged = copy.deepcopy(objects)
        damaged[0]["manifest"]["object"]["frames"][1]["timestamp"] += 0.0001
        with self.assertRaisesRegex(ValueError, "timescale"):
            build_mpd(damaged)

    @unittest.skipUnless(shutil.which("xmllint"), "xmllint required")
    def test_xsd_rejects_invalid_bandwidth(self):
        self.fixture()
        encoded, _ = build_mpd(load_prepared(self.root)[0])
        xml, ns = self.parse(encoded)
        xml.find("d:Period/d:AdaptationSet/d:Representation", ns).set("bandwidth", "invalid")
        path = self.root / "invalid.mpd"
        path.write_bytes(ET.tostring(xml))
        with self.assertRaisesRegex(ValueError, "schema validation"):
            validate_mpd_schema(path)

    @unittest.skipUnless(shutil.which("xmllint"), "xmllint required")
    def test_overflow_keeps_exact_required_rate_and_mandatory_extension(self):
        # Short fixture durations exercise uint32 overflow using real small
        # container files, without allocating multi-gigabyte media artifacts.
        self.fixture(frame_count=2, fps=4_000_000, qualities=2)
        export_mpd(self.root)
        self.assertTrue(validate_mpd(self.root / "manifest.mpd")["passed"])
        xml, ns = self.parse((self.root / "manifest.mpd").read_bytes())
        reps = xml.findall("d:Period/d:AdaptationSet/d:Representation", ns)
        sidecar = json.loads((self.root / "content_index.json").read_text())
        object_record = sidecar["objects"][0]
        cumulative = 0
        for rep, record in zip(reps, object_record["representations"]):
            own_peak = max(s["bytes"] * 8 * object_record["timescale"] // s["duration_ticks"]
                           for s in record["segments"])
            cumulative += own_peak
            self.assertGreater(cumulative, UINT32_MAX)
            self.assertEqual(int(rep.attrib["bandwidth"]), UINT32_MAX)
            self.assertEqual(record["bandwidth"], UINT32_MAX)
            self.assertEqual(record["required_bandwidth_bits_per_second"], cumulative)
            self.assertEqual(record["cumulative_peak_bound_bits_per_second"], cumulative)
            self.assertTrue(record["bandwidth_saturated"])
            self.assertEqual(record["extended_bandwidth_scheme"], EXTENDED_BANDWIDTH_SCHEME)
            extension = [p for p in rep.findall("d:EssentialProperty", ns)
                         if p.attrib["schemeIdUri"] == EXTENDED_BANDWIDTH_SCHEME]
            self.assertEqual(len(extension), 1)
            self.assertEqual(extension[0].attrib["value"], str(cumulative))
        self.assertEqual(sidecar["bandwidth_signaling"]["standard_attribute_max"], UINT32_MAX)
        # Dropping the required extension remains schema-valid XML but violates
        # the experimental project's accounting/decoder contract.
        reps[-1].remove(extension[0])
        path = self.root / "manifest.mpd"
        path.write_bytes(ET.tostring(xml))
        self.assertTrue(validate_mpd_schema(path)["validated"])
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            validate_mpd(path)


if __name__ == "__main__":
    unittest.main()
