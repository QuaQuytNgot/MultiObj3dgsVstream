"""Small CPU-only checks of bytes, dependency access and manifest semantics."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.content_preparation.manifest import build_manifest, load_manifest, write_manifest
from tools.content_preparation.packaging import (
    cold_access, extract_segment_member, package_object, read_segment_header,
)
from tools.content_preparation.validation import (
    safe_relative_path, validate_dependency_graph, validate_manifest, validate_package_index,
)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frames = [{"frame": 1051 + i, "timestamp": i / 30} for i in range(6)]
        self.qualities = ["Q0", "Q1", "Q2"]
        self.records = []
        self.data = {}
        for mode in ("independent", "progressive"):
            for qi, quality in enumerate(self.qualities):
                layer = quality if mode == "independent" else ("Base" if qi == 0 else f"E{qi}")
                for frame in self.frames:
                    value = f"{mode},{quality},{frame['frame']}\n".encode() * (qi + 1)
                    relative = f"encoded/{mode}/{layer}/{frame['frame']}.bin"
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value)
                    identifier = f"{mode}:{layer}:{frame['frame']}"
                    self.data[identifier] = value
                    self.records.append({"id": identifier, "mode": mode, "quality": quality,
                                         "layer": layer, "frame": frame["frame"], "path": relative,
                                         "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(),
                                         "codec": {"name": "test-lossless", "version": "1"},
                                         "decoded_state_hash": hashlib.sha256(value).hexdigest()})
        self.config = {"gof_frames": 4, "segment_frames": 3,
                       "refresh_frames": {"default": 4, "Base": 4, "E1": 2, "E2": 1}}

    def tearDown(self):
        self.temporary.cleanup()

    def package(self):
        return package_object("object", self.frames, self.qualities, self.records, self.config, self.root)

    def test_actual_container_roundtrip_and_accounting(self):
        index = self.package()
        self.assertEqual(index["media_bytes"], sum((self.root / segment["path"]).stat().st_size for segment in index["segments"]))
        for segment in index["segments"]:
            header, offset = read_segment_header(self.root / segment["path"])
            self.assertEqual(offset, segment["header_bytes"])
            self.assertEqual(header["offset_origin"], "data_section")
            self.assertEqual(segment["bytes"], segment["header_bytes"] + segment["payload_bytes"])
            for member in segment["members"]:
                self.assertEqual(extract_segment_member(self.root / segment["path"], member["id"]), self.data[member["id"]])
        validate_package_index(index, self.root)

    def test_deterministic_packaging_and_no_input_mutation(self):
        original = copy.deepcopy(self.records)
        index = self.package()
        self.assertEqual(index, self.package())
        self.assertEqual(self.records, original)
        self.assertEqual(json.loads((self.root / "package_index.json").read_text()), index)

    def test_independent_and_progressive_access_graph(self):
        index = self.package()
        for payload in index["payloads"]:
            access = cold_access(index, payload["id"])
            closure = access["dependencies_in_decode_order"]
            if payload["mode"] == "independent":
                self.assertEqual(closure, [payload["id"]])
                self.assertTrue(payload["self_contained"])
            else:
                qi = self.qualities.index(payload["quality"])
                self.assertEqual(len(closure), qi + 1)
                self.assertTrue(all(p.endswith(str(payload["frame"])) for p in closure))
            self.assertGreater(access["request_bytes"], access["required_payload_bytes"])
            self.assertEqual(access["request_bytes"], sum(s["bytes"] for s in index["segments"] if s["id"] in access["segment_ids"]))

    def test_refresh_epoch_network_cut_semantics(self):
        index = self.package()
        self.assertEqual(len(index["epochs"]), 2)
        self.assertFalse(index["temporal_prediction"])
        selected = [s for s in index["segments"] if s["mode"] == "progressive" and s["layer"] == "E1" and s["gof"] == 0]
        self.assertEqual([s["frames"] for s in selected], [[1051, 1052], [1053], [1054]])
        self.assertEqual([p["frame"] for p in index["payloads"] if p["mode"] == "progressive" and p["layer"] == "E1" and p["configured_refresh"]], [1051, 1053, 1055])
        self.assertTrue(all(p["codec_access_point"] for p in index["payloads"]))
        self.assertEqual([p["frame"] for p in index["payloads"] if p["mode"] == "progressive" and p["layer"] == "Base" and p["full_reset"]], [1051, 1055])

    def test_common_refresh_supported(self):
        self.config["refresh_frames"] = 2
        index = self.package()
        self.assertTrue(all(r["refresh_interval_frames"] == 2 for r in index["representations"]))

    def test_refresh_map_aliases_apply_to_both_modes(self):
        self.config["refresh_frames"] = {"Base": 4, "E1": 2, "E2": 1}
        index = self.package()
        self.assertEqual([r["refresh_interval_frames"] for r in index["representations"]], [4, 2, 1, 4, 2, 1])
        self.config["refresh_frames"] = {"Q0": 4, "Q1": 2, "Q2": 1}
        self.assertEqual(index, self.package())

    def test_adversarial_epoch_frame_access_and_parent_validation(self):
        index = self.package()
        edits = [lambda value: value["epochs"][0].update({"full_reset_frame": 1052}),
                 lambda value: value["segments"][0].update({"frame_range": [1052, 1053]}),
                 lambda value: value["payloads"][0].update({"timestamp": 100}),
                 lambda value: value["payloads"][0].update({"segment_offset": 0}),
                 lambda value: value["representations"][0].update({"access_points": [1051]}),
                 lambda value: value["representations"].append(copy.deepcopy(value["representations"][0]))]
        for edit in edits:
            damaged = copy.deepcopy(index)
            edit(damaged)
            with self.assertRaises(ValueError):
                validate_package_index(damaged)
        records = copy.deepcopy(self.records)
        for record in records:
            if record["id"] == "progressive:E2:1051":
                record["parent_layer"] = "Base"
                record["dependencies"] = ["progressive:Base:1051"]
        with self.assertRaisesRegex(ValueError, "preceding quality"):
            package_object("object", self.frames, self.qualities, records, self.config, self.root)

    def test_manifest_roundtrip_hashes_and_provenance(self):
        index = self.package()
        (self.root / "proxy.json").write_text('{"fixed_quality": "Q0"}')
        (self.root / "profile.json").write_text('{"samples": []}')
        provenance = {"checkpoint": "sha256:" + "a" * 64, "config_hash": "b" * 64,
                      "upstream_renderer": "gaussian_renderer.render"}
        manifest = build_manifest("object", self.frames, self.qualities, index,
                                  provenance=provenance, proxy_path="proxy.json",
                                  quality_profile_path="profile.json", output_dir=self.root)
        path = self.root / "manifest.json"
        write_manifest(path, manifest)
        self.assertEqual(load_manifest(path, validate_files=True), manifest)
        self.assertEqual(manifest["provenance"], provenance)
        self.assertEqual(len(manifest["assets"]), 2)
        (self.root / "proxy.json").write_text("changed")
        with self.assertRaises(ValueError):
            load_manifest(path, validate_files=True)

    def test_reject_corrupt_bytes_and_bad_graphs(self):
        bad = copy.deepcopy(self.records)
        bad[0]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            package_object("object", self.frames, self.qualities, bad, self.config, self.root)
        index = self.package()
        payloads = copy.deepcopy(index["payloads"])
        payloads[0]["dependencies"] = [payloads[-1]["id"]]
        with self.assertRaisesRegex(ValueError, "Independent"):
            validate_dependency_graph(payloads)
        payloads = copy.deepcopy(index["payloads"])
        by_id = {p["id"]: p for p in payloads}
        by_id["progressive:E1:1051"]["dependencies"] = ["progressive:Base:1052"]
        with self.assertRaisesRegex(ValueError, "same frame"):
            validate_dependency_graph(payloads)
        segment = index["segments"][0]
        path = self.root / segment["path"]
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        with self.assertRaises(ValueError):
            validate_package_index(index, self.root)
        with self.assertRaises(ValueError):
            extract_segment_member(path, segment["members"][-1]["id"])

    def test_reject_unsafe_paths_incomplete_mapping_and_stale_refresh(self):
        for path in ("/absolute.bin", "../escape.bin", "folder/../../escape.bin", "a\\b", "."):
            with self.assertRaises(ValueError):
                safe_relative_path(path)
        with self.assertRaisesRegex(ValueError, "Missing payload"):
            package_object("object", self.frames, self.qualities, self.records[:-1], self.config, self.root)
        index = self.package()
        index["payloads"][0]["configured_refresh"] = False
        with self.assertRaisesRegex(ValueError, "refresh"):
            validate_package_index(index)
        index = self.package()
        manifest = build_manifest("object", self.frames, self.qualities, index)
        manifest["object"]["proxy_path"] = "../escape.json"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_temporal_prediction_is_never_silently_relabelled(self):
        records = copy.deepcopy(self.records)
        records[0]["temporal_dependencies"] = ["previous-frame"]
        with self.assertRaisesRegex(ValueError, "independently initialized"):
            package_object("object", self.frames, self.qualities, records, self.config, self.root)
        frames = copy.deepcopy(self.frames)
        frames[0]["timestamp"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            package_object("object", frames, self.qualities, self.records, self.config, self.root)


if __name__ == "__main__":
    unittest.main()
