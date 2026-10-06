"""Decoded diagnostic exports must preserve packaged state and resume safely."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.content_preparation.assets import synthetic_state, save_state, read_ply, state_hash
from tools.content_preparation.codec_adapter import get_codec
from tools.content_preparation.manifest import build_manifest, write_manifest
from tools.content_preparation.packaging import package_object
from tools.export_decoded_viewer_assets import export_viewer, CATALOG_VERSION
from tools.content_trial_viewer.vendor_dependencies import vendor


class DecodedViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = {}
        objects = []
        codec = get_codec({"name": "gaussian_attribute_zlib"})
        config = {"name": "gaussian_attribute_zlib", "precision": "f32", "compression": "zlib", "level": 6}
        for object_id in ("alpha", "beta"):
            output = self.root / object_id
            output.mkdir()
            frames = [{"frame": 10, "timestamp": 0.}, {"frame": 11, "timestamp": 1 / 30}]
            payloads, states, profile_rows = [], [], []
            for frame in frames:
                parent = None
                for qi, quality in enumerate(("Q0", "Q1")):
                    state = synthetic_state(frame["frame"], 8 + qi * 4)
                    # Exercise complete standard SH3 fields, including sign bits.
                    state.arrays["sh"] = np.pad(state.arrays["sh"], ((0, 0), (0, 0), (0, 15)))
                    state.arrays["sh"][0, 0, 15] = -0.0
                    layer = "Base" if qi == 0 else "E1"
                    packet = output / "encoded" / "progressive" / layer / f"{frame['frame']}.cpgs"
                    meta = codec.encode(state, packet, config, parent)
                    decoded = codec.decode(packet, parent)
                    cache = output / "decoded" / "progressive" / layer / f"{frame['frame']}.npz"
                    save_state(cache, decoded)
                    payloads.append({**meta, "id": f"progressive:{layer}:{frame['frame']}", "mode": "progressive", "quality": quality,
                        "layer": layer, "frame": frame["frame"], "path": packet.relative_to(output).as_posix(),
                        "dependencies": [] if qi == 0 else [f"progressive:Base:{frame['frame']}"], "parent_layer": None if qi == 0 else "Base"})
                    states.append({"mode": "progressive", "quality": quality, "layer": layer, "frame": frame["frame"],
                                   "path": cache.relative_to(output).as_posix(), "state_hash": state_hash(decoded)})
                    self.expected[(object_id, quality, frame["frame"])] = state_hash(decoded)
                    image = output / "profiles/images" / quality / f"{frame['frame']}.npz"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(image, rgb=np.full((16, 20, 3), .4, dtype=np.float32))
                    profile_rows.append({"mode": "progressive", "quality": quality, "frame": frame["frame"], "timestamp": frame["timestamp"],
                        "azimuth": 0, "elevation": 0, "scale": 1, "view_bin": "a0_e0", "sample_id": f"{quality}_{frame['frame']}",
                        "image_path": image.relative_to(output).as_posix(), "metrics": {"mse": 0, "psnr": "inf", "lpips": 0}, "reference_quality": "Q1"})
                    parent = decoded
            package = package_object(object_id, frames, ["Q0", "Q1"], payloads,
                {"gof_frames": 2, "segment_frames": 1, "refresh_frames": {"Base": 2, "E1": 1}}, output)
            (output / "package.json").write_text(json.dumps(package))
            (output / "decoding.json").write_text(json.dumps({"states": states, "source": "requestable segment containers only"}))
            (output / "training.json").write_text(json.dumps({"backend": "synthetic", "commands": []}))
            profile = output / "profiles/profile.json"
            profile.write_text(json.dumps({"rows": profile_rows, "aggregates": [], "reference": "highest decoded"}))
            write_manifest(output / "manifest.json", build_manifest(object_id, frames, ["Q0", "Q1"], package,
                quality_profile_path="profiles/profile.json", output_dir=output))
            objects.append({"id": object_id, "qualities": [{"id": "Q0"}, {"id": "Q1"}], "fps": 30})
        preparation = self.root / ".preparation"
        preparation.mkdir()
        (preparation / "resolved_config.json").write_text(json.dumps({"objects": objects,
            "renderer": {"backend": "synthetic", "width": 20, "height": 16, "up_axis": "z"}}))

    def test_multiple_objects_sh3_roundtrip_and_resume(self):
        result = export_viewer(self.root)
        self.assertEqual(result["asset_count"], 8)
        index = json.loads((self.root / "viewer_assets/index.json").read_text())
        self.assertFalse(index["included_in_encoded_media_bytes"])
        for row in index["assets"]:
            self.assertEqual(row["sh_degree"], 3)
            self.assertEqual(state_hash(read_ply(self.root / row["path"])), self.expected[(row["object"], row["quality"], row["frame"])])
            self.assertEqual(row["decoded_state_hash"], self.expected[(row["object"], row["quality"], row["frame"])])
            self.assertEqual(len(row["dependencies_in_decode_order"]), 1 if row["quality"] == "Q0" else 2)
        catalog_bytes = (self.root / "catalog.json").read_bytes()
        catalog = json.loads(catalog_bytes)
        self.assertEqual(catalog["schema_version"], CATALOG_VERSION)
        self.assertEqual([obj["id"] for obj in catalog["variants"][0]["objects"]], ["alpha", "beta"])
        self.assertEqual(export_viewer(self.root, resume=True)["executed"], 0)
        self.assertEqual((self.root / "catalog.json").read_bytes(), catalog_bytes)
        self.assertTrue((self.root / "vendor/spark/dist/spark.module.js").is_file())

    def test_foreign_artifacts_and_changes_are_protected(self):
        (self.root / "catalog.json").write_text("foreign")
        with self.assertRaises(FileExistsError):
            export_viewer(self.root)
        (self.root / "catalog.json").unlink()
        export_viewer(self.root)
        asset = self.root / "viewer_assets/alpha/Q0/10.ply"
        asset.write_bytes(asset.read_bytes() + b"tampered")
        with self.assertRaises(FileExistsError):
            export_viewer(self.root, resume=True)

    def test_direct_checkpoint_source_is_rejected(self):
        path = self.root / "alpha/decoding.json"
        document = json.loads(path.read_text())
        document["source"] = "training checkpoint"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "requestable segment"):
            export_viewer(self.root)

    def test_uint32_ids_and_requested_frames_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Requested frames"):
            export_viewer(self.root, frames=[99])
        from tools.content_preparation.assets import write_ply
        state = synthetic_state(0, 2)
        state.ids[0] = 2**32
        with self.assertRaisesRegex(ValueError, "uint32"):
            write_ply(self.root / "invalid.ply", state)

    def test_offline_dependency_verification(self):
        self.assertEqual(len(vendor(verify=True)), 7)


if __name__ == "__main__":
    unittest.main()
