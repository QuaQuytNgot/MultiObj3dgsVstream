"""Real public-Draco bridge tests; build the binary before executing this file."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tools.content_preparation.assets import GaussianState, file_hash, state_hash, synthetic_state
from tools.content_preparation.codec_adapter import CodecAdapter, get_codec, inspect_payload
from tools.content_preparation.draco_adapter import CODEC_NAME, DracoCodecAdapter, normalize_config, runtime_info


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "output/build/content_preparation/draco_byte_codec"


class DracoConfigurationTests(unittest.TestCase):
    def test_config_dispatch_and_no_silent_quantization(self):
        config = {"name": CODEC_NAME, "precision": "float32", "compression": "zlib", "level": 6}
        self.assertIsInstance(get_codec(config), DracoCodecAdapter)
        self.assertNotIn("compression", normalize_config(config))
        for bad in ({"precision": "f16"}, {"precision": "q8"}, {"encoder_speed": True},
                    {"decoder_speed": 11}, {"compression": "none"}, {"level": 1},
                    {"attribute_precision": {"xyz": "f32"}}, {"unknown": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_config(bad)


@unittest.skipUnless(BINARY.is_file(), "Build the public-Draco native bridge first")
class DracoCodecTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.codec = get_codec({"name": CODEC_NAME})
        self.base = synthetic_state(0, 8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_attribute_bits_order_duplicates_and_provenance(self):
        state = synthetic_state(0, 16)
        values = np.array([0., -0., np.finfo(np.float32).smallest_subnormal,
                           -np.finfo(np.float32).smallest_subnormal, np.finfo(np.float32).max,
                           -np.finfo(np.float32).max, 1., -1.], dtype="<f4")
        for attribute in state.arrays:
            state.arrays[attribute].flat[:8] = values
        state.arrays["xyz"][8:] = state.arrays["xyz"][8]
        state.arrays["sh"] = np.resize(values, (16, 3, 16)).astype("<f4")
        state.ids = np.array([2**60 + i for i in reversed(range(16))], dtype="<i8")
        record = self.codec.encode(state, self.root / "first.bin")
        self.codec.encode(state, self.root / "second.bin")
        self.assertEqual((self.root / "first.bin").read_bytes(), (self.root / "second.bin").read_bytes())
        decoded = self.codec.decode(self.root / "first.bin")
        self.assertEqual(state_hash(decoded), state_hash(state))
        self.assertEqual(decoded.count, state.count)
        np.testing.assert_array_equal(decoded.ids, state.ids)
        for attribute in state.arrays:
            self.assertEqual(decoded.arrays[attribute].tobytes(), state.arrays[attribute].tobytes())
        header = inspect_payload(self.root / "first.bin")
        self.assertEqual(record["bytes"], (self.root / "first.bin").stat().st_size)
        self.assertEqual(record["sha256"], file_hash(self.root / "first.bin"))
        self.assertEqual(header["compressor"]["runtime"]["binary_sha256"], file_hash(BINARY))
        self.assertEqual(header["compressor"]["runtime"]["draco_version"], "1.5.7")
        self.assertFalse(header["temporal_prediction"])

    def test_progressive_corrects_shared_attributes_new_deleted_reordered(self):
        target = synthetic_state(0, 12)
        for attribute in target.arrays:
            target.arrays[attribute][:4] += np.float32(.125)
        target = target.subset(np.array([11, 1, 4, 9, 0, 6, 8, 5, 10, 7, 3]))
        self.codec.encode(self.base, self.root / "base.bin")
        parent = self.codec.decode(self.root / "base.bin")
        record = self.codec.encode(target, self.root / "enh.bin", parent=parent)
        decoded = self.codec.decode(self.root / "enh.bin", parent=parent)
        self.assertEqual(state_hash(decoded), state_hash(target))
        self.assertEqual(record["new_ids"], 4)
        self.assertEqual(record["deleted_ids"], 1)
        self.assertTrue(all(record["changed_rows"][a] > 4 for a in target.arrays))
        with self.assertRaisesRegex(ValueError, "requires its decoded parent"):
            self.codec.decode(self.root / "enh.bin")
        with self.assertRaisesRegex(ValueError, "parent.*hash mismatch"):
            self.codec.decode(self.root / "enh.bin", parent=parent.subset(np.arange(7)))
        foreign_frame = GaussianState(parent.arrays, {**parent.metadata, "frame": 1}, parent.ids)
        with self.assertRaisesRegex(ValueError, "same-frame"):
            self.codec.decode(self.root / "enh.bin", parent=foreign_frame)

    def test_closed_loop_baseline_parent_sh_change_empty_and_deletion(self):
        baseline = CodecAdapter()
        baseline.encode(self.base, self.root / "lossy.bin", {"precision": "q8"})
        parent = baseline.decode(self.root / "lossy.bin")
        self.codec.encode(self.base, self.root / "correction.bin", parent=parent)
        self.assertEqual(state_hash(self.codec.decode(self.root / "correction.bin", parent)), state_hash(self.base))
        richer = GaussianState({**self.base.arrays, "sh": np.repeat(self.base.arrays["sh"], 16, axis=2)},
                               dict(self.base.metadata), self.base.ids)
        self.codec.encode(richer, self.root / "richer.bin", parent=self.base)
        self.assertEqual(state_hash(self.codec.decode(self.root / "richer.bin", self.base)), state_hash(richer))
        empty = self.base.subset(np.array([], dtype=int))
        for parent in (None, self.base):
            self.codec.encode(empty, self.root / "empty.bin", parent=parent)
            self.assertEqual(state_hash(self.codec.decode(self.root / "empty.bin", parent)), state_hash(empty))
        self.assertEqual(inspect_payload(self.root / "lossy.bin")["codec"]["name"], baseline.name)

    def test_native_byte_chunk_roundtrip_and_rejected_dimensions(self):
        raw = np.tile(np.arange(197, dtype=np.uint8), (9, 1)).tobytes()
        source = self.root / "raw.bin"
        encoded = self.root / "encoded.bin"
        decoded = self.root / "decoded.bin"
        source.write_bytes(raw)
        argv = [str(BINARY), "encode", "--input", str(source), "--output", str(encoded),
                "--rows", "9", "--row-bytes", "197"]
        subprocess.run(argv, check=True, capture_output=True)
        subprocess.run([str(BINARY), "decode", "--input", str(encoded), "--output", str(decoded),
                        "--rows", "9", "--row-bytes", "197"], check=True, capture_output=True)
        self.assertEqual(decoded.read_bytes(), raw)
        for rows, width in (("0", "197"), ("1", "4097"), ("9", "198"), ("-1", "197")):
            result = subprocess.run(argv[:-4] + ["--rows", rows, "--row-bytes", width], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        encoded.write_bytes(encoded.read_bytes() + b"trailing")
        result = subprocess.run([str(BINARY), "decode", "--input", str(encoded), "--output", str(decoded),
                                 "--rows", "9", "--row-bytes", "197"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_corruption_truncation_and_wrong_packet(self):
        self.codec.encode(self.base, self.root / "valid.bin")
        valid = (self.root / "valid.bin").read_bytes()
        for blob in (valid[:-1], valid[:100], valid + b"x", valid[:30] + b"x" + valid[31:]):
            (self.root / "bad.bin").write_bytes(blob)
            with self.assertRaises(ValueError):
                self.codec.decode(self.root / "bad.bin")
        info = runtime_info()
        self.assertEqual(info["maximum_components"], 64)
        self.assertEqual(info["prediction"], "none")

    def test_decoder_runtime_override_does_not_depend_on_encoder_path(self):
        relocated_encoder = self.root / "encoder-binary"
        relocated_encoder.symlink_to(BINARY)
        self.codec.encode(self.base, self.root / "portable.bin",
                          {"executable": str(relocated_encoder)})
        relocated_encoder.unlink()
        decoded = get_codec({"name": CODEC_NAME}).decode(self.root / "portable.bin")
        self.assertEqual(state_hash(decoded), state_hash(self.base))


if __name__ == "__main__":
    unittest.main()
