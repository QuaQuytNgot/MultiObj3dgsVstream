"""Bounded CPU tests for file identity, accounting and progressive correction."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.content_preparation.assets import (GaussianState, file_hash, load_state,
    read_ply, save_state, state_hash, synthetic_state, write_ply)
from tools.content_preparation.codec_adapter import CodecAdapter, inspect_payload


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.codec = CodecAdapter()
        self.base = synthetic_state(frame=0, quality_count=8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_assets_deterministic_and_provenance(self):
        self.base.metadata["provenance"] = {"checkpoint_sha256": "a" * 64, "label": "đối tượng"}
        save_state(self.root / "one.npz", self.base)
        save_state(self.root / "two.npz", self.base)
        self.assertEqual(file_hash(self.root / "one.npz"), file_hash(self.root / "two.npz"))
        loaded = load_state(self.root / "one.npz")
        self.assertEqual(state_hash(loaded), state_hash(self.base))
        self.assertEqual(loaded.metadata, self.base.metadata)
        write_ply(self.root / "one.ply", self.base)
        ply = read_ply(self.root / "one.ply")
        self.assertEqual(state_hash(ply), state_hash(self.base))
        self.assertEqual(ply.metadata["provenance"], self.base.metadata["provenance"])

    def test_full_codec_roundtrip_and_actual_byte_accounting(self):
        for precision in ("f32", "f16", "q8", "q10", "q12", "q16"):
            with self.subTest(precision=precision):
                config = {"precision": precision, "compression": "zlib", "level": 6}
                first, second = self.root / "first.bin", self.root / "second.bin"
                record = self.codec.encode(self.base, first, config)
                self.codec.encode(self.base, second, config)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                decoded = self.codec.decode(first)
                self.assertEqual(record["bytes"], first.stat().st_size)
                self.assertEqual(record["sha256"], file_hash(first))
                self.assertEqual(record["decoded_state_hash"], state_hash(decoded))
                if precision == "f32":
                    self.assertEqual(state_hash(decoded), state_hash(self.base))
                else:
                    for attr in self.base.arrays:
                        np.testing.assert_allclose(decoded.arrays[attr], self.base.arrays[attr], atol=0.015, rtol=0.015)

    def test_progressive_refines_all_attributes_and_handles_ids(self):
        target = synthetic_state(frame=0, quality_count=12)
        for attribute in target.arrays:
            target.arrays[attribute][:4] += np.float32(0.1)
        # Delete one old Gaussian and change output row order as well as add IDs.
        target = target.subset(np.array([11, 1, 4, 9, 0, 6, 8, 5, 10, 7, 3]))
        for precision in ("f32", "f16", "q12"):
            config = {"precision": precision}
            self.codec.encode(self.base, self.root / "base.bin", config)
            parent = self.codec.decode(self.root / "base.bin")
            record = self.codec.encode(target, self.root / "enh.bin", config, parent=parent)
            progressive = self.codec.decode(self.root / "enh.bin", parent=parent)
            self.codec.encode(target, self.root / "full.bin", config)
            independent = self.codec.decode(self.root / "full.bin")
            self.assertEqual(state_hash(progressive), state_hash(independent))
            self.assertEqual(record["deleted_ids"], 1)
            self.assertEqual(record["new_ids"], 4)
            self.assertTrue(all(record["changed_rows"][a] > 4 for a in target.arrays))
            with self.assertRaisesRegex(ValueError, "requires its decoded parent"):
                self.codec.decode(self.root / "enh.bin")
            wrong = parent.subset(np.arange(parent.count - 1))
            with self.assertRaisesRegex(ValueError, "parent.*hash mismatch"):
                self.codec.decode(self.root / "enh.bin", parent=wrong)

    def test_closed_loop_parent_not_original_parent(self):
        self.codec.encode(self.base, self.root / "base.bin", {"precision": "q8"})
        decoded_parent = self.codec.decode(self.root / "base.bin")
        self.codec.encode(self.base, self.root / "enh.bin", {"precision": "f32"}, parent=decoded_parent)
        decoded = self.codec.decode(self.root / "enh.bin", parent=decoded_parent)
        self.assertEqual(state_hash(decoded), state_hash(self.base))
        self.assertGreater(inspect_payload(self.root / "enh.bin")["changed_rows"]["xyz"], 0)

    def test_refinement_identity_must_be_verified(self):
        unverified = GaussianState(self.base.arrays, metadata={"stable_ids": False})
        with self.assertRaisesRegex(ValueError, "stable IDs"):
            self.codec.encode(unverified, self.root / "bad.bin", parent=self.base)
        unverified.metadata["lineage_verified"] = True
        self.codec.encode(unverified, self.root / "ok.bin", parent=self.base)

    def test_no_shared_ids_and_sh_schema_change(self):
        target = synthetic_state(0, 4)
        target.ids += 100
        self.codec.encode(target, self.root / "no_shared.bin", parent=self.base)
        self.assertEqual(state_hash(self.codec.decode(self.root / "no_shared.bin", parent=self.base)), state_hash(target))
        richer = GaussianState({**self.base.arrays,
            "sh": np.repeat(self.base.arrays["sh"], 4, axis=2)}, dict(self.base.metadata), self.base.ids)
        self.codec.encode(richer, self.root / "richer.bin", parent=self.base)
        self.assertEqual(state_hash(self.codec.decode(self.root / "richer.bin", parent=self.base)), state_hash(richer))

    def test_tamper_detection_and_configuration_failure(self):
        self.codec.encode(self.base, self.root / "corrupt.bin")
        blob = bytearray((self.root / "corrupt.bin").read_bytes())
        blob[-50] ^= 1
        (self.root / "corrupt.bin").write_bytes(blob)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.codec.decode(self.root / "corrupt.bin")
        for config in ({"name": "LTS"}, {"precision": "q7"}, {"unexpected": True}, {"level": 10}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.codec.encode(self.base, self.root / "bad.bin", config)

    def test_identity_validation(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            GaussianState(self.base.arrays, ids=np.zeros(self.base.count, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "integers"):
            GaussianState(self.base.arrays, ids=np.arange(self.base.count, dtype=np.float32))

    def test_signed_bitpacking_bounds_all_precisions(self):
        rng = np.random.default_rng(123)
        state = synthetic_state(0, 13)
        state.arrays["sh"] = rng.normal(size=(13, 3, 16)).astype(np.float32)
        for bits in range(8, 17):
            with self.subTest(bits=bits):
                self.codec.encode(state, self.root / "quantized.bin", {"precision": f"q{bits}"})
                decoded = self.codec.decode(self.root / "quantized.bin")
                header = inspect_payload(self.root / "quantized.bin")
                for attribute in state.arrays:
                    step = header["arrays"][attribute]["quantizer"]["step"]
                    error = np.max(np.abs(state.arrays[attribute].astype(np.float64) - decoded.arrays[attribute]))
                    self.assertLessEqual(error, step / 2 + 1e-6)

    def test_empty_and_complete_deletion_state(self):
        empty = self.base.subset(np.empty(0, dtype=int))
        self.codec.encode(empty, self.root / "empty.bin")
        self.assertEqual(state_hash(self.codec.decode(self.root / "empty.bin")), state_hash(empty))
        record = self.codec.encode(empty, self.root / "deletion.bin", parent=self.base)
        self.assertEqual(record["deleted_ids"], self.base.count)
        self.assertEqual(state_hash(self.codec.decode(self.root / "deletion.bin", parent=self.base)), state_hash(empty))


if __name__ == "__main__":
    unittest.main()
