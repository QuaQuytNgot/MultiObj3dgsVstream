"""Codec dispatch and portable decoding through actual packaged bytes."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import yaml

from tools.content_preparation.assets import canonical_json, file_hash, load_state, state_hash, synthetic_state
from tools.content_preparation.codec_adapter import get_codec, get_decoder, inspect_payload
from tools.content_preparation.config import load_config
from tools.content_preparation.draco_adapter import CODEC_NAME as DRACO_NAME
from tools.content_preparation.pipeline import Pipeline
from tests.content_preparation.test_pipeline import mini_config


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "output/build/content_preparation/draco_byte_codec"
BASELINE_NAME = "gaussian_attribute_zlib"


def make_pipeline(directory, encoding, override=None):
    config = mini_config(directory / "run")
    config["delivery_modes"] = ["progressive"]
    config["encoding"] = encoding
    config["proxy"] = {"enabled": False}
    if override:
        config["objects"][0]["qualities"][1]["encoding"] = override
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    pipeline = Pipeline(load_config(path), path)
    with redirect_stdout(io.StringIO()):
        for stage in ("preprocess", "train", "export", "encode"):
            pipeline.run(stage)
    return pipeline, path


class DecoderMetadataTests(unittest.TestCase):
    def test_decoder_preserves_packet_codec_and_validates_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.cpgs"
            codec = get_codec({"name": BASELINE_NAME, "precision": "f16"})
            record = codec.encode(synthetic_state(0, 4), path, {"precision": "f16"})
            decoder = get_decoder(record["codec"], {"name": DRACO_NAME, "precision": "f32"})
            self.assertEqual(decoder.name, BASELINE_NAME)
            self.assertEqual(state_hash(decoder.decode(path)), record["decoded_state_hash"])
            for field, value in (("name", DRACO_NAME), ("version", "99.0.0")):
                bad = {**record["codec"], field: value}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "codec name/version"):
                    get_decoder(bad)
            with self.assertRaisesRegex(ValueError, "declared codec metadata"):
                get_decoder({"name": BASELINE_NAME})


@unittest.skipUnless(BINARY.is_file(), "Build the public-Draco native bridge first")
class CodecRuntimeIntegrationTests(unittest.TestCase):
    def test_each_quality_selects_its_own_codec_in_both_directions(self):
        for default, override in ((DRACO_NAME, BASELINE_NAME), (BASELINE_NAME, DRACO_NAME)):
            with self.subTest(default=default), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                pipeline, _ = make_pipeline(directory, {"name": default, "precision": "f32"},
                                            {"name": override, "precision": "f32"})
                root = pipeline.output / "cpu_mini"
                records = json.loads((root / "encoding.json").read_text())["payloads"]
                self.assertEqual([record["codec"]["name"] for record in records], [default, override])
                self.assertEqual([inspect_payload(root / record["path"])["codec"]["name"]
                                  for record in records], [default, override])
                with redirect_stdout(io.StringIO()):
                    pipeline.run("package")
                # Delivery reconstruction must work without the encoder's files.
                shutil.rmtree(root / "encoded")
                shutil.rmtree(root / ".work/encoder_decoded")
                with redirect_stdout(io.StringIO()):
                    pipeline.run("decode")
                states = json.loads((root / "decoding.json").read_text())["states"]
                exported = json.loads((root / "export.json").read_text())["states"]
                for actual, expected in zip(states, exported):
                    self.assertEqual(state_hash(load_state(root / actual["path"])), expected["state_hash"])

    def test_segment_decode_uses_local_bridge_instead_of_historical_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pipeline, config_path = make_pipeline(directory, {
                "name": DRACO_NAME, "precision": "f32", "executable": str(BINARY),
                "encoder_speed": 7, "decoder_speed": 3})
            root = pipeline.output / "cpu_mini"
            index = json.loads((root / "encoding.json").read_text())
            for record in index["payloads"]:
                path = root / record["path"]
                blob = path.read_bytes()
                header_size = struct.unpack("<Q", blob[8:16])[0]
                header = json.loads(blob[16:16 + header_size])
                header["codec"]["config"]["executable"] = "/missing/encoder-machine/draco_byte_codec"
                header_data = canonical_json(header)
                packet = blob[:8] + struct.pack("<Q", len(header_data)) + header_data + blob[16 + header_size:-32]
                packet += hashlib.sha256(packet).digest()
                path.write_bytes(packet)
                record.update(codec=header["codec"], bytes=len(packet), payload_bytes=len(packet), sha256=file_hash(path))
            (root / "encoding.json").write_bytes(canonical_json(index))
            with redirect_stdout(io.StringIO()):
                pipeline.run("package")
            shutil.rmtree(root / "encoded")
            shutil.rmtree(root / ".work/encoder_decoded")

            local_binary = directory / "local_decoder/draco_byte_codec"
            local_binary.parent.mkdir()
            shutil.copy2(BINARY, local_binary)
            config = yaml.safe_load(config_path.read_text())
            config["encoding"]["executable"] = str(local_binary)
            config["encoding"]["encoder_speed"] = 1
            config["encoding"]["decoder_speed"] = 9
            config_path.write_text(yaml.safe_dump(config))
            decoder = get_decoder(index["payloads"][0]["codec"], config["encoding"])
            self.assertEqual(decoder.runtime_config["executable"], str(local_binary))
            self.assertEqual(decoder.runtime_config["encoder_speed"], 7)
            self.assertEqual(decoder.runtime_config["decoder_speed"], 9)
            default_runtime = get_decoder(index["payloads"][0]["codec"], {"name": DRACO_NAME, "precision": "f32"})
            self.assertEqual(default_runtime.runtime_config["executable"], "output/build/content_preparation/draco_byte_codec")
            self.assertEqual(default_runtime.runtime_config["decoder_speed"], 3)
            with self.assertRaisesRegex(RuntimeError, "native bridge missing"):
                # Packet metadata alone intentionally retains historical provenance.
                from tools.content_preparation.packaging import extract_segment_member
                package = json.loads((root / "package.json").read_text())
                standalone = directory / "historical.cpgs"
                standalone.write_bytes(extract_segment_member(root / package["segments"][0]["path"], index["payloads"][0]["id"]))
                get_decoder(index["payloads"][0]["codec"]).decode(standalone)
            portable = Pipeline(load_config(config_path), config_path)
            with redirect_stdout(io.StringIO()):
                portable.run("decode")
            decoded = json.loads((root / "decoding.json").read_text())["states"]
            self.assertEqual([state["state_hash"] for state in decoded],
                             [record["decoded_state_hash"] for record in index["payloads"]])


if __name__ == "__main__":
    unittest.main()
