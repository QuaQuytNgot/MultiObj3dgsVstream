"""Exact codec reuse, progressive identities and bounded CPU state lifetime."""
from dataclasses import replace
import io
from pathlib import Path

import numpy as np
import pytest

from src.client.adaptation_inputs import PayloadDescriptor
from src.client.decoder import MissingDependencyError, ProgressiveDecoder
from tools.content_preparation.assets import state_hash, synthetic_state
from tools.content_preparation.codec_adapter import CodecAdapter, get_decoder, inspect_payload


def ladder(tmp_path, frame=4, object_id="actor", epoch=0, precision="f32"):
    codec, previous, result = CodecAdapter(), None, []
    for quality, count in enumerate((4, 7, 10, 13)):
        target = synthetic_state(frame, count)
        target.metadata["quality"] = f"Q{quality}"
        path = tmp_path / f"{object_id}-{frame}-{quality}.cpgs"
        record = codec.encode(target, path, {"precision": precision}, parent=previous)
        reference = codec.decode(path, parent=previous)
        descriptor = PayloadDescriptor(
            f"{object_id}:{frame}:Q{quality}", object_id, frame, quality, epoch,
            frame / 30, f"{object_id}:segment:{quality}", record["decoded_state_hash"],
            () if quality == 0 else (result[-1][0].payload_id,),
            record["parent_state_hash"], codec_id=codec.name + "/" + codec.version)
        result.append((descriptor, path.read_bytes(), reference))
        previous = reference
    return result


@pytest.mark.parametrize("precision", ["f32", "f16", "q12"])
def test_progressive_states_match_reference_and_switching_reuses_states(tmp_path, precision):
    packets = ladder(tmp_path, precision=precision)
    with ProgressiveDecoder(max_cached_frames=1) as decoder:
        for descriptor, blob, reference in packets:
            actual = decoder.decode_payload(descriptor, blob)
            assert state_hash(actual.state) == state_hash(reference)
            for name, values in reference.arrays.items():
                np.testing.assert_array_equal(actual.state.arrays[name], values)
                assert not actual.state.arrays[name].flags.writeable
            np.testing.assert_array_equal(actual.state.ids, reference.ids)
            assert (actual.frame_index, actual.timestamp, actual.quality) == (
                descriptor.frame, descriptor.media_time_s, descriptor.quality)
            assert actual.decode_seconds >= 0
            assert decoder.decoded_states[descriptor.payload_id].matches(descriptor)
        total = decoder.total_decode_seconds
        for quality in (0, 3, 1, 2, 0):
            descriptor, blob, _ = packets[quality]
            original = decoder.get_frame("actor", 4, quality)
            cached = decoder.decode_payload(descriptor, memoryview(blob))
            assert cached.state is original.state
            assert cached.cache_hit and cached.decode_seconds == 0
        assert decoder.decode_count == 4 and decoder.cache_hits == 5
        assert decoder.total_decode_seconds == total
        assert decoder.retained_bytes > 0
    assert not decoder.decoded_states and decoder.retained_bytes == 0


def test_codec_accepts_encoded_buffers_and_preserves_binary_stream_position(tmp_path):
    descriptor, blob, reference = ladder(tmp_path)[0]
    stream = io.BytesIO(b"prefix" + blob)
    stream.seek(6)
    header = inspect_payload(stream)
    assert stream.tell() == 6
    codec = get_decoder(header["codec"])
    assert state_hash(codec.decode(stream)) == state_hash(reference)
    assert stream.tell() == 6
    assert state_hash(codec.decode(bytearray(blob))) == descriptor.decoded_state_hash


def test_missing_parent_never_decodes_or_mutates_cache(tmp_path):
    packets = ladder(tmp_path)
    decoder = ProgressiveDecoder()
    with pytest.raises(MissingDependencyError, match="Missing same-frame"):
        decoder.decode_payload(*packets[1][:2])
    assert decoder.decode_count == 0 and not decoder.decoded_states
    decoder.decode_payload(*packets[0][:2])
    descriptor, blob, _ = packets[1]
    for changed in (replace(descriptor, object_id="other"), replace(descriptor, epoch=3),
                    replace(descriptor, media_time_s=1.0)):
        with pytest.raises(ValueError, match="same object, frame, epoch and timestamp"):
            decoder.decode_payload(changed, blob)
    assert decoder.decode_count == 1 and len(decoder.decoded_states) == 1


@pytest.mark.parametrize("field,value,message", [
    ("decoded_state_hash", "wrong", "decoded-state hash"),
    ("parent_state_hash", "wrong", "parent-state hash"),
    ("codec_id", "unknown/1", "codec differs"),
    ("state_version", "future-state", "version mismatch"),
    ("frame", 100, "frame metadata"),
    ("quality", 1, "quality metadata"),
])
def test_packet_descriptor_mismatches_are_rejected(tmp_path, field, value, message):
    descriptor, blob, _ = ladder(tmp_path)[0]
    with pytest.raises(ValueError, match=message):
        ProgressiveDecoder().decode_payload(replace(descriptor, **{field: value}), blob)


def test_corrupt_bytes_cannot_reuse_cached_identity(tmp_path):
    descriptor, blob, _ = ladder(tmp_path)[0]
    bad = blob[:-1] + bytes((blob[-1] ^ 1,))
    with pytest.raises(ValueError, match="checksum"):
        ProgressiveDecoder().decode_payload(descriptor, bad)
    decoder = ProgressiveDecoder()
    decoder.decode_payload(descriptor, blob)
    with pytest.raises(ValueError, match="different bytes"):
        decoder.decode_payload(descriptor, bad)
    assert decoder.decode_count == 1 and decoder.cache_hits == 0


def test_cache_evicts_whole_frames_and_isolates_object_lifetimes(tmp_path):
    decoder = ProgressiveDecoder(max_cached_frames=2)
    for object_id in ("first", "second"):
        for descriptor, blob, _ in ladder(tmp_path, object_id=object_id):
            decoder.decode_payload(descriptor, blob)
    first = decoder.get_frame("first", 4, 3)
    second = decoder.get_frame("second", 4, 3)
    assert first.state is not second.state
    descriptor, blob, _ = ladder(tmp_path, frame=5, object_id="first")[0]
    decoder.decode_payload(descriptor, blob)
    with pytest.raises(KeyError):
        decoder.get_frame("first", 4, 3)
    assert len(decoder.decoded_states) == 5
    decoder.release_object("second")
    assert len(decoder.decoded_states) == 1
    decoder.release_frame("first", 5)
    assert not decoder.decoded_states


def test_sparse_replacements_handle_deletion_reorder_and_sh_schema_change(tmp_path):
    codec = CodecAdapter()
    base = synthetic_state(4, 5)
    target = synthetic_state(4, 7).subset([6, 2, 0, 5])
    target.arrays["sh"] = np.repeat(target.arrays["sh"], 4, axis=2)
    base.metadata["quality"], target.metadata["quality"] = "Q0", "Q1"
    base_path, target_path = tmp_path / "base.cpgs", tmp_path / "refinement.cpgs"
    first = codec.encode(base, base_path)
    base_decoded = codec.decode(base_path)
    second = codec.encode(target, target_path, parent=base_decoded)
    a = PayloadDescriptor("base", "actor", 4, 0, 0, 0, "segment-a", first["decoded_state_hash"])
    b = PayloadDescriptor("refinement", "actor", 4, 1, 0, 0, "segment-b",
                          second["decoded_state_hash"], ("base",), second["parent_state_hash"])
    decoder = ProgressiveDecoder()
    decoder.decode_payload(a, base_path)
    result = decoder.decode_payload(b, target_path).state
    np.testing.assert_array_equal(result.ids, [6, 2, 0, 5])
    assert result.sh_degree == 1
    assert state_hash(result) == state_hash(codec.decode(target_path, parent=base_decoded))


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "output/build/content_preparation/draco_byte_codec"


@pytest.mark.skipif(not BRIDGE.is_file(), reason="Existing Draco native bridge is unavailable")
def test_draco_encoded_buffers_match_file_reference(tmp_path):
    config = {"name": "gaussian_attribute_draco_byteplanes", "precision": "f32"}
    from tools.content_preparation.codec_adapter import get_codec
    codec, parent = get_codec(config), None
    decoder = ProgressiveDecoder(runtime_config=config)
    previous_id = None
    for quality, count in enumerate((3, 6)):
        target = synthetic_state(8, count)
        target.metadata["quality"] = f"Q{quality}"
        path = tmp_path / f"Q{quality}.cpgs"
        record = codec.encode(target, path, config, parent=parent)
        descriptor = PayloadDescriptor(
            f"actor:{quality}", "actor", 8, quality, 0, 0, "segment",
            record["decoded_state_hash"], () if previous_id is None else (previous_id,),
            record["parent_state_hash"], codec_id=codec.name + "/" + codec.version)
        parent = codec.decode(path, parent=parent)
        actual = decoder.decode_payload(descriptor, io.BytesIO(path.read_bytes()))
        assert state_hash(actual.state) == state_hash(parent)
        previous_id = descriptor.payload_id
