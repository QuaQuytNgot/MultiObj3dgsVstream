"""Lossless project Gaussian packets over public Draco byte attributes.

This adapter reproduces no unpublished LTS encoder. All identities and numeric
bits are delivered in the packet, including shared-state replacements. The
native bridge uses sequential DT_UINT8 attributes with <=64 components and no
prediction, quantization, or deduplication, avoiding unbounded float-bit symbols.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from .assets import ATTRIBUTES, STATE_VERSION, file_hash
from .codec_adapter import CodecAdapter, _sha

MAGIC = b"CPDRAC01"
CODEC_NAME = "gaussian_attribute_draco_byteplanes"
CODEC_VERSION = "1.0.0"
BRIDGE_VERSION = "1.0.0"
MAX_STREAM_BYTES = 2 * 1024 ** 3


def normalize_config(config=None):
    config = dict(config or {})
    supported = {"name", "version", "precision", "attribute_precision", "executable",
                 "encoder_speed", "decoder_speed", "compression", "level"}
    if set(config) - supported:
        raise ValueError(f"Unknown Draco codec fields: {sorted(set(config) - supported)}")
    if config.get("name", CODEC_NAME) != CODEC_NAME or config.get("version", CODEC_VERSION) != CODEC_VERSION:
        raise ValueError("Unsupported project Draco codec name/version")
    if config.get("precision", "f32") not in ("f32", "float32"):
        raise ValueError("Draco byte-attribute codec is lossless f32 only; no silent quantization")
    if config.get("attribute_precision"):
        raise ValueError("Draco f32 codec does not accept per-attribute quantization")
    # Legacy config defaults are injected by schema v1. They are not a second
    # compression layer and are deliberately excluded from normalized metadata.
    if config.get("compression", "zlib") != "zlib" or config.get("level", 6) != 6:
        raise ValueError("Draco compression is selected by its adapter; remove baseline compression/level overrides")
    speeds = {}
    for field in ("encoder_speed", "decoder_speed"):
        value = config.get(field, 3)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise ValueError(f"{field} must be an integer in 0..10")
        speeds[field] = value
    executable = str(config.get("executable") or "output/build/content_preparation/draco_byte_codec")
    return {"name": CODEC_NAME, "version": CODEC_VERSION, "precision": "f32",
            "attribute_precision": {}, "executable": executable, **speeds}


def executable_path(config):
    path = Path(config["executable"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Draco native bridge missing: {path}. Build tools/content_preparation/native with CMake first.")
    return path


@lru_cache(maxsize=4)
def _bridge_info(path, binary_hash):
    try:
        result = subprocess.run([path, "--version"], check=True, capture_output=True,
                                text=True, timeout=30)
        info = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot identify Draco native bridge: {path}: {error}") from error
    required = {"bridge": "draco_byte_codec", "bridge_version": BRIDGE_VERSION,
                "method": "sequential", "prediction": "none", "attribute_type": "DT_UINT8",
                "maximum_components": 64, "deduplication": False, "quantization": False}
    if any(info.get(k) != v for k, v in required.items()):
        raise RuntimeError("Unsupported native bridge version or coding policy")
    if not isinstance(info.get("draco_commit"), str) or len(info["draco_commit"]) != 40:
        raise RuntimeError("Native bridge must identify its pinned Draco commit")
    return {**info, "binary_sha256": binary_hash}


def runtime_info(config=None):
    config = normalize_config(config)
    path = executable_path(config)
    return dict(_bridge_info(str(path), file_hash(path)))


def _bridge(operation, data, rows, row_bytes, config):
    path = executable_path(config)
    with tempfile.TemporaryDirectory(prefix="cp-draco-") as temporary:
        source = Path(temporary) / "input.bin"
        destination = Path(temporary) / "output.bin"
        source.write_bytes(data)
        argv = [str(path), operation, "--input", str(source), "--output", str(destination),
                "--rows", str(rows), "--row-bytes", str(row_bytes)]
        if operation == "encode":
            argv += ["--encoder-speed", str(config["encoder_speed"]),
                     "--decoder-speed", str(config["decoder_speed"])]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(f"Draco native {operation} failed: {error}") from error
        if result.returncode:
            raise ValueError(f"Draco native {operation} rejected stream: {result.stderr.strip()}")
        if not destination.is_file() or destination.stat().st_size > MAX_STREAM_BYTES:
            raise ValueError("Draco native output missing or exceeds stream size limit")
        return destination.read_bytes()


def _descriptor_layout(descriptors, raw_bytes):
    streams, offset = [], 0
    expected_names = {"ids", "new_ids", "deleted_ids"} | set(ATTRIBUTES) | {a + ".indices" for a in ATTRIBUTES}
    if set(descriptors) != expected_names:
        raise ValueError("Wrong Draco Gaussian attribute stream set")
    # Empty streams may share an offset with a nonempty stream. Canonical JSON
    # sorts field names, so order ties by size/name instead of insertion order.
    for name, descriptor in sorted(descriptors.items(),
                                  key=lambda pair: (int(pair[1]["offset"]), int(pair[1]["bytes"]), pair[0])):
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or not shape or any(type(v) is not int or v < 0 for v in shape):
            raise ValueError("Invalid Draco stream dimensions")
        rows = shape[0]
        width = 1
        for dimension in shape[1:]:
            width *= dimension
        itemsize = 8 if name.endswith(".indices") or name in ("ids", "new_ids", "deleted_ids") else 4
        row_bytes = width * itemsize
        size = int(descriptor["bytes"])
        if int(descriptor["offset"]) != offset or size != rows * row_bytes or size > MAX_STREAM_BYTES or not 0 < row_bytes <= 4096 or rows > (2**31 - 1) // 64:
            raise ValueError("Invalid, overlapping or oversized Draco stream layout")
        if itemsize == 4 and descriptor.get("quantizer") != {"precision": "f32"}:
            raise ValueError("Draco Gaussian attributes require lossless f32")
        streams.append((name, offset, size, rows, row_bytes))
        offset += size
    if offset != raw_bytes:
        raise ValueError("Draco raw stream byte count mismatch")
    return streams


def _read_draco_packet(path):
    blob = Path(path).read_bytes()
    if len(blob) < 48 or blob[:8] != MAGIC:
        raise ValueError("Not a project Draco Gaussian packet")
    if hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
        raise ValueError("Draco Gaussian packet checksum mismatch")
    header_bytes = struct.unpack("<Q", blob[8:16])[0]
    if header_bytes > len(blob) - 48:
        raise ValueError("Truncated Draco packet header")
    header = json.loads(blob[16:16 + header_bytes].decode("utf-8"))
    if header.get("format") != "cp-gaussian-packet-v1" or header.get("codec", {}).get("name") != CODEC_NAME or header["codec"].get("version") != CODEC_VERSION or header.get("state_version") != STATE_VERSION:
        raise ValueError("Unsupported Draco packet schema/version")
    config = normalize_config(header["codec"]["config"])
    payload = blob[16 + header_bytes:-32]
    if header.get("compressed_bytes") != len(payload) or _sha(payload) != header.get("compressed_sha256"):
        raise ValueError("Draco encoded stream checksum mismatch")
    if type(header.get("raw_bytes")) is not int or header["raw_bytes"] < 0:
        raise ValueError("Invalid Draco raw byte count")
    layouts = _descriptor_layout(header["arrays"], header["raw_bytes"])
    compressor = header.get("compressor", {})
    if compressor.get("name") != "draco_byte_attributes" or compressor.get("version") != CODEC_VERSION:
        raise ValueError("Unsupported Draco compressor")
    records = compressor.get("streams", {})
    if set(records) != {layout[0] for layout in layouts}:
        raise ValueError("Draco compressed stream set mismatch")
    cursor = 0
    for name, _, size, rows, row_bytes in layouts:
        record = records[name]
        if type(record.get("bytes")) is not int or record["bytes"] < 0 or record.get("offset") != cursor or record.get("rows") != rows or record.get("row_bytes") != row_bytes:
            raise ValueError("Malformed Draco encoded stream layout")
        length = record["bytes"]
        if cursor + length > len(payload) or (size == 0) != (length == 0) or _sha(payload[cursor:cursor + length]) != record.get("sha256"):
            raise ValueError("Draco compressed attribute checksum mismatch")
        cursor += length
    if cursor != len(payload):
        raise ValueError("Trailing Draco stream bytes")
    return header, payload, layouts, config


def inspect_draco_payload(path):
    """Verify container and all encoded attribute hashes without native decode."""
    return _read_draco_packet(path)[0]


class DracoCodecAdapter(CodecAdapter):
    name = CODEC_NAME
    version = CODEC_VERSION
    packet_magic = MAGIC

    def __init__(self, config=None):
        # Decoder execution belongs to the local runtime, not the payload's
        # historical executable path. get_codec(config) selects that runtime.
        self.runtime_config = normalize_config(config) if config is not None else None

    def _normalize(self, config):
        return normalize_config(config)

    def encode(self, input_state, path, config=None, parent=None):
        if parent is not None:
            target_frame = input_state.metadata.get("frame")
            parent_frame = parent.metadata.get("frame")
            if target_frame is not None and parent_frame is not None and target_frame != parent_frame:
                raise ValueError("Draco progressive refinement requires a same-frame decoded parent; no temporal prediction")
        return super().encode(input_state, path, config, parent)

    def decode(self, path, parent=None):
        header = inspect_draco_payload(path)
        if header.get("kind") == "refinement" and parent is not None:
            target_frame = header.get("metadata", {}).get("frame")
            parent_frame = parent.metadata.get("frame")
            if target_frame is not None and parent_frame is not None and target_frame != parent_frame:
                raise ValueError("Draco progressive refinement requires a same-frame decoded parent; no temporal prediction")
        return super().decode(path, parent)

    def _compress_stream(self, raw, config, descriptors):
        info = runtime_info(config)
        blocks, records, cursor = [], {}, 0
        for name, start, size, rows, row_bytes in _descriptor_layout(descriptors, len(raw)):
            compressed = _bridge("encode", raw[start:start + size], rows, row_bytes, config) if size else b""
            records[name] = {"offset": cursor, "bytes": len(compressed), "sha256": _sha(compressed),
                             "rows": rows, "row_bytes": row_bytes}
            blocks.append(compressed)
            cursor += len(compressed)
        return b"".join(blocks), {"name": "draco_byte_attributes", "version": CODEC_VERSION,
                                   "runtime": info, "streams": records}

    def _read(self, path):
        header, payload, layouts, config = _read_draco_packet(path)
        # Decode compatibility is version/policy-based. Encoding binary hashes
        # remain provenance; a rebuilt compatible decoder need not be identical.
        decoder_config = self.runtime_config or config
        runtime_info(decoder_config)
        blocks = []
        for name, _, size, rows, row_bytes in layouts:
            record = header["compressor"]["streams"][name]
            data = payload[record["offset"]:record["offset"] + record["bytes"]]
            decoded = _bridge("decode", data, rows, row_bytes, decoder_config) if size else b""
            if len(decoded) != size or _sha(decoded) != header["arrays"][name]["sha256"]:
                raise ValueError(f"Draco decoded attribute checksum mismatch: {name}")
            blocks.append(decoded)
        raw = b"".join(blocks)
        if _sha(raw) != header["raw_sha256"]:
            raise ValueError("Draco decoded stream checksum mismatch")
        return header, raw
