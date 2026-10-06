"""Deterministic conventional Gaussian attribute codec, explicitly not LTS.

Full states and sparse replacements use the same target quantizer. Refinements
are derived from the *decoded* parent and cover every attribute, additions,
deletions and row ordering. This module never reads a training checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from .assets import ATTRIBUTES, STATE_VERSION, GaussianState, canonical_json, file_hash, state_hash

MAGIC = b"CPGAUS01"
CODEC_NAME = "gaussian_attribute_zlib"
CODEC_VERSION = "1.0.0"


def _sha(data) -> str:
    return hashlib.sha256(data).hexdigest()


def _precision(value):
    value = str(value).lower()
    value = {"float32": "f32", "float16": "f16"}.get(value, value)
    if value not in ("f32", "f16"):
        if not value.startswith("q") or not value[1:].isdigit() or not 8 <= int(value[1:]) <= 16:
            raise ValueError(f"Unsupported precision {value}; use f32, f16, or q8..q16")
    return value


def _normalize_config(config):
    config = dict(config or {})
    supported = {"name", "version", "precision", "attribute_precision", "compression", "level"}
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"Unknown codec configuration fields: {sorted(unknown)}")
    if config.get("name", CODEC_NAME) not in (CODEC_NAME, "baseline", "attribute_zlib"):
        raise ValueError("Exact LTS enhanced-Draco codec is unavailable; select gaussian_attribute_zlib")
    if config.get("version", CODEC_VERSION) != CODEC_VERSION:
        raise ValueError(f"Unsupported codec version {config['version']}")
    compression = config.get("compression", "zlib")
    if compression not in ("zlib", "none", "zstd"):
        raise ValueError("compression must be zlib, none or zstd")
    level = int(config.get("level", 6))
    if compression == "zlib" and not 0 <= level <= 9:
        raise ValueError("zlib compression level must be in 0..9")
    if compression == "zstd" and not 1 <= level <= 22:
        raise ValueError("zstd compression level must be in 1..22")
    attributes = config.get("attribute_precision", {})
    if set(attributes) - set(ATTRIBUTES):
        raise ValueError("attribute_precision contains unknown Gaussian attributes")
    return {"name": CODEC_NAME, "version": CODEC_VERSION,
            "precision": _precision(config.get("precision", "f32")),
            "attribute_precision": {k: _precision(v) for k, v in sorted(attributes.items())},
            "compression": compression, "level": level}


def _quantize(values, precision):
    values = np.ascontiguousarray(values, dtype="<f4")
    if precision in ("f32", "f16"):
        encoded = values.astype("<f4" if precision == "f32" else "<f2")
        if not np.isfinite(encoded).all():
            raise ValueError(f"{precision} overflow; choose f32 or a calibrated q precision")
        return encoded, {"precision": precision}, encoded.astype("<f4")
    bits = int(precision[1:])
    maximum = float(np.abs(values.astype(np.float64)).max()) if values.size else 0.0
    levels = (1 << (bits - 1)) - 1
    step = maximum / levels if maximum else 1.0
    encoded = np.rint(values.astype(np.float64) / step).astype("<i4")
    if encoded.size and int(np.abs(encoded).max()) > levels:
        raise ValueError("Quantizer overflow; no silent clipping is permitted")
    decoded = (encoded.astype(np.float64) * step).astype("<f4")
    return encoded, {"precision": precision, "bits": bits, "step": step,
                     "calibration": "per_attribute_full_target_symmetric_max_abs",
                     "rounding": "nearest_even"}, decoded


def _pack(values, meta):
    if meta["precision"] in ("f32", "f16"):
        return values.tobytes(order="C")
    bits = meta["bits"]
    unsigned = (values.reshape(-1).astype(np.int64) & ((1 << bits) - 1)).astype(np.uint16)
    if bits == 8:
        return unsigned.astype(np.uint8).tobytes()
    if bits == 16:
        return unsigned.astype("<u2").tobytes()
    planes = ((unsigned[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def _unpack(data, shape, meta):
    count = int(np.prod(shape, dtype=np.int64))
    precision = meta["precision"]
    if precision in ("f32", "f16"):
        dtype = "<f4" if precision == "f32" else "<f2"
        if len(data) != count * np.dtype(dtype).itemsize:
            raise ValueError("Float attribute byte count mismatch")
        return np.frombuffer(data, dtype=dtype).reshape(shape).astype("<f4")
    bits = int(meta["bits"])
    if not 8 <= bits <= 16 or len(data) != (count * bits + 7) // 8:
        raise ValueError("Packed integer attribute byte count mismatch")
    if bits in (8, 16):
        unsigned = np.frombuffer(data, dtype="u1" if bits == 8 else "<u2").astype(np.int32)
    else:
        planes = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:count * bits]
        unsigned = np.sum(planes.reshape(count, bits).astype(np.int32) *
                          (1 << np.arange(bits, dtype=np.int32)), axis=1)
    signed = np.where(unsigned & (1 << (bits - 1)), unsigned - (1 << bits), unsigned)
    return (signed.astype(np.float64) * float(meta["step"])).astype("<f4").reshape(shape)


def _compress(data, config):
    method = config["compression"]
    if method == "none":
        return data, {"name": "none", "version": "1"}
    if method == "zlib":
        return zlib.compress(data, config["level"]), {"name": "zlib", "version": zlib.ZLIB_VERSION}
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("zstd requested; install zstandard or configure compression: zlib") from error
    return zstandard.ZstdCompressor(level=config["level"], threads=0,
                                   write_checksum=True).compress(data), {
        "name": "zstd", "version": zstandard.__version__}


def _decompress(data, method, expected_bytes):
    if method == "none":
        raw = data
    elif method == "zlib":
        decoder = zlib.decompressobj()
        raw = decoder.decompress(data, expected_bytes + 1)
        if decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
            raise ValueError("Truncated, trailing or oversized compressed payload")
    elif method == "zstd":
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError("Decoding zstd payload requires the zstandard package") from error
        raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=max(1, expected_bytes))
    else:
        raise ValueError(f"Unknown compression method {method}")
    if len(raw) != expected_bytes:
        raise ValueError("Decompressed payload byte count mismatch")
    return raw


def _read_packet(path):
    blob = Path(path).read_bytes()
    if len(blob) < 16 + 32 or blob[:8] != MAGIC:
        raise ValueError("Not a content-preparation Gaussian packet")
    if hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
        raise ValueError("Gaussian packet checksum mismatch")
    header_bytes = struct.unpack("<Q", blob[8:16])[0]
    if header_bytes > len(blob) - 48:
        raise ValueError("Truncated Gaussian packet header")
    header = json.loads(blob[16:16 + header_bytes].decode("utf-8"))
    if header.get("codec", {}).get("version") != CODEC_VERSION or header.get("format") != "cp-gaussian-packet-v1":
        raise ValueError("Unsupported Gaussian packet version")
    payload = blob[16 + header_bytes:-32]
    if _sha(payload) != header["compressed_sha256"]:
        raise ValueError("Compressed stream checksum mismatch")
    raw = _decompress(payload, header["compressor"]["name"], int(header["raw_bytes"]))
    if _sha(raw) != header["raw_sha256"]:
        raise ValueError("Attribute stream checksum mismatch")
    return header, raw


def inspect_payload(path) -> dict:
    """Return the validated header without constructing a Gaussian model."""
    with Path(path).open("rb") as source:
        magic = source.read(8)
    if magic == b"CPDRAC01":
        from .draco_adapter import inspect_draco_payload
        return inspect_draco_payload(path)
    header, _ = _read_packet(path)
    return header


def _matching_ids(target, parent):
    sort = np.argsort(parent.ids)
    sorted_ids = parent.ids[sort]
    positions = np.searchsorted(sorted_ids, target.ids)
    common = positions < len(sorted_ids)
    if len(sorted_ids):
        common &= sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == target.ids
    indices = np.full(target.count, -1, dtype=np.int64)
    indices[common] = sort[positions[common]]
    return common, indices


def _stable(state):
    return bool(state.metadata.get("stable_ids") or state.metadata.get("lineage_verified"))


class CodecAdapter:
    """Baseline facade; another codec can implement this encode/decode interface."""
    name = CODEC_NAME
    version = CODEC_VERSION
    packet_magic = MAGIC

    def _normalize(self, config):
        return _normalize_config(config)

    def _compress_stream(self, raw, config, descriptors):
        return _compress(raw, config)

    def _read(self, path):
        return _read_packet(path)

    def encode(self, input_state: GaussianState, path, config=None,
               parent: GaussianState | None = None) -> dict:
        input_state.validate()
        config = self._normalize(config)
        if parent is not None:
            parent.validate()
            if not _stable(input_state) or not _stable(parent):
                raise ValueError("Progressive encoding requires explicit stable IDs or verified checkpoint lineage")
        target = input_state
        blocks, descriptors, offset = [], {}, 0

        def emit(name, raw, **descriptor):
            nonlocal offset
            descriptors[name] = {"offset": offset, "bytes": len(raw), "sha256": _sha(raw), **descriptor}
            blocks.append(raw)
            offset += len(raw)

        emit("ids", target.ids.tobytes(), dtype="<i8", shape=[target.count])
        common, parent_indices = (_matching_ids(target, parent) if parent is not None else
                                  (np.zeros(target.count, dtype=bool), np.full(target.count, -1)))
        new_ids = target.ids[~common]
        deleted_ids = np.setdiff1d(parent.ids, target.ids) if parent is not None else np.empty(0, dtype="<i8")
        emit("new_ids", new_ids.astype("<i8").tobytes(), dtype="<i8", shape=[len(new_ids)])
        emit("deleted_ids", deleted_ids.astype("<i8").tobytes(), dtype="<i8", shape=[len(deleted_ids)])
        decoded_arrays, changed_rows = {}, {}
        for attribute in ATTRIBUTES:
            precision = config["attribute_precision"].get(attribute, config["precision"])
            encoded, quantizer, decoded = _quantize(target.arrays[attribute], precision)
            decoded_arrays[attribute] = decoded
            changes = np.ones(target.count, dtype=bool)
            if parent is not None and parent.arrays[attribute].shape[1:] == decoded.shape[1:]:
                width = int(np.prod(decoded.shape[1:]))
                a = decoded[common].reshape(int(common.sum()), width).view(np.uint8)
                b = parent.arrays[attribute][parent_indices[common]].reshape(int(common.sum()), width).view(np.uint8)
                changes[common] = np.any(a != b, axis=1)
            indices = np.flatnonzero(changes).astype("<i8")
            changed_rows[attribute] = len(indices)
            emit(attribute + ".indices", indices.tobytes(), dtype="<i8", shape=[len(indices)])
            emit(attribute, _pack(encoded[indices], quantizer), shape=list(encoded[indices].shape),
                 target_shape=list(decoded.shape), quantizer=quantizer)
        decoded = GaussianState(decoded_arrays, dict(target.metadata), target.ids.copy())
        raw = b"".join(blocks)
        compressed, compressor = self._compress_stream(raw, config, descriptors)
        header = {
            "format": "cp-gaussian-packet-v1", "codec": {"name": self.name, "version": self.version, "config": config},
            "state_version": STATE_VERSION, "kind": "refinement" if parent is not None else "full",
            "self_contained": parent is None, "parent_state_hash": state_hash(parent) if parent is not None else None,
            "input_state_hash": state_hash(target), "decoded_state_hash": state_hash(decoded),
            "metadata": target.metadata, "arrays": descriptors, "changed_rows": changed_rows,
            "new_ids": len(new_ids), "deleted_ids": len(deleted_ids),
            "raw_bytes": len(raw), "raw_sha256": _sha(raw),
            "compressed_bytes": len(compressed), "compressed_sha256": _sha(compressed), "compressor": compressor,
            "refinement_method": "sparse_absolute_replacements_against_decoded_parent",
            "temporal_prediction": False,
        }
        header_data = canonical_json(header)
        packet = self.packet_magic + struct.pack("<Q", len(header_data)) + header_data + compressed
        packet += hashlib.sha256(packet).digest()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(packet)
        temporary.replace(path)
        return {"path": str(path), "bytes": len(packet), "payload_bytes": len(packet),
                "sha256": file_hash(path), "codec": header["codec"], "compressor": compressor,
                "decoded_state_hash": header["decoded_state_hash"], "input_state_hash": header["input_state_hash"],
                "parent_state_hash": header["parent_state_hash"], "self_contained": parent is None,
                "changed_rows": changed_rows, "new_ids": len(new_ids), "deleted_ids": len(deleted_ids),
                "decoded_state_version": STATE_VERSION}

    def decode(self, path, parent: GaussianState | None = None) -> GaussianState:
        header, raw = self._read(path)
        if header["kind"] == "refinement":
            if parent is None:
                raise ValueError("Refinement packet requires its decoded parent state")
            if state_hash(parent) != header["parent_state_hash"]:
                raise ValueError("Refinement decoded-parent state hash mismatch")
        elif header["kind"] != "full":
            raise ValueError(f"Unknown packet kind {header['kind']}")
        else:
            parent = None

        def read(name):
            descriptor = header["arrays"][name]
            start, length = int(descriptor["offset"]), int(descriptor["bytes"])
            if start < 0 or length < 0 or start + length > len(raw):
                raise ValueError("Attribute descriptor lies outside the stream")
            data = raw[start:start + length]
            if _sha(data) != descriptor["sha256"]:
                raise ValueError(f"Attribute checksum mismatch: {name}")
            return data, descriptor

        def indices(name):
            data, descriptor = read(name)
            if descriptor.get("dtype") != "<i8" or descriptor.get("shape") != [len(data) // 8] or len(data) % 8:
                raise ValueError("Malformed ID/index stream")
            return np.frombuffer(data, dtype="<i8").copy()

        ids = indices("ids")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate target Gaussian IDs")
        # Identity and byte accounting are transmitted, not inferred side data.
        new_ids, deleted_ids = indices("new_ids"), indices("deleted_ids")
        if parent is not None:
            skeleton = type("IdState", (), {"ids": ids, "count": len(ids)})()
            common, parent_indices = _matching_ids(skeleton, parent)
            if not np.array_equal(new_ids, ids[~common]) or not np.array_equal(deleted_ids, np.setdiff1d(parent.ids, ids)):
                raise ValueError("Invalid new/deleted Gaussian ID accounting")
        else:
            common, parent_indices = np.zeros(len(ids), dtype=bool), np.full(len(ids), -1)
            if not np.array_equal(new_ids, ids) or len(deleted_ids):
                raise ValueError("Invalid full-state identity accounting")
        arrays = {}
        for attribute in ATTRIBUTES:
            selected = indices(attribute + ".indices")
            data, descriptor = read(attribute)
            shape = tuple(descriptor["target_shape"])
            if not shape or shape[0] != len(ids) or any(int(x) < 0 for x in shape):
                raise ValueError("Malformed target attribute shape")
            if len(np.unique(selected)) != len(selected) or (selected < 0).any() or (selected >= len(ids)).any():
                raise ValueError("Malformed sparse replacement indices")
            values = _unpack(data, tuple(descriptor["shape"]), descriptor["quantizer"])
            if values.shape != (len(selected),) + shape[1:]:
                raise ValueError("Sparse attribute shape mismatch")
            covered = np.zeros(len(ids), dtype=bool)
            covered[selected] = True
            can_inherit = parent is not None and parent.arrays[attribute].shape[1:] == shape[1:]
            inherited = common if can_inherit else np.zeros(len(ids), dtype=bool)
            if (~covered & ~inherited).any():
                raise ValueError("New Gaussian/SH schema requires complete attribute replacement")
            result = np.empty(shape, dtype="<f4")
            if can_inherit:
                result[common] = parent.arrays[attribute][parent_indices[common]]
            result[selected] = values
            arrays[attribute] = result
        state = GaussianState(arrays, header["metadata"], ids)
        if state_hash(state) != header["decoded_state_hash"]:
            raise ValueError("Decoded Gaussian state checksum mismatch")
        return state


class BaselineCodecAdapter(CodecAdapter):
    """Explicit name for the current conventional adapter implementation."""


def get_codec(config=None) -> CodecAdapter:
    if (config or {}).get("name") == "gaussian_attribute_draco_byteplanes":
        from .draco_adapter import DracoCodecAdapter
        adapter = DracoCodecAdapter(config)
        return adapter
    _normalize_config(config)
    return BaselineCodecAdapter()


def get_decoder(codec_metadata, runtime_config=None) -> CodecAdapter:
    """Decode the packet's codec using an explicitly compatible local runtime.

    Coding parameters and codec/schema versions belong to the encoded payload.
    A Draco executable path belongs to the machine performing the decode; its
    historical encoder path can be replaced without changing the bitstream.
    A runtime configured for another codec does not override packet metadata.
    """
    if not isinstance(codec_metadata, dict) or not isinstance(codec_metadata.get("config"), dict):
        raise ValueError("Packet decoder requires declared codec metadata and config")
    declared = dict(codec_metadata["config"])
    decoder = get_codec(declared)
    if codec_metadata.get("name") != decoder.name or codec_metadata.get("version") != decoder.version:
        raise ValueError("Packet codec name/version differs from its declared configuration")
    if decoder.name == "gaussian_attribute_draco_byteplanes" and runtime_config:
        runtime = get_codec(runtime_config)
        if runtime.name == decoder.name:
            # Validate the runtime, but retain the packet's encoding parameters.
            for field in ("executable", "decoder_speed"):
                declared[field] = runtime.runtime_config[field]
            decoder = get_codec(declared)
    return decoder
