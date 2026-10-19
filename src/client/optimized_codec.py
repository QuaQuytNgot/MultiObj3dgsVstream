"""Opt-in, bit-exact decoding of existing CPDRAC01 packets.

Strict mode checks every encoded/raw checksum and reconstructed-state digest.
Runtime mode reuses trusted ingestion proofs and immutable validation; see
docs/RUNTIME_VERIFICATION_AND_WORKERS.md for its distinct integrity boundary.
Baseline codecs are intact.
"""
from collections import Counter, OrderedDict
import ctypes
import hashlib
import threading
import re
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np

from tools.content_preparation.assets import ATTRIBUTES, STATE_VERSION, GaussianState, canonical_json, state_hash
from tools.content_preparation.codec_adapter import _matching_ids
from tools.content_preparation.draco_adapter import _read_draco_packet, _bridge, normalize_config, runtime_info

OPTIMIZATION_MODES = {
    "baseline": None,
    "verified": dict(native="subprocess", stream_cache_bytes=0, array_cache_bytes=0),
    "streams": dict(native="subprocess", array_cache_bytes=0),
    "cached": dict(native="subprocess"),
    "native": dict(native="memory", stream_cache_bytes=0, array_cache_bytes=0),
    "native-cached": dict(native="memory"),
}


class ByteCache:
    def __init__(self, budget):
        if type(budget) is not int or budget < 0:
            raise ValueError("Cache budget must be a nonnegative integer")
        self.budget, self.size = budget, 0
        self.items = OrderedDict()
        self.lock = threading.RLock()
        # In-flight results are bounded separately from retained cache payload.
        # At most one factory runs per content key; unrelated keys never wait.
        self._flights = {}
        self._generation = 0
        self.max_flights = 32

    def get(self, key):
        with self.lock:
            entry = self.items.get(key)
            if entry is not None:
                self.items.move_to_end(key)
                return entry[0]

    def put(self, key, value, size):
        with self.lock:
            self._put(key, value, size)

    def _put(self, key, value, size):
        if size > self.budget or not self.budget:
            return
        old = self.items.pop(key, None)
        if old:
            self.size -= old[1]
        while self.items and (self.size + size > self.budget or len(self.items) >= 4096):
            self.size -= self.items.popitem(last=False)[1][1]
        self.items[key] = value, size
        self.size += size

    def get_or_compute(self, key, factory, publish):
        """Return a cached/shared value, publishing only a successful factory.

        The producer must verify its result before returning. Errors reach all
        current waiters and are never cached; a later request may retry. A clear
        during computation prevents that old producer repopulating this cache.
        Factories are never called under the cache lock. The bounded fallback
        performs independent work when the in-flight table is full.
        """
        with self.lock:
            generation = self._generation
            value = self.get(key)
            if value is not None:
                return value, "hit"
            flight = self._flights.get(key)
            if flight is not None:
                if flight["owner"] == threading.get_ident():
                    raise RuntimeError("Recursive computation for the same cache key")
                flight["waiters"] += 1
                producer = False
            elif self.budget and len(self._flights) < self.max_flights:
                flight = {"event": threading.Event(), "owner": threading.get_ident(),
                          "generation": self._generation, "waiters": 0}
                self._flights[key] = flight
                producer = True
            else:
                flight, producer = None, True
        if not producer:
            flight["event"].wait()
            if "error" in flight:
                raise flight["error"]
            return flight["value"], "shared"
        try:
            value = factory()
            with self.lock:
                if generation == self._generation:
                    publish(value)
                if flight is not None:
                    flight["value"] = value
            return value, "miss"
        except BaseException as error:
            if flight is not None:
                with self.lock:
                    flight["error"] = error
            raise
        finally:
            if flight is not None:
                with self.lock:
                    self._flights.pop(key, None)
                    flight["event"].set()

    def clear(self):
        with self.lock:
            self.items.clear()
            self.size = 0
            self._generation += 1


class SharedCodecCache:
    """Pool-owned immutable values, bounded once for the entire thread pool."""
    def __init__(self, stream_bytes=128*1024**2, array_bytes=128*1024**2):
        self.streams = ByteCache(stream_bytes)
        self.arrays = ByteCache(array_bytes)
        self.validation = ByteCache(1024*1024)
        self.history = ByteCache(512*1024)
        self.idmaps = ByteCache(16*1024**2)

    def clear(self):
        for cache in (self.streams, self.arrays, self.validation, self.history, self.idmaps):
            cache.clear()


def _immutable(array):
    root = array
    while isinstance(root, np.ndarray):
        root = root.base
    if isinstance(root, bytes) and array.flags.c_contiguous:
        return array
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _layout(array):
    # Read-only arrays still allow changing strides, dtype and buffer metadata.
    # Bind the verified digest to its exact immutable buffer interpretation.
    return (array.shape, array.dtype.str, array.strides,
            array.__array_interface__["data"], array.flags.c_contiguous)


def _unique(values):
    # Strictly ascending IDs/index streams are common; verify, never assume.
    return len(values) < 2 or bool(np.all(values[1:] > values[:-1])) or len(np.unique(values)) == len(values)


class ImmutableGaussianState(GaussianState):
    def __init__(self, arrays, metadata, ids, keys, *, expected_digest=None, validation=None):
        self._sealed = False
        self._validation = validation
        self._validation_keys = keys
        self.canonical_digest_computed = expected_digest is None
        super().__init__({k: _immutable(v) for k, v in arrays.items()}, metadata, _immutable(ids))
        self.arrays = MappingProxyType(self.arrays)
        self.array_keys = MappingProxyType(keys)
        owners = []
        for array in (self.ids, *self.arrays.values()):
            owner = array
            while isinstance(owner, np.ndarray):
                owner = owner.base
            owners.append(owner)
        # Keep original buffers alive even if a caller replaces ndarray metadata.
        self._owners = tuple(owners)
        self._layouts = tuple(_layout(a) for a in (self.ids, *self.arrays.values()))
        if expected_digest is None:
            digest = hashlib.sha256(STATE_VERSION.encode("ascii"))
            for name, array in [("ids", self.ids), *self.arrays.items()]:
                digest.update(canonical_json([name, array.dtype.str, list(array.shape)]))
                if array.size:
                    digest.update(memoryview(array).cast("B"))
            self._digest = digest.hexdigest()
        else:
            # This is the publisher's declared canonical identity. Runtime does
            # not independently verify the final reconstruction's canonical SHA.
            self._digest = expected_digest
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Decoded Gaussian states are immutable")
        super().__setattr__(name, value)

    def validate(self):
        if getattr(self, "_sealed", False):
            if self._layouts != tuple(_layout(a) for a in (self.ids, *self.arrays.values())):
                raise ValueError("Immutable state array layout was modified")
            canonical_json(self.metadata)
            return self
        n = len(self.ids)
        for name, width in (("xyz", 3), ("rotation", 4), ("scale", 3), ("opacity", 1)):
            if self.arrays[name].shape != (n, width):
                raise ValueError(f"Malformed {name} shape")
        sh = self.arrays["sh"]
        if sh.ndim != 3 or sh.shape[:2] != (n, 3) or sh.shape[2] < 1 or int(np.sqrt(sh.shape[2]))**2 != sh.shape[2]:
            raise ValueError("Malformed SH shape")
        if self.ids.ndim != 1 or (self._validation is None and not _unique(self.ids)):
            raise ValueError("Gaussian IDs must be unique")
        for name, value in self.arrays.items():
            key = (name, self._validation_keys[name]) if self._validation is not None else None
            if self._validation is not None and self._validation.get(key):
                continue
            if not np.isfinite(value).all():
                raise ValueError("Gaussian state contains nonfinite values")
            if name == "rotation" and n and (np.linalg.norm(value.astype(np.float64), axis=1) == 0).any():
                raise ValueError("A raw wxyz quaternion may not be zero")
            if self._validation is not None:
                self._validation.put(key, True, 256)
        canonical_json(self.metadata)
        return self

    @property
    def state_identity(self):
        """Computed strict identity or publisher-declared runtime identity."""
        self.validate()
        return self._digest

    @property
    def verified_digest(self):
        self.validate()
        if not self.canonical_digest_computed:
            raise ValueError("Runtime state has a declared identity; use state_hash for an independent digest")
        return self._digest


class MemoryDraco:
    def __init__(self, library=None):
        path = Path(library) if library else Path(__file__).resolve().parents[2] / "output/build/content_preparation/libdraco_memory.so"
        self.library = ctypes.CDLL(str(path.resolve()))
        if self.library.cp_draco_abi() != 1:
            raise RuntimeError("Unsupported in-memory Draco ABI")
        self.call = self.library.cp_draco_decode
        self.call.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint,
                              ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
                              ctypes.c_void_p, ctypes.c_size_t]
        self.call.restype = ctypes.c_int

    def decode(self, data, rows, width, stats):
        start = perf_counter()
        output = ctypes.create_string_buffer(rows * width)
        stats["native_allocation_ms"] += (perf_counter()-start)*1000
        timing, error = (ctypes.c_double*2)(), ctypes.create_string_buffer(2048)
        start = perf_counter()
        if self.call(data, len(data), rows, width, output, rows*width, timing, error, len(error)):
            raise ValueError("Native Draco rejected stream: " + error.value.decode(errors="replace"))
        stats["native_call_ms"] += (perf_counter()-start)*1000
        stats["draco_algorithm_ms"] += timing[0]
        stats["native_extraction_ms"] += timing[1]
        start = perf_counter()
        result = output.raw
        stats["native_output_copy_ms"] += (perf_counter()-start)*1000
        return result


class OptimizedDraco:
    def __init__(self, runtime_config=None, *, native="subprocess", stream_cache_bytes=128*1024**2,
                 array_cache_bytes=128*1024**2, library=None, verification="strict", shared_cache=None,
                 reuse_attributes=True, repeat_stream_admission=False, singleflight=True,
                 reuse_index_classification=True):
        if native not in ("subprocess", "memory"):
            raise ValueError("native must be subprocess or memory")
        self.config = normalize_config(runtime_config)
        if verification not in ("strict", "runtime"):
            raise ValueError("verification must be strict or runtime")
        self.verification = verification
        self.reuse_attributes = verification == "runtime" and reuse_attributes
        self.repeat_stream_admission = repeat_stream_admission
        self.singleflight = singleflight
        self.reuse_index_classification = verification == "runtime" and reuse_index_classification
        self.native = MemoryDraco(library) if native == "memory" else None
        self.shared_cache = shared_cache
        self.streams = shared_cache.streams if shared_cache else ByteCache(stream_cache_bytes)
        self.arrays = shared_cache.arrays if shared_cache else ByteCache(array_cache_bytes)
        self.validation = shared_cache.validation if shared_cache else ByteCache(1024*1024)
        self.history = shared_cache.history if shared_cache else ByteCache(512*1024)
        self.idmaps = shared_cache.idmaps if shared_cache else ByteCache(16*1024**2)
        self.stats = Counter()
        self._runtime_checked = False

    def _retain(self, kind, cache, key, value, size):
        # Generic repeat admission prevents one-use dynamic streams/arrays from
        # evicting repeatedly used content. It assumes no static attribute names.
        history_key = kind, key
        # Index/ID streams often repeat within the same packet. Admit streams
        # immediately; repeat admission there caused redundant native decodes.
        if (kind == "stream" and not self.repeat_stream_admission) or not self.reuse_attributes or self.history.get(history_key):
            cache.put(key,value,size)
        else:
            self.history.put(history_key,True,256)

    def inspect(self, blob, *, externally_verified=False):
        start = perf_counter()
        if self.verification == "runtime" and not externally_verified:
            raise ValueError("Runtime verification requires the trusted MPD payload checksum")
        packet = _read_draco_packet(blob, verify_checksums=self.verification == "strict")
        if self.verification == "runtime":
            header = packet[0]
            digests = [header.get("decoded_state_hash"), header.get("raw_sha256"),
                       header.get("compressed_sha256")]
            digests += [d.get("sha256") for d in header["arrays"].values()]
            digests += [d.get("sha256") for d in header["compressor"]["streams"].values()]
            if any(not isinstance(d, str) or re.fullmatch("[0-9a-f]{64}", d) is None for d in digests):
                raise ValueError("Invalid integrity digest in runtime packet")
        self.stats["packet_verification_ms"] += (perf_counter()-start)*1000
        self.stats["packet_parses"] += 1
        return packet

    def decode(self, packet, parent):
        header, payload, layouts, _ = packet
        if header["kind"] == "refinement":
            if parent is not None:
                target_frame, parent_frame = header.get("metadata",{}).get("frame"), parent.metadata.get("frame")
                if target_frame is not None and parent_frame is not None and target_frame != parent_frame:
                    raise ValueError("Draco refinement requires a same-frame decoded parent")
            reusable = isinstance(parent, ImmutableGaussianState) and (self.verification == "runtime" or parent.canonical_digest_computed)
            digest = parent.state_identity if reusable else state_hash(parent) if parent is not None else None
            if digest != header["parent_state_hash"] or parent is None:
                raise ValueError("Refinement decoded-parent state hash mismatch")
        elif header["kind"] == "full":
            parent = None
        else:
            raise ValueError("Unknown packet kind")
        if not self._runtime_checked and self.native is None:
            runtime_info(self.config)
            self._runtime_checked = True
        layout_by_name = {layout[0]:layout for layout in layouts}
        def decode_stream(name):
            start = perf_counter()
            _, _, size, rows, width = layout_by_name[name]
            record = header["compressor"]["streams"][name]
            expected = header["arrays"][name]["sha256"]
            key = record["sha256"], rows, width, expected
            def compute():
                data = payload[record["offset"]:record["offset"]+record["bytes"]]
                if size:
                    self.stats["native_invocations"] += 1
                    decoded = self.native.decode(bytes(data), rows, width, self.stats) if self.native else _bridge("decode", bytes(data), rows, width, self.config)
                else:
                    decoded = b""
                if len(decoded) != size or hashlib.sha256(decoded).hexdigest() != expected:
                    raise ValueError(f"Draco decoded attribute checksum mismatch: {name}")
                return decoded
            def publish(decoded):
                self._retain("stream",self.streams,key,decoded,len(decoded))
            if self.singleflight:
                decoded, status = self.streams.get_or_compute(key, compute, publish)
            else:
                decoded = self.streams.get(key)
                status = "hit" if decoded is not None else "miss"
                if decoded is None:
                    decoded = compute()
                    publish(decoded)
            if status == "miss":
                self.stats["stream_misses"] += 1
            else:
                self.stats["stream_hits"] += 1
                if status == "shared":
                    self.stats["stream_singleflight_reuses"] += 1
                    self.stats["stream_singleflight_wait_ms"] += (perf_counter()-start)*1000
            self.stats["stream_decode_and_verify_ms"] += (perf_counter()-start)*1000
            return decoded
        class RawStreams(dict):
            def __missing__(self, name):
                self[name] = value = decode_stream(name)
                return value
        raw, raw_digest = RawStreams(), hashlib.sha256()
        stream_before = self.stats["stream_decode_and_verify_ms"]
        start = perf_counter()
        for name, *_ in layouts:
            if self.reuse_attributes and name in ATTRIBUTES:
                continue
            decoded = raw[name]
            if self.verification == "strict":
                raw_digest.update(decoded)
        if self.verification == "strict" and raw_digest.hexdigest() != header["raw_sha256"]:
            raise ValueError("Draco decoded stream checksum mismatch")
        self.stats["stream_decode_and_verify_ms"] += (perf_counter()-start)*1000-(self.stats["stream_decode_and_verify_ms"]-stream_before)
        return self.reconstruct(header, raw, parent)

    def reconstruct(self, header, raw, parent):
        start = perf_counter()
        stream_before = self.stats["stream_decode_and_verify_ms"]
        descriptors = header["arrays"]
        def indices(name):
            data, d = raw[name], descriptors[name]
            if d.get("dtype") != "<i8" or d.get("shape") != [len(data)//8] or len(data)%8:
                raise ValueError("Malformed ID/index stream")
            return np.frombuffer(data, dtype="<i8")
        ids = indices("ids")
        identity_key = ("ids", descriptors["ids"]["sha256"])
        if not (self.verification == "runtime" and self.validation.get(identity_key)) and not _unique(ids):
            raise ValueError("Duplicate target Gaussian IDs")
        self.validation.put(identity_key, True, 256)
        new, deleted = indices("new_ids"), indices("deleted_ids")
        if parent is not None:
            key = (descriptors["ids"]["sha256"],parent.array_keys["ids"],
                   descriptors["new_ids"]["sha256"],descriptors["deleted_ids"]["sha256"]) if self.reuse_attributes and isinstance(parent,ImmutableGaussianState) else None
            cached_map = self.idmaps.get(key) if key else None
            if cached_map is None:
                skeleton = type("IdState", (), {"ids": ids, "count": len(ids)})()
                common, mapping = _matching_ids(skeleton, parent)
                # Both inputs were verified unique. assume_unique avoids rechecking.
                if not np.array_equal(new, ids[~common]) or not np.array_equal(deleted, np.setdiff1d(parent.ids, ids, assume_unique=True)):
                    raise ValueError("Invalid new/deleted Gaussian ID accounting")
                if key:
                    common,mapping = _immutable(common),_immutable(mapping)
                    self.idmaps.put(key,(common,mapping),common.nbytes+mapping.nbytes)
            else:
                common,mapping = cached_map
                self.stats["id_mapping_hits"] += 1
        else:
            common, mapping = np.zeros(len(ids), dtype=bool), None
            if not np.array_equal(new, ids) or len(deleted):
                raise ValueError("Invalid full-state identity accounting")
        arrays, keys = {}, {}
        ids_hash = descriptors["ids"]["sha256"]
        for name in ATTRIBUTES:
            d = descriptors[name]
            selected = indices(name+".indices")
            shape = tuple(d["target_shape"])
            if not shape or shape[0] != len(ids) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError("Malformed target attribute shape")
            index_key = ("indices", descriptors[name+".indices"]["sha256"], len(ids))
            checked = self.verification == "runtime" and self.validation.get(index_key)
            if not checked and (not _unique(selected) or (selected < 0).any() or (selected >= len(ids)).any()):
                raise ValueError("Malformed sparse replacement indices")
            self.validation.put(index_key, True, 256)
            if tuple(d["shape"]) != (len(selected),)+shape[1:]:
                raise ValueError("Sparse attribute shape mismatch")
            inherit = parent is not None and parent.arrays[name].shape[1:] == shape[1:]
            classification_key = ("full_indices",descriptors[name+".indices"]["sha256"],len(ids))
            classification = self.validation.get(classification_key) if self.reuse_index_classification else None
            if classification is None:
                full = len(selected) == len(ids) and np.array_equal(selected,np.arange(len(ids)))
                if self.reuse_index_classification:
                    self.validation.put(classification_key,(full,),256)
            else:
                full = classification[0]
                self.stats["index_classification_hits"] += 1
            parent_key = None
            if inherit and not full:
                parent_key = (parent.array_keys[name], parent.array_keys["ids"]) if isinstance(parent, ImmutableGaussianState) else (state_hash(parent),)
            key = (name, shape, ids_hash, descriptors[name+".indices"]["sha256"], d["sha256"], parent_key)
            cached = self.arrays.get(key)
            if cached is not None and cached[1] != _layout(cached[0]):
                raise ValueError("Cached attribute array layout was modified")
            result = None if cached is None else cached[0]
            if result is None:
                values = np.frombuffer(raw[name], dtype="<f4").reshape(d["shape"])
                if full:
                    # Every row is explicitly supplied in verified ID order;
                    # no coverage mask or parent gather can add information.
                    result = values
                else:
                    covered = np.zeros(len(ids), dtype=bool)
                    covered[selected] = True
                    if (~covered & ~(common if inherit else np.zeros(len(ids), dtype=bool))).any():
                        raise ValueError("New Gaussian/SH schema requires complete attribute replacement")
                    result = np.empty(shape, dtype="<f4")
                    if inherit:
                        result[common] = parent.arrays[name][mapping[common]]
                    result[selected] = values
                    result = _immutable(result)
                self._retain("array",self.arrays,key,(result,_layout(result)),result.nbytes)
                self.stats["array_misses"] += 1
            else:
                self.stats["array_hits"] += 1
                self.stats["attribute_stream_reuses"] += int(self.reuse_attributes)
            arrays[name], keys[name] = result, key
        keys["ids"] = ids_hash
        self.stats["reconstruction_ms"] += (perf_counter()-start)*1000-(self.stats["stream_decode_and_verify_ms"]-stream_before)
        start = perf_counter()
        runtime = self.verification == "runtime"
        state = ImmutableGaussianState(arrays, header["metadata"], ids, keys,
                                       expected_digest=header["decoded_state_hash"] if runtime else None,
                                       validation=self.validation if runtime else None)
        if state.state_identity != header["decoded_state_hash"]:
            raise ValueError("Decoded Gaussian state checksum mismatch")
        self.stats["state_verification_ms"] += (perf_counter()-start)*1000
        return state

    def close(self):
        if self.shared_cache is None:
            self.streams.clear()
            self.arrays.clear()
            self.validation.clear()
            self.history.clear()
            self.idmaps.clear()
