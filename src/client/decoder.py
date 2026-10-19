"""Verified progressive decoding through the content preparation codec.

GaussianState keeps the codec's CPU numpy representation: xyz, raw wxyz
quaternions, log scales, opacity logits and N x RGB x SH coefficients. Renderer
backends apply their own activation/layout conversions; no PLY is involved.
The codec currently predicts no temporal frames. Enhancements require the
retained, exact decoded parent for the same object, frame and representation
epoch, including when payloads arrive out of order over the network.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

from tools.content_preparation.assets import GaussianState, STATE_VERSION, state_hash
from tools.content_preparation.codec_adapter import _read_blob, get_decoder, inspect_payload

from .adaptation_inputs import DecodedState, PayloadDescriptor, integer
from .verified_input import VerifiedPayload


class MissingDependencyError(ValueError):
    """An enhancement arrived before its same-frame parent was retained."""


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    descriptor: PayloadDescriptor
    state: GaussianState
    decode_seconds: float
    cache_hit: bool = False

    @property
    def object_id(self) -> str:
        return self.descriptor.object_id

    @property
    def frame_index(self) -> int:
        return self.descriptor.frame

    @property
    def timestamp(self) -> float:
        return self.descriptor.media_time_s

    @property
    def quality(self) -> int:
        return self.descriptor.quality


class ProgressiveDecoder:
    """Small bounded cache of exact decoded quality states.

    The downloader owns encoded segment storage. Call ``decode_payload`` in
    RequirementsResolver's decode order, then obtain any retained quality with
    ``get_frame``. All quality states of a cached frame are retained together,
    so Q0 -> Q3 -> Q0 does not repeat codec work. ``max_cached_frames`` counts
    distinct (object, media frame, epoch) tuples across all objects.
    """

    def __init__(self, runtime_config: Mapping | None = None,
                 max_cached_frames: int = 2, *, optimization: Mapping | None = None):
        self.runtime_config = dict(runtime_config) if runtime_config is not None else None
        self.max_cached_frames = integer(max_cached_frames, "max_cached_frames", 1)
        self._frames: OrderedDict[tuple[str, int, int], dict[int, DecodedFrame]] = OrderedDict()
        self._payloads: dict[str, DecodedFrame] = {}
        self._packet_hashes: dict[str, str] = {}
        self.total_decode_seconds = 0.0
        self.decode_count = 0
        self.cache_hits = 0
        self.optimized = None
        if optimization is not None:
            from .optimized_codec import OptimizedDraco
            self.optimized = OptimizedDraco(self.runtime_config, **optimization)

    @property
    def decoded_states(self) -> Mapping[str, DecodedState]:
        """Verified identities suitable for AvailabilitySnapshot/planner input."""
        return MappingProxyType({key: DecodedState.from_payload(frame.descriptor)
                                 for key, frame in tuple(self._payloads.items())})

    @property
    def retained_bytes(self) -> int:
        return sum(frame.state.ids.nbytes + sum(a.nbytes for a in frame.state.arrays.values())
                   for frame in tuple(self._payloads.values()))

    @staticmethod
    def _frame_key(descriptor: PayloadDescriptor) -> tuple[str, int, int]:
        return descriptor.object_id, descriptor.frame, descriptor.epoch

    def _validate_header(self, descriptor: PayloadDescriptor, header: dict) -> None:
        if descriptor.state_version != STATE_VERSION or header.get("state_version") != descriptor.state_version:
            raise ValueError("Payload decoded-state version mismatch")
        if header.get("decoded_state_hash") != descriptor.decoded_state_hash:
            raise ValueError("Payload decoded-state hash differs from its descriptor")
        if header.get("parent_state_hash") != descriptor.parent_state_hash:
            raise ValueError("Payload parent-state hash differs from its descriptor")
        codec = header["codec"]
        if descriptor.codec_id not in ("default", codec["name"] + "/" + codec["version"]):
            raise ValueError("Payload codec differs from its descriptor")
        if header.get("temporal_prediction") is not False:
            raise ValueError("Temporal prediction is unsupported by the current codec contract")
        metadata = header.get("metadata", {})
        if "frame" in metadata and metadata["frame"] != descriptor.frame:
            raise ValueError("Payload frame metadata differs from its descriptor")
        if "quality" in metadata and metadata["quality"] != f"Q{descriptor.quality}":
            raise ValueError("Payload quality metadata differs from its descriptor")

    def _parent(self, descriptor: PayloadDescriptor, header: dict) -> GaussianState | None:
        if header.get("kind") == "full":
            if descriptor.dependencies or descriptor.parent_state_hash is not None or not header.get("self_contained"):
                raise ValueError("Full payload must be self contained without dependencies")
            return None
        if header.get("kind") != "refinement" or descriptor.quality == 0:
            raise ValueError("Only enhancement qualities can contain refinement packets")
        if header.get("self_contained") is not False or not descriptor.parent_state_hash:
            raise ValueError("Refinement payload must declare its decoded parent")
        dependencies = []
        for identifier in descriptor.dependencies:
            dependency = self._payloads.get(identifier)
            if dependency is None:
                raise MissingDependencyError(f"Missing same-frame decoded parent: {identifier}")
            other = dependency.descriptor
            if (self._frame_key(other) != self._frame_key(descriptor)
                    or other.media_time_s != descriptor.media_time_s):
                raise ValueError("Progressive dependency must belong to the same object, frame, epoch and timestamp")
            dependencies.append(dependency)
        parents = [dependency for dependency in dependencies
                   if (dependency.quality == descriptor.quality - 1
                       and dependency.descriptor.decoded_state_hash == descriptor.parent_state_hash)]
        if len(parents) != 1:
            raise MissingDependencyError("Missing exact preceding-quality decoded parent")
        return parents[0].state

    def decode_payload(self, descriptor: PayloadDescriptor, payload, *, expected_checksum=None) -> DecodedFrame:
        """Decode actual .cpgs bytes/path/stream, verifying index and codec hashes.

        Malformed packets and missing dependencies raise; no substitute state is
        manufactured. Failed decoding leaves retained states unchanged.
        """
        if not isinstance(descriptor, PayloadDescriptor):
            raise TypeError("descriptor must be a PayloadDescriptor")
        started = perf_counter()
        token = payload if isinstance(payload,VerifiedPayload) else None
        runtime = self.optimized is not None and self.optimized.verification == "runtime"
        if token is not None and token.request_unit_id != descriptor.request_unit_id:
            raise ValueError("Verified payload segment differs from descriptor")
        blob = token.data if token is not None and runtime else _read_blob(token.data if token else payload)
        hash_started = perf_counter()
        digest = token.checksum if token is not None and runtime else hashlib.sha256(blob).hexdigest()
        if expected_checksum is not None and digest != expected_checksum:
            raise ValueError("Payload hash mismatch against trusted MPD checksum")
        if self.optimized is not None:
            self.optimized.stats["ingestion_verification_ms"] += (perf_counter()-hash_started)*1000
            self.optimized.stats["verified_segment_reuses"] += int(token is not None and runtime)
        previous = self._payloads.get(descriptor.payload_id)
        if previous is not None:
            if previous.descriptor != descriptor or self._packet_hashes[descriptor.payload_id] != digest:
                raise ValueError("Retained payload identity was reused with different bytes or metadata")
            self.cache_hits += 1
            self._frames.move_to_end(self._frame_key(descriptor))
            return replace(previous, decode_seconds=0.0, cache_hit=True)
        packet = self.optimized.inspect(blob, externally_verified=expected_checksum is not None or token is not None) if self.optimized is not None and blob[:8] == b"CPDRAC01" else None
        header = packet[0] if packet is not None else inspect_payload(blob)
        self._validate_header(descriptor, header)
        parent = self._parent(descriptor, header)
        key = self._frame_key(descriptor)
        existing_quality = self._frames.get(key, {}).get(descriptor.quality)
        if existing_quality is not None:
            raise ValueError("Multiple payload identities target the same object/frame/quality")
        if packet is not None:
            state = self.optimized.decode(packet, parent)
            verified_hash = state.state_identity
        else:
            decoder = get_decoder(header["codec"], self.runtime_config)
            state = decoder.decode(blob, parent=parent)
            verified_hash = state_hash(state)
        if verified_hash != descriptor.decoded_state_hash:
            raise ValueError("Reconstructed state differs from the declared decoded-state hash")
        # Backends must never transform these shared cached arrays in place.
        for array in (*state.arrays.values(), state.ids):
            array.flags.writeable = False
        elapsed = perf_counter() - started
        frame = DecodedFrame(descriptor, state, elapsed)
        self._frames.setdefault(key, {})[descriptor.quality] = frame
        self._frames.move_to_end(key)
        self._payloads[descriptor.payload_id] = frame
        self._packet_hashes[descriptor.payload_id] = digest
        self.total_decode_seconds += elapsed
        self.decode_count += 1
        while len(self._frames) > self.max_cached_frames:
            self._release_key(next(iter(self._frames)))
        return frame

    def get_frame(self, object_id: str, frame: int, quality: int,
                  epoch: int | None = None) -> DecodedFrame:
        frame, quality = integer(frame, "frame"), integer(quality, "quality")
        matches = [key for key in self._frames
                   if key[0] == object_id and key[1] == frame and (epoch is None or key[2] == epoch)
                   and quality in self._frames[key]]
        if not matches:
            raise KeyError((object_id, frame, quality, epoch))
        if len(matches) != 1:
            raise ValueError("Multiple representation epochs retained; specify epoch")
        key = matches[0]
        self._frames.move_to_end(key)
        return self._frames[key][quality]

    def _release_key(self, key: tuple[str, int, int]) -> None:
        for frame in self._frames.pop(key).values():
            self._payloads.pop(frame.descriptor.payload_id, None)
            self._packet_hashes.pop(frame.descriptor.payload_id, None)

    def release_frame(self, object_id: str, frame: int, epoch: int | None = None) -> None:
        for key in tuple(self._frames):
            if key[:2] == (object_id, frame) and (epoch is None or key[2] == epoch):
                self._release_key(key)

    def release_object(self, object_id: str) -> None:
        for key in tuple(self._frames):
            if key[0] == object_id:
                self._release_key(key)

    def close(self) -> None:
        self._frames.clear()
        self._payloads.clear()
        self._packet_hashes.clear()
        if self.optimized is not None:
            self.optimized.close()

    def __enter__(self) -> ProgressiveDecoder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
