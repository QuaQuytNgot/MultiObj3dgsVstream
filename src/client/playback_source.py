"""Small playback adapter over the existing MPD, dependency and transport APIs."""
from __future__ import annotations

from bisect import bisect_right
import asyncio
from collections import OrderedDict, Counter
from dataclasses import dataclass
import hashlib
import math
import threading
from time import perf_counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .adaptation_inputs import AvailabilitySnapshot, QualityTarget, RequirementsResolver
from .decoder import ProgressiveDecoder
from .mpd_parser import MPDParser
from .request_api import TransferClient
from .verified_input import VerifiedSegment


class SegmentBuffer:
    """Verified complete containers, bounded by bytes with LRU eviction.

    A single container larger than the budget is usable for the current decode
    but is not retained. Cache credit is removed as soon as its bytes are evicted.
    """
    def __init__(self, max_bytes: int = 512 * 1024**2):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._items = OrderedDict()

    def get(self, unit_id):
        item = self.get_verified(unit_id)
        return None if item is None else item.data

    def get_verified(self, unit_id):
        item = self._items.get(unit_id)
        if item is not None:
            self._items.move_to_end(unit_id)
        return None if item is None else item[1]

    def put(self, unit, body: bytes):
        self._put_verified(unit, VerifiedSegment(unit,body))

    def _put_verified(self, unit, body):
        """Internal publication after _verify_segment; do not use for ingestion."""
        if not isinstance(body,VerifiedSegment) or body.unit_id != unit.unit_id or body.checksum != unit.checksum:
            raise ValueError("Cannot publish unverified segment bytes")
        previous = self._items.pop(unit.unit_id, None)
        if previous is not None:
            self.size_bytes -= len(previous[1])
        if len(body) > self.max_bytes:
            return
        while self.size_bytes + len(body) > self.max_bytes:
            _, (_, evicted) = self._items.popitem(last=False)
            self.size_bytes -= len(evicted)
        self._items[unit.unit_id] = (unit.checksum, body)
        self.size_bytes += len(body)

    @property
    def cached_units(self):
        return {key: checksum for key, (checksum, _) in self._items.items()}

    def clear(self):
        self._items.clear()
        self.size_bytes = 0


@dataclass(frozen=True)
class FrameSelection:
    frame: int
    media_time_s: float
    presentation_time_s: float
    cycle: int


class PlaybackSource:
    """Base-first dependency scheduling and in-memory media delivery.

    Fixed quality selection belongs to the application. An AdaptationPlanner can
    supply targets through the same ``get_frame`` and ``availability`` boundary.
    No calibrated density or visibility data is invented here.
    """
    def __init__(self, parser: MPDParser, decoder: ProgressiveDecoder, *,
                 client: TransferClient | None = None, local_root: Path | None = None,
                 media_base_url: str = "http://offline.invalid/", buffer_bytes=512 * 1024**2,
                 decode_in_worker=False, stream_package=None, stream_base_url=None, workers=None,
                 result_capacity=8):
        if (client is None) == (local_root is None):
            raise ValueError("Provide exactly one of client or local_root")
        self.parser, self.decoder = parser, decoder
        self.index = parser.get_content_index()
        self.resolver = RequirementsResolver(self.index)
        self.client = client
        self.local_root = None if local_root is None else Path(local_root).resolve()
        self.media_base_url = media_base_url
        self.buffer = SegmentBuffer(buffer_bytes)
        self.read_bytes = 0
        self.fetch_count = 0
        self.segment_verification_seconds = 0.
        self.input_copy_seconds = 0.
        self.input_copy_bytes = 0
        self._generation = 0
        self._decode_lock = asyncio.Lock()
        self.decode_in_worker = decode_in_worker
        self.workers = workers
        self._pool_slots = asyncio.Semaphore(workers.pending) if workers else None
        self._input_slots = asyncio.Semaphore(result_capacity) if workers else None
        self._frame_tasks = {}
        self._frame_waiters = Counter()
        self._active_requests = set()
        self._results = OrderedDict()
        self.result_capacity = result_capacity
        self._receive_tasks = {}
        self.stream_store = None
        self.stream_base_url = stream_base_url
        if stream_package is not None:
            if client is not None and not stream_base_url:
                raise ValueError("HTTP stream packages require stream_base_url")
            from .static_dynamic import StaticDynamicStore
            self.stream_store = StaticDynamicStore(stream_package)
        self._timelines = {}
        for obj in parser.get_objects():
            metadata = parser.get_object_metadata(obj)
            frames = tuple(metadata["frames"])
            first = frames[0]["timestamp"]
            self._timelines[obj] = (frames, tuple(f["timestamp"] - first for f in frames),
                                    float(metadata["duration_seconds"]))

    def select_frame(self, object_id: str, time_s: float, *, loop=True) -> FrameSelection:
        if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Playback time must be finite and nonnegative")
        frames, times, duration = self._timelines[object_id]
        if not loop and time_s >= duration:
            raise EOFError(f"End of sequence: {object_id}")
        cycle, local_time = 0, time_s
        if loop:
            quotient = time_s / duration
            boundary = round(quotient)
            # Decimal durations need not divide binary floats exactly (.6/.2).
            # Snap only floating-point rounding error at a cycle boundary.
            if abs(quotient - boundary) <= 4 * math.ulp(quotient):
                cycle, local_time = boundary, 0.
            else:
                cycle = math.floor(quotient)
                local_time = time_s - cycle * duration
        tolerance = 4 * max(math.ulp(time_s), math.ulp(duration))
        position = max(0, bisect_right(times, local_time + tolerance) - 1)
        frame = frames[position]
        return FrameSelection(frame["frame"], frame["timestamp"], times[position] + cycle * duration, cycle)

    def availability(self) -> AvailabilitySnapshot:
        self._generation += 1
        return AvailabilitySnapshot(self.index.content_revision, f"playback:{self._generation}",
                                    self.buffer.cached_units, self.decoder.decoded_states)

    async def _receive(self, unit):
        return (await self._receive_verified(unit)).data

    async def _receive_verified(self, unit):
        body = self.buffer.get_verified(unit.unit_id)
        if body is not None:
            return body
        task = self._receive_tasks.get(unit.unit_id)
        if task is None:
            task = asyncio.create_task(self._receive_unit(unit))
            self._receive_tasks[unit.unit_id] = task
            def received(done):
                if self._receive_tasks.get(unit.unit_id) is done:
                    self._receive_tasks.pop(unit.unit_id, None)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(received)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep admission charged until a non-cancellable disk/hash operation
            # has drained, rather than launching unlimited obsolete I/O work.
            await asyncio.gather(task,return_exceptions=True)
            raise
        finally:
            if task.done() and self._receive_tasks.get(unit.unit_id) is task:
                self._receive_tasks.pop(unit.unit_id, None)

    async def _receive_unit(self, unit):
        body = self.buffer.get_verified(unit.unit_id)
        if body is not None:
            return body
        if self.local_root is not None:
            if not unit.request.url.startswith(self.media_base_url):
                raise ValueError("Offline media URL is outside configured media base")
            relative = unquote(urlsplit(unit.request.url[len(self.media_base_url):]).path)
            path = (self.local_root / relative).resolve()
            if not path.is_relative_to(self.local_root):
                raise ValueError("Offline media path escapes content root")
            if not path.is_file():
                raise FileNotFoundError(f"Media unavailable (possibly still transferring): {path}")
            if path.stat().st_size != unit.size_bytes:
                raise ValueError(f"Incomplete media segment: {path}")
            body = await asyncio.to_thread(path.read_bytes)
        else:
            result = await self.client.fetch(unit.request)
            if not result.success or result.body is None:
                raise RuntimeError(f"Media transfer failed: {unit.unit_id}: {result.error}")
            body = result.body
        # Input verification and disk I/O must not block camera/event polling.
        # Receive tasks are coalesced by unit ID; cache mutation occurs below on
        # the event-loop thread after the expensive hash has finished.
        began = perf_counter()
        verified = await asyncio.to_thread(VerifiedSegment, unit, body)
        self.segment_verification_seconds += perf_counter()-began
        self.buffer._put_verified(unit, verified)
        self.read_bytes += len(body)
        self.fetch_count += 1
        return verified

    async def prepare_chain(self, object_id, frame, quality):
        """Transport only; each worker reconstructs the complete local chain."""
        if self.stream_store is None:
            descriptors = [self.index.payload_for(object_id,frame,q) for q in range(quality+1)]
            locations = [self.parser.get_payload_location(d.payload_id) for d in descriptors]
            containers = await asyncio.gather(*[self._receive_verified(self.index.request_units[d.request_unit_id])
                                                for d in descriptors])
            # Detach in an I/O thread. Copying verified immutable bytes preserves
            # their certificate; only payload bytes count toward the worker's
            # input allowance, rather than whole evicted containers.
            began = perf_counter()
            members = await asyncio.to_thread(lambda:[c.member(l).detached() for c,l in zip(containers,locations)])
            self.input_copy_seconds += perf_counter()-began
            self.input_copy_bytes += sum(map(len,members))
            return [(d,m,l.checksum) for d,m,l in zip(descriptors,members,locations)]
        chain = []
        for q in range(quality+1):
            descriptor = self.index.payload_for(object_id, frame, q)
            location = self.parser.get_payload_location(descriptor.payload_id)
            if self.stream_store is not None:
                if self.client is not None:
                    member = await self.stream_store.read_http(descriptor.payload_id, location.checksum,
                                                              location.length, self.client, self.stream_base_url)
                else:
                    member = await asyncio.to_thread(self.stream_store.read, descriptor.payload_id,
                                                     location.checksum, location.length)
            else:
                unit = self.index.request_units[descriptor.request_unit_id]
                container = await self._receive_verified(unit)
                member = container.member(location)
            chain.append((descriptor, member, location.checksum))
        return chain

    def is_active(self,key):
        return key in self._active_requests

    async def _pooled_frame(self, key):
        async with self._input_slots:
            self._active_requests.add(key)
            try:
                chain = await self.prepare_chain(*key)
            finally:
                self._active_requests.discard(key)
            # Prepared input and native execution have separate admission limits.
            # Waiting for I/O must not consume a native decoder slot.
            async with self._pool_slots:
                self._active_requests.add(key)
                try:
                    return await self._decode_prepared_frame(key,chain)
                finally:
                    self._active_requests.discard(key)

    async def _decode_prepared_frame(self, key, chain):
        cancelled = threading.Event()
        future = asyncio.wrap_future(self.workers.submit(chain, cancelled))
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await future
            except (asyncio.CancelledError, Exception):
                pass
            raise
        self._results[key] = result
        self._results.move_to_end(key)
        while len(self._results) > self.result_capacity:
            self._results.popitem(last=False)
        return result

    async def get_frame(self, object_id: str, frame: int, quality: int):
        if self.workers is not None:
            key = object_id, frame, quality
            if key in self._results:
                self._results.move_to_end(key)
                return self._results[key]
            task = self._frame_tasks.get(key)
            if task is None:
                task = asyncio.create_task(self._pooled_frame(key))
                self._frame_tasks[key] = task
            self._frame_waiters[key] += 1
            cancelled = False
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                self._frame_waiters[key] -= 1
                if not self._frame_waiters[key]:
                    del self._frame_waiters[key]
                    if cancelled and not task.done():
                        task.cancel()
                        await asyncio.gather(task,return_exceptions=True)
                if task.done() and self._frame_tasks.get(key) is task:
                    self._frame_tasks.pop(key, None)
        # Retain parents throughout an awaited dependency chain, including when
        # multiple scene consumers request frames concurrently.
        async with self._decode_lock:
            return await self._decode_frame(object_id, frame, quality)

    async def _decode_frame(self, object_id: str, frame: int, quality: int):
        target = QualityTarget(object_id, (frame,), quality)
        requirements = self.resolver.required_units(target, self.availability())
        if not requirements.valid:
            raise ValueError(f"Cannot schedule progressive decode: {requirements.reason}")
        # Decode as each parent container arrives; do not retain an entire
        # multi-quality batch before allowing useful decoding work.
        for identifier in requirements.decode_order:
            payload = self.index.payloads[identifier]
            unit = self.index.request_units[payload.request_unit_id]
            location = self.parser.get_payload_location(identifier)
            if self.stream_store is not None:
                old_bytes, old_reads = self.stream_store.stats["read_bytes"], self.stream_store.stats["stream_reads"]
                if self.client is not None:
                    member = await self.stream_store.read_http(identifier,location.checksum,location.length,self.client,self.stream_base_url)
                else:
                    member = self.stream_store.read(identifier, location.checksum, location.length)
                self.read_bytes += self.stream_store.stats["read_bytes"]-old_bytes
                self.fetch_count += self.stream_store.stats["stream_reads"]-old_reads
            else:
                body = await self._receive(unit)
                end = location.offset + location.length
                if location.offset < 0 or end > len(body):
                    raise ValueError(f"Payload extent exceeds segment: {identifier}")
                member = memoryview(body)[location.offset:end]
            runtime = self.decoder.optimized is not None and self.decoder.optimized.verification == "runtime"
            if not runtime and hashlib.sha256(member).hexdigest() != location.checksum:
                raise ValueError(f"Payload hash mismatch: {identifier}")
            if self.decode_in_worker:
                # Keep the dependency lock held until the native/CPU task ends,
                # including cancellation, before another request touches caches.
                task = asyncio.create_task(asyncio.to_thread(self.decoder.decode_payload,payload,member,
                                                            expected_checksum=location.checksum))
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            else:
                self.decoder.decode_payload(payload, member, expected_checksum=location.checksum)
        descriptor = self.index.payload_for(object_id, frame, quality)
        return self.decoder.get_frame(object_id, frame, quality, epoch=descriptor.epoch)

    def close(self):
        self.buffer.clear()
        self.decoder.close()
        if self.stream_store is not None:
            self.stream_store.close()
        self._results.clear()

    async def aclose(self):
        tasks = list(self._frame_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*self._receive_tasks.values(), return_exceptions=True)
        self._frame_tasks.clear()
        self._receive_tasks.clear()
        self.close()
