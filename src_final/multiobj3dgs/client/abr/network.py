"""Production observation adapter for the existing DASH HTTP/cache runtime.

``ResourceCache`` calls ``started`` before admission and delivers each completed
``TransferResult`` once. This adapter measures the union of actual HTTP busy
intervals, including TTFB, and passes one aggregate sample to the existing
harmonic estimator. It never sums competing requests' individual rates. Cache
hits create no request and therefore no fictitious bandwidth observation.
"""
from __future__ import annotations

from collections import Counter
import asyncio
import math
import time

from multiobj3dgs.client.network.bandwidth import BandwidthEstimator
from multiobj3dgs.client.cache import ResourceCache
from multiobj3dgs.client.cache.resources import digest
from multiobj3dgs.shared.transfer import BandwidthSample, ErrorKind


class PipelineResourceCache(ResourceCache):
    """Existing cache with conservative resident-byte credit for ABR planning.

    The base cache retains its verification history after LRU eviction. ABR cost
    must additionally require residency; an evicted, previously verified blob
    needs another HTTP transfer. No credit is given to pending downloads.
    """

    async def verify_resident(self, catalog):
        """Validate prior-session blobs before the first adaptation decision.

        The production cache discovers resident files at startup, but verifies
        them lazily on ``obtain``. Planning before that lazy check would charge
        bytes which a warm session never transfers. Validate only resident
        resources in this publication, off the I/O thread, without fetching any
        payload or generating bandwidth samples. Call during startup, before
        admitting materializations.
        """
        if self._pending or self._packages:
            raise RuntimeError('Resident verification must precede cache admission')
        refs = {}
        for content in catalog.contents.values():
            for representation in content.representations.values():
                for segment in representation.segments:
                    for ref in segment.resources:
                        previous = refs.get(ref.sha256)
                        if previous is not None and previous.bytes != ref.bytes:
                            raise ValueError('Conflicting resource sizes for identical SHA256')
                        refs[ref.sha256] = ref

        began = time.monotonic()

        def verify():
            counts = dict(verified_entries=0, verified_bytes=0, removed_entries=0)
            with self._guard:
                residents = [(key, path, size, refs[key]) for key, (path, size) in self._entries.items()
                             if key in refs]
            for key, path, size, ref in residents:
                try:
                    valid = size == ref.bytes and path.stat().st_size == ref.bytes and digest(path) == key
                except OSError:
                    valid = False
                with self._guard:
                    if not valid:
                        self._entries.pop(key, None)
                        self._verified.discard(key)
                        path.unlink(missing_ok=True)
                        counts['removed_entries'] += 1
                    else:
                        self._verified.add(key)
                        counts['verified_entries'] += 1
                        counts['verified_bytes'] += size
            return counts

        counts = await asyncio.to_thread(verify)
        if self.events:
            self.events.emit('resident_cache_verification', **counts,
                             duration_s=time.monotonic() - began)
        return counts

    def missing_bytes(self, segment):
        segments = (segment,) if hasattr(segment, "resource") else tuple(segment)
        unique = {}
        for seg in segments:
            refs = getattr(seg, "resources", None)
            if refs is None:
                refs = (seg.resource,) + tuple(seg.shared_assets)
            for ref in refs:
                previous = unique.get(ref.sha256)
                if previous is not None and previous.bytes != ref.bytes:
                    raise ValueError("Conflicting resource sizes for identical SHA256")
                unique[ref.sha256] = ref
        with self._guard:
            return sum(ref.bytes for key, ref in unique.items()
                       if key not in self._verified or key not in self._entries)

    async def materialize(self, segment):
        # A GOP consists of a bundle followed by small shared assets. Giving
        # each tiny file equal harmonic weight would make round-trip overhead
        # dominate link prediction. Observe the actual selected resource group
        # once, while preserving the existing cache/materializer implementation.
        grouped = self.observer is not None and hasattr(self.observer, "begin_batch")
        if grouped:
            self.observer.begin_batch()
        try:
            return await super().materialize(segment)
        finally:
            if grouped:
                self.observer.end_batch()


def busy_interval_seconds(results):
    """Union duration of completed request intervals (not their duration sum)."""
    spans = sorted((r.start_time, r.end_time) for r in results)
    if not spans:
        return 0.0
    if any(not math.isfinite(a) or not math.isfinite(b) or b < a for a, b in spans):
        raise ValueError("Invalid HTTP transfer interval")
    start, end = spans[0]
    busy = 0.0
    for a, b in spans[1:]:
        if a > end:
            busy += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return busy + end - start


class PipelineTransferObserver:
    """ResourceCache-compatible observer with consistent concurrent accounting.

    All calls run on the runtime's existing I/O loop. Even with concurrency one,
    effective goodput includes request latency; this conservative policy avoids
    a large first body chunk turning a paced transfer into an inflated sample.
    A batch containing failed or cancelled eligible transfers supplies no
    capacity sample, but all received bytes remain in the traffic ledger.
    Startup/staleness and the harmonic window are owned by BandwidthEstimator.
    """

    KINDS = frozenset(("metadata", "media", "shared_asset", "profile", "other"))

    def __init__(self, estimator=None, concurrency=1, events=None):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.estimator = estimator if estimator is not None else BandwidthEstimator()
        self.concurrency = concurrency
        self.events = events
        self.totals = Counter()
        self._active = {}
        self._batch = []
        self._cancelled = False
        self._group_depth = 0
        self.last_observed_bps = None
        self.last_batch_duration_s = None

    def begin_batch(self):
        """Keep sequential resources/concurrent GOPs in one capacity sample."""
        self._group_depth += 1

    def end_batch(self):
        if self._group_depth <= 0:
            raise ValueError("No bandwidth observation batch is open")
        self._group_depth -= 1
        self._flush_if_idle()

    def _emit(self, kind, **values):
        if self.events is not None:
            self.events.emit(kind, **values)

    def started(self, spec):
        if spec.request_id in self._active:
            raise ValueError(f"Duplicate request admission: {spec.request_id}")
        self._active[spec.request_id] = spec
        self._emit("request_started", request_id=spec.request_id, url=spec.url,
                   traffic_kind=spec.metadata.get("traffic_kind", "other"))

    def transport_completion(self, result):
        """Subscribe once to Request API for cancelled transfers' partial bytes.

        The cache normally observes returned results itself. A cancelled fetch
        raises after Request API emits its final result, so only that result is
        forwarded here. Normal completion stays on the cache's existing path.
        """
        if (result.error is not None and result.error.kind == ErrorKind.CANCELLED
                and result.request_id in self._active):
            self(result)

    def __call__(self, result):
        if result.request_id not in self._active:
            raise ValueError("Observation requires started(spec) before completion")
        if isinstance(result.received_bytes, bool) or not isinstance(result.received_bytes, int) or result.received_bytes < 0:
            raise ValueError("received_bytes must be a nonnegative integer")
        self._active.pop(result.request_id)
        kind = result.metadata.get("traffic_kind", "other")
        kind = kind if isinstance(kind, str) and kind in self.KINDS else "other"
        self.totals[kind] += result.received_bytes
        self.totals["total_bytes"] += result.received_bytes
        self.totals["transfer_count"] += 1
        if not result.success:
            self.totals["failed_bytes"] += result.received_bytes
            self.totals["failed_transfer_count"] += 1
        self._emit("transfer", request_id=result.request_id, url=result.url,
                   object_id=result.object_id, representation_id=result.representation_id,
                   segment_id=result.segment_id, bytes=result.received_bytes,
                   success=result.success, status_code=result.status_code,
                   duration_s=result.total_duration, ttfb_s=result.ttfb,
                   http_version=result.http_version, traffic_kind=kind,
                   effective_goodput_bps=result.effective_goodput_bps,
                   error=str(result.error) if result.error else None)
        if kind in ("media", "shared_asset") and result.metadata.get("bandwidth_eligible", True) is True:
            self._batch.append(result)
        self._flush_if_idle()

    def cancelled(self, spec):
        # ResourceCache also invokes this in its finally block after a normal
        # completion. Such a call must not erase another request's observation.
        active = self._active.pop(spec.request_id, None)
        if active is not None:
            self.totals["cancelled_before_completion_count"] += 1
            if active.metadata.get("traffic_kind") in ("media", "shared_asset"):
                self._cancelled = True
            self._emit("request_cancelled", request_id=spec.request_id, url=spec.url)
            self._flush_if_idle()

    def _flush_if_idle(self):
        if self._active or self._group_depth:
            return
        results, cancelled = self._batch, self._cancelled
        self._batch, self._cancelled = [], False
        if not results:
            return
        duration = busy_interval_seconds(results)
        size = sum(r.received_bytes for r in results)
        accepted = False
        rate = size * 8 / duration if size > 0 and duration > 0 else None
        complete = all(r.success for r in results) and not cancelled
        if rate is not None and complete:
            accepted = self.estimator.accept_transfer_sample(BandwidthSample(
                timestamp=time.time(), received_bytes=size, ttfb=None,
                transfer_duration=duration, total_duration=duration,
                body_goodput_bps=rate, effective_goodput_bps=rate,
                http_version=None, success=True,
            ))
        if accepted:
            self.last_observed_bps = rate
            self.last_batch_duration_s = duration
        self.totals["bandwidth_batch_count"] += 1
        self.totals["bandwidth_sample_count"] += int(accepted)
        self._emit("bandwidth_sample", accepted=accepted, bytes=size,
                   duration_s=duration, observed_bps=rate if complete else None,
                   transfers=len(results), cancelled=cancelled,
                   failed_transfers=sum(not r.success for r in results),
                   policy="aggregate bytes / union HTTP busy intervals including TTFB")

    def snapshot(self):
        values = dict(self.totals)
        for name in ("total_bytes", "transfer_count", "failed_bytes", "failed_transfer_count"):
            values.setdefault(name, 0)
        values.update(active_requests=len(self._active),
                      open_resource_groups=self._group_depth,
                      observed_bps=self.last_observed_bps,
                      last_batch_duration_s=self.last_batch_duration_s)
        return values
