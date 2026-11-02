"""Virtual HTTP/2-style streams over one shared trace-capacity connection.

Open streams include response latency and body service; ready bodies share the
connection equally. This models multiplexing, not HTTP/2 negotiation/framing,
TCP congestion control, or flow-control window stalls. No real sockets are used.
The estimator sees completed batches only, never future trace capacity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping

from experiments.abr_network_evaluation.network import NetworkTrace, SharedLink, Transfer
from experiments.abr_network_evaluation.predictor import HarmonicPredictor


def load_trace(config: Mapping, base_dir: str | Path | None = None) -> NetworkTrace:
    """Load existing CSV/JSON traces without silently extending their last rate.

    Time is seconds and rates are decimal Mbps; production JSON ``bps`` points
    are normalized to that existing contract. A finite service horizon must be
    supplied by ``duration_s`` (possibly in JSON), unless ``end_policy=hold``
    or independent trace looping is explicitly requested. Missing trace samples
    are not interpolated. RTT/latency is a separate simulation configuration.
    """
    cfg = dict(config)
    if cfg.get("mode") == "unthrottled":
        raise ValueError("Streaming evaluation requires a bandwidth service curve")
    if "end_policy" not in cfg:
        cfg["end_policy"] = "stop"
    path = cfg.get("path", cfg.get("trace_file"))
    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(base_dir or ".") / resolved
        if resolved.suffix.lower() == ".json":
            raw = json.loads(resolved.read_text())
            if isinstance(raw, dict):
                for key in ("duration_s", "loop", "end_policy"):
                    if key in raw and key not in config:
                        cfg[key] = raw[key]
                raw = raw["points"]
            cfg.pop("path", None)
            cfg.pop("trace_file", None)
            cfg["points"] = raw
    if "points" in cfg:
        normalized = []
        for point in cfg["points"]:
            if isinstance(point, dict) and "bps" in point:
                point = {
                    "time_s": point.get("time_s", point.get("at_s", point.get("time"))),
                    "mbps": float(point["bps"]) / 1e6,
                }
            normalized.append(point)
        cfg["points"] = normalized
    if cfg.get("duration_s") is None and cfg.get("end_policy") != "hold":
        raise ValueError("Network duration_s is required; last-rate extension requires explicit end_policy=hold")
    return NetworkTrace.from_config(cfg, base_dir=base_dir)


@dataclass(frozen=True)
class DownloadResult:
    request_id: str
    kind: str
    requested_bytes: int
    delivered_bytes: float
    start_s: float | None
    ready_s: float | None
    end_s: float
    completed_at_s: float | None
    capacity_bytes: float
    status: str
    queued_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s if self.start_s is not None else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self),
                "queue_duration_s": (self.start_s if self.start_s is not None else self.end_s) - self.queued_s,
                "latency_duration_s": (max(0., min(self.ready_s, self.end_s) - self.start_s)
                                       if self.start_s is not None else 0.),
                "body_duration_s": max(0., self.end_s - self.ready_s) if self.ready_s is not None else 0.}


class NetworkLink:
    """One persistent connection, with FIFO admission to bounded stream slots.

    ``download`` advances virtual time and returns either a complete transfer or
    explicit partial delivery at trace exhaustion/no future service. A failed
    link cannot accept more requests. No real sockets or clocks are consulted.
    Per-request capacity is an overlapping diagnostic, never additive across
    streams. The single shared service ledger is authoritative for conservation.
    """

    def __init__(self, trace: NetworkTrace, latency_s: float = 0.0,
                 max_concurrent_streams: int = 16):
        if trace.unthrottled:
            raise ValueError("Streaming evaluation requires a bandwidth service curve")
        self.trace = trace
        # SharedLink may inspect rate at an exact completion boundary. Treat
        # error-ended traces as stopped internally and report exhaustion here.
        service = (NetworkTrace(trace.points, duration_s=trace.duration_s, loop=trace.loop,
                                end_policy="stop") if trace.end_policy == "error" else trace)
        self._link = SharedLink(service, concurrency=max_concurrent_streams, latency_s=latency_s)
        self.max_concurrent_streams = max_concurrent_streams
        self.latency_s = float(latency_s)
        self.results: list[DownloadResult] = []
        self.failed = False
        self.failure_status: str | None = None

    @property
    def now(self) -> float:
        return self._link.now

    @property
    def ledger(self) -> list[dict]:
        return self._link.ledger

    @property
    def delivered_bytes(self) -> float:
        return self._link.transferred_bytes

    @property
    def requested_bytes(self) -> int:
        return sum(row.requested_bytes for row in self.results)

    def capacity_between(self, start_s: float, end_s: float) -> float:
        """Offline available body capacity; never supplied to the estimator."""
        return self._link.trace.integral_bits(start_s, end_s) / 8

    def advance_to(self, wall_time_s: float) -> None:
        if not self._link.idle:
            raise ValueError("Cannot idle-advance a blocked incomplete request")
        # Idle progress must not ask an error-ended trace for future capacity.
        if not math.isfinite(wall_time_s) or wall_time_s < self.now - 1e-9:
            raise ValueError("Network time must be finite and nondecreasing")
        self._link.now = max(self.now, float(wall_time_s))

    def download(self, size_bytes: int, start_time_s: float | None = None, *,
                 request_id: str | None = None, kind: str = "media",
                 deadline_s: float | None = None) -> DownloadResult:
        """Compatibility wrapper for a single blocking request."""
        return self.download_batch([dict(requested_bytes=size_bytes,
            request_id=request_id, kind=kind)], start_time_s, deadline_s=deadline_s)[0]

    def download_batch(self, requests, start_time_s: float | None = None, *,
                       deadline_s: float | None = None) -> list[DownloadResult]:
        """Queue a complete GOP at once, then advance shared service to idle/limit.

        Results retain submission order and each stream's own completion time.
        At exhaustion, active streams retain partial bytes; unadmitted streams
        are explicitly cancelled. Validate the whole batch before submission.
        """
        if self.failed:
            raise ValueError("The network link is blocked after an incomplete request")
        requests = tuple(requests)
        existing = {row.request_id for row in self.results}
        normalized = []
        for i, request in enumerate(requests):
            key = request.get("request_id")
            key = str(key if key is not None else f"request-{len(self.results) + i}")
            size = request["requested_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("Transfer size must be a nonnegative integer")
            if key in existing:
                raise ValueError("Transfer keys must be unique")
            existing.add(key)
            normalized.append((key, size, str(request.get("kind", "media"))))
        if start_time_s is not None:
            self.advance_to(start_time_s)
        if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s < self.now):
            raise ValueError("Download deadline_s must be finite and at or after the current clock")
        transfers = [self._link.submit(key, size, self.now) for key, size, _ in normalized]
        status = "complete"
        # Stop/error both expose the known finite service boundary. A hold or
        # independently looped trace is intentionally allowed to continue.
        horizon = (self.trace.duration_s if not self.trace.loop and self.trace.end_policy != "hold"
                   and self.trace.duration_s is not None else math.inf)
        limit = min(horizon, deadline_s if deadline_s is not None else math.inf)
        while not self._link.idle:
            if self.now >= limit:
                status = "trace_exhausted" if horizon <= limit else "time_limit"
                break
            if self.trace.loop and not any(rate > 0 for _, rate in self.trace.points) and any(
                    t.remaining_bytes for t in transfers):
                status = "no_future_service"
                break
            target = min(limit, self._link.next_event_time())
            if not math.isfinite(target):
                status = "no_future_service"
                break
            self._link.advance(target)
        if not self._link.idle:
            self.failed = True
            self.failure_status = status
        results = []
        for transfer, (_, _, kind) in zip(transfers, normalized):
            end = transfer.end_s if transfer.end_s is not None else self.now
            results.append(DownloadResult(
                request_id=transfer.key, kind=kind, requested_bytes=transfer.bytes,
                delivered_bytes=transfer.transferred_bytes, queued_s=transfer.queued_s,
                start_s=transfer.start_s, ready_s=transfer.ready_s, end_s=end,
                completed_at_s=transfer.end_s,
                capacity_bytes=(self.capacity_between(transfer.start_s, end)
                                if transfer.start_s is not None else 0.),
                status=("complete" if transfer.end_s is not None else
                        "cancelled_before_service" if transfer.start_s is None else status)))
        self.results.extend(results)
        return results

    def latency_reserve(self, request_count: int) -> float:
        """Conservative bound on connection-wide time with no ready body.

        While the queue is nonempty, all M slots are occupied: global idle time
        is at most the sum of K response delays divided by M. Once the last
        stream opens, all remaining response delays expire within one L.
        For K <= M all delays overlap exactly. Body-service time is additional.
        """
        if request_count == 0:
            return 0.
        if request_count <= self.max_concurrent_streams:
            return self.latency_s
        return self.latency_s * (1 + request_count / self.max_concurrent_streams)

    def expected_duration(self, sizes, body_bps: float) -> float:
        """Exact same scheduler, constant *causal estimate*, no trace lookahead."""
        if not math.isfinite(body_bps) or body_bps <= 0:
            raise ValueError("Predicted body throughput must be finite and positive")
        predicted = NetworkLink(NetworkTrace([(0., body_bps / 1e6)]),
            latency_s=self.latency_s, max_concurrent_streams=self.max_concurrent_streams)
        predicted.download_batch([dict(requested_bytes=size) for size in sizes])
        return predicted.now


class CausalBandwidthEstimator:
    """Separate harmonic body-throughput and application-goodput histories.

    Body throughput aggregates *all* completed stream bytes over the union of
    [ready,end] intervals, including trace outages but excluding response delay.
    Summing individual stream durations would undercount shared-link throughput.
    """

    def __init__(self, startup_bps: float = 20e6, window: int = 5, stale_s: float = 30.0):
        self.predictor = HarmonicPredictor(startup_bps=startup_bps, window=window, stale_s=stale_s)
        self.goodput_predictor = HarmonicPredictor(startup_bps=startup_bps, window=window, stale_s=stale_s)
        self._now = 0.0
        self._observed_ids: set[str] = set()

    @property
    def samples(self) -> list[dict]:
        return self.predictor.samples

    def snapshot(self, now: float) -> dict:
        self._advance_clock(now)
        body, goodput = self.predictor.snapshot(now), self.goodput_predictor.snapshot(now)
        return {**body, "body_bps": body["bps"], "application_goodput_bps": goodput["bps"],
                "measurement": "completed_batch_body_interval_union",
                "application_goodput_estimate": goodput}

    def predict(self, now: float) -> float:
        return self.snapshot(now)["bps"]

    def _advance_clock(self, now: float) -> None:
        if not math.isfinite(now) or now < self._now - 1e-9:
            raise ValueError("Estimator observation time must be finite and nondecreasing")
        self._now = max(self._now, float(now))

    def observe(self, result: DownloadResult, now: float | None = None) -> dict | None:
        received_at = result.end_s if now is None else now
        if result.completed_at_s is None or result.status != "complete":
            self._advance_clock(received_at)
            return None
        return self.observe_batch([result], now=received_at)

    def observe_batch(self, results, now: float | None = None) -> dict | None:
        """Observe one fully completed scene GOP as one sample of each metric.

        The body predictor uses ready-to-completion intervals; the application
        predictor uses admitted-to-completion intervals, including latency. Validation
        precedes mutation: a partial, future, or repeated member rejects the
        entire batch and cannot leak an observation into a later decision.
        """
        results = tuple(results)
        received_at = (max((row.end_s for row in results), default=self._now)
                       if now is None else now)
        if not math.isfinite(received_at) or received_at < self._now - 1e-9:
            raise ValueError("Estimator observation time must be finite and nondecreasing")
        identifiers = set()
        for row in results:
            if row.completed_at_s is None or row.status != "complete":
                raise ValueError("Bandwidth batches require complete transfers; partial delivery is not a sample")
            if row.completed_at_s > received_at + 1e-9 or row.end_s > received_at + 1e-9:
                raise ValueError("Cannot observe a future transfer")
            if row.request_id in identifiers or row.request_id in self._observed_ids:
                raise ValueError("Transfer observations must not be counted twice")
            if (row.start_s is None or row.ready_s is None or
                    any(not math.isfinite(t) for t in (row.queued_s, row.start_s, row.ready_s, row.end_s)) or
                    not 0 <= row.queued_s <= row.start_s <= row.ready_s <= row.end_s or
                    row.end_s != row.completed_at_s or not math.isfinite(row.delivered_bytes) or
                    row.delivered_bytes != row.requested_bytes or row.requested_bytes < 0 or
                    (row.requested_bytes > 0 and row.end_s <= row.ready_s)):
                raise ValueError("Invalid completed transfer bytes or intervals")
            identifiers.add(row.request_id)
        transfers = [Transfer(row.request_id, row.requested_bytes, row.queued_s,
                              start_s=row.start_s, ready_s=row.ready_s,
                              end_s=row.completed_at_s, transferred_bytes=row.delivered_bytes)
                     for row in results]
        bodies = [Transfer(row.request_id, row.requested_bytes, row.queued_s,
                           start_s=row.ready_s, ready_s=row.ready_s,
                           end_s=row.completed_at_s, transferred_bytes=row.delivered_bytes)
                  for row in results]
        sample = self.predictor.observe(bodies, received_at, flush=True)
        goodput = self.goodput_predictor.observe(transfers, received_at, flush=True)
        if sample is not None:
            sample.update(body_duration_s=sample["duration_s"],
                application_duration_s=goodput["duration_s"],
                observed_body_bps=sample["observed_bps"],
                observed_goodput_bps=goodput["observed_bps"],
                policy="aggregate completed bytes / union body intervals; separate application goodput")
        self._advance_clock(received_at)
        self._observed_ids.update(identifiers)
        return sample
