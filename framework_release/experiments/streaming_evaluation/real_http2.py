"""Actual httpx HTTP/2 requests against a local TLS/ALPN Hypercorn origin.

This transport retains the synchronous experiment Request API while a private
event-loop thread owns one persistent AsyncClient and one server. Completions,
TTFB and bytes come from real socket I/O and monotonic wall time. The optional
experiment trace is always enforced as aggregate *application egress* shaping,
never advertised as kernel/link shaping or unshaped localhost performance.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import time
from urllib.parse import quote

from .http2_server import AggregateTraceShaper, UnshapedBodyService, ResourceOrigin, create_local_certificate
from .network import DownloadResult, NetworkLink


@dataclass(frozen=True)
class RealDownloadResult(DownloadResult):
    http_version: str | None = None
    alpn_protocol: str | None = None
    stream_id: int | None = None
    connection_id: str | None = None
    server_client_port: int | None = None
    first_body_s: float | None = None
    server_ready_s: float | None = None
    server_open_streams: int | None = None

    def to_dict(self) -> dict:
        row = asdict(self)
        body_duration = max(0., self.end_s - self.ready_s) if self.ready_s is not None else 0.
        application_duration = self.duration_s
        return {**row,
            "queue_duration_s": (self.start_s if self.start_s is not None else self.end_s) - self.queued_s,
            "latency_duration_s": (max(0., (self.ready_s or self.end_s) - self.start_s)
                                   if self.start_s is not None else 0.),
            "body_duration_s": body_duration,
            "ttfb_s": (self.first_body_s - self.start_s if self.first_body_s is not None and
                       self.start_s is not None else None),
            "response_headers_ttfb_s": (self.ready_s - self.start_s if self.ready_s is not None and
                                       self.start_s is not None else None),
            "body_throughput_bps": self.delivered_bytes * 8 / body_duration if body_duration else None,
            "application_goodput_bps": self.delivered_bytes * 8 / application_duration
                                       if application_duration else None,
            "transport": "real_http2",
        }


class RealHTTP2Link:
    """One negotiated h2 connection; FIFO admission to bounded actual streams.

    TLS/ALPN warmup precedes the trace clock. Its cost is reported separately in
    metadata; subsequent experiment planning and waits consume real trace time.
    No reconnect, HTTP/1.1 fallback or synthetic request completion is accepted.
    """

    def __init__(self, trace, resource_paths: dict[str, str | Path], *, latency_s: float = .02,
                 max_concurrent_streams: int = 16, shaping_tick_s: float = .002,
                 chunk_bytes: int = 1048576, max_requests_per_connection: int = 1000000,
                 shaping_enabled: bool = True, diagnostics: bool = False,
                 profile_path: str | Path | None = None, flow_probe: bool = False):
        if isinstance(max_concurrent_streams, bool) or not isinstance(max_concurrent_streams, int) or max_concurrent_streams < 1:
            raise ValueError("max_concurrent_streams must be a positive integer")
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError("request latency must be finite and nonnegative")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ValueError("chunk_bytes must be a positive integer")
        if (isinstance(max_requests_per_connection, bool) or not isinstance(max_requests_per_connection,int)
                or max_requests_per_connection < 1):
            raise ValueError("max_requests_per_connection must be a positive integer")
        self.trace, self.latency_s = trace, float(latency_s)
        self.max_concurrent_streams = max_concurrent_streams
        self.max_requests_per_connection = max_requests_per_connection
        if not isinstance(shaping_enabled, bool):
            raise ValueError("shaping_enabled must be boolean")
        self.shaping_enabled = shaping_enabled
        self._diagnostics_enabled = diagnostics or profile_path is not None or flow_probe
        self._profile_path, self._flow_probe = profile_path, flow_probe
        self.resources = {str(key): Path(path).resolve() for key, path in resource_paths.items()}
        for path in self.resources.values():
            if not path.is_file():
                raise ValueError(f"HTTP/2 resource does not exist: {path}")
        self.shaper = (AggregateTraceShaper if shaping_enabled else UnshapedBodyService)(trace, tick_s=shaping_tick_s)
        self.origin_app = ResourceOrigin(self.resources, self.shaper,
            latency_s=self.latency_s, chunk_bytes=chunk_bytes)
        self.results: list[RealDownloadResult] = []
        self.failed, self.failure_status = False, None
        self._origin = None
        self._connection = None
        self._stream_ids: set[int] = set()
        self._closed = False
        self._temporary = tempfile.TemporaryDirectory(prefix="streaming-real-http2-")
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="streaming-real-http2", daemon=True)
        self._thread.start()
        self._loop_ready.wait()
        setup_start = time.monotonic()
        try:
            self._call(self._start())
            self.setup_wall_s = time.monotonic() - setup_start
        except BaseException:
            self.close()
            raise

    def _run_loop(self):
        # The loop and its wakeup descriptor belong to this thread. In
        # particular, do not create the selector loop on the caller's thread.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def _call(self, awaitable):
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop).result()

    async def _start(self):
        import httpx
        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        certificate, private_key = create_local_certificate(Path(self._temporary.name))
        config = Config()
        config.certfile, config.keyfile = str(certificate), str(private_key)
        config.alpn_protocols = ["h2"]
        config.h2_max_concurrent_streams = self.max_concurrent_streams
        config.keep_alive_timeout = 3600
        # Hypercorn otherwise recycles h2 after1000 requests (including warmup).
        # A single N15/10s case already has1275 resources; reconnecting would
        # invalidate the requested persistent-connection experiment contract.
        config.keep_alive_max_requests = self.max_requests_per_connection
        config.graceful_timeout = 1
        config.accesslog = None
        config.errorlog = None
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        # Transfer the already-bound descriptor to Hypercorn without a free-port
        # race; Hypercorn wraps/owns this descriptor via its public fd:// bind.
        config.bind = [f"fd://{listener.detach()}"]
        self._shutdown = asyncio.Event()
        self._server_task = asyncio.create_task(serve(self.origin_app, config,
            shutdown_trigger=self._shutdown.wait, mode="asgi"))
        self.base_url = f"https://127.0.0.1:{port}"
        context = ssl.create_default_context(cafile=str(certificate))
        self._client = httpx.AsyncClient(http2=True, http1=False, verify=context, trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=None),
            timeout=None)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_streams)
        # Establish TLS once, then retain the same actual network stream for all
        # resource requests. Startup attempts perform no experiment transfers.
        warmup_start = time.monotonic()
        for attempt in range(100):
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("HTTP/2 server stopped during startup")
            try:
                response = await asyncio.wait_for(self._client.get(self.base_url + "/_warmup"), timeout=5.)
                response.raise_for_status()
                self._check_protocol(response)
                break
            except httpx.ConnectError:
                if attempt == 99:
                    raise
                await asyncio.sleep(.01)
        self.warmup_wall_s = time.monotonic() - warmup_start
        self._origin = time.monotonic()
        self.shaper.start(self._origin)
        if self._diagnostics_enabled:
            from .http2_profiling import HTTP2Profiler
            self.profiler = HTTP2Profiler(self._profile_path, flow_probe=self._flow_probe)
            await self.profiler.start()

    def _check_protocol(self, response) -> tuple[int, str, str]:
        if response.http_version != "HTTP/2":
            raise RuntimeError(f"HTTP/2 validation failed: negotiated {response.http_version}")
        stream = response.extensions.get("network_stream")
        tls = stream.get_extra_info("ssl_object") if stream is not None else None
        alpn = tls.selected_alpn_protocol() if tls is not None else None
        if alpn != "h2":
            raise RuntimeError(f"HTTP/2 validation failed: ALPN is {alpn!r}")
        stream_id = response.extensions.get("stream_id")
        if not isinstance(stream_id, int) or stream_id <= 0 or not stream_id % 2 or stream_id in self._stream_ids:
            raise RuntimeError(f"HTTP/2 validation failed: invalid/repeated client stream ID {stream_id!r}")
        if self._connection is None:
            self._connection = stream
        if stream is not self._connection:
            raise RuntimeError("HTTP/2 validation failed: persistent connection was replaced")
        self._stream_ids.add(stream_id)
        return stream_id, "persistent-connection-1", alpn

    @property
    def now(self) -> float:
        return max(0., time.monotonic() - self._origin) if self._origin is not None else 0.

    @property
    def ledger(self):
        return self.shaper.ledger

    @property
    def delivered_bytes(self) -> float:
        return math.fsum(row.delivered_bytes for row in self.results)

    @property
    def requested_bytes(self) -> int:
        return sum(row.requested_bytes for row in self.results)

    def capacity_between(self, start_s: float, end_s: float) -> float:
        return self.shaper.capacity(start_s, end_s)

    def advance_to(self, wall_time_s: float) -> None:
        if not math.isfinite(wall_time_s) or wall_time_s < 0:
            raise ValueError("Network wall time must be finite and nonnegative")
        # A target computed by the caller can already have passed while its
        # Python code ran. Never rewind or synthesize a later clock value.
        delay = wall_time_s - self.now
        if delay > 0:
            self._call(asyncio.sleep(delay))

    latency_reserve = NetworkLink.latency_reserve
    expected_duration = NetworkLink.expected_duration

    def download(self, size_bytes: int, start_time_s: float | None = None, *,
                 request_id: str | None = None, kind: str = "media", resource_sha256: str | None = None,
                 deadline_s: float | None = None):
        return self.download_batch([dict(requested_bytes=size_bytes, request_id=request_id,
            kind=kind, resource_sha256=resource_sha256)], start_time_s, deadline_s=deadline_s)[0]

    def download_batch(self, requests, start_time_s: float | None = None, *,
                       deadline_s: float | None = None) -> list[RealDownloadResult]:
        if self._closed:
            raise ValueError("Real HTTP/2 link is closed")
        if self.failed:
            raise ValueError("The network link is blocked after an incomplete request")
        normalized = []
        identifiers = {row.request_id for row in self.results}
        for index, original in enumerate(requests):
            row = dict(original)
            key = row.get("request_id")
            row["request_id"] = str(key if key is not None else f"request-{len(self.results) + index}")
            size, resource = row["requested_bytes"], row.get("resource_sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("Transfer size must be a nonnegative integer")
            if row["request_id"] in identifiers:
                raise ValueError("Transfer keys must be unique")
            identifiers.add(row["request_id"])
            if resource not in self.resources or self.resources[resource].stat().st_size != size:
                raise ValueError("Real request must identify an existing resource with the exact byte size")
            normalized.append(row)
        if start_time_s is not None:
            self.advance_to(start_time_s)
        if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s < 0):
            raise ValueError("Download deadline must be finite and nonnegative")
        results = self._call(self._download_batch(normalized, deadline_s))
        self.results.extend(results)
        if any(row.status != "complete" for row in results):
            self.failed = True
            self.failure_status = self._last_batch_failure_status
        return results

    async def _download_batch(self, requests, deadline_s):
        import httpx
        horizon = (self.trace.duration_s if self.shaping_enabled and not self.trace.loop and self.trace.duration_s is not None
                   and self.trace.end_policy != "hold" else math.inf)
        deadline = deadline_s if deadline_s is not None else math.inf
        limit = min(horizon, deadline)
        limit_status = "trace_exhausted" if horizon <= deadline else "time_limit"
        queued = self.now
        states = [dict(request_id=row["request_id"], kind=str(row.get("kind", "media")),
            requested_bytes=row["requested_bytes"], delivered_bytes=0, queued_s=queued,
            start_s=None, ready_s=None, end_s=queued, completed_at_s=None, capacity_bytes=0.,
            status="cancelled_before_service") for row in requests]

        async def fetch(row, state):
            try:
                async with self._semaphore:
                    if self.now >= limit:
                        return
                    state["start_s"] = self.now
                    state["status"] = limit_status
                    async with self._client.stream("GET", self.base_url + "/resources/" +
                            quote(row["resource_sha256"], safe=""),
                            headers={"x-experiment-request-id": row["request_id"]}) as response:
                        state["ready_s"] = self.now
                        state["http_version"] = response.http_version
                        state["stream_id"], state["connection_id"], state["alpn_protocol"] = self._check_protocol(response)
                        response.raise_for_status()
                        if int(response.headers.get("content-length", -1)) != row["requested_bytes"]:
                            raise RuntimeError("HTTP/2 Content-Length disagrees with requested resource bytes")
                        state["server_ready_s"] = float(response.headers["x-server-ready-s"])
                        state["server_client_port"] = int(response.headers["x-server-client-port"])
                        state["server_open_streams"] = int(response.headers["x-server-open-streams"])
                        async for chunk in response.aiter_raw():
                            if chunk and state.get("first_body_s") is None:
                                state["first_body_s"] = self.now
                            state["delivered_bytes"] += len(chunk)
                    if state["delivered_bytes"] != row["requested_bytes"]:
                        raise RuntimeError("HTTP/2 body byte count disagrees with requested resource bytes")
                    state["status"] = "complete"
                    state["completed_at_s"] = self.now
            except (httpx.RemoteProtocolError, httpx.ReadError):
                # Only the finite service boundary explains an expected abort.
                # A protocol failure before it remains a hard validation error.
                if not math.isfinite(horizon) or self.now < horizon:
                    raise
                state["status"] = "trace_exhausted"
            finally:
                state["end_s"] = state["completed_at_s"] or self.now
                if state["start_s"] is not None:
                    state["capacity_bytes"] = self.capacity_between(state["start_s"], state["end_s"])

        tasks = [asyncio.create_task(fetch(row, state)) for row, state in zip(requests, states)]
        group = asyncio.gather(*tasks)
        try:
            if not math.isfinite(limit):
                await group
            else:
                await asyncio.wait_for(group, timeout=max(0., limit - self.now))
        except asyncio.TimeoutError:
            await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        self._last_batch_failure_status = (limit_status if any(state["status"] != "complete"
                                                              for state in states) else None)
        if self._last_batch_failure_status is not None:
            # httpcore 1.0.9 closes a response iterator without necessarily
            # sending RST_STREAM. A failed experiment link cannot be reused:
            # close its sole actual connection so ASGI receives disconnect and
            # cancels all body grants, rather than silently serving discarded
            # responses after the client's wall deadline.
            await self._client.aclose()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            async def wait_origin_idle():
                while self.origin_app.active:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_origin_idle(), timeout=1.)
        return [RealDownloadResult(**state) for state in states]

    def transport_metadata(self) -> dict:
        ports = sorted({row.server_client_port for row in self.results if row.server_client_port is not None})
        return dict(mode="real_http2", protocol_negotiated=True, http_version="HTTP/2", alpn="h2",
            client="httpx.AsyncClient(http2=True,http1=False)", server="Hypercorn TLS",
            connection_count=1 if self._connection is not None else 0, observed_server_client_ports=ports,
            stream_ids=sorted(self._stream_ids), max_concurrent_streams=self.max_concurrent_streams,
            server_keep_alive_max_requests=self.max_requests_per_connection,
            max_observed_server_streams=self.origin_app.max_open_streams,
            actual_multiplexing_verified=self.origin_app.max_open_streams > 1,
            request_latency_s=self.latency_s, clock="time.monotonic actual wall time",
            tls_setup_and_warmup_s=getattr(self, "setup_wall_s", None),
            warmup_request_s=getattr(self, "warmup_wall_s", None),
            startup_clock="warm TLS connection; setup separately reported; planning consumes trace time",
            shaping=("aggregate_server_application_egress_trace_pacing" if self.shaping_enabled
                     else "explicit_unshaped_localhost_control_no_trace_capacity_claim"),
            shaping_enabled=self.shaping_enabled,
            shaping_tick_s=self.shaper.tick_s, kernel_link_shaping=False,
            shaping_chunk_bytes=self.origin_app.chunk_bytes,
            shaping_granularity="actual tick-integral release; per-stream grant bounded by chunk_bytes",
            shaping_bytes="resource body only; HTTP/2 TLS TCP overhead excluded",
            shaping_idle_credit="discarded", shaping_sharing="max_min_fair_integer_bytes_per_tick",
            body_sharing="fair_application_release_grants; actual HTTP/2 flow control applies",
            flow_control="actual httpcore/h2 and Hypercorn defaults",
            server_execution="same experiment process; private asyncio thread",
            expected_duration_model="causal constant-rate fluid scheduler using completed real body samples",
            server_body_bytes_released=self.shaper.released_bytes,
            server_events=[dict(row) for row in self.origin_app.events])

    def diagnostics_snapshot(self):
        """End optional profiling after transfers; normal requests are unchanged."""
        if not self._diagnostics_enabled:
            return {}
        return self._call(self.profiler.stop())

    async def _close(self):
        if hasattr(self, "profiler"):
            await self.profiler.stop()
        if hasattr(self, "_client"):
            await self._client.aclose()
        await self.shaper.close()
        if hasattr(self, "_shutdown"):
            self._shutdown.set()
        if hasattr(self, "_server_task"):
            await asyncio.gather(self._server_task, return_exceptions=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._thread.is_alive():
                self._call(self._close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def validate_real_http2() -> dict:
    """Bounded independent protocol + aggregate shaping measurement (~2 s).

    This uses a small synthetic 8 Mbps trace and real payload files; it is a
    transport correctness check, never presented as a Lumos5G experiment.
    """
    from .network import CausalBandwidthEstimator, NetworkTrace
    with tempfile.TemporaryDirectory(prefix="streaming-h2-proof-") as directory:
        resource = Path(directory) / "existing-resource.bin"
        resource.write_bytes(bytes(range(256)) * 625)
        trace = NetworkTrace([(0., 8.), (.25, 2.), (.55, 8.)], duration_s=5., end_policy="stop")
        with RealHTTP2Link(trace, {"payload": resource}, max_concurrent_streams=4) as link:
            first = link.download_batch([dict(request_id=f"proof-{i}", resource_sha256="payload",
                requested_bytes=resource.stat().st_size) for i in range(4)], deadline_s=4.)
            second = link.download_batch([dict(request_id="proof-reuse", resource_sha256="payload",
                requested_bytes=resource.stat().st_size)], deadline_s=4.)
            rows = first + second
            metadata = link.transport_metadata()
            # Independent closed-form integration, not the shaper's helper or
            # its reported capacity field, validates each actual release tick.
            def analytical_bits(start, end):
                return sum(max(0., min(end, right) - max(start, left)) * rate
                    for left, right, rate in ((0., .25, 8e6), (.25, .55, 2e6), (.55, 5., 8e6)))
            checks = dict(
                all_requests_complete=all(row.status == "complete" for row in rows),
                negotiated_h2=all(row.http_version == "HTTP/2" and row.alpn_protocol == "h2" for row in rows),
                one_persistent_connection=len({row.server_client_port for row in rows}) == 1,
                unique_odd_stream_ids=len({row.stream_id for row in rows}) == len(rows) and
                    all(row.stream_id % 2 == 1 for row in rows),
                concurrent_stream_overlap=max(row.start_s for row in first) < min(row.end_s for row in first),
                bounded_streams=1 < metadata["max_observed_server_streams"] <= 4,
                exact_response_bytes=sum(row.delivered_bytes for row in rows) == 5 * resource.stat().st_size,
                per_tick_aggregate_capacity_conserved=all(
                    tick["delivered_bytes"] * 8 <= tick["capacity_bits"] + 1e-6 for tick in link.ledger),
                independent_analytical_tick_capacity_conserved=all(
                    tick["delivered_bytes"] * 8 <= analytical_bits(tick["start_s"], tick["end_s"]) + 1e-6
                    for tick in link.ledger),
                total_aggregate_capacity_conserved=link.delivered_bytes <= link.capacity_between(0., link.now),
                body_release_equals_received=link.shaper.released_bytes == link.delivered_bytes,
                rate_change_exercised=any(row.start_s < .55 < row.end_s for row in rows),
            )
            estimator = CausalBandwidthEstimator()
            sample = estimator.observe_batch(first, max(row.end_s for row in first))
            checks["body_goodput_separated"] = sample["observed_body_bps"] > sample["observed_goodput_bps"]
            result = dict(schema="streaming-evaluation.real-http2-proof.v1",
                status="passed" if all(checks.values()) else "failed", checks=checks,
                trace=dict(kind="synthetic_piecewise_shaping_validation", points=trace.points,
                           duration_s=trace.duration_s),
                analytical_minimum_8mbps_body_s=sum(row.delivered_bytes for row in first) * 8 / 8e6,
                first_batch_download_s=max(row.end_s for row in first) - min(row.start_s for row in first),
                first_batch_estimator_sample=sample, requests=[row.to_dict() for row in rows],
                transport=metadata, shaping_ledger=list(link.ledger))
    if result["status"] != "passed":
        raise RuntimeError(f"Real HTTP/2 proof failed: {result['checks']}")
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    proof = validate_real_http2()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": proof["status"], "checks": proof["checks"],
        "output": str(arguments.output), "first_batch_download_s": proof["first_batch_download_s"],
        "first_batch_estimator_sample": proof["first_batch_estimator_sample"]}, indent=2))
