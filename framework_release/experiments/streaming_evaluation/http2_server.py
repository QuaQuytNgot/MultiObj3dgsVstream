"""Local TLS HTTP/2 resource origin and aggregate application egress shaper.

The shaper gates bytes passed to ASGI, across *all* streams. It does not change
kernel interfaces, TCP congestion control, or count HTTP/TLS framing bytes.
Each service tick spends only the trace integral since the preceding tick;
unused/idle capacity is discarded. This is intentionally distinct from a
kernel bottleneck, and achievable throughput can be lower than the trace.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import time


@dataclass
class _Demand:
    stream: str
    amount: int
    future: asyncio.Future


def fair_byte_grants(demands: list[int], capacity_bytes: int) -> list[int]:
    """Deterministic max-min fair integer grants with no capacity creation."""
    if capacity_bytes < 0 or any(n < 0 for n in demands):
        raise ValueError("Negative byte demand/capacity")
    grants = [0] * len(demands)
    active = [i for i, n in enumerate(demands) if n]
    remaining = int(capacity_bytes)
    while active and remaining:
        small = [i for i in active if demands[i] <= remaining // len(active)]
        if small:
            for index in small:
                grants[index] = demands[index]
                remaining -= demands[index]
            active = [i for i in active if i not in small]
            continue
        share, extra = divmod(remaining, len(active))
        for position, index in enumerate(active):
            grants[index] = share + (position < extra)
        break
    return grants


class AggregateTraceShaper:
    """One nonbanking trace service clock and fair queue for every response."""

    def __init__(self, trace, *, tick_s: float = .002):
        if not math.isfinite(tick_s) or tick_s <= 0:
            raise ValueError("shaping tick_s must be finite and positive")
        if trace.unthrottled:
            raise ValueError("real_http2 requires an explicit aggregate trace")
        self.trace, self.tick_s = trace, float(tick_s)
        self.origin: float | None = None
        self.pending: OrderedDict[str, _Demand] = OrderedDict()
        self.ledger: list[dict] = []
        self._task: asyncio.Task | None = None
        self._last_s: float | None = None
        self.released_bytes = 0
        self.pump_starts = 0

    def start(self, origin: float) -> None:
        if self.origin is not None:
            raise RuntimeError("Trace clock cannot be reset")
        self.origin = origin

    def now(self) -> float:
        if self.origin is None:
            raise RuntimeError("Trace shaper has not started")
        return max(0., time.monotonic() - self.origin)

    def capacity(self, start: float, end: float) -> float:
        # error-ended traces must stop cleanly at their finite service horizon.
        if not self.trace.loop and self.trace.duration_s is not None and self.trace.end_policy != "hold":
            horizon = self.trace.duration_s
            start, end = min(start, horizon), min(end, horizon)
        return self.trace.integral_bits(start, end) / 8

    async def acquire(self, stream: str, amount: int) -> int:
        if amount <= 0:
            return 0
        if stream in self.pending:
            raise RuntimeError("A stream may have only one outstanding body grant")
        future = asyncio.get_running_loop().create_future()
        self.pending[stream] = _Demand(stream, amount, future)
        if self._task is None or self._task.done():
            # Never accrue idle trace credit before a ready body appears.
            self._last_s = self.now()
            self.pump_starts += 1
            self._task = asyncio.create_task(self._pump())
        try:
            return await future
        finally:
            current = self.pending.get(stream)
            if current is not None and current.future is future:
                self.pending.pop(stream, None)

    async def _pump(self) -> None:
        while self.pending:
            await asyncio.sleep(self.tick_s)
            end = self.now()
            start = self._last_s
            self._last_s = end
            demands = [d for d in self.pending.values() if not d.future.done()]
            for key, demand in tuple(self.pending.items()):
                if demand.future.done():
                    self.pending.pop(key, None)
            capacity = self.capacity(start, end)
            exhausted = (not self.trace.loop and self.trace.duration_s is not None and
                         end >= self.trace.duration_s and self.trace.end_policy != "hold")
            # A delayed event-loop tick must never release pre-boundary credit
            # after the finite trace has ended. Discard that last unused credit.
            grants = fair_byte_grants([d.amount for d in demands], 0 if exhausted else math.floor(capacity))
            delivered = sum(grants)
            self.ledger.append(dict(start_s=start, end_s=end, capacity_bits=capacity * 8,
                delivered_bytes=delivered, active_bodies=len(demands),
                tick_overshoot_s=max(0., end - start - self.tick_s),
                demanded_bytes=sum(d.amount for d in demands),
                accounting="actual_ASGI_release_grants_not_wire_or_simulated_delivery"))
            self.released_bytes += delivered
            for demand, grant in zip(demands, grants):
                if grant:
                    self.pending.pop(demand.stream, None)
                    demand.future.set_result(grant)
            # Allow each real response to submit its next body demand before
            # deciding whether the shared link has become idle.
            await asyncio.sleep(0)
            if exhausted:
                for demand in self.pending.values():
                    if not demand.future.done():
                        demand.future.set_exception(EOFError("Network trace exhausted"))
                self.pending.clear()

    async def close(self) -> None:
        for demand in self.pending.values():
            demand.future.cancel()
        self.pending.clear()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class UnshapedBodyService(AggregateTraceShaper):
    """Explicit localhost control: no trace pacing or finite capacity claim."""

    def __init__(self, trace, *, tick_s=.002):
        # Retain the same clock/accounting API, never consult trace capacity.
        self.trace, self.tick_s = trace, tick_s
        self.origin = None
        self.pending = OrderedDict()
        self.ledger = []
        self._task = None
        self._last_s = None
        self.released_bytes = self.pump_starts = 0

    def capacity(self, start, end):
        return math.inf

    async def acquire(self, stream, amount):
        self.released_bytes += amount
        return amount


class ResourceOrigin:
    """ASGI origin, serving only explicitly registered existing resource files."""

    def __init__(self, resources: dict[str, Path], shaper: AggregateTraceShaper,
                 *, latency_s: float = .02, chunk_bytes: int = 1048576):
        self.resources, self.shaper = resources, shaper
        self.latency_s, self.chunk_bytes = latency_s, chunk_bytes
        self.events: list[dict] = []
        self.active = 0
        self.max_open_streams = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        if scope["path"] == "/_warmup":
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-length", b"0"), (b"cache-control", b"no-store") ]})
            await send({"type": "http.response.body", "body": b""})
            return
        resource = self.resources.get(scope["path"].removeprefix("/resources/"))
        headers = dict(scope["headers"])
        request_id = headers.get(b"x-experiment-request-id", b"missing").decode("utf-8")
        if resource is None or not scope["path"].startswith("/resources/") or scope["method"] != "GET":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        event = dict(request_id=request_id, resource=str(resource), start_s=self.shaper.now(),
                     client=list(scope["client"]), http_version=scope["http_version"],
                     bytes_released=0, completed=False, grant_wait_s=0., file_read_wall_s=0.,
                     body_send_await_s=0., max_body_send_await_s=0., asgi_body_calls=0)
        self.active += 1
        self.max_open_streams = max(self.max_open_streams, self.active)
        self.events.append(event)

        async def watch_disconnect():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    event["disconnected"] = True
                    return

        async def send_resource():
            await asyncio.sleep(self.latency_s)
            event["ready_s"] = self.shaper.now()
            event["latency_overshoot_s"] = max(0., event["ready_s"] - event["start_s"] - self.latency_s)
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-length", str(resource.stat().st_size).encode()),
                (b"content-type", b"application/octet-stream"),
                (b"cache-control", b"no-store"),
                (b"x-server-client-port", str(scope["client"][1]).encode()),
                (b"x-server-ready-s", repr(event["ready_s"]).encode()),
                (b"x-server-open-streams", str(self.active).encode()),
            ]})
            with resource.open("rb") as source:
                remaining = resource.stat().st_size
                while remaining:
                    grant_start = time.monotonic()
                    try:
                        grant = await self.shaper.acquire(request_id, min(self.chunk_bytes, remaining))
                    except EOFError:
                        # An incomplete HTTP body cannot become a successful
                        # response. The real client cancels at the same finite
                        # horizon; wait for its disconnect rather than emitting
                        # an invalid end-of-body or an unhandled ASGI exception.
                        event["trace_exhausted"] = True
                        await asyncio.Future()
                    event["grant_wait_s"] += time.monotonic() - grant_start
                    read_start = time.monotonic()
                    data = source.read(grant)
                    event["file_read_wall_s"] += time.monotonic() - read_start
                    if len(data) != grant:
                        raise RuntimeError("Resource changed during its HTTP response")
                    send_start = time.monotonic()
                    await send({"type": "http.response.body", "body": data, "more_body": True})
                    send_wait = time.monotonic() - send_start
                    event["body_send_await_s"] += send_wait
                    event["max_body_send_await_s"] = max(event["max_body_send_await_s"], send_wait)
                    event["asgi_body_calls"] += 1
                    event["bytes_released"] += len(data)
                    remaining -= len(data)
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            event["completed"] = True

        body_task = asyncio.create_task(send_resource())
        disconnect_task = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait((body_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED)
            if body_task in done:
                await body_task
        finally:
            body_task.cancel()
            disconnect_task.cancel()
            await asyncio.gather(body_task, disconnect_task, return_exceptions=True)
            event["end_s"] = self.shaper.now()
            self.active -= 1


def create_local_certificate(directory: Path) -> tuple[Path, Path]:
    """Ephemeral trusted localhost certificate; private key stays in temp dir."""
    from datetime import datetime, timedelta, timezone
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256()))
    certificate_path, key_path = directory / "certificate.pem", directory / "private-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    return certificate_path, key_path
