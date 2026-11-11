"""Local HTTP origin with optional shared, time-varying delivery pacing.

The ordinary localhost demo uses HTTP without an artificial rate limit. An
explicit Mbps rate or schedule enables the existing emulation mode: all response
bodies share one paced byte scheduler, charging chunks before writing so parallel
requests cannot bypass the configured throughput. Both modes use the same client
Request API and preserve transfer observations.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class RatePoint:
    at_s: float
    mbps: float


class PacedHTTPOrigin:
    """Serve a publication; ``mbps=None, schedule=None`` is unthrottled.

    The historic class name is retained for callers running network experiments.
    ``schedule`` remains a tuple of rate points; an empty tuple denotes the
    unthrottled mode, rather than a synthetic high-bandwidth rate.
    """

    def __init__(self, root, mbps=None, *, schedule=None, chunk_bytes=16 * 1024,
                 host="127.0.0.1", port=0, faults=None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Origin root must be an existing directory")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self.chunk_bytes = chunk_bytes
        self.host, self.port = host, port
        self._server = None
        self._tasks = set()
        self._writers = set()
        self._lock = asyncio.Lock()
        self._epoch = time.monotonic()
        self.requests = []
        self.faults = faults or {}
        self.set_schedule(schedule if schedule is not None else
                          ([RatePoint(0.0, mbps)] if mbps is not None else None))

    def set_schedule(self, points):
        if points is None:
            self.schedule = ()
            self._epoch = time.monotonic()
            return
        points = tuple(p if isinstance(p, RatePoint) else RatePoint(**p) for p in points)
        if not points or points[0].at_s != 0:
            raise ValueError("Delivery schedule must start at 0 seconds")
        for point in points:
            if (not math.isfinite(point.at_s) or point.at_s < 0 or
                    not math.isfinite(point.mbps) or point.mbps <= 0):
                raise ValueError("Delivery schedule needs finite times and positive rates")
        if any(a.at_s >= b.at_s for a, b in zip(points, points[1:])):
            raise ValueError("Delivery schedule times must strictly increase")
        self.schedule = points
        self._epoch = time.monotonic()

    def set_rate(self, mbps):
        self.set_schedule([RatePoint(0.0, mbps)] if mbps is not None else None)

    @property
    def unthrottled(self):
        return not self.schedule

    def _rate(self, now):
        elapsed = now - self._epoch
        index = max(i for i, p in enumerate(self.schedule) if p.at_s <= elapsed)
        point = self.schedule[index]
        boundary = (self._epoch + self.schedule[index + 1].at_s
                    if index + 1 < len(self.schedule) else math.inf)
        return point.mbps * 1e6 / 8, boundary

    async def _charge(self, size):
        if self.unthrottled:
            return
        async with self._lock:
            remaining = float(size)
            while remaining > 1e-6:
                if self.unthrottled:
                    return
                now = time.monotonic()
                rate, boundary = self._rate(now)
                interval = min(remaining / rate, max(0.0, boundary - now))
                if interval <= 0:
                    await asyncio.sleep(0)
                    continue
                # Recheck rate at most every 20 ms so an explicit set_rate()
                # affects an in-flight response as well as later requests.
                interval = min(interval, 0.02)
                await asyncio.sleep(interval)
                remaining -= min(interval, max(0.0, time.monotonic() - now)) * rate

    @property
    def base_url(self):
        if self._server is None:
            raise RuntimeError("HTTP origin is not running")
        return f"http://{self.host}:{self._server.sockets[0].getsockname()[1]}/"

    async def __aenter__(self):
        self._epoch = time.monotonic()
        self._server = await asyncio.start_server(self._serve, self.host, self.port)
        return self

    async def __aexit__(self, *_):
        self._server.close()
        await self._server.wait_closed()
        for writer in tuple(self._writers):
            writer.close()
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._server = None

    async def _serve(self, reader, writer):
        task = asyncio.current_task()
        self._tasks.add(task)
        self._writers.add(writer)
        observation = None
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            lines = raw.decode("iso-8859-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {k.lower(): v.strip() for k, v in
                       (line.split(":", 1) for line in lines[1:] if line)}
            relative = unquote(urlsplit(target).path).lstrip("/")
            path = (self.root / relative).resolve()
            status = 200 if path.is_relative_to(self.root) and path.is_file() else 404
            fault = self.faults.get(relative, {})
            status = int(fault.get("status", status))
            if method not in ("GET", "HEAD"):
                status = 405
            start, end = 0, path.stat().st_size - 1 if status == 200 else -1
            extra = {}
            if status == 200 and "range" in headers:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", headers["range"])
                if match:
                    start = int(match[1])
                    end = min(end, int(match[2]) if match[2] else end)
                if match is None or not 0 <= start <= end < path.stat().st_size:
                    status = 416
                else:
                    status = 206
                    extra["Content-Range"] = f"bytes {start}-{end}/{path.stat().st_size}"
            payload = b"HTTP delivery failure\n" if status not in (200, 206) else None
            size = len(payload) if payload is not None else end - start + 1
            response = {"Content-Length": str(size), "Connection": "close",
                        "Content-Type": "application/octet-stream", "Accept-Ranges": "bytes", **extra}
            writer.write((f"HTTP/1.1 {status} Response\r\n" + "".join(
                f"{k}: {v}\r\n" for k, v in response.items()) + "\r\n").encode("ascii"))
            await writer.drain()
            observation = {"path": relative, "range": headers.get("range"),
                           "status": status, "expected_bytes": size, "sent_bytes": 0,
                           "start_time": time.monotonic()}
            self.requests.append(observation)
            if method == "HEAD":
                return
            limit = min(size, int(fault.get("truncate_after", size)))
            stream = path.open("rb") if payload is None else None
            try:
                if stream is not None:
                    stream.seek(start)
                while observation["sent_bytes"] < limit:
                    if self.unthrottled:
                        quantum = 1024**2
                    else:
                        rate, _ = self._rate(time.monotonic())
                        # Millisecond timer granularity must not cap a fast link
                        # at one tiny chunk per wakeup. Charge about 10 ms of
                        # bytes, bounded to 1 MiB; low-rate tests keep small chunks.
                        quantum = min(1024**2, max(self.chunk_bytes, int(rate * 0.01)))
                    amount = min(quantum, limit - observation["sent_bytes"])
                    chunk = stream.read(amount) if stream is not None else payload[
                        observation["sent_bytes"]:observation["sent_bytes"] + amount]
                    if not chunk:
                        break
                    if not self.unthrottled:
                        await self._charge(len(chunk))
                    writer.write(chunk)
                    await writer.drain()
                    observation["sent_bytes"] += len(chunk)
            finally:
                if stream is not None:
                    stream.close()
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                asyncio.TimeoutError):
            pass
        finally:
            if observation is not None:
                observation["end_time"] = time.monotonic()
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self._writers.discard(writer)
            self._tasks.discard(task)
