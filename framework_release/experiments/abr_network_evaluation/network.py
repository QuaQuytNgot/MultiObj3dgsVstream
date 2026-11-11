"""Deterministic virtual-time HTTP body delivery over one shared bottleneck.

Trace rates use decimal Mbps. Transfers share body capacity equally while their
requests occupy bounded concurrency slots; fixed latency consumes time but no
body bytes. No wall clock, sockets, or payload files are involved.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


class NetworkTrace:
    """Piecewise-constant bandwidth, optionally repeated independently of media.

    A finite non-looped trace uses ``hold`` (last rate), ``stop`` (zero rate),
    or ``error`` after duration_s. Looping requires an explicit duration.
    """

    def __init__(self, points, duration_s=None, loop=False, end_policy="hold"):
        self.unthrottled = False
        self.points = tuple((float(t), float(r)) for t, r in points)
        if not self.points or self.points[0][0] != 0:
            raise ValueError("Network trace must begin at time zero")
        if any(not math.isfinite(t) or not math.isfinite(r) or t < 0 or r < 0
               for t, r in self.points):
            raise ValueError("Trace times and rates must be finite and nonnegative")
        if any(a[0] >= b[0] for a, b in zip(self.points, self.points[1:])):
            raise ValueError("Network trace timestamps must strictly increase")
        if end_policy not in {"hold", "stop", "error"}:
            raise ValueError("end_policy must be hold, stop, or error")
        if duration_s is not None:
            duration_s = float(duration_s)
            if not math.isfinite(duration_s) or duration_s <= 0 or duration_s < self.points[-1][0]:
                raise ValueError("Trace duration must be positive and cover all points")
        if loop and duration_s is None:
            raise ValueError("A repeated network trace requires duration_s")
        self.duration_s, self.loop, self.end_policy = duration_s, bool(loop), end_policy
        self._times = [p[0] for p in self.points]

    def _local(self, t):
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Trace time must be finite and nonnegative")
        if not self.loop:
            return t
        local = t % self.duration_s
        # Decimal trace periods (e.g. 1.1) can leave an exact repetition
        # boundary one ULP below the period after floating modulo.
        tolerance = max(1e-12, 4 * math.ulp(t))
        if local <= tolerance or self.duration_s - local <= tolerance:
            return 0.0
        return local

    def rate_bps(self, t):
        if self.unthrottled:return None
        local = self._local(t)
        if not self.loop and self.duration_s is not None and local >= self.duration_s:
            if self.end_policy == "stop":
                return 0.0
            if self.end_policy == "error":
                raise ValueError("Network trace exhausted")
        return self.points[bisect_right(self._times, local + 1e-12) - 1][1] * 1e6

    def next_boundary(self, t):
        local = self._local(t)
        index = bisect_right(self._times, local + 1e-12)
        if self.loop:
            origin = float(t) - local
            point = self._times[index] if index < len(self._times) else self.duration_s
            # A point exactly at the end belongs to the next repetition.
            return origin + min(point, self.duration_s)
        candidates = self._times[index:index + 1]
        if self.duration_s is not None and t < self.duration_s and self.end_policy != "hold":
            candidates = candidates + [self.duration_s]
        return min(candidates, default=math.inf)

    def integral_bits(self, start, end):
        if self.unthrottled:return None
        if not math.isfinite(end) or end < start:
            raise ValueError("Invalid trace integration interval")
        current, total = float(start), 0.0
        while current < end:
            stop = min(end, self.next_boundary(current))
            if stop <= current:
                raise RuntimeError("Network trace failed to advance")
            total += self.rate_bps(current) * (stop - current)
            current = stop
        return total

    @classmethod
    def from_config(cls, config, base_dir=None):
        config = dict(config)
        if config.get('mode')=='unthrottled':
            trace=cls([(0.,0.)]);trace.unthrottled=True;return trace
        path = config.get("path", config.get("trace_file"))
        if path is not None:
            path = Path(path)
            if not path.is_absolute():
                path = Path(base_dir or ".") / path
            if path.suffix.lower() == ".csv":
                with path.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                points = [(float(r.get("time_seconds", r.get("time_s", r.get("at_s", r.get("time"))))),
                           float(r.get("mbps", r.get("bandwidth_mbps")))) for r in rows]
            elif path.suffix.lower() == ".json":
                raw = json.loads(path.read_text())
                if isinstance(raw, dict):
                    for key in ("duration_s", "loop", "end_policy"):
                        if key in raw and key not in config:
                            config[key] = raw[key]
                    raw = raw["points"]
                points = cls._parse_points(raw)
            else:
                raise ValueError("Network traces must be CSV or JSON")
        elif "points" in config:
            points = cls._parse_points(config["points"])
        else:
            points = [(0.0, float(config.get("mbps", config.get("bandwidth_mbps", 20.0))))]
        return cls(points, duration_s=config.get("duration_s"),
                   loop=config.get("loop", config.get("network_trace_loop", False)),
                   end_policy=config.get("end_policy", "hold"))

    @staticmethod
    def _parse_points(raw):
        return [(p.get("time_s", p.get("at_s", p.get("time"))),
                 p.get("mbps", p.get("bandwidth_mbps"))) if isinstance(p, dict) else tuple(p)
                for p in raw]


@dataclass
class Transfer:
    key: str
    bytes: int
    queued_s: float
    start_s: float | None = None
    ready_s: float | None = None
    end_s: float | None = None
    transferred_bytes: float = 0.0

    @property
    def remaining_bytes(self):
        return max(0.0, self.bytes - self.transferred_bytes)


class SharedLink:
    """Work-conserving equal body-rate sharing with FIFO request admission."""

    def __init__(self, trace, concurrency=4, latency_s=0.02):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError("latency_s must be nonnegative")
        self.trace, self.concurrency, self.latency_s = trace, concurrency, float(latency_s)
        self.now = 0.0
        self._queue, self._active, self._keys = deque(), [], set()
        self.transfers = []
        self.ledger = []
        self.transferred_bytes = 0.0

    @property
    def idle(self):
        return not self._queue and not self._active

    @property
    def pending_count(self):
        return len(self._queue) + len(self._active)

    def submit(self, key, size, now):
        if abs(now - self.now) > 1e-8:
            raise ValueError("Advance the link before submitting at a new time")
        if key in self._keys:
            raise ValueError("Transfer keys must be unique")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Transfer size must be a nonnegative integer")
        self._keys.add(key)
        transfer = Transfer(key=str(key), bytes=size, queued_s=float(now))
        self._queue.append(transfer)
        self._admit()
        return transfer

    def _admit(self):
        while self._queue and len(self._active) < self.concurrency:
            transfer = self._queue.popleft()
            transfer.start_s = self.now
            transfer.ready_s = self.now + self.latency_s
            self._active.append(transfer)

    def _bodies(self):
        return [r for r in self._active if r.ready_s <= self.now + 1e-12]

    def _byte_tolerance(self):
        if self.trace.unthrottled:return 1e-7
        return max(1e-7,4*math.ulp(self.now)*self.trace.rate_bps(self.now)/8)

    def next_event_time(self, now=None):
        if now is not None and abs(now - self.now) > 1e-8:
            raise ValueError("next_event_time requires the current link time")
        if self.idle:
            return math.inf
        candidates = [r.ready_s for r in self._active if r.ready_s > self.now + 1e-12]
        bodies = self._bodies()
        if bodies:
            if self.trace.unthrottled:return self.now
            minimum = min(r.remaining_bytes for r in bodies)
            if minimum <= self._byte_tolerance():
                return self.now
            rate = self.trace.rate_bps(self.now) / 8
            if rate > 0:
                candidates.append(self.now + minimum * len(bodies) / rate)
            candidates.append(self.trace.next_boundary(self.now))
        return min(candidates, default=math.inf)

    def _finish(self):
        if self.trace.unthrottled:
            bodies=self._bodies()
            amount=sum(t.remaining_bytes for t in bodies)
            if bodies:
                for t in bodies:t.transferred_bytes=float(t.bytes)
                self.transferred_bytes+=amount
                self.ledger.append(dict(start_s=self.now,end_s=self.now,capacity_bits=None,
                    delivered_bytes=amount,active_bodies=len(bodies),unthrottled=True))
        done = [r for r in self._active
                if r.ready_s <= self.now + 1e-12 and r.remaining_bytes <= self._byte_tolerance()]
        for transfer in done:
            # Only unrepresentable virtual-time byte dust; never add it to the ledger.
            transfer.transferred_bytes = float(transfer.bytes)
            transfer.end_s = self.now
            self._active.remove(transfer)
            self.transfers.append(transfer)
        self._admit()
        return done

    def advance(self, to_time):
        if not math.isfinite(to_time) or to_time < self.now - 1e-9:
            raise ValueError("Virtual time must be finite and nondecreasing")
        target, done = max(self.now, float(to_time)), []
        done.extend(self._finish())
        while self.now < target:
            stop = min(target, self.next_event_time())
            if stop <= self.now:
                more = self._finish()
                if not more:
                    raise RuntimeError("Shared link event failed to advance")
                done.extend(more)
                continue
            bodies = self._bodies()
            capacity_bits = self.trace.integral_bits(self.now, stop)
            amount = capacity_bits / 8 / len(bodies) if bodies else 0.0
            delivered = 0.0
            for transfer in bodies:
                progress = min(transfer.remaining_bytes, amount)
                transfer.transferred_bytes += progress
                delivered += progress
            self.transferred_bytes += delivered
            self.ledger.append(dict(start_s=self.now, end_s=stop,
                                    capacity_bits=capacity_bits, delivered_bytes=delivered,
                                    active_bodies=len(bodies)))
            self.now = stop
            done.extend(self._finish())
        # Reaching an exact latency/completion event also admits FIFO successors.
        self.now = target
        done.extend(self._finish())
        return done
