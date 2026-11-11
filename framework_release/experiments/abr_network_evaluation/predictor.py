"""Standalone production-compatible harmonic completed-transfer estimator.

Samples aggregate concurrent transfer bytes over the union of request intervals,
including fixed request latency. Call record_batch only after all admitted work
is idle, before a new request batch. No trace or link capacity is visible here.
"""
from __future__ import annotations

from collections import deque
import math


def busy_interval_seconds(transfers):
    intervals = sorted((t.start_s, t.end_s) for t in transfers)
    if not intervals:
        return 0.0
    if any(a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b < a
           for a, b in intervals):
        raise ValueError("Bandwidth observations require valid completed intervals")
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


class HarmonicPredictor:
    def __init__(self, startup_bps=20e6, window=5, stale_s=30.0):
        if not math.isfinite(startup_bps) or startup_bps <= 0:
            raise ValueError("startup_bps must be finite and positive")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        if not math.isfinite(stale_s) or stale_s <= 0:
            raise ValueError("stale_s must be finite and positive")
        self.startup_bps, self.window, self.stale_s = float(startup_bps), window, float(stale_s)
        self._samples = deque(maxlen=window)
        self._pending = []
        self._seen = set()
        self.samples = []

    def observe(self, transfers, now, flush=False):
        """Accept completed observations; partial/future observations are rejected."""
        for transfer in transfers:
            if transfer.end_s is None or transfer.end_s > now + 1e-9:
                raise ValueError("Cannot observe an incomplete or future transfer")
            if transfer.key in self._seen:
                raise ValueError("Transfer observations must not be counted twice")
            if transfer.bytes < 0:
                raise ValueError("Negative observation bytes")
            self._seen.add(transfer.key)
            self._pending.append(transfer)
        return self.record_batch(now) if flush else None

    def record_batch(self, now):
        """Flush an idle request group into one harmonic sample."""
        if not self._pending:
            return None
        pending, self._pending = self._pending, []
        duration = busy_interval_seconds(pending)
        size = sum(t.bytes for t in pending)
        if size <= 0 or duration <= 0:
            return None
        rate = size * 8 / duration
        sample = dict(time_s=float(now), bytes=size, duration_s=duration,
                      observed_bps=rate, transfer_count=len(pending),
                      policy="aggregate bytes / union busy request intervals including latency")
        self._samples.append((rate, float(now)))
        self.samples.append(sample)
        return sample

    def snapshot(self, now):
        fresh = [(rate, t) for rate, t in self._samples if max(0.0, now - t) <= self.stale_s]
        age = max(0.0, now - self._samples[-1][1]) if self._samples else None
        if fresh:
            rates = [r for r, _ in fresh]
            minimum = min(rates)
            rate = minimum / (math.fsum(minimum / r for r in rates) / len(rates))
            return dict(bps=rate, sample_count=len(rates),
                        status="ready" if len(rates) == self.window else "warming_up",
                        sample_age_s=age, stale=False)
        return dict(bps=self.startup_bps, sample_count=0, status="bootstrap",
                    sample_age_s=age, stale=bool(self._samples))

    def predict(self, now):
        return self.snapshot(now)["bps"]

    def reset(self):
        self._samples.clear()
        self._pending.clear()
