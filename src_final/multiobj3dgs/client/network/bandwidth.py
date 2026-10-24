"""Body-goodput estimation and completion-event traffic accounting.

The estimator consumes the existing public network sample contract and never
imports the transport. Context filtering belongs in :class:`TransferObserver`:
metadata/profile traffic is counted, while only eligible media transfers feed
the harmonic mean. Register the observer once with ``client.subscribe(observer)``
and retain the returned unsubscribe function. Do not feed fetched results again.

These measurements describe application body goodput in bits/second. They are
neither encoded media bitrate nor a guarantee of playback-deadline feasibility.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real
import time

from multiobj3dgs.shared.transfer import BandwidthSample, TransferResult, to_bandwidth_sample

__all__ = [
    "BandwidthEstimate", "BandwidthEstimator",
]


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True, slots=True)
class BandwidthEstimate:
    """A raw harmonic mean (or explicit startup prior), before budget safety.

    ``sample_age_s`` is the newest accepted observation's receipt age, including
    when it has expired. ``stale`` means history exists but has no fresh samples.
    ``sample_count`` always counts fresh observations, not the startup prior.
    """

    bps: float | None
    sample_count: int
    status: str
    sample_age_s: float | None
    stale: bool = False


class BandwidthEstimator:
    """Windowed harmonic mean of fresh, successful body-goodput observations.

    Receipt times use a monotonic clock, independently of sample Unix timestamps.
    The 10-second default lifetime is provisional and should be evaluated for the
    deployment. Call ``reset`` on known path/origin changes. Low positive rates
    are valid observations; there is no rate/size outlier filter.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_sample_age_s: float = 10.0,
        startup_bandwidth_bps: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not _positive_finite(max_sample_age_s):
            raise ValueError("max_sample_age_s must be finite and positive")
        if startup_bandwidth_bps is not None and not _positive_finite(startup_bandwidth_bps):
            raise ValueError("startup_bandwidth_bps must be finite and positive or None")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.window_size = window_size
        self.max_sample_age_s = float(max_sample_age_s)
        self.startup_bandwidth_bps = (
            None if startup_bandwidth_bps is None else float(startup_bandwidth_bps)
        )
        self._clock = clock
        self._observations: deque[tuple[BandwidthSample, float]] = deque(maxlen=window_size)

    def accept_transfer_sample(self, sample: BandwidthSample) -> bool:
        """Accept a valid body measurement; rejection leaves history unchanged."""
        if not isinstance(sample, BandwidthSample):
            return False
        if (
            sample.success is not True
            or isinstance(sample.received_bytes, bool)
            or not isinstance(sample.received_bytes, int)
            or sample.received_bytes <= 0
            or not _positive_finite(sample.transfer_duration)
            or not _positive_finite(sample.total_duration)
            or not _positive_finite(sample.body_goodput_bps)
        ):
            return False
        self._observations.append((sample, self._clock()))
        return True

    def estimate_bandwidth(self) -> BandwidthEstimate:
        """Estimate from fresh samples, or label the explicit startup fallback."""
        now = self._clock()
        fresh = [
            sample for sample, received_at in self._observations
            if max(0.0, now - received_at) <= self.max_sample_age_s
        ]
        age = (
            max(0.0, now - self._observations[-1][1])
            if self._observations else None
        )
        if fresh:
            # Scaling the reciprocal sum avoids overflow for very small rates.
            # At least one ratio equals one, so the denominator cannot underflow.
            rates = [float(sample.body_goodput_bps) for sample in fresh]
            minimum = min(rates)
            estimate = minimum / (math.fsum(minimum / rate for rate in rates) / len(rates))
            return BandwidthEstimate(
                bps=estimate,
                sample_count=len(fresh),
                status="ready" if len(fresh) == self.window_size else "warming_up",
                sample_age_s=age,
            )
        return BandwidthEstimate(
            bps=self.startup_bandwidth_bps,
            sample_count=0,
            status="bootstrap" if self.startup_bandwidth_bps is not None else "unavailable",
            sample_age_s=age,
            stale=bool(self._observations),
        )

    def get_sample_history(self) -> tuple[BandwidthSample, ...]:
        """Snapshot newest accepted samples, including expired ones for diagnosis."""
        return tuple(sample for sample, _ in self._observations)

    def reset(self) -> None:
        """Forget observations after a known path/origin change; retain config."""
        self._observations.clear()


