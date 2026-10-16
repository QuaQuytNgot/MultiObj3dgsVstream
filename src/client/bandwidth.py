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

from .request_api import BandwidthSample, TransferResult, to_bandwidth_sample

__all__ = [
    "BandwidthEstimate", "BandwidthEstimator", "TrafficSnapshot", "TransferObserver",
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


@dataclass(frozen=True, slots=True)
class TrafficSnapshot:
    """Actual received body bytes; failed bytes overlap the disjoint kind totals.

    Container headers are included. HTTP/TLS overhead is not. Planned remaining
    bytes belong in the plan result and are never added to this actual ledger.
    """

    metadata_bytes: int = 0
    profile_bytes: int = 0
    base_bytes: int = 0
    enhancement_bytes: int = 0
    other_bytes: int = 0
    total_bytes: int = 0
    failed_bytes: int = 0
    transfer_count: int = 0
    failed_transfer_count: int = 0


class TransferObserver:
    """One completion subscription for accounting and contextual eligibility.

    Set ``RequestSpec.metadata['traffic_kind']`` to ``metadata``, ``profile``,
    ``base``, or ``enhancement``; missing/unknown kinds are counted as ``other``.
    Only Base/enhancement requests with ``bandwidth_eligible is True`` (default)
    feed the estimator. Failed transfers remain accounted but are rejected by
    the estimator. Individual request rates require network concurrency=1.

    This observer does not deduplicate requests or infer retries, cache state,
    or traffic from payload lengths: each completion reports actual body bytes.
    """

    _KINDS = frozenset({"metadata", "profile", "base", "enhancement", "other"})

    def __init__(self, estimator: BandwidthEstimator, *, concurrency: int = 1) -> None:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency != 1:
            raise ValueError("body-goodput observation requires concurrency=1")
        if not isinstance(estimator, BandwidthEstimator):
            raise TypeError("estimator must be a BandwidthEstimator")
        self.estimator = estimator
        self._totals = dict.fromkeys(self._KINDS, 0)
        self._failed_bytes = 0
        self._transfer_count = 0
        self._failed_transfer_count = 0

    def __call__(self, result: TransferResult) -> None:
        if not isinstance(result, TransferResult):
            raise TypeError("result must be a TransferResult")
        if (
            isinstance(result.received_bytes, bool)
            or not isinstance(result.received_bytes, int)
            or result.received_bytes < 0
        ):
            raise ValueError("received_bytes must be a nonnegative integer")
        kind = result.metadata.get("traffic_kind", "other")
        if not isinstance(kind, str) or kind not in self._KINDS:
            kind = "other"
        self._totals[kind] += result.received_bytes
        self._transfer_count += 1
        if not result.success:
            self._failed_bytes += result.received_bytes
            self._failed_transfer_count += 1
        if kind in {"base", "enhancement"} and result.metadata.get("bandwidth_eligible", True) is True:
            self.estimator.accept_transfer_sample(to_bandwidth_sample(result))

    def snapshot(self) -> TrafficSnapshot:
        return TrafficSnapshot(
            metadata_bytes=self._totals["metadata"],
            profile_bytes=self._totals["profile"],
            base_bytes=self._totals["base"],
            enhancement_bytes=self._totals["enhancement"],
            other_bytes=self._totals["other"],
            total_bytes=sum(self._totals.values()),
            failed_bytes=self._failed_bytes,
            transfer_count=self._transfer_count,
            failed_transfer_count=self._failed_transfer_count,
        )
