"""Exact media looping on an independent, monotonically increasing timeline.

The production scene timeline supplies the same modulo-content convention.
This experimental timeline additionally identifies every *request*, so repeated
content never acquires a repeated network-cache key. Decimal inputs are treated
as exact rationals while constructing boundaries; binary float roundoff cannot
create a tiny extra GOP or move a source boundary into another loop.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from fractions import Fraction
import math


def exact_time(value, name="time", *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    try:
        finite = math.isfinite(float(result))
    except OverflowError:
        finite = False
    if result < 0 or (positive and result == 0) or not finite:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


@dataclass(frozen=True)
class TimelineSegment:
    global_segment_index: int
    loop_index: int
    source_gop_index: int
    global_start_s: float
    global_end_s: float
    content_start_s: float
    content_end_s: float
    source_segment_duration_s: float
    duration_s: float
    request_id: str

    @property
    def segment_id(self):
        return self.request_id

    @property
    def gop_id(self):
        """Global GOP identity used by the common adaptation interface."""
        return str(self.global_segment_index)

    @property
    def source_gop_id(self):
        return str(self.source_gop_index)

    def to_dict(self):
        return asdict(self)


class Timeline:
    """Split requests at source GOP, loop, and final experiment boundaries.

    A final partial interval still identifies its original full source GOP.
    Representation accounting must charge that GOP's actual package size; a
    short playback interval does not imply a proportional byte discount.
    """

    def __init__(self, source_duration_s, total_duration_s, gop_duration_s):
        self._source = exact_time(source_duration_s, "source_duration_s", positive=True)
        self._total = exact_time(total_duration_s, "total_duration_s", positive=True)
        self._gop = exact_time(gop_duration_s, "gop_duration_s", positive=True)
        self.source_duration_s, self.total_duration_s, self.gop_duration_s = map(
            float, (self._source, self._total, self._gop))
        segments, starts, exact_bounds = [], [], []
        current = Fraction(0)
        while current < self._total:
            loop = int(current // self._source)
            phase = current - loop*self._source
            source_gop = int(phase // self._gop)
            content_end = min((source_gop+1)*self._gop, self._source)
            end = min(current+content_end-phase, self._total)
            index = len(segments)
            request_id = f"segment-{index:08d}/loop-{loop:08d}/source-gop-{source_gop:08d}"
            source_start = source_gop*self._gop
            segments.append(TimelineSegment(index, loop, source_gop, float(current), float(end),
                float(phase), float(phase+end-current), float(content_end-source_start),
                float(end-current), request_id))
            starts.append(current)
            exact_bounds.append((current, end))
            current = end
        self.segments = tuple(segments)
        self._starts, self._bounds = tuple(starts), tuple(exact_bounds)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def content_time_exact(self, global_time_s):
        timestamp = exact_time(global_time_s, "global_time_s")
        if timestamp > self._total:
            raise ValueError("global_time_s exceeds the experiment duration")
        return timestamp % self._source

    def content_time(self, global_time_s):
        return float(self.content_time_exact(global_time_s))

    def segment_at(self, global_time_s):
        timestamp = exact_time(global_time_s, "global_time_s")
        if timestamp >= self._total:
            raise ValueError("Segment lookup requires a time in the half-open experiment interval")
        return self.segments[bisect_right(self._starts, timestamp)-1]

    def sample_times_exact(self, segment, samples_per_gop):
        """Uniform, half-open sample times; every GOP starts with its own pose."""
        if type(samples_per_gop) is not int or samples_per_gop < 1:
            raise ValueError("samples_per_gop must be a positive integer")
        start, end = self.segment_bounds_exact(segment)
        return tuple(start+(end-start)*i/samples_per_gop for i in range(samples_per_gop))

    def segment_bounds_exact(self, segment):
        if (not isinstance(segment, TimelineSegment) or
                not 0 <= segment.global_segment_index < len(self.segments) or
                self.segments[segment.global_segment_index] != segment):
            raise ValueError("Segment does not belong to this timeline")
        return self._bounds[segment.global_segment_index]

    def to_dict(self):
        return {"source_duration_s": self.source_duration_s, "total_duration_s": self.total_duration_s,
                "gop_duration_s": self.gop_duration_s, "content_looping": True,
                "segment_cache_enabled": False, "segments": [row.to_dict() for row in self.segments]}


def build_timeline(config):
    """Construct from a timeline subsection or a mapping with the three durations."""
    row = config.get("timeline", config)
    return Timeline(row["source_duration_s"], row["total_duration_s"], row["gop_duration_s"])
