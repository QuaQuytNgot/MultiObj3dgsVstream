"""Deterministic continuous scene buffer, independent of source-video loops.

Whole scene GOPs become playable only after every selected object dependency is
delivered. Inputs use global media time, so content repetition cannot reset the
buffer, startup state, stalls, or wall clock. No segment cache exists here.
"""
from __future__ import annotations

import math


class PlaybackBuffer:
    """Continuous virtual-time playback with contiguous whole-GOP admission."""

    def __init__(self, duration_s: float, startup_buffer_s: float = 0.0):
        if isinstance(duration_s, bool) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Playback duration_s must be finite and positive")
        if (isinstance(startup_buffer_s, bool) or not math.isfinite(startup_buffer_s)
                or startup_buffer_s < 0):
            raise ValueError("startup_buffer_s must be finite and nonnegative")
        self.duration_s = float(duration_s)
        self.startup_buffer_s = min(float(startup_buffer_s), self.duration_s)
        self.now = 0.0
        self.position_s = 0.0
        self.available_until_s = 0.0
        self.started = False
        self.startup_delay_s: float | None = None
        self.completion_wall_s: float | None = None
        self._stall_since: float | None = None
        self.stalls: list[dict] = []
        self.events: list[dict] = []
        self.intervals: list[dict] = []
        self.snapshots: list[dict] = [self.snapshot()]

    @property
    def buffer_s(self) -> float:
        return max(0.0, self.available_until_s - self.position_s)

    @property
    def stall_count(self) -> int:
        return len(self.stalls) + int(self._stall_since is not None and self.now > self._stall_since + 1e-12)

    @property
    def rebuffer_duration_s(self) -> float:
        active = self.now - self._stall_since if self._stall_since is not None else 0.0
        return math.fsum(row["duration_s"] for row in self.stalls) + active

    def snapshot(self) -> dict:
        return dict(wall_time_s=self.now, media_time_s=self.position_s, buffer_s=self.buffer_s,
                    started=self.started, stalled=self._stall_since is not None,
                    completed=self.completion_wall_s is not None)

    def _interval(self, end: float, playing: bool) -> None:
        if end <= self.now:
            return
        start, position, buffer = self.now, self.position_s, self.buffer_s
        if playing:
            self.position_s = min(self.duration_s, self.available_until_s, self.position_s + end - start)
        self.now = end
        self.intervals.append(dict(start_s=start, end_s=end, playing=playing,
                                   media_start_s=position, media_end_s=self.position_s,
                                   buffer_start_s=buffer, buffer_end_s=self.buffer_s))

    def _event(self, kind: str, **extra) -> None:
        self.events.append(dict(kind=kind, wall_time_s=self.now, media_time_s=self.position_s, **extra))

    def advance_to(self, wall_time_s: float) -> None:
        if not math.isfinite(wall_time_s) or wall_time_s < self.now - 1e-9:
            raise ValueError("Playback time must be finite and nondecreasing")
        target = max(self.now, float(wall_time_s))
        if self.started and self.completion_wall_s is None and self.buffer_s > 0:
            self._interval(min(target, self.now + self.buffer_s), True)
            if self.position_s >= self.duration_s - 1e-10:
                self.position_s = self.duration_s
                self.completion_wall_s = self.now
                self._event("playback_complete")
        if target > self.now:
            if self.started and self.completion_wall_s is None and self._stall_since is None:
                self._stall_since = self.now
                self._event("rebuffer_start")
            self._interval(target, False)
        self.snapshots.append(self.snapshot())

    def append(self, media_start_s: float, duration_s: float, completion_wall_s: float) -> None:
        """Admit a newly downloaded global scene GOP exactly once and in order."""
        if not all(math.isfinite(x) for x in (media_start_s, duration_s, completion_wall_s)):
            raise ValueError("GOP admission times must be finite")
        if duration_s <= 0 or abs(media_start_s - self.available_until_s) > 1e-8:
            raise ValueError("Scene GOPs must have positive duration and contiguous unique global ranges")
        end = media_start_s + duration_s
        if end > self.duration_s + 1e-8:
            raise ValueError("Scene GOP extends past configured playback duration")
        self.advance_to(completion_wall_s)
        self.available_until_s = min(end, self.duration_s)
        self._event("gop_available", media_start_s=media_start_s, duration_s=duration_s)
        if not self.started and self.buffer_s > 0 and self.buffer_s >= self.startup_buffer_s - 1e-10:
            self.started = True
            self.startup_delay_s = self.now
            self._event("startup_complete")
        elif self._stall_since is not None:
            duration = self.now - self._stall_since
            if duration > 1e-12:
                self.stalls.append(dict(start_s=self._stall_since, end_s=self.now, duration_s=duration))
                self._event("rebuffer_end", duration_s=duration)
            self._stall_since = None
        self.snapshots.append(self.snapshot())

    def finish(self) -> float | None:
        """Drain delivered media; incomplete runs remain explicitly incomplete."""
        if self.started and self.buffer_s > 0:
            self.advance_to(self.now + self.buffer_s)
        return self.completion_wall_s

    def summary(self) -> dict:
        return dict(startup_delay_s=self.startup_delay_s,
                    completion_wall_time_s=self.completion_wall_s,
                    media_duration_s=self.duration_s, played_duration_s=self.position_s,
                    stall_count=self.stall_count, total_rebuffering_s=self.rebuffer_duration_s,
                    wall_time_s=self.now, buffer_s=self.buffer_s)
