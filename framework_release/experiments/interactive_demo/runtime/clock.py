"""Presentation clock: frame stepping never converts lateness into dropped frames."""
from __future__ import annotations
import math
import time


class SceneClock:
    def __init__(self, duration: float, fps: float, *, loop=False, clock=time.monotonic):
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('duration and fps must be finite and positive')
        self.duration, self.fps, self.loop = duration, fps, loop
        self.clock = clock
        self.position = 0.0
        self.playing = True
        self.ended = False
        self._deadline = clock()
        self._stall_start = None
        self.rebuffer_seconds = 0.0
        self.rebuffer_count = 0
        self.generation = 0
        self.last_lateness = 0.0

    @property
    def position(self):
        # Derive each presentation timestamp independently. Repeated addition
        # drifts below exact frame/GOP boundaries and can repeat source frames.
        return self._position_anchor+self._position_ticks/self.fps

    @position.setter
    def position(self,seconds):
        # Preserve a seek's arbitrary phase and existing direct assignments.
        # Only committed frame steps advance the integer tick counter.
        self._position_anchor=float(seconds)
        self._position_ticks=0

    @property
    def deadline(self):
        return self._deadline

    def make_due(self):
        self._deadline = self.clock()

    @property
    def stalled(self):
        return self._stall_start is not None

    @property
    def total_rebuffer_seconds(self):
        ongoing = max(0.0,self.clock()-self._stall_start) if self._stall_start is not None else 0.0
        return self.rebuffer_seconds+ongoing

    def due(self):
        return self.playing and not self.ended and self.clock() >= self._deadline

    def pause(self):
        self.playing = False
        self.unstall()

    def play(self):
        if self.ended:
            self.seek(0)
        self.playing = True
        self._deadline = self.clock()

    def seek(self, seconds):
        if not math.isfinite(seconds):
            raise ValueError('seek time must be finite')
        self.unstall()
        self.position = max(0.0, min(float(seconds), max(0.0, self.duration - 1 / self.fps)))
        self.ended = False
        self.generation += 1
        self._deadline = self.clock()

    def stall(self):
        if self.playing and self._stall_start is None:
            self._stall_start = self.clock()
            self.rebuffer_count += 1

    def unstall(self):
        if self._stall_start is not None:
            self.rebuffer_seconds += max(0.0, self.clock() - self._stall_start)
            self._stall_start = None
            self._deadline = self.clock()

    def commit_frame(self):
        """Call only after every active instance and the joint draw completed."""
        now = self.clock()
        self.last_lateness = max(0.0, now - self._deadline)
        self.unstall()
        next_tick = self._position_ticks+1
        next_time = self._position_anchor+next_tick/self.fps
        if next_time >= self.duration - 1e-9:
            if self.loop:
                self.position = 0.0
                # Immutable prefetched GOPs remain valid at a natural wrap.
                # Only explicit seek/restart invalidates asynchronous jobs.
            else:
                self.ended = True
                self.playing = False
        else:
            self._position_ticks = next_tick
        self._deadline = max(self._deadline + 1 / self.fps, now)
