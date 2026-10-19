"""Decoded media-window readiness over the existing bounded decode-ahead queue.

Readiness uses actual object/frame/quality keys, not received encoded bytes.
The presentation clock and rendering remain owned by playback.py.
"""
from dataclasses import dataclass
import asyncio
import math
from time import perf_counter


@dataclass(frozen=True)
class BufferWindow:
    requests: tuple
    tracks: dict
    active: frozenset
    fps: float
    requested_s: float
    effective_s: float
    reason: str | None = None

    def availability(self, ahead, known=()):
        """Contiguous decoded durations, capped at this window's media horizon."""
        available = set(ahead.ready) | set(known)
        durations = {}
        for identifier, track in self.tracks.items():
            count = 0
            for key in track:
                if key not in available:
                    break
                count += 1
            durations[identifier] = count / self.fps if identifier in self.active else 0.
        first_ready = all(not track or track[0] in available for track in self.tracks.values())
        duration = min((durations[name] for name in self.active), default=0.)
        ready = first_ready and all(key in available for key in self.requests)
        return {"ready":ready,"first_ready":first_ready,"decoded_duration_s":duration,
                "per_object":durations,"ready_frames":sum(key in available for key in self.requests),
                "required_frames":len(self.requests)}


def plan_window(source, objects, clocks, quality_at, advancing, *, fps, duration_s,
                capacity, loop=True, max_steps=None):
    """Admit complete presentation steps fairly, clipping to a common RAM bound.

    The same Longdress source/frame/quality can satisfy multiple scene instances.
    Paused instances require their current state only. Sampling on presentation
    steps naturally deduplicates a source with a lower frame rate.
    """
    steps = max(1, math.ceil(duration_s*fps-1e-9))
    reason = None
    if max_steps is not None and steps > max_steps:
        steps = max(1,max_steps)
        reason = "finite presentation end"
    keys, tracks, active = [], {obj['id']:[] for obj in objects}, set()
    for obj in objects:
        if advancing(obj): active.add(obj['id'])
    seen = set()
    for offset in range(steps):
        additions=[]
        try:
            for obj in objects:
                moving = obj['id'] in active
                sample=source.select_frame(obj['source'],clocks[obj['id']]+(offset/fps if moving else 0.),loop=loop)
                q=quality_at(obj,offset if moving else 0)
                additions.append((obj['id'],(obj['source'],sample.frame,q)))
        except EOFError:
            if offset == 0: raise
            reason = "end of media"
            break
        new = [key for _,key in additions if key not in seen]
        if len(seen | set(new)) > capacity:
            if offset == 0:
                raise ValueError("Decoded buffer capacity cannot admit every initial object state")
            reason = "decoded queue memory/capacity limit"
            break
        for identifier,key in additions:
            tracks[identifier].append(key)
            if key not in seen:
                keys.append(key);seen.add(key)
    effective = min((len(tracks[name])/fps for name in active),default=0.)
    return BufferWindow(tuple(keys),tracks,frozenset(active),fps,duration_s,effective,reason)


async def prime_window(ahead, window, *, timeout_s, fallback='shrink'):
    """Wait for a contiguous decoded window; caller keeps viewer polling alive.

    A timeout can accept an actually decoded prefix only when all objects have
    their initial state. Corrupt input/decode errors always propagate.
    """
    began=perf_counter()
    await ahead.prefetch(window.requests)
    while True:
        ahead.collect()
        available=window.availability(ahead)
        if available['ready']:
            status='capacity_limited' if window.reason else 'ready'
            break
        if perf_counter()-began >= timeout_s:
            if fallback == 'error' or not available['first_ready']:
                raise RuntimeError("Startup decoded buffer timed out before readiness")
            status='timeout'
            break
        await asyncio.sleep(.005)
    return {"requested_s":window.requested_s,"effective_target_s":window.effective_s,
            **available,"status":status,"reason":window.reason if status!='timeout' else 'startup timeout; decoded prefix accepted',
            "wait_s":perf_counter()-began,"peak_slots":len(ahead.pending)+len(ahead.ready)+len(ahead.draining)}
