"""Bounded deadline-ordered prefetch over the existing source interface."""
import asyncio
from collections import OrderedDict, Counter


class DecodeAhead:
    def __init__(self, source, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Decode-ahead capacity must be positive")
        self.source, self.capacity = source, capacity
        self.pending = {}
        self.draining = set()
        self.ready = OrderedDict()
        self.stats = Counter()

    async def get_frame(self, object_id, frame, quality):
        key = object_id, frame, quality
        if key in self.ready:
            return self.ready.pop(key)
        task = self.pending.pop(key, None)
        if task is None:
            return await self.source.get_frame(*key)
        return await task

    async def prefetch(self, requests, *, retain_running=False, admit=None):
        requests = list(dict.fromkeys(tuple(key) for key in requests))
        wanted = set(requests)
        for key in list(self.ready):
            if key not in wanted:
                del self.ready[key]
        qualities = {(key[0], key[2]) for key in requests}
        active = getattr(self.source,"is_active",lambda key:True)
        obsolete = [self.pending.pop(key) for key in list(self.pending)
                    if key not in requests and (not retain_running or (key[0],key[2]) not in qualities or not active(key))]
        for task in obsolete:
            task.cancel()
            self.draining.add(task)
            task.add_done_callback(self._drained)
            self.stats["cancelled_obsolete"] += 1
        for key in requests:
            key = tuple(key)
            if admit is not None and key not in admit:
                continue
            if key not in self.pending and key not in self.ready and len(self.pending)+len(self.draining)+len(self.ready) < self.capacity:
                self.pending[key] = asyncio.create_task(self.source.get_frame(*key))
        self.stats["peak_pending"] = max(self.stats["peak_pending"], len(self.pending)+len(self.draining))

    def _drained(self, task):
        self.draining.discard(task)
        if not task.cancelled():
            task.exception()  # Cancellation drainage must retrieve worker errors.

    def collect(self):
        for key, task in list(self.pending.items()):
            if task.done():
                del self.pending[key]
                if not task.cancelled():
                    self.ready[key] = task.result()  # Propagate genuine decode/transport errors.
                    self.stats["completed_chains"] += 1
        while len(self.ready) > self.capacity:
            self.ready.popitem(last=False)

    def latest(self, source_id, media_time, quality):
        candidates = [frame for (obj, _, q),frame in self.ready.items()
                      if obj == source_id and q == quality and frame.timestamp <= media_time+1e-9]
        return max(candidates, key=lambda frame:frame.timestamp) if candidates else None

    async def close(self):
        tasks = list(self.pending.values()) + list(self.draining)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.ready.clear()
        self.draining.clear()
