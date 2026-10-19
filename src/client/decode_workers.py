"""Bounded independent frame-chain workers for throughput experiments.

Each worker owns a decoder and its caches. Never share mutable decoder state
across threads; same-frame dependencies are decoded together in one worker.
The caller supplies verified transport payloads and consumes results in order.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import CancelledError
from collections import Counter
import os
import json
from pathlib import Path
import threading
import time

from .decoder import ProgressiveDecoder
from .optimized_codec import SharedCodecCache

_DEFAULT = object()


def cpu_model():
    try:
        return next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name"))
    except (OSError,StopIteration):
        return "unknown"


def available_memory_mib():
    limits = []
    try:
        fields = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
        limits.append(int(fields["MemAvailable"].split()[0])//1024)
    except (OSError,ValueError,KeyError):
        pass
    try:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if maximum != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            limits.append(max(0,int(maximum)-used)//1024**2)
    except (OSError,ValueError):
        pass
    return min(limits) if limits else 4096


def physical_cores():
    """Respect affinity and Linux CPU quotas, not just logical CPU count."""
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    cores = set()
    try:
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            fields = {k.strip(): v.strip() for k, v in fields.items()}
            if int(fields.get("processor", -1)) in affinity:
                cores.add((fields.get("physical id", "0"), fields.get("core id", fields["processor"])))
    except (OSError, ValueError):
        pass
    limit = len(cores) or len(affinity)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limit = min(limit, max(1, int(quota)//int(period)))
    except (OSError, ValueError):
        pass
    return max(1, limit)


def select_workers(requested, quality, *, max_workers=6, memory_mib=2048,
                   shared_mib=256, profile=None, verification="strict", native="memory"):
    # Conservative allowance: two retained progressive chains, transient native
    # buffers and queued encoded input. This is admission accounting, not an RSS cap.
    working_mib = (96, 160, 288, 480)[quality]
    available_mib = available_memory_mib()
    memory_mib = min(memory_mib,int(.8*available_mib))
    if memory_mib < shared_mib+working_mib:
        raise ValueError("Decode memory budget cannot admit one worker and shared caches")
    cap = min(16, max_workers, physical_cores(), max(1, (memory_mib-shared_mib)//working_mib))
    if requested != "auto":
        if type(requested) is not int or requested < 1 or requested > cap:
            raise ValueError(f"Requested workers exceed hardware/memory cap {cap}")
        return requested, {"policy": "fixed", "cap": cap, "working_mib_per_worker": working_mib,
                           "admission_memory_mib":memory_mib,"available_memory_mib":available_mib}
    choice, reason = min(cap, 4 if quality >= 2 else 6), "conservative hardware/memory fallback"
    if profile:
        document = json.loads(Path(profile).read_text())
        if (document.get("physical_cores") == physical_cores()
                and document.get("cpu_model",cpu_model()) == cpu_model()
                and document.get("native",native) == native):
            candidates = document.get("fps", {}).get(verification, {}).get(str(quality), {})
            allowed = [(float(fps), int(n)) for n, fps in candidates.items() if 0 < int(n) <= cap]
            if allowed:
                choice = max(allowed)[1]
                reason = "offline measured throughput profile"
    return choice, {"policy": "auto", "cap": cap, "reason": reason,
                    "working_mib_per_worker": working_mib,"admission_memory_mib":memory_mib,
                    "available_memory_mib":available_mib}


def queue_capacity(quality, workers, memory_mib, shared_mib, desired, requested=None):
    """Longdress admission estimates for owned encoded input / ready CPU states.

    Segment-cache bytes and current scene/GPU objects are separate allowances.
    These are conservative planning estimates, not an allocator-enforced limit.
    """
    resident_mib = (48,80,144,240)[quality]*workers
    queued_mib = (9,18,36,68)[quality]
    cap = min(128,(memory_mib-shared_mib-resident_mib)//queued_mib)
    if cap < workers:
        raise ValueError("Decode memory budget cannot admit a bounded input queue")
    if requested is not None:
        if type(requested) is not int or requested < 1 or requested > cap:
            raise ValueError(f"Requested ahead capacity exceeds memory cap {cap}")
        return requested
    return max(1,min(desired,cap))


class DecodeWorkers:
    def __init__(self, workers=1, pending=None, optimization=_DEFAULT, *, runtime_config=None,
                 shared=True, max_cached_frames=1):
        if type(workers) is not int or workers < 1 or workers > 16:
            raise ValueError("workers must be in 1..16")
        self.pending = pending if pending is not None else workers*2
        if type(self.pending) is not int or self.pending < 1:
            raise ValueError("pending must be positive")
        self.optimization = ({"native":"memory"} if optimization is _DEFAULT else
                             dict(optimization) if optimization is not None else None)
        self.runtime_config = runtime_config
        self.max_cached_frames = max_cached_frames
        self.shared = SharedCodecCache(self.optimization.get("stream_cache_bytes",128*1024**2),
                                       self.optimization.get("array_cache_bytes",128*1024**2)) if shared and self.optimization is not None else None
        if self.shared:
            self.optimization["shared_cache"] = self.shared
        self.workers = workers
        self.active = 0
        self.futures = set()
        self.closed = False
        self.stats = Counter()
        self.timings = deque(maxlen=4096)
        self.local = threading.local()
        self.decoders = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=workers,thread_name_prefix="gaussian-decode")

    def _decode(self, chain, cancelled=None):
        if not hasattr(self.local,"decoder"):
            self.local.decoder = ProgressiveDecoder(self.runtime_config, optimization=self.optimization,
                                                    max_cached_frames=self.max_cached_frames)
            with self.lock:
                self.decoders.append(self.local.decoder)
        decoder = self.local.decoder
        before_count = decoder.decode_count
        result = None
        for item in chain:
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Obsolete frame dependency chain")
            descriptor, payload, *checksum = item
            result = decoder.decode_payload(descriptor,payload,expected_checksum=checksum[0] if checksum else None)
        if result is None:
            raise ValueError("Empty frame dependency chain")
        with self.lock:
            self.stats["decoded_frame_chains"] += 1
            self.stats["cache_hit_chains"] += int(decoder.decode_count == before_count)
        return result

    def submit(self, chain, cancelled=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("Decoder workers are closed")
            self.futures = {f for f in self.futures if not f.done()}
            if len(self.futures) >= self.pending:
                raise RuntimeError("Decode queue capacity exceeded")
            self.active += 1
            self.stats["submitted"] += 1
            self.stats["peak_pending"] = max(self.stats["peak_pending"], self.active)
        def execute():
            started = time.perf_counter()
            cpu = time.thread_time()
            try:
                return self._decode(chain, cancelled)
            finally:
                with self.lock:
                    self.stats["worker_seconds"] += time.perf_counter()-started
                    self.timings.append({"started":started,"finished":time.perf_counter(),
                                         "latency_ms":(time.perf_counter()-started)*1000,
                                         "thread_cpu_s":time.thread_time()-cpu})
        future = self.executor.submit(execute)
        with self.lock:
            self.futures.add(future)
        def completed(f):
            with self.lock:
                self.active -= 1
                self.futures.discard(f)
                self.stats["completed"] += 1
        future.add_done_callback(completed)
        return future

    def map(self, chains):
        iterator, pending = iter(chains), deque()
        try:
            for _ in range(self.pending):
                chain = next(iterator,None)
                if chain is None: break
                pending.append(self.submit(chain))
            while pending:
                yield pending.popleft().result()
                chain = next(iterator,None)
                if chain is not None:
                    pending.append(self.submit(chain))
        finally:
            for future in pending: future.cancel()

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True,cancel_futures=True)
        for decoder in self.decoders: decoder.close()
        self.decoders.clear()
        if self.shared:
            self.shared.clear()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
