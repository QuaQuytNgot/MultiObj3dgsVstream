"""Opt-in measurements for the existing real HTTP/2 experiment transport.

Frame probes wrap the installed Hypercorn in this isolated process only and
are restored on close. Profiled runs are separate from throughput timings.
Await durations overlap between streams and are never summed as CPU time.
"""
from __future__ import annotations

import asyncio
import cProfile
from collections import defaultdict
from pathlib import Path
import pstats
import time


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, p50=None, p95=None, maximum=None)
    return dict(count=len(values), p50=values[(len(values)-1)//2],
                p95=values[min(len(values)-1, int(len(values)*.95))], maximum=values[-1])


class HTTP2Profiler:
    def __init__(self, profile_path=None, *, flow_probe=False, interval_s=.005):
        self.profile_path = Path(profile_path) if profile_path else None
        self.flow_probe, self.interval_s = flow_probe, interval_s
        self.lags = []
        self.flow = dict(send_attempts=0, zero_window_attempts=0,
                         flush_calls=0, flush_bytes=0, window_updates=0,
                         connection_window_min=None, stream_window_min=None)
        self._originals = {}
        # Thread CPU excludes time asleep in select()/socket I/O. The custom
        # timer adds overhead, so profiled rates are never benchmark controls.
        self._profiler = cProfile.Profile(timer=time.thread_time) if self.profile_path else None
        self._closed = False

    async def start(self):
        self.wall_start, self.cpu_start = time.monotonic(), time.thread_time()
        self._sampler = asyncio.create_task(self._sample())
        if self.flow_probe:
            self._install_flow_probe()
        if self._profiler:
            self._profiler.enable()

    async def _sample(self):
        while True:
            expected = time.monotonic() + self.interval_s
            await asyncio.sleep(self.interval_s)
            self.lags.append(max(0., time.monotonic() - expected))

    def _install_flow_probe(self):
        from hypercorn.protocol.h2 import H2Protocol
        metrics = self.flow
        for name in ('_send_data', '_flush', '_window_updated'):
            self._originals[name] = getattr(H2Protocol, name)
        send_original = self._originals['_send_data']
        flush_original = self._originals['_flush']
        update_original = self._originals['_window_updated']

        async def send(protocol, stream_id):
            metrics['send_attempts'] += 1
            try:
                connection = protocol.connection.outbound_flow_control_window
                window = protocol.connection.local_flow_control_window(stream_id)
                for key, value in (('connection_window_min', connection), ('stream_window_min', window)):
                    metrics[key] = value if metrics[key] is None else min(metrics[key], value)
                if window <= 0:
                    metrics['zero_window_attempts'] += 1
            except Exception:
                # The original implementation owns stream-close handling.
                metrics['closed_stream_probe_count'] = metrics.get('closed_stream_probe_count', 0) + 1
            return await send_original(protocol, stream_id)

        async def flush(protocol):
            metrics['flush_calls'] += 1
            metrics['flush_bytes'] += len(getattr(protocol.connection, '_data_to_send', b''))
            return await flush_original(protocol)

        async def update(protocol, stream_id):
            metrics['window_updates'] += 1
            return await update_original(protocol, stream_id)

        H2Protocol._send_data, H2Protocol._flush, H2Protocol._window_updated = send, flush, update

    async def stop(self):
        if self._closed:
            return self.result
        self._closed = True
        if self._profiler:
            self._profiler.disable()
        cpu, wall = time.thread_time() - self.cpu_start, time.monotonic() - self.wall_start
        self._sampler.cancel()
        await asyncio.gather(self._sampler, return_exceptions=True)
        if self._originals:
            from hypercorn.protocol.h2 import H2Protocol
            for name, original in self._originals.items():
                setattr(H2Protocol, name, original)
        self.result = dict(loop_thread_cpu_s=cpu, loop_wall_s=wall,
            loop_thread_cpu_percent=100*cpu/wall if wall else None,
            scheduling_lag_s=distribution(self.lags), flow_control=self.flow,
            cprofile_enabled=bool(self._profiler), flow_probe_enabled=self.flow_probe)
        if self._profiler:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.dump_stats(str(self.profile_path))
            stats = pstats.Stats(self._profiler)
            groups = defaultdict(float)
            functions = []
            for (filename, line, function), (cc, nc, tt, ct, callers) in stats.stats.items():
                normalized = filename.replace('\\', '/')
                group = next((name for name in ('httpx', 'httpcore', 'hypercorn', 'h2', 'hpack',
                    'hyperframe', 'ssl', 'asyncio', 'http2_server', 'http2_profiling')
                    if f'/{name}/' in normalized or f'/{name}.py' in normalized), 'other')
                groups[group] += tt
                functions.append(dict(file=filename, line=line, function=function,
                                      calls=nc, self_cpu_s=tt, cumulative_s=ct))
            self.result.update(profile_path=str(self.profile_path),
                exclusive_cpu_s_by_component=dict(sorted(groups.items())),
                top_functions_by_self_time=sorted(functions, key=lambda r:-r['self_cpu_s'])[:40],
                profile_note='Deterministic profiler overhead applies; throughput controls run separately. '
                             'Exclusive component CPU is additive; cumulative/await durations are not.')
        return self.result
