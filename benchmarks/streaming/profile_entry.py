"""Instrumentation entry for an unchanged native player (including snapshots).

Invoke as: python profile_entry.py --timing FILE -- PLAYER_ARGUMENTS
The target package is selected through PYTHONPATH by profile_native.py.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

ENTRY_MONOTONIC = time.monotonic()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing', required=True, type=Path)
    parser.add_argument('player_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    player_args = args.player_args[1:] if args.player_args[:1] == ['--'] else args.player_args
    lock = threading.RLock()
    records = []
    first_render = None
    first_swap = None
    originals = []
    traces=[]

    def record(stage, start, **fields):
        event = dict(stage=stage, start_monotonic_s=start,
                     completed_monotonic_s=time.monotonic(), **fields)
        event['duration_s'] = event['completed_monotonic_s'] - start
        with lock:
            records.append(event)
        return event

    def metadata(value):
        reconstruction = getattr(value, 'reconstruction', {}) or {}
        segment = reconstruction.get('segment', {})
        stamps = segment.get('timestamps', [])
        return dict(package=str(getattr(value, 'package', '')),
                    crf=reconstruction.get('crf'), gop_index=segment.get('gop_index'),
                    content_id=reconstruction.get('object', reconstruction.get('scene')),
                    source_frame_start=stamps[0].get('frame') if stamps else None,
                    source_frame_end=stamps[-1].get('frame') if stamps else None,
                    nbytes=getattr(value, 'nbytes', None))

    def patch(cls, name, make):
        old = getattr(cls, name)
        originals.append((cls, name, old))
        setattr(cls, name, make(old))

    imported = time.monotonic()
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    from multiobj3dgs.client.viewer.window import NativeWindow
    from multiobj3dgs.client.__main__ import main as player_main
    record('entry_imports', imported)

    def initialization(stage):
        def make(old):
            def wrapped(self, *a, **kw):
                start = time.monotonic()
                try:
                    result=old(self, *a, **kw)
                    if stage=='renderer_init' and os.environ.get('STREAMSTGS_PROFILE_FAST')=='1':
                        from trace_fast import install
                        traces.append(install(self,args.timing.with_suffix('.fast_trace.json')))
                    return result
                finally:
                    record(stage, start)
            return wrapped
        return make

    patch(StreamSTGSAdapter, '__init__', initialization('adapter_init'))
    patch(FASTRenderer, '__init__', initialization('renderer_init'))
    patch(NativeWindow, '__init__', initialization('window_init'))

    def prepare(old):
        def wrapped(self, *a, **kw):
            start = time.monotonic()
            value = old(self, *a, **kw)
            record('prepare_cpu', start, **metadata(value), stages=dict(value.stages),
                   thread_id=threading.get_ident())
            return value
        return wrapped
    # Progressive preparation enters _prepare_cpu from start_cpu, bypassing
    # prepare_cpu. Wrap the physical implementation exactly once so worker
    # counts and CPU timings remain comparable with the legacy snapshot.
    prepare_method = '_prepare_cpu' if hasattr(StreamSTGSAdapter, '_prepare_cpu') else 'prepare_cpu'
    patch(StreamSTGSAdapter, prepare_method, prepare)

    def realize(old):
        def wrapped(self, prepared, *a, **kw):
            start = time.monotonic()
            value = old(self, prepared, *a, **kw)
            record('realize_gpu', start, **metadata(value), stages=dict(value.stages),
                   thread_id=threading.get_ident(), handle_id=id(value))
            return value
        return wrapped
    patch(StreamSTGSAdapter, 'realize_gpu', realize)

    def reconstruct(old):
        def wrapped(self, handle, model_time, *a, **kw):
            start = time.monotonic()
            value = old(self, handle, model_time, *a, **kw)
            record('reconstruct', start, handle_id=id(handle), model_time=float(model_time),
                   **metadata(handle))
            return value
        return wrapped
    patch(StreamSTGSAdapter, 'reconstruct', reconstruct)

    def release(old):
        def wrapped(self, handle, *a, **kw):
            start = time.monotonic()
            try:
                return old(self, handle, *a, **kw)
            finally:
                record('release', start, handle_id=id(handle), **metadata(handle))
        return wrapped
    patch(StreamSTGSAdapter, 'release', release)

    def draw(old):
        def wrapped(self, *a, **kw):
            nonlocal first_render
            start = time.monotonic()
            result = old(self, *a, **kw)
            event = record('draw', start, frame_id=kw.get('frame_id'))
            if first_render is None:
                first_render = event['completed_monotonic_s']
            return result
        return wrapped
    patch(FASTRenderer, 'draw', draw)

    def swap(old):
        def wrapped(self, *a, **kw):
            nonlocal first_swap
            start = time.monotonic()
            result = old(self, *a, **kw)
            if first_render is not None and first_swap is None:
                first_swap = time.monotonic()
                record('first_content_swap', start)
            return result
        return wrapped
    patch(NativeWindow, 'swap', swap)

    status = 'failed'
    try:
        result = player_main(player_args)
        status = 'completed'
        return result
    finally:
        for trace in traces:trace.close()
        for cls, name, old in reversed(originals):
            setattr(cls, name, old)
        args.timing.parent.mkdir(parents=True, exist_ok=True)
        args.timing.write_text(json.dumps(dict(schema='multiobj3dgs.profile.v1',
            status=status, entry_monotonic_s=ENTRY_MONOTONIC,
            first_render_monotonic_s=first_render, first_swap_monotonic_s=first_swap,
            player_args=player_args, records=records), indent=2, default=str)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
