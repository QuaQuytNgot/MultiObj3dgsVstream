#!/usr/bin/env python3
"""Bounded N=3/4 real CUDA/GLFW smoke, adapted from tools/smoke_abr_migration_native.py.

Only the isolated diagnostic subclass is temporarily replaced. Native callbacks,
codec and FAST rendering are the existing implementations. No production class
or module is monkey-patched. No physical-input/FPS claim is made.
"""
from contextlib import contextmanager
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src_final'))

def window_with_checks(base):
    class CheckedWindow(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._checks = set()
            self._resume_at = None
            self._manual_rep = None
            self.runtime.emit('demo_native_context',
                vendor=self.GL.glGetString(self.GL.GL_VENDOR).decode(),
                renderer=self.GL.glGetString(self.GL.GL_RENDERER).decode(),
                viewer_class=base.__module__+'.'+base.__name__)

        @contextmanager
        def _uncaptured_input(self):
            # imgui exposes capture flags as read-only. Keep the real UI alive,
            # replacing only this window's capture query during injected input.
            original = self.imgui
            self.imgui = SimpleNamespace(get_io=lambda: SimpleNamespace(
                want_capture_keyboard=False, want_capture_mouse=False))
            try:
                yield
            finally:
                self.imgui = original

        def _press(self, key):
            with self._uncaptured_input():
                self._key(self.window, key, 0, self.glfw.PRESS, 0)

        def _injected_poll(self, *, button=None, move=False):
            g = self.glfw
            names = ('get_cursor_pos', 'get_mouse_button', 'get_key')
            original = {name: getattr(g, name) for name in names}
            cursor = self._cursor
            try:
                self._cursor = (0., 0.)
                g.get_cursor_pos = lambda _: (8., 3.)
                g.get_mouse_button = lambda _, key: g.PRESS if key == button else g.RELEASE
                g.get_key = lambda _, key: g.PRESS if move and key == g.KEY_W else g.RELEASE
                self._last = time.monotonic()-.02
                # The inherited native mouse/fly code handles the inputs.
                with self._uncaptured_input():
                    super().poll()
            finally:
                for name, function in original.items():
                    setattr(g, name, function)
                self._cursor = cursor

        def _done(self, name, **fields):
            self._checks.add(name)
            self.runtime.emit('demo_control_checked', control=name, **fields)

        def poll(self):
            super().poll()
            frame = self.runtime.frame_id
            clock = self.runtime.clock
            if self._resume_at is not None and time.monotonic() >= self._resume_at:
                assert not clock.playing and frame == 1
                self._press(self.glfw.KEY_SPACE)
                assert clock.playing
                self._resume_at = None
                self._done('pause_resume')
            if frame >= 1 and 'pause_started' not in self._checks:
                self._press(self.glfw.KEY_SPACE)
                assert not clock.playing
                self._resume_at = time.monotonic()+.03
                self._done('pause_started')
            if frame >= 2 and 'camera' not in self._checks:
                import numpy as np
                c = self.controller
                before_yaw, before_target, before_radius = c.yaw, c.target.copy(), c.radius
                self._injected_poll(button=self.glfw.MOUSE_BUTTON_LEFT)
                assert c.yaw != before_yaw
                self._injected_poll(button=self.glfw.MOUSE_BUTTON_RIGHT)
                assert not np.array_equal(c.target, before_target)
                with self._uncaptured_input():
                    self._scroll(self.window, 0., .3)
                assert c.radius != before_radius
                self._press(self.glfw.KEY_F)
                assert c.free
                before_fly = c.target.copy()
                self._injected_poll(move=True)
                assert not np.array_equal(c.target, before_fly)
                self._press(self.glfw.KEY_F)
                assert not c.free
                self._resize(self.window, *self.glfw.get_framebuffer_size(self.window))
                handles = [self.runtime.jobs[key].handle for key in self.runtime.required.values()]
                assert len(handles) == len(self.runtime.catalog.instances) and len({id(h) for h in handles}) == len(handles)
                assert len({tuple(i.translation) for i in self.runtime.catalog.instances}) == len(handles)
                self._done('camera', orbit=True, pan=True, zoom=True, fly=True, resize=True,
                           distinct_decoder_handles=True)
            if frame >= 3 and 'manual_quality' not in self._checks:
                identifier = self.runtime.catalog.instances[0].id
                peer = self.runtime.catalog.instances[1].id
                selected = dict(self.runtime.selector.selected)
                active = dict(self.runtime.active_representations)
                self._selected_object = 0
                self._press(self.glfw.KEY_Q)
                self._manual_rep = self.runtime.selector.selected[identifier]
                assert self._manual_rep != selected[identifier]
                assert self.runtime.selector.selected[peer] == selected[peer]
                assert self.runtime.active_representations == active
                self._done('manual_quality', instance_id=identifier, requested=self._manual_rep)
            if frame >= 3 and 'seek' not in self._checks:
                old_generation=clock.generation
                self._press(self.glfw.KEY_RIGHT)
                assert clock.generation==old_generation+1
                self._press(self.glfw.KEY_L)
                assert clock.loop
                self._done('seek', position=clock.position)
            if frame >= 4 and 'seek_render' not in self._checks:
                assert all(row['media_time']>=2 for row in self.runtime._last_frame_sources)
                assert self.runtime._last_frame_sources[0]['representation_id']==self._manual_rep
                self._done('seek_render')
        
    return CheckedWindow

def main(argv=None):
    faulthandler.enable()
    faulthandler.dump_traceback_later(25, repeat=False)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects',type=int,choices=(3,4),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-time-s',type=float,default=90.)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'experiments/interactive_demo/output'):
        parser.error('--output must be inside experiments/interactive_demo/output')
    output.mkdir(parents=True,exist_ok=False)
    from experiments.interactive_demo.abr.config import load_config
    from experiments.interactive_demo.abr import diagnostics,runner
    config=load_config(['--config',str(ROOT/'experiments/interactive_demo/configs'/f'n{args.objects}.json'),
        '--frames','8','--prefetch','0','--startup-mbps','24',
        '--width','256','--height','192','--analysis-width','64',
        '--wall-time-s',str(args.wall_time_s),'--startup-timeout',str(args.wall_time_s),
        '--cache',str(output/'cache'),'--report',str(output/'report.json'),
        '--capture-dir',str(output/'captures'),'--capture-frames','0,7'])
    (output/'config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    base=diagnostics.ABRWindow
    diagnostics.ABRWindow=window_with_checks(base)
    try:code=runner.run(config)
    finally:
        diagnostics.ABRWindow=base
        faulthandler.cancel_dump_traceback_later()
    summary=inspect_result(output,args.objects,code)
    print(json.dumps(summary,indent=2))
    return 0 if summary['status']=='passed' else 1


def inspect_result(output,count,code=0):
    """Also permits stronger post-run validation without another GPU workload."""
    output=Path(output)
    report=json.loads((output/'report.json').read_text())
    events=[json.loads(line) for line in (output/'report.jsonl').read_text().splitlines()]
    controls={e['control'] for e in events if e['stage']=='demo_control_checked'}
    required={'pause_started','pause_resume','camera','manual_quality','seek','seek_render'}
    frames=[e for e in events if e['stage']=='scene_frame' and e.get('fresh',True)]
    transfers=[e for e in events if e['stage']=='transfer' and e.get('object_id')]
    manual=next((e for e in events if e['stage']=='demo_control_checked' and e.get('control')=='manual_quality'),None)
    manual_applied=manual is not None and any(row['instance_id']==manual['instance_id'] and
        row['representation_id']==manual['requested'] and row['media_time']>=2.
        for frame in frames for row in frame['source_frames'])
    checks={'completed':report['status']=='completed',
            'eight_fresh_frames':len(frames)==8,
            'eight_native_swaps':report.get('native_fresh_frame_swap_completions')==8,
            'controls':controls==required,
            'all_instances_in_each_frame':bool(frames) and all(len(e['source_frames'])==count for e in frames),
            'manual_representation_applied_after_seek':manual_applied,
            'no_shared_cpu_preparation':not any(e['stage']=='cpu_preparation_shared' for e in events),
            'all_instance_transfers':len({e['object_id'] for e in transfers})==count,
            'original_sha_preserved':all(len(e.get('resource_sha256',''))==64 for e in transfers),
            'mse_dp_rows':bool(report.get('decisions')) and all(len(d['solver_groups'])==count for d in report.get('decisions',[])),
            'grid_150000':all(d.get('delta_bps')==150000 for d in report.get('decisions',[]))}
    summary={'status':'passed' if all(checks.values()) and code==0 else 'failed',
             'objects':count,'checks':checks,'controls':sorted(controls),
             'frames':len(frames),'error':report.get('error'),'report':str(output/'report.json'),
             'scope':'bounded actual GLFW/CUDA decode+FAST render; injected existing callbacks; no benchmark'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':raise SystemExit(main())
