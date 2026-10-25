"""Exercise real native callbacks, paused redraw and runtime on a local MPD.

Programmatic input on an actual GLFW/CUDA context; not a physical-display FPS test.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mpd',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from multiobj3dgs.client.viewer.window import NativeWindow
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    from multiobj3dgs.client.__main__ import main as player
    original_poll,original_draw=NativeWindow.poll,FASTRenderer.draw
    actions=[];stage=0;resume_at=None;window_ref=None

    def poll(window):
        nonlocal stage,resume_at,window_ref
        window_ref=window
        original_poll(window)
        runtime=window.runtime;g=window.glfw
        def key(code):window._key(window.window,code,0,g.PRESS,0)
        if stage==0 and runtime.frame_id>=1:
            g.set_cursor_pos(window.window,950,700)
            key(g.KEY_SPACE);assert not runtime.clock.playing
            actions.append('space_pause');resume_at=time.monotonic()+.15;stage=1
        elif stage==1 and time.monotonic()>=resume_at:
            previous=window.controller.camera()
            window.controller.orbit(80,15);window.controller.pan(12,-8)
            window._scroll(window.window,0,2)
            assert not np.allclose(previous.center,window.controller.camera().center)
            actions.append('paused_orbit_pan_scroll');stage=2;resume_at=time.monotonic()+.15
        elif stage==2 and time.monotonic()>=resume_at:
            key(g.KEY_SPACE);assert runtime.clock.playing
            actions.append('space_resume');stage=3
        elif stage==3 and runtime.frame_id>=3:
            key(g.KEY_Q);assert runtime.pending_representations
            key(g.KEY_RIGHT);assert runtime.clock.position>=2
            actions.append('quality_Q_and_seek_right');stage=4
        elif stage==4 and runtime.frame_id>=5:
            key(g.KEY_HOME);assert runtime.clock.position==0
            key(g.KEY_L);assert runtime.clock.loop
            key(g.KEY_F);assert window.controller.free
            before=window.controller.camera().center
            window.controller.move(1,0,1,.1)
            assert not np.allclose(before,window.controller.camera().center)
            g.set_window_size(window.window,1280,720)
            actions.append('home_loop_fly_move_resize');stage=5

    def draw(renderer,state,camera,*a,**kw):
        receipt=original_draw(renderer,state,camera,*a,**kw)
        if window_ref and stage and not (args.output/f'stage_{stage}.png').exists():
            from OpenGL import GL
            from PIL import Image
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT,1)
            raw=GL.glReadPixels(0,0,camera.width,camera.height,GL.GL_RGB,GL.GL_UNSIGNED_BYTE)
            pixels=np.frombuffer(raw,np.uint8).reshape(camera.height,camera.width,3)[::-1]
            Image.fromarray(pixels).save(args.output/f'stage_{stage}.png')
        return receipt

    NativeWindow.poll=poll;FASTRenderer.draw=draw
    try:
        player(['--mpd',args.mpd,'--mode','native','--frames','12',
                '--prefetch','0','--report',str(args.output/'playback.json'),
                '--cache',str(args.output/'cache')])
        assert stage==5 and len(actions)==5,actions
        report=json.loads((args.output/'playback.json').read_text())
        assert report['fresh_frame_swap_completions']==12
        (args.output/'controls.json').write_text(json.dumps(dict(status='passed',actions=actions,
            scope='real GLFW keyboard/scroll callbacks; camera orbit/pan/fly methods; resize, paused refresh, native swaps'),indent=2))
    finally:
        NativeWindow.poll=original_poll;FASTRenderer.draw=original_draw


if __name__=='__main__':main()
