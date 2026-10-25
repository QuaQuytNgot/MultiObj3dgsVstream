"""Real-package EGL/native parity, resize/UI regression and warm GPU timings.

Run each mode in a separate process. Native requires NVIDIA GLX/PRIME and DISPLAY.
Images are framebuffer readbacks, not photographs or ground-truth quality metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('native', 'headless'), required=True)
    p.add_argument('--soldier', type=Path, required=True, help='Root containing gop_NNN/encoded/crf_20')
    p.add_argument('--longdress', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from multiobj3dgs.client.viewer import CameraController
    from multiobj3dgs.shared.geometry import Camera, similarity_matrix
    from multiobj3dgs.shared.gaussian import concatenate
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter

    controller = CameraController(1024, 768)
    window = None
    if args.mode == 'native':
        from multiobj3dgs.client.viewer.window import NativeWindow
        runtime = SimpleNamespace(
            clock=SimpleNamespace(playing=False, position=0., duration=10., loop=False),
            catalog=SimpleNamespace(instances=[]),
            diagnostics=lambda: dict(scene_time=0., frames=0, stalled=False, playing=False,
                gpu_bytes=0, fresh_wall_fps=None, torch_allocated_bytes=None,
                rebuffer_count=0, rebuffer_seconds=0., host_bytes=0, objects=[]))
        window = NativeWindow(controller, runtime, 1024, 768)
        window.glfw.swap_interval(0)
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    renderer = FASTRenderer(mode=args.mode)
    from OpenGL import GL
    adapter = StreamSTGSAdapter()
    records = []

    def read_native(camera):
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glReadBuffer(GL.GL_BACK)
        data = GL.glReadPixels(0, 0, camera.width, camera.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, np.uint8).reshape(camera.height, camera.width, 3)[::-1].copy()

    def capture(state, camera):
        receipt = renderer.draw(state, camera, readback=True)
        if window:
            return read_native(camera)
        return (receipt.image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)

    def reconstruct(handles, camera, frame, count, single=0):
        states = []
        for i in range(count):
            handle = handles[i % 2] if count > 1 else handles[single]
            matrix = similarity_matrix(
                translation=((i-(count-1)/2)*1.8, 0., .3*(i % 2)) if count > 1 else (0, 0, 0),
                rotation=(np.cos(.1*i), 0, np.sin(.1*i), 0), scale=1-.08*(i % 2))
            center = np.linalg.solve(matrix[:3, :3], camera.center-matrix[:3, 3])
            state = adapter.reconstruct(handle, frame/handle.model.time_line, center)
            states.append(state.transformed(matrix))
        return concatenate(states)

    with torch.inference_mode():
        for gop in ([0] if args.quick else [0, 4]):
            handles = [adapter.realize_gpu(adapter.prepare_cpu(root/f'gop_{gop:03d}/encoded/crf_20'))
                       for root in (args.soldier, args.longdress)]
            for count, single in ([(2, 0)] if args.quick else [(1, 0), (1, 1), (2, 0), (4, 0)]):
                for angle, frame in enumerate([0, 29, 59]):
                    width, height = [(1024, 768), (1280, 720), (800, 600)][angle]
                    controller.adopt(Camera.from_package(handles[0].cameras[0], width, height))
                    if count > 1:
                        controller.fit([((i-(count-1)/2)*1.8, 0, 0) for i in range(count)])
                    controller.orbit(angle*50, angle*9)
                    camera = controller.camera()
                    if window:
                        window.glfw.set_window_size(window.window, width, height)
                        window.poll()
                        window._resize(window.window, width, height)
                        # Reproduce a clip rectangle left by a previous framebuffer/UI.
                        GL.glEnable(GL.GL_SCISSOR_TEST)
                        GL.glScissor(0, 0, 640, 480)
                    state = reconstruct(handles, camera, frame, count, single)
                    name = f'g{gop}_n{count}_f{frame}_a{angle}'+('_longdress' if single else '')
                    before = {'viewport': GL.glGetIntegerv(GL.GL_VIEWPORT).tolist(),
                              'scissor': GL.glGetIntegerv(GL.GL_SCISSOR_BOX).tolist()}
                    pixels = capture(state, camera)
                    Image.fromarray(pixels).save(args.output/f'{name}.png')
                    hashes = {k: hashlib.sha256(getattr(state, k).cpu().numpy().tobytes()).hexdigest()
                              for k in ('xyz', 'scales', 'rotations', 'opacities', 'colors')}
                    row = dict(name=name, count=count, frame=frame, gop=gop, angle=angle,
                               state_sha256=hashes, gl_before=before)
                    if args.reference:
                        ref = np.asarray(Image.open(args.reference/f'{name}.png'))
                        error = np.abs(pixels.astype(float)-ref.astype(float))
                        row.update(mae_u8=float(error.mean()), max_u8=float(error.max()),
                                   fraction_over_1=float((error > 1).mean()))
                    if window:
                        window.draw_ui();window.swap();window.poll()
                        # Destroy the swapped backbuffer and leave a viewport /
                        # scissor unrelated to the scene. Restoring the clean
                        # scene must use its cache, without running FAST again.
                        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
                        GL.glDrawBuffer(GL.GL_BACK)
                        GL.glDisable(GL.GL_SCISSOR_TEST)
                        GL.glColorMask(True, True, True, True)
                        GL.glClearColor(.8, .1, .7, 1.)
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                        GL.glEnable(GL.GL_SCISSOR_TEST)
                        GL.glScissor(0, 0, 64, 48)
                        GL.glViewport(13, 7, width//2, height//2)
                        mismatch = SimpleNamespace(width=width+1, height=height)
                        assert not renderer.present_cached(mismatch), name
                        row['cached_size_mismatch_rejected'] = True
                        assert renderer.present_cached(camera), name
                        cached = read_native(camera)
                        row['cached_restore_max_u8'] = int(np.abs(cached.astype(int)-pixels.astype(int)).max())
                        assert row['cached_restore_max_u8'] <= 1, row
                        # The next scene draw must not depend on the UI pass.
                        repeat = capture(state, camera)
                        row['after_ui_max_u8'] = int(np.abs(repeat.astype(int)-pixels.astype(int)).max())
                        assert row['after_ui_max_u8'] <= 1, row
                        # Empty/startup scenes must clear every pixel despite a
                        # stale clip region and invalidate the previous scene.
                        GL.glScissor(0, 0, 64, 48)
                        GL.glViewport(13, 7, width//2, height//2)
                        renderer.clear(camera)
                        cleared = read_native(camera)
                        background = np.round(np.asarray(renderer.background)*255).astype(np.uint8)
                        row['clear_background_max_u8'] = int(np.abs(cleared.astype(int)-background.astype(int)).max())
                        assert row['clear_background_max_u8'] <= 1, row
                        assert not renderer.present_cached(camera), name
                        row['clear_invalidates_cache'] = True
                    for _ in range(4):
                        renderer.draw(state, camera)
                    torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for _ in range(12):
                        start = time.perf_counter()
                        state = reconstruct(handles, camera, frame, count, single)
                        renderer.draw(state, camera)
                        samples.append(time.perf_counter()-start)
                    row.update(warm_fps=1/np.mean(samples), p50_ms=float(np.median(samples)*1000),
                               p95_ms=float(np.percentile(samples, 95)*1000),
                               torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                               torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
                    records.append(row)
                    print(json.dumps(row), flush=True)
            for handle in handles:
                adapter.release(handle)
    report = dict(mode=args.mode, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  gl_renderer=GL.glGetString(GL.GL_RENDERER).decode(), cases=records)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    renderer.close()
    if window:
        window.close()
    if args.reference:
        reference = {r['name']: r for r in json.loads((args.reference/'report.json').read_text())['cases']}
        for row in records:
            assert row['state_sha256'] == reference[row['name']]['state_sha256'], row['name']
            # Native GL converts the CV view to GL coordinates; floating point
            # rounding may cross an 8-bit framebuffer quantization boundary.
            assert row['mae_u8'] < .03 and row['fraction_over_1'] < .001 and row['max_u8'] <= 8, row
            if window:
                assert row['after_ui_max_u8'] <= 1, row


if __name__ == '__main__':
    main()
