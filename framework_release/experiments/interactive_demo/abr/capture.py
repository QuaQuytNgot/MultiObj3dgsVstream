"""Optional framebuffer evidence from existing draws; no extra RGB render."""
import hashlib
from pathlib import Path


class CaptureRenderer:
    def __init__(self, renderer, directory, frames, events):
        self.renderer=renderer
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.frames=set(frames)
        self.events=events
        self.capture_enabled=True

    def __getattr__(self,name):return getattr(self.renderer,name)

    def draw(self,state,camera,frame_id=None):
        capture=self.capture_enabled and frame_id in self.frames
        receipt=self.renderer.draw(state,camera,frame_id=frame_id,readback=capture)
        if capture:
            import numpy as np
            from PIL import Image
            if self.renderer.mode=='native':
                from OpenGL import GL
                GL.glPixelStorei(GL.GL_PACK_ALIGNMENT,1);GL.glReadBuffer(GL.GL_BACK)
                data=GL.glReadPixels(0,0,camera.width,camera.height,GL.GL_RGB,GL.GL_UNSIGNED_BYTE)
                pixels=np.frombuffer(data,np.uint8).reshape(camera.height,camera.width,3)[::-1].copy()
            elif receipt.image is not None:
                pixels=(receipt.image.float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            else:raise RuntimeError('Renderer did not supply requested framebuffer readback')
            destination=self.directory/f'frame-{frame_id:06d}.png'
            if destination.exists():raise FileExistsError(destination)
            Image.fromarray(pixels).save(destination)
            self.frames.discard(frame_id)
            self.events.emit('frame_capture',frame_id=frame_id,path=str(destination),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                shape=list(pixels.shape),pixel_std=float(pixels.std()))
        return receipt
