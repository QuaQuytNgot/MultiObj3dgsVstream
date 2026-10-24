"""FAST presentation adapter. A single owner controls CUDA/OpenGL completion."""
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from ...codecs.streamstgs.vendor import vendor_root as resolve_vendor_root


@dataclass(frozen=True)
class RenderReceipt:
    frame_id: Any
    completed_at: float
    elapsed_s: float
    image: Any = None
    backend: str = 'fast'
    physical_presented_at: Any = None


class FASTRenderer:
    _session_started = False

    def __init__(self,mode='headless',vendor_root=None,background=(1.,1.,1.),device='cuda',jit_optimized=False):
        if mode not in ('headless','native'):raise ValueError('Render mode must be native or headless')
        if type(self)._session_started:raise RuntimeError('FAST owns one OpenGL session per process; start a fresh process for another window/context')
        self.mode=mode;self.device=device;self.background=background;self.owner=threading.get_ident();self.closed=False
        self.jit_optimized=jit_optimized
        self._native_fbo=0;self._native_texture=0
        self._native_texture_size=(0,0);self._cached_size=None
        root=resolve_vendor_root(vendor_root)
        source=root/'submodules/fast-gaussian-rasterization'
        if 'fast_gauss' in sys.modules:
            loaded=Path(sys.modules['fast_gauss'].__file__).resolve()
            if not loaded.is_relative_to(source):raise RuntimeError('A different FAST build is already imported; start a fresh process')
        if mode=='headless':
            if 'OpenGL' in sys.modules:raise RuntimeError('Headless FAST must be initialized before OpenGL imports')
            os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        else:
            if 'OpenGL.GL' not in sys.modules:raise RuntimeError('Native FAST requires a current window context and OpenGL.GL imported first')
            from OpenGL.GL import glGetString,GL_VERSION
            if glGetString(GL_VERSION) is None:raise RuntimeError('No current native OpenGL context')
        sys.path.insert(0,str(source))
        self.fast=importlib.import_module('fast_gauss')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('FAST requires an available CUDA device')
        self.torch=torch
        self.bg=torch.tensor(background,device=device,dtype=torch.float32)
        self._check_interop()
        type(self)._session_started = True

    def _assert_owner(self):
        if self.owner!=threading.get_ident():raise RuntimeError('FAST must run on its OpenGL owner thread')
        if self.closed:raise RuntimeError('FAST renderer is closed')

    def _check_interop(self):
        # Query the actual current GL device instead of assuming CUDA ordinal 0.
        try:
            from cuda import cudart
            result=cudart.cudaGLGetDevices(16,cudart.cudaGLDeviceList.cudaGLDeviceListAll)
            if int(result[0])!=0:raise RuntimeError(f'CUDA/GL device query failed: {result[0]}')
            devices=list(result[2])[:int(result[1])]
            selected=self.torch.cuda.device(self.device).idx
            if selected is None:selected=self.torch.cuda.current_device()
            if selected not in devices:raise RuntimeError(f'CUDA device {selected} does not own current GL context; GL devices={devices}')
        except ImportError as exc:
            raise RuntimeError('FAST CUDA/GL interop requires cuda-python (CUDA 11/12 API)') from exc

    def draw(self,state,camera,frame_id=None,readback=False):
        self._assert_owner();started=time.perf_counter();torch=self.torch
        from OpenGL.GL import glFinish
        with torch.cuda.device(self.device),torch.inference_mode():
            view=camera.to_torch(self.device) if hasattr(camera,'to_torch') else camera
            settings=self.fast.GaussianRasterizationSettings(image_height=view.image_height,image_width=view.image_width,
                tanfovx=math.tan(view.FoVx/2),tanfovy=math.tan(view.FoVy/2),bg=self.bg,scale_modifier=1.,
                viewmatrix=view.world_view_transform,projmatrix=view.full_proj_transform,campos=view.camera_center,
                sh_degree=0,prefiltered=False,debug=False)
            rasterizer=self.fast.GaussianRasterizer(settings)
            if self.mode=='native':
                # FAST's internal texture growth binds framebuffer 0. Grow it
                # before binding our scene target, including on the first draw.
                self.fast.raster_context.resize_textures(view.image_height,view.image_width)
                self._ensure_native_framebuffer(view.image_width,view.image_height)
                self._cached_size=None
                self._prepare_native_frame(view.image_width,view.image_height,self._native_fbo)
            else:
                # EGL clears its FBO before updating viewport/scissor. Set the
                # full extent first so growing the target clears new pixels too.
                from OpenGL.GL import glViewport,glScissor
                glViewport(0,0,view.image_width,view.image_height)
                glScissor(0,0,view.image_width,view.image_height)
            # FAST's ScriptFunctions otherwise specialize on a new Gaussian
            # cardinality during live draws (measured 0.14–0.33 s compilation).
            # The generic graph executes identical operations without profiling
            # or compilation pauses. Keep the policy scoped to this draw.
            with torch.jit.optimized_execution(self.jit_optimized):
                image=rasterizer(**state.raster_args())[0]
            if self.mode=='native':
                self._cached_size=(view.image_width,view.image_height)
                self._present_native_cache(*self._cached_size)
            glFinish();torch.cuda.current_stream(self.device).synchronize()
            output=image.clone() if readback and image is not None else None
            if output is not None:torch.cuda.current_stream(self.device).synchronize()
        return RenderReceipt(frame_id,time.perf_counter(),time.perf_counter()-started,output)

    def _ensure_native_framebuffer(self,width,height):
        from OpenGL import GL
        if not self._native_fbo:
            self._native_fbo=int(GL.glGenFramebuffers(1))
            self._native_texture=int(GL.glGenTextures(1))
        previous_width,previous_height=self._native_texture_size
        if width<=previous_width and height<=previous_height:
            return
        width=max(width,previous_width);height=max(height,previous_height)
        previous_texture=int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        GL.glBindTexture(GL.GL_TEXTURE_2D,self._native_texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S,GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T,GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA8,width,height,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,None)
        GL.glBindTexture(GL.GL_TEXTURE_2D,previous_texture)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self._native_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,GL.GL_TEXTURE_2D,self._native_texture,0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)!=GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Native scene framebuffer is incomplete')
        self._native_texture_size=(width,height)
        self._cached_size=None

    def _prepare_native_frame(self,width,height,framebuffer=0):
        from OpenGL import GL
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,framebuffer)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0 if framebuffer else GL.GL_BACK)
        GL.glViewport(0,0,width,height)
        # FAST enables scissor but only updates its box when viewport != render
        # size. Window resize/ImGui can leave a smaller box, clipping clears and
        # splats alike. The scene pass owns the entire framebuffer every frame.
        GL.glScissor(0,0,width,height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glColorMask(True,True,True,True)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_ALPHA_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glBlendFunc(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(*self.background,1.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)

    def _present_native_cache(self,width,height):
        from OpenGL import GL
        # ImGui draws into the default framebuffer. A cached scene must be
        # copied back before every UI/swap pass, without accumulating the UI.
        self._prepare_native_frame(width,height)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER,self._native_fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBlitFramebuffer(0,0,width,height,0,0,width,height,GL.GL_COLOR_BUFFER_BIT,GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        GL.glReadBuffer(GL.GL_BACK)

    def present_cached(self,camera):
        """Copy the last native scene before UI/swap without running FAST again.

        The caller must invalidate scene content when the camera or timeline
        changes. A framebuffer-size mismatch always requires a fresh draw.
        """
        self._assert_owner()
        if self.mode!='native' or self._cached_size is None:
            return False
        width=getattr(camera,'width',getattr(camera,'image_width',None))
        height=getattr(camera,'height',getattr(camera,'image_height',None))
        if (width,height)!=self._cached_size:
            return False
        self._present_native_cache(width,height)
        return True

    def clear(self,camera,frame_id=None,readback=False):
        """Present the background for a valid scene interval with no active objects."""
        self._assert_owner();started=time.perf_counter();output=None
        if self.mode=='native':
            from OpenGL.GL import glFinish
            self._cached_size=None
            self._prepare_native_frame(camera.width,camera.height)
            glFinish()
        elif readback:
            with self.torch.cuda.device(self.device),self.torch.inference_mode():
                output=self.bg[:,None,None].expand(3,camera.height,camera.width).clone()
                self.torch.cuda.current_stream(self.device).synchronize()
        return RenderReceipt(frame_id,time.perf_counter(),time.perf_counter()-started,output)

    def close(self):
        self._assert_owner()
        from OpenGL.GL import glFinish
        glFinish();self.torch.cuda.synchronize(self.device)
        if self.mode=='native':
            from OpenGL import GL
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
            if self._native_fbo:GL.glDeleteFramebuffers(1,[self._native_fbo])
            if self._native_texture:GL.glDeleteTextures(1,[self._native_texture])
            self._native_fbo=0;self._native_texture=0;self._cached_size=None
        # FAST owns one process-wide GL singleton. Keep its resources alive until
        # the owning context is destroyed; reopening with another context is forbidden.
        self.closed=True
