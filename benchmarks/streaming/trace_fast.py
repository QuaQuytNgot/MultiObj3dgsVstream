"""Opt-in CPU wall-time tracing of FAST/CUDA-GL spikes.

Install only after FASTRenderer initialization (so platform/context selection has
already happened). No synchronization is added. Nested intervals overlap and
must not be summed as independent GPU work. Only calls above threshold are kept.

    if os.environ.get('STREAMSTGS_PROFILE_FAST') == '1':
        from trace_fast import install
        trace = install(renderer, timing_path.with_suffix('.fast_trace.json'))

Call trace.close() from the entry's finally block; atexit is also registered.
"""
from __future__ import annotations
import atexit
from collections import defaultdict
import functools
import json
import os
from pathlib import Path
import threading
import time


class FASTTrace:
    def __init__(self,renderer,output_path=None,threshold_s=.010):
        self.renderer=renderer
        self.output_path=Path(output_path) if output_path is not None else None
        self.threshold_s=float(threshold_s)
        self.records=[];self.patch_failures=[];self.originals=[]
        self.local=threading.local();self.lock=threading.RLock()
        self.next_span=0;self.closed=False
        self.started_monotonic_s=time.monotonic()
        self.call_counts=defaultdict(int)
        self._install()
        atexit.register(self.close)

    def _patch(self,target,name,stage):
        try:
            old=getattr(target,name)
        except AttributeError:
            self.patch_failures.append(dict(stage=stage,error='unavailable'))
            return
        @functools.wraps(old)
        def wrapped(*args,**kwargs):
            if getattr(self.local,'frame_active',False) is not True:
                return old(*args,**kwargs)
            return self._span(stage,old,args,kwargs)
        try:
            setattr(target,name,wrapped)
        except (AttributeError,TypeError) as error:
            self.patch_failures.append(dict(stage=stage,error=str(error)))
            return
        self.originals.append((target,name,old))

    def _span(self,stage,function,args,kwargs):
        stack=getattr(self.local,'stack',[])
        self.next_span+=1;identifier=self.next_span
        parent=stack[-1][0] if stack else None
        path=[item[1] for item in stack]
        stack.append((identifier,stage));self.local.stack=stack
        start=time.monotonic();error=None
        try:
            return function(*args,**kwargs)
        except BaseException as exc:
            error=type(exc).__name__
            raise
        finally:
            completed=time.monotonic();stack.pop()
            self.call_counts[stage]+=1
            if completed-start>=self.threshold_s:
                row=dict(stage=stage,span_id=identifier,parent_span_id=parent,
                    parent_stages=path,frame_id=getattr(self.local,'frame_id',None),
                    thread_id=threading.get_ident(),start_monotonic_s=start,
                    completed_monotonic_s=completed,duration_s=completed-start)
                if stage=='cudart.cudaMemcpyAsync' and len(args)>2:
                    try:row['copy_bytes']=int(args[2])
                    except (TypeError,ValueError):pass
                if stage=='gl.glGetIntegerv' and args:
                    try:row['pname']=int(args[0])
                    except (TypeError,ValueError):pass
                if error:row['error_type']=error
                with self.lock:self.records.append(row)

    def _install(self):
        # Importing these modules before renderer initialization changes FAST's
        # native/EGL context detection, so all imports stay in this late hook.
        import fast_gauss.gsplat_utils as gsplat
        from cuda import cudart
        from OpenGL import GL
        import torch
        for name in ('build_cov6','in_frustum'):
            self._patch(gsplat,name,'fast.'+name)
        context=gsplat.GSplatContextManager
        for name in ('__init__','compile_shaders','init_gl_buffers','init_textures',
                     'resize_buffers','resize_textures','upload_gl_uniforms',
                     'render','rasterize_gaussians'):
            self._patch(context,name,'fast.'+name)
        for name in ('cudaGraphicsMapResources','cudaGraphicsResourceGetMappedPointer',
                     'cudaMemcpyAsync','cudaGraphicsUnmapResources',
                     'cudaGraphicsSubResourceGetMappedArray','cudaMemcpy2DFromArrayAsync'):
            self._patch(cudart,name,'cudart.'+name)
        for name in ('glFinish','glDrawArrays','glGetIntegerv','glGetInteger',
                     'glBlitFramebuffer','glTexImage2D','glBufferData'):
            self._patch(GL,name,'gl.'+name)
        # These Python entry points expose implicit waits that can occur before
        # the interop API (nonzero size readback, allocation, indexed copies and
        # camera CPU readback). They retain exactly the original call semantics.
        for name in ('cat','empty','zeros_like','tensor'):
            self._patch(torch,name,'torch.'+name)
        for name in ('nonzero','argsort','__getitem__','contiguous','to','cpu','clone'):
            self._patch(torch.Tensor,name,'tensor.'+name)
        self._patch(torch.cuda.Stream,'synchronize','cuda.stream_synchronize')
        self._patch(torch.cuda,'synchronize','cuda.device_synchronize')
        old_draw=self.renderer.draw
        @functools.wraps(old_draw)
        def draw(*args,**kwargs):
            previous=(getattr(self.local,'frame_active',False),
                      getattr(self.local,'frame_id',None),getattr(self.local,'stack',[]))
            self.local.frame_active=True
            self.local.frame_id=kwargs.get('frame_id',args[2] if len(args)>2 else None)
            self.local.stack=[]
            try:
                return self._span('renderer.draw',old_draw,args,kwargs)
            finally:
                self.local.frame_active,self.local.frame_id,self.local.stack=previous
        # Instance hook composes with profile_entry's class-level draw wrapper.
        self.renderer.draw=draw
        self.originals.append((self.renderer,'draw',old_draw))

    def snapshot(self):
        with self.lock:
            return dict(schema='multiobj3dgs.fast_cpu_spikes.v1',
                scope='CPU wall duration; no added GPU synchronization; nested intervals overlap',
                threshold_s=self.threshold_s,started_monotonic_s=self.started_monotonic_s,
                completed_monotonic_s=time.monotonic(),call_counts=dict(self.call_counts),
                patch_failures=list(self.patch_failures),
                records=sorted(self.records,key=lambda row:row['start_monotonic_s']))

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            for target,name,old in reversed(self.originals):
                try:setattr(target,name,old)
                except (AttributeError,TypeError):pass
            self.originals.clear()
            report=self.snapshot()
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True,exist_ok=True)
            self.output_path.write_text(json.dumps(report,indent=2,default=str)+'\n')


def install(renderer,output_path=None,threshold_s=.010):
    """Return an installed trace when STREAMSTGS_PROFILE_FAST=1, else None."""
    if os.environ.get('STREAMSTGS_PROFILE_FAST')!='1':return None
    existing=getattr(renderer,'_fast_cpu_trace',None)
    if existing is not None:return existing
    trace=FASTTrace(renderer,output_path,threshold_s)
    renderer._fast_cpu_trace=trace
    return trace
