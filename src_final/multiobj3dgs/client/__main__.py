"""Native or EGL multi-object StreamSTGS playback from a verified DASH catalog."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpd',required=True,help='Local MPD path or HTTP(S) URL')
    parser.add_argument('--mode',choices=('native','headless'),default='native')
    parser.add_argument('--frames',type=int,default=None,help='Stop after this many fresh scene frames')
    parser.add_argument('--report',type=Path,help='Write a new JSON report and adjacent JSONL events')
    parser.add_argument('--cache',type=Path,default=Path('.cache/multiobj3dgs'))
    parser.add_argument('--cache-mib',type=int,default=4096)
    parser.add_argument('--host-mib',type=int,default=2048)
    parser.add_argument('--raw-mib',type=int,default=512)
    parser.add_argument('--gpu-mib',type=int,default=4096)
    parser.add_argument('--gpu-transient-mib',type=int,default=256)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--fetch-concurrency',type=int,default=1)
    parser.add_argument('--ffmpeg-threads',type=int,default=2)
    parser.add_argument('--upload-budget-ms',type=float,default=2.,help='Incremental GPU upload owner budget per pump; 0 uses synchronous upload')
    parser.add_argument('--fast-jit-specialization',action='store_true',help='Enable FAST TorchScript specialization (may compile during live playback; benchmark ablation)')
    parser.add_argument('--prefetch',type=int,default=1,help='Target GOPs ahead per instance, subject to byte admission')
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--fps',type=float,help='Presentation clock rate; default maximum content fps')
    parser.add_argument('--loop',action='store_true')
    parser.add_argument('--unpaced',action='store_true',help='Headless throughput experiment; still never skips frames')
    parser.add_argument('--representation',action='append',default=[],metavar='INSTANCE=ID')
    parser.add_argument('--visibility-every',type=int,default=0,help='Contribution analysis sampling interval; 0 disables')
    parser.add_argument('--control-trace',type=Path,help='JSON list of manual actions triggered by after_frame; actions quality, seek, pause, play, loop')
    parser.add_argument('--vendor-root',type=Path,help='Pinned StreamSTGS source root')
    parser.add_argument('--startup-timeout',type=float,default=300)
    args=parser.parse_args(argv)
    for key in ('width','height','workers','fetch_concurrency','ffmpeg_threads','startup_timeout'):
        if getattr(args,key)<=0:parser.error(f'--{key.replace("_","-")} must be positive')
    if args.frames is not None and args.frames<=0:parser.error('--frames must be positive')
    if args.visibility_every<0:parser.error('--visibility-every must be nonnegative')
    if args.upload_budget_ms<0:parser.error('--upload-budget-ms must be nonnegative')
    if args.mode=='native' and args.unpaced:parser.error('--unpaced is a headless experiment option')
    return args


def main(argv=None):
    started=time.monotonic()
    args=arguments(argv)
    from .runtime import IOLoop, PlaybackRuntime, MemoryLimits
    from .metadata import MPDParser
    from .cache import ResourceCache
    from .network.observation import TransferObserver
    from ..shared.telemetry import EventLog
    from .viewer import CameraController
    if args.report and args.report.exists():
        raise FileExistsError(f'Report already exists: {args.report}')
    event_path=args.report.with_suffix('.jsonl') if args.report else None
    events=EventLog(event_path)
    observer=TransferObserver(concurrency=args.fetch_concurrency,events=events)
    io=IOLoop()
    runtime=window=renderer=cache=client=None
    first_frame=None;first_presented=None;presentations=0
    status='failed';error=None;catalog=None
    report={}
    try:
        async def setup():
            from ..shared.transfer import NetworkClient
            is_http=urlsplit(args.mpd).scheme in ('http','https')
            transport=NetworkClient(concurrency=args.fetch_concurrency) if is_http else None
            try:
                class ObservedMetadataClient:
                    async def fetch(self, spec):
                        observer.started(spec)
                        try:
                            result = await transport.fetch(spec)
                        except BaseException:
                            observer.cancelled(spec)
                            raise
                        observer(result)
                        return result
                catalog=await MPDParser.from_http(args.mpd,ObservedMetadataClient()) if is_http else MPDParser.from_local(args.mpd)
                if transport is None and any(urlsplit(r.url).scheme in ('http','https')
                        for c in catalog.contents.values() for rep in c.representations.values()
                        for segment in rep.segments for r in segment.resources):
                    transport=NetworkClient(concurrency=args.fetch_concurrency)
                cache=ResourceCache(args.cache,args.cache_mib*1024**2,client=transport,observer=observer,events=events)
                return catalog,transport,cache
            except BaseException:
                if transport:await transport.aclose()
                raise
        catalog,client,cache=io.submit(setup()).result(timeout=args.startup_timeout)
        from ..codecs.streamstgs import StreamSTGSAdapter
        from .adaptation import ManualSelector
        defaults={}
        for spec in args.representation:
            if '=' not in spec:raise ValueError('--representation requires INSTANCE=ID')
            key,value=spec.split('=',1);defaults[key]=value
        unknown=set(defaults)-{i.id for i in catalog.instances}
        if unknown:raise ValueError(f'Unknown instance selections: {unknown}')
        selector=ManualSelector(catalog,defaults)
        adapter=StreamSTGSAdapter(vendor_root=args.vendor_root,ffmpeg_threads=args.ffmpeg_threads)
        limits=MemoryLimits(host_bytes=args.host_mib*1024**2,raw_bytes=args.raw_mib*1024**2,
                            gpu_bytes=args.gpu_mib*1024**2,transient_gpu_bytes=args.gpu_transient_mib*1024**2)
        runtime=PlaybackRuntime(catalog,adapter,cache,io,limits=limits,workers=args.workers,
                                prefetch=args.prefetch,loop=args.loop,selector=selector,telemetry=events,fps=args.fps,
                                upload_budget_s=args.upload_budget_ms/1000)
        # Fetch immutable current/next packages while the native GL context and
        # CUDA renderer initialize, rather than waiting for the first UI loop.
        runtime.pump()
        controller=CameraController(args.width,args.height)
        if args.mode=='native':
            from .viewer.window import NativeWindow
            with events.timed('native_window_init'):
                window=NativeWindow(controller,runtime,args.width,args.height)
        runtime.pump()
        from .rendering.fast import FASTRenderer
        with events.timed('renderer_cuda_init'):
            renderer=FASTRenderer(mode=args.mode,vendor_root=args.vendor_root,jit_optimized=args.fast_jit_specialization)
        import torch
        properties=torch.cuda.get_device_properties(torch.cuda.current_device())
        events.emit('gpu_environment',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                    gpu_name=properties.name,total_vram_bytes=properties.total_memory,
                    compute_capability=[properties.major,properties.minor],
                    cuda_device=torch.cuda.current_device())
        events.emit('runtime_configuration',mode=args.mode,workers=args.workers,
                    fetch_concurrency=args.fetch_concurrency,ffmpeg_threads=args.ffmpeg_threads,
                    prefetch=args.prefetch,budgets=vars(limits),
                    upload_budget_ms=args.upload_budget_ms,
                    fast_jit_specialization=args.fast_jit_specialization,
                    presentation_fps=runtime.clock.fps,instances=len(catalog.instances))
        analyzer=None
        if args.visibility_every:
            from .visibility.analyzer import VisibilityAnalyzer
            analyzer=VisibilityAnalyzer()
        controls=[]
        if args.control_trace:
            controls=json.loads(args.control_trace.read_text())
            if not isinstance(controls,list):raise ValueError('control trace must be a JSON list')
            previous=0
            for item in controls:
                if not isinstance(item,dict) or type(item.get('after_frame')) is not int or item['after_frame']<previous:
                    raise ValueError('trace actions require nondecreasing integer after_frame')
                if item.get('action') not in ('quality','seek','pause','play','loop'):
                    raise ValueError('unsupported trace action')
                previous=item['after_frame']
        control_index=0;resume_at=None;bandwidth_future=None;last_bandwidth_request=0.;next_ui_present=0.
        async def bandwidth_snapshot():
            return observer.estimator.estimate_bandwidth()
        calibrated=False
        while (window is not None or not runtime.clock.ended) and (args.frames is None or runtime.frame_id<args.frames):
            if window:
                window.poll()
                if window.closed:break
            now=time.monotonic()
            if bandwidth_future is not None and bandwidth_future.done():
                runtime.bandwidth_estimate=bandwidth_future.result()
                bandwidth_future=None
            if bandwidth_future is None and now-last_bandwidth_request>=.5:
                bandwidth_future=io.submit(bandwidth_snapshot());last_bandwidth_request=now
            if resume_at is not None and now>=resume_at:
                runtime.clock.play();resume_at=None
            while control_index<len(controls) and controls[control_index]['after_frame']<=runtime.frame_id:
                item=controls[control_index];action=item['action'];control_index+=1
                events.emit('manual_control',**item)
                if action=='quality':runtime.request_quality(item['instance_id'],item['representation_id'])
                elif action=='seek':runtime.seek(float(item['seconds']))
                elif action=='play':runtime.clock.play()
                elif action=='loop':runtime.clock.loop=bool(item['enabled'])
                elif action=='pause':
                    runtime.clock.pause()
                    if 'duration_s' in item:
                        if float(item['duration_s'])<0:raise ValueError('pause duration must be nonnegative')
                        resume_at=now+float(item['duration_s'])
            runtime.pump()
            if not calibrated:
                handle=next((j.handle for j in runtime.jobs.values() if j.handle is not None),None)
                if handle is not None and handle.cameras:
                    from ..shared.geometry import Camera
                    controller.adopt(Camera.from_package(handle.cameras[0],controller.width,controller.height))
                    if len(catalog.instances)>1:
                        import numpy as np
                        controller.fit([np.asarray(i.translation)+np.asarray(catalog.contents[i.content_id].pivot)
                                        for i in catalog.instances],[i.scale for i in catalog.instances])
                    calibrated=True
            if first_frame is None and time.monotonic()-started>args.startup_timeout:
                raise TimeoutError('No decoder-ready joint frame before startup deadline')
            if args.unpaced:runtime.clock.make_due()
            camera=controller.camera()
            result=runtime.render(camera,renderer)
            if result:
                controller.changed=False
                if first_frame is None:first_frame=time.monotonic()
                if analyzer and runtime.frame_id%args.visibility_every==0:
                    ids=[i.id for i in catalog.instances if i.id in runtime.required]
                    visibility=analyzer.analyze(dict(zip(ids,runtime.last_states)),camera,
                                                timestamp=result.scene_time,revision=catalog.revision)
                    runtime.visibility={key:value.A_visible for key,value in visibility.per_object.items()}
                    for job in runtime.jobs.values():
                        job.visibility=runtime.visibility.get(job.instance_id,1.)
                    events.emit('visibility',scene_time=result.scene_time,
                                backend='streamstgs_cuda_contribution',
                                duration_s=visibility.metadata.get('duration_s',0.),
                                visible_area=runtime.visibility)
            # Input/GOP polling can run faster than the display. Repaint an
            # unchanged UI at 60 Hz instead of repeatedly blitting/querying CUDA
            # diagnostics on every idle poll. Fresh frames/camera input present
            # immediately and retain the same fixed-step playback clock.
            present=window and (result is not None or controller.changed or time.monotonic()>=next_ui_present)
            if present:
                if result is None:
                    if runtime.last_states:
                        if controller.changed or not renderer.present_cached(camera):
                            runtime.refresh(camera,renderer)
                    else:renderer.clear(camera)
                    controller.changed=False
                window.draw_ui();window.swap()
                next_ui_present=time.monotonic()+1/60
                if result:
                    presentations+=1
                    if first_presented is None:
                        first_presented=time.monotonic()
                        events.emit('first_scene_swap',frame_id=result.frame_id,elapsed_s=first_presented-started)
                events.emit('presentation',fresh=bool(result),frame_id=runtime.frame_id)
            if not result:time.sleep(.001)
        status='completed'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        events.emit('failure',error=error)
        raise
    finally:
        elapsed=time.monotonic()-started
        if runtime:report.update(runtime.diagnostics())
        report.update(schema='multiobj3dgs.playback.v1',status=status,error=error,mode=args.mode,
                      publication_revision=catalog.revision if catalog else None,
                      elapsed_s=elapsed,startup_to_first_frame_s=first_frame-started if first_frame else None,
                      startup_to_first_swap_s=first_presented-started if first_presented else None,
                      fresh_scene_frames=runtime.frame_id if runtime else 0,
                      end_to_end_fresh_scene_fps=(runtime.frame_id/elapsed) if runtime else 0,
                      fresh_frame_swap_completions=presentations if args.mode=='native' else None,
                      physical_display_fps=None,
                      pacing_policy='fixed presentation steps; no frame drops; wall deadline resynchronized after stalls/lateness',
                      scope='network/cache + GOP preparation + reconstruction + joint rendering; EGL completion is not physical display',
                      traffic=observer.snapshot(),telemetry=events.summary())
        try:
            if runtime:runtime.close()
            if renderer and hasattr(renderer,'close'):renderer.close()
            if window:window.close()
            async def cleanup():
                if cache:await cache.close()
                if client:await client.aclose()
            io.submit(cleanup()).result(timeout=60)
        finally:
            io.close();events.close()
            if args.report:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                with args.report.open('x') as out:json.dump(report,out,indent=2)
            print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
