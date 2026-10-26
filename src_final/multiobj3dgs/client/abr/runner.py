"""Composition root; codec, transfer, cache, rendering and scheduling are reused."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from .config import ROOT, path


def _publication(config):
    if config.get('mpd'):
        value=str(config['mpd'])
        if urlsplit(value).scheme in ('http','https'):return value
        result=path(value)
        return result/'manifest.mpd' if result.is_dir() else result
    from multiobj3dgs.content.publication import publish
    archives=[path(p) for p in config.get('archives',[])]
    if not archives:
        for name in dict.fromkeys(config['contents']):
            choices=sorted((ROOT/'data_stream').glob('streamstgs_'+name+'_*crf20_28_36_40.tar.gz'))
            if len(choices)!=1:raise ValueError(f'Specify --archive for {name}: found {len(choices)} candidates')
            archives.append(choices[0])
    signature=hashlib.sha256(''.join(str(p.resolve()) for p in archives).encode()).hexdigest()[:12]
    destination=path(config.get('publication') or 'output/abr_publications/'+signature)
    # Existing publication is immutable. Avoid republishing on each playback.
    mpd=destination/'manifest.mpd'
    if not mpd.exists():publish(archives,destination)
    return mpd


def _scene(catalog, config):
    from multiobj3dgs.shared.metadata import SceneInstance
    if config.get('scene'):
        scene=config['scene']
        if not isinstance(scene,dict):scene=json.loads(path(scene).read_text())
        instances=tuple(SceneInstance(**item) for item in scene['instances'])
        return replace(catalog,instances=instances,duration=float(scene['duration']))
    names=config['contents']
    count=config.get('objects',len(names))
    instances=[]
    duration=max(catalog.contents[name].duration for name in names)
    for i in range(count):
        name=names[i%len(names)]
        content=catalog.contents[name]
        instances.append(SceneInstance(f'{name}-{i}',name,
            translation=((i-(count-1)/2)*1.8,0.,0.),
            active_duration=content.duration,loop=bool(config['loop'])))
    return replace(catalog,instances=tuple(instances),duration=duration)


async def _setup(config, mpd, events):
    from multiobj3dgs.shared.transfer import NetworkClient
    from multiobj3dgs.client.metadata import MPDParser
    from multiobj3dgs.client.network.bandwidth import BandwidthEstimator
    from .network import PipelineTransferObserver
    from .network_delivery import PacedHTTPOrigin
    from .catalog import QualityCatalog
    from .binding import QualityBoundCache
    estimator=BandwidthEstimator(window_size=config['bandwidth_window'],
        max_sample_age_s=config['bandwidth_max_age_s'],startup_bandwidth_bps=config['startup_mbps']*1e6)
    observer=PipelineTransferObserver(estimator=estimator,concurrency=config['fetch_concurrency'],events=events)
    origin=None;client=None;cache=None
    try:
        if isinstance(mpd,Path):
            if config.get('base_url'):
                url=config['base_url'].rstrip('/')+'/'+mpd.name
            else:
                schedule=config.get('delivery_trace')
                if schedule and not isinstance(schedule,list):schedule=json.loads(path(schedule).read_text())
                origin=PacedHTTPOrigin(mpd.parent,mbps=config['server_mbps'],schedule=schedule)
                await origin.__aenter__()
                url=origin.base_url+mpd.name
        else:url=mpd
        client=NetworkClient(http_version='h1',concurrency=config['fetch_concurrency'],total_timeout_s=config['startup_timeout'])
        client.subscribe(observer.transport_completion)
        class MetadataClient:
            async def fetch(self,spec):
                observer.started(spec)
                try:
                    result=await client.fetch(spec)
                    observer(result)
                    return result
                finally:observer.cancelled(spec)
        catalog=await MPDParser.from_http(url,MetadataClient())
        catalog=_scene(catalog,config)
        quality=QualityCatalog(catalog,path(config['quality_metadata']))
        cache=QualityBoundCache(path(config['cache']),config['cache_mib']*1024**2,client=client,observer=observer,events=events)
        cache.bind_quality(catalog,quality)
        await cache.verify_resident(catalog)
        events.emit('abr_configuration',config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()},
                    mpd=url,publication_revision=catalog.revision,scene=[asdict(i) for i in catalog.instances],
                    network_path='HTTP Request API only; local files served by origin',
                    local_delivery_mode=('unthrottled' if origin.unthrottled else 'network-emulated') if origin else 'external-origin')
        return catalog,quality,cache,client,observer,origin
    except BaseException:
        if cache:await cache.close()
        if client:await client.aclose()
        if origin:await origin.__aexit__()
        raise


def _planner(config,catalog,quality,cache,events):
    from .planner import GOPPlanner
    return GOPPlanner(catalog,quality,cache,mode=config['abr_mode'],fixed_crf=config['fixed_crf'],
        safety_factor=config['safety_factor'],delta_bps=round(config['delta_mbps']*1e6),
        startup_bandwidth_bps=config['startup_mbps']*1e6,events=events)


async def _network_evaluation(config,catalog,cache,observer,planner,events):
    """Actual selected-GOP delivery/materialization, explicitly without a decoder."""
    from multiobj3dgs.client.scene import media_time
    rows=[]
    for gop in range(config['network_gops']):
        scene_time=gop*min(s.duration for c in catalog.contents.values() for r in c.representations.values() for s in r.segments)
        targets={}
        for instance in catalog.instances:
            content=catalog.contents[instance.content_id]
            stamp=media_time(instance,content,scene_time)
            if stamp is not None:targets[instance.id]=content.segment_at(planner.selected[instance.id],stamp).gop_index
        if not targets:break
        estimate=observer.estimator.estimate_bandwidth()
        selected=planner.plan(targets,estimate.bps,{},context={'scene_time':scene_time,'scope':'network-eval-no-render'})
        segments=[catalog.contents[planner.instances[i].content_id].representations[rep].segments[targets[i]] for i,rep in selected.items()]
        expected=cache.missing_bytes(segments)
        before=observer.snapshot().get('total_bytes',0)
        began=time.monotonic()
        packages=await asyncio.gather(*(cache.materialize(segment) for segment in segments))
        try:
            actual=observer.snapshot().get('total_bytes',0)-before
            row={'scene_time':scene_time,'selected':planner.last_decision['selected'],
                 'feasible':planner.last_decision['feasible'],'budget_bps':planner.last_decision['budget_bps'],
                 'expected_missing_union_bytes':expected,'actual_http_bytes':actual,'duration_s':time.monotonic()-began,
                 'predicted_bandwidth_bps':estimate.bps,'next_bandwidth_bps':observer.estimator.estimate_bandwidth().bps,
                 'timestamps':{i:[f.seconds for f in segment.frames] for i,segment in zip(selected,segments)}}
            if actual!=expected:raise AssertionError(f'HTTP bytes {actual} differ from exact selected resource union {expected}')
            rows.append(row);events.emit('network_gop_complete',**row)
        finally:
            for package in packages:cache.release_package(package)
    return {'network_gops':rows,'fresh_scene_frames':0,'gpu_validation':'not_attempted_in_network_eval',
            'scope':'HTTP → harmonic bandwidth → DP → chosen GOP requests → verified materialization; no visibility/decode/render',
            'viewport_quality':None,'rebuffering':None}


def _native(config,catalog,quality,cache,io,observer,planner,events,*,application_started=None):
    from multiobj3dgs.client.runtime import MemoryLimits
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.client.viewer import CameraController
    from .runtime import QualityAwarePlaybackRuntime
    from .visibility import RuntimeVisibility
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU validation blocked: NVIDIA CUDA device/driver unavailable')
    properties=torch.cuda.get_device_properties(torch.cuda.current_device())
    events.emit('abr_gpu_environment',gpu_name=properties.name,total_vram_bytes=properties.total_memory,
                torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                device=torch.cuda.current_device())
    adapter=StreamSTGSAdapter(vendor_root=path(config['vendor_root']) if config.get('vendor_root') else None,
                             ffmpeg_threads=config['ffmpeg_threads'])
    limits=MemoryLimits(host_bytes=config['host_mib']*1024**2,raw_bytes=config['raw_mib']*1024**2,
        gpu_bytes=config['gpu_mib']*1024**2,transient_gpu_bytes=config['gpu_transient_mib']*1024**2)
    runtime=QualityAwarePlaybackRuntime(catalog,adapter,cache,io,selector=planner,
        visibility_sampler=RuntimeVisibility(interval_s=config['visibility_interval_s'],analysis_width=config['analysis_width']),
        limits=limits,workers=config['workers'],prefetch=config['prefetch'],loop=config['loop'],
        telemetry=events,upload_budget_s=config['upload_budget_ms']/1000)
    window=renderer=None;began=time.monotonic();first=None;native_presentations=0
    controller=CameraController(config['width'],config['height'])
    try:
        runtime.pump()
        if config['mode']=='native':
            from .diagnostics import ABRWindow
            window=ABRWindow(controller,runtime,config['width'],config['height'],diagnostics=config['diagnostics'])
        from multiobj3dgs.client.rendering.fast import FASTRenderer
        renderer=FASTRenderer(mode=config['mode'],vendor_root=path(config['vendor_root']) if config.get('vendor_root') else None)
        if config.get('capture_dir'):
            from .capture import CaptureRenderer
            frames=config.get('capture_frames',[0])
            if isinstance(frames,str):frames=[int(x) for x in frames.split(',')]
            renderer=CaptureRenderer(renderer,path(config['capture_dir']),frames,events)
        calibrated=False;estimate_future=None;next_estimate=0.;next_ui=0.;next_diagnostic=0.
        controls=json.loads(path(config['control_trace']).read_text()) if config.get('control_trace') else []
        control_index=0;resume_at=None
        async def estimate():return observer.estimator.estimate_bandwidth(),observer.snapshot()
        while (window or not runtime.clock.ended) and (config['frames'] is None or runtime.frame_id<config['frames']):
            if window:
                window.poll()
                if window.closed:break
            now=time.monotonic();force_visibility=False
            if estimate_future is not None and estimate_future.done():
                runtime.bandwidth_estimate,runtime.network_diagnostics=estimate_future.result();estimate_future=None
            if estimate_future is None and now>=next_estimate:
                estimate_future=io.submit(estimate());next_estimate=now+.25
            if resume_at is not None and now>=resume_at:runtime.clock.play();resume_at=None
            while control_index<len(controls) and controls[control_index]['after_frame']<=runtime.frame_id:
                item=controls[control_index];control_index+=1;action=item['action']
                events.emit('manual_control',**item)
                if action=='quality':runtime.request_quality(item['instance_id'],item['representation_id'])
                elif action=='seek':runtime.seek(float(item['seconds']))
                elif action=='play':runtime.clock.play()
                elif action=='loop':runtime.clock.loop=bool(item['enabled'])
                elif action=='camera':
                    if 'orbit' in item:controller.orbit(*item['orbit'])
                    if 'pan' in item:controller.pan(*item['pan'])
                    if 'zoom' in item:controller.zoom(float(item['zoom']))
                    if 'target' in item:
                        import numpy as np
                        target=np.asarray(item['target'],dtype=float)
                        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('Camera target must be finite XYZ')
                        controller.target=target;controller._invalidate()
                    force_visibility=True
                elif action=='pause':
                    runtime.clock.pause()
                    if 'duration_s' in item:resume_at=now+float(item['duration_s'])
                else:raise ValueError('Unsupported control action: '+action)
            runtime.pump()
            if not calibrated:
                handle=next((j.handle for j in runtime.jobs.values() if j.handle is not None),None)
                if handle is not None and handle.cameras:
                    from multiobj3dgs.shared.geometry import Camera
                    controller.adopt(Camera.from_package(handle.cameras[0],controller.width,controller.height))
                    needs_framing=len(catalog.instances)>1 or any(
                        tuple(i.translation)!=(0.,0.,0.) or tuple(i.rotation)!=(1.,0.,0.,0.) or i.scale!=1.
                        for i in catalog.instances)
                    if needs_framing:
                        from .camera import fit_scene_camera
                        fit_scene_camera(controller,catalog)
                    calibrated_camera=controller.camera()
                    import numpy as np
                    centers={}
                    for instance in catalog.instances:
                        center=np.asarray(instance.translation)+np.asarray(catalog.contents[instance.content_id].pivot)
                        point=calibrated_camera.world_to_camera@np.array([*center,1.])
                        projected=calibrated_camera.K@point[:3]
                        centers[instance.id]={'pixel':(projected[:2]/projected[2]).tolist(),'depth':float(point[2])}
                    events.emit('abr_camera_calibrated',width=calibrated_camera.width,height=calibrated_camera.height,
                        K=calibrated_camera.K.tolist(),world_to_camera=calibrated_camera.world_to_camera.tolist(),
                        instance_centers=centers)
                    calibrated=True
            if first is None and now-began>config['startup_timeout']:raise TimeoutError('No joint decoder-ready frame before startup deadline')
            if config['unpaced']:runtime.clock.make_due()
            camera=controller.camera()
            result=runtime.render(camera,renderer)
            if result:
                if first is None:first=time.monotonic()
                controller.changed=False
                visibility=runtime.update_visibility(camera,force=force_visibility)
                if visibility is not None:
                    from .metrics import viewport_quality_proxy
                    events.emit('abr_viewport_quality_proxy',**viewport_quality_proxy(runtime,quality))
                if config['diagnostics'] and now>=next_diagnostic:
                    events.emit('abr_playback_snapshot',**runtime.diagnostics());next_diagnostic=now+.5
            present=window and (result is not None or controller.changed or now>=next_ui)
            if present:
                if result is None:
                    if hasattr(renderer,'capture_enabled'):renderer.capture_enabled=False
                    if runtime.last_states:
                        if controller.changed or not renderer.present_cached(camera):runtime.refresh(camera,renderer)
                        if controller.changed:runtime.update_visibility(camera)
                    else:renderer.clear(camera)
                    if hasattr(renderer,'capture_enabled'):renderer.capture_enabled=True
                    controller.changed=False
                window.draw_ui();window.swap();next_ui=now+1/60
                if result is not None:
                    native_presentations+=1
                    events.emit('abr_native_present',frame_id=result.frame_id,scene_time=result.scene_time,
                                sources=list(runtime._last_frame_sources),physical_presented_at=None)
            if result is None:time.sleep(.001)
        return {**runtime.diagnostics(),'fresh_scene_frames':runtime.frame_id,
                'startup_to_first_frame_s':first-began if first else None,'gpu_validation':'rendered',
                'end_to_end_startup_to_first_frame_s':first-application_started if first and application_started is not None else None,
                'native_fresh_frame_swap_completions':native_presentations if window else None,
                'physical_display_fps':None,
                'scope':'HTTP → bandwidth → measured PSNR DP → decode → joint native rendering; current-pose visibility'}
    finally:
        runtime.close()
        if renderer:renderer.close()
        if window:window.close()


def run(config):
    from multiobj3dgs.client.runtime import IOLoop
    from multiobj3dgs.shared.telemetry import EventLog
    stamp=time.strftime('%Y%m%dT%H%M%S')
    report_path=path(config.get('report') or f'output/abr_pipeline/{stamp}-{time.time_ns()%1000000}/report.json')
    if report_path.exists() or report_path.with_suffix('.jsonl').exists():raise FileExistsError('Choose a new report path')
    events=EventLog(report_path.with_suffix('.jsonl'))
    io=IOLoop();resources=None;planner=None;began=time.monotonic()
    report={'schema':'multiobj3dgs.quality-aware-pipeline.v1','mode':config['mode'],'status':'failed'}
    code=1
    try:
        mpd=_publication(config)
        resources=io.submit(_setup(config,mpd,events)).result(timeout=config['startup_timeout'])
        catalog,quality,cache,client,observer,origin=resources
        planner=_planner(config,catalog,quality,cache,events)
        if config['mode']=='network-eval':
            result=io.submit(_network_evaluation(config,catalog,cache,observer,planner,events)).result()
        else:result=_native(config,catalog,quality,cache,io,observer,planner,events,application_started=began)
        report.update(result,status='completed',publication_revision=catalog.revision,instances=[asdict(i) for i in catalog.instances])
        code=0
    except KeyboardInterrupt:
        report.update(status='interrupted');code=130
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        report.update(status='blocked_gpu' if 'NVIDIA CUDA device/driver unavailable' in error else 'failed',error=error)
        events.emit('failure',error=error)
    finally:
        if planner:report['decisions']=planner.decisions
        if resources:
            report['traffic_at_playback_end']=resources[4].snapshot()
            async def cleanup():
                await resources[2].close();await resources[3].aclose()
                if resources[5]:await resources[5].__aexit__()
            try:io.submit(cleanup()).result(timeout=60)
            except Exception as exc:report['cleanup_error']=str(exc)
            report['traffic']=resources[4].snapshot()
        io.close()
        report.update(elapsed_s=time.monotonic()-began,telemetry=events.summary())
        events.close();report_path.parent.mkdir(parents=True,exist_ok=True)
        with report_path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False,default=str)
    print(json.dumps({'status':report['status'],'report':str(report_path),'error':report.get('error')},indent=2))
    return code
