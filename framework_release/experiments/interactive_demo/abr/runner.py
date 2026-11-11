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
    """Reference an existing immutable publication; never copy/publish media."""
    if config.get('mpd') or config.get('publication'):
        value=str(config.get('mpd') or config['publication'])
        if urlsplit(value).scheme in ('http','https'):return value
        result=path(value)
        mpd=result/'manifest.mpd' if result.is_dir() else result
        if not mpd.is_file():raise FileNotFoundError('Existing publication required: '+str(mpd))
        return mpd
    from multiobj3dgs.client.metadata import MPDParser
    wanted=set(config['contents'])
    for mpd in sorted((ROOT/'output/abr_publications').glob('*/manifest.mpd')):
        catalog=MPDParser.from_local(mpd)
        if wanted <= set(catalog.contents):return mpd
    raise ValueError('No existing publication covers '+', '.join(sorted(wanted))+
                     '; supply --mpd with an existing publication. This isolated demo never publishes/copies media.')


def _scene(catalog, config):
    from ..placement import build_scene
    return build_scene(catalog,config)


def _output_path(value):
    from .config import EXPERIMENT_ROOT
    result=path(value).resolve()
    if not result.is_relative_to(EXPERIMENT_ROOT/'output'):
        raise ValueError('All demo outputs must stay under '+str(EXPERIMENT_ROOT/'output'))
    return result


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
        quality=QualityCatalog(catalog,path(config['quality_metadata']),mse_root=path(config['mse_profiles']))
        cache=QualityBoundCache(_output_path(config['cache']),config['cache_mib']*1024**2,client=client,observer=observer,events=events)
        cache.bind_quality(catalog,quality)
        await cache.verify_resident(catalog)
        from ..placement import placement_report
        events.emit('demo_placement',**placement_report(catalog,config))
        events.emit('mse_profile_binding',**quality.mse_audit)
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
        scopes={identifier:(identifier,0,0,gop,targets[identifier]) for identifier in targets}
        selected=planner.plan(targets,estimate.bps,None,context={
            'scene_time':scene_time,'scope':'network-eval-no-render','delivery_scopes':scopes})
        segments={identifier:catalog.contents[planner.instances[identifier].content_id].representations[rep].segments[targets[identifier]]
                  for identifier,rep in selected.items()}
        expected=sum(cache.missing_bytes(segment,scope=scopes[i]) for i,segment in segments.items())
        before=observer.snapshot().get('total_bytes',0)
        began=time.monotonic()
        packages=[]
        try:
            # A task that succeeds before a peer fails still owns a lease.
            results=await asyncio.gather(*(cache.materialize(segment,scope=scopes[i],representation_id=selected[i])
                for i,segment in segments.items()),return_exceptions=True)
            packages=[p for p in results if not isinstance(p,BaseException)]
            error=next((p for p in results if isinstance(p,BaseException)),None)
            if error is not None:raise error
            actual=observer.snapshot().get('total_bytes',0)-before
            row={'scene_time':scene_time,'selected':planner.last_decision['selected'],
                 'feasible':planner.last_decision['feasible'],'budget_bps':planner.last_decision['budget_bps'],
                 'expected_independent_bytes':expected,'actual_http_bytes':actual,'duration_s':time.monotonic()-began,
                 'predicted_bandwidth_bps':estimate.bps,'next_bandwidth_bps':observer.estimator.estimate_bandwidth().bps,
                 'timestamps':{i:[f.seconds for f in segment.frames] for i,segment in segments.items()}}
            if actual!=expected:raise AssertionError(f'HTTP bytes {actual} differ from independent demand {expected}')
            rows.append(row);events.emit('network_gop_complete',**row)
        finally:
            for package in packages:cache.release_package(package)
    return {'network_gops':rows,'fresh_scene_frames':0,'gpu_validation':'not_attempted_in_network_eval',
            'scope':'HTTP → harmonic bandwidth → DP → chosen GOP requests → verified materialization; no visibility/decode/render',
            'viewport_quality':None,'rebuffering':None}


def _native(config,catalog,quality,cache,io,observer,planner,events,*,application_started=None):
    from ..runtime import MemoryLimits
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
            renderer=CaptureRenderer(renderer,_output_path(config['capture_dir']),frames,events)
        calibrated=False;estimate_future=None;next_estimate=0.;next_ui=0.;next_diagnostic=0.
        controls=json.loads(path(config['control_trace']).read_text()) if config.get('control_trace') else []
        control_index=0;resume_at=None
        async def estimate():return observer.estimator.estimate_bandwidth(),observer.snapshot()
        while (window or not runtime.clock.ended) and (config['frames'] is None or runtime.frame_id<config['frames']):
            if window:
                window.poll()
                if window.closed:break
            now=time.monotonic();force_visibility=False
            if config.get('wall_time_s') is not None and now-began>=config['wall_time_s']:
                raise TimeoutError('Native smoke wall-time limit reached')
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
                        from ..placement import frame_scene_camera
                        frame_scene_camera(controller,catalog,config)
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
                'scope':'HTTP → bandwidth → Visibility × MSE DP (150 kbps default) → independent decode → joint native rendering; current-pose visibility'}
    finally:
        runtime.close()
        if renderer:renderer.close()
        if window:window.close()


def run(config):
    from ..runtime import IOLoop
    from multiobj3dgs.shared.telemetry import EventLog
    stamp=time.strftime('%Y%m%dT%H%M%S')
    report_path=_output_path(config.get('report') or f'experiments/interactive_demo/output/runs/{stamp}-{time.time_ns()%1000000}/report.json')
    if report_path.exists() or report_path.with_suffix('.jsonl').exists():raise FileExistsError('Choose a new report path')
    events=EventLog(report_path.with_suffix('.jsonl'))
    io=IOLoop();resources=None;planner=None;began=time.monotonic()
    report={'schema':'interactive-demo.v1','mode':config['mode'],'status':'failed'}
    code=1
    try:
        mpd=_publication(config)
        resources=io.submit(_setup(config,mpd,events)).result(timeout=config['startup_timeout'])
        catalog,quality,cache,client,observer,origin=resources
        planner=_planner(config,catalog,quality,cache,events)
        if config['mode']=='network-eval':
            result=io.submit(_network_evaluation(config,catalog,cache,observer,planner,events)).result()
        else:result=_native(config,catalog,quality,cache,io,observer,planner,events,application_started=began)
        from ..placement import placement_report
        report.update(result,status='completed',publication_revision=catalog.revision,instances=[asdict(i) for i in catalog.instances],
                      placement=placement_report(catalog,config),mse_profile_binding=dict(quality.mse_audit))
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
            report['resource_spool_after_cleanup']=resources[2].snapshot()
            if resources[5]:
                report['origin_requests']=resources[5].requests
        io.close()
        report.update(elapsed_s=time.monotonic()-began,telemetry=events.summary())
        events.close();report_path.parent.mkdir(parents=True,exist_ok=True)
        with report_path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False,default=str)
    print(json.dumps({'status':report['status'],'report':str(report_path),'error':report.get('error')},indent=2))
    return code
