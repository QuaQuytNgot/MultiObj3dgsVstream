#!/usr/bin/env python3
"""Reproducible cold-cache native runtime profiles, one GPU process at a time.

Example: python profile_native.py --case n2=PATH/manifest.mpd --output NEW_DIR
Use --source-root SNAPSHOT/src_final to run an unchanged baseline implementation.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, max=None)
    def percentile(q):
        p=(len(values)-1)*q; i=int(p)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(p-i)
    return dict(count=len(values), mean=sum(values)/len(values), p50=percentile(.5),
                p95=percentile(.95), max=values[-1])


def _rational_time(value):
    """Recover configuration fractions only within a small floating-point bound.

    A value such as the stored float of 7/30 represents that exact frame phase.
    A deliberately fractional phase just below it must remain below it. This
    helper is independent of production timeline code and never uses a broad
    epsilon to move a genuinely earlier timestamp across a frame boundary.
    """
    exact=Fraction(str(value));candidate=exact.limit_denominator(1_000_000_000)
    return candidate if abs(float(candidate)-float(value))<=4*math.ulp(float(value)) else exact


def validate_source_mappings(frames,index,fps=None,scene_loop=False,quality_controls=()):
    """Check source indices from rational frame steps, not observed timestamps.

    The legacy snapshot has no source_frames field; retain an explicit unknown
    result instead of treating absent evidence as a successful mapping check.
    No seeks are supported by this oracle. Quality intents use the explicit
    after_frame control trace and must activate at the next local GOP boundary.
    """
    present=sum('source_frames' in frame for frame in frames)
    if not present:
        return dict(valid=None,scope='source_frames telemetry unavailable (legacy snapshot)',
                    checked_frames=0,checked_objects=0,error_count=0,errors=[])
    errors=[];error_count=0;checked_objects=0
    def error(frame,message,instance=None,**values):
        nonlocal error_count
        error_count+=1
        if len(errors)<20:
            errors.append(dict(frame_id=frame.get('frame_id'),scene_time=frame.get('scene_time'),
                               instance_id=instance,message=message,**values))
    if index is None:
        return dict(valid=False,scope='source telemetry present but adjacent local index.json unavailable',
                    checked_frames=0,checked_objects=0,error_count=1,errors=[])
    contents={content['id']:content for content in index['contents']}
    frame_rate=_rational_time(fps if fps is not None else max(c['fps'] for c in contents.values()))
    scene_frames=math.ceil(_rational_time(index['duration'])*frame_rate) if scene_loop else None
    controls=sorted((c for c in quality_controls if c.get('action')=='quality'),
                    key=lambda c:c['after_frame'])
    control_cursor=0;pending={};active_reps={};previous_segments={};checked_intents=0
    for frame in frames:
        if 'source_frames' not in frame:
            error(frame,'Missing source_frames telemetry');continue
        actual={}
        for source in frame['source_frames']:
            identifier=source.get('instance_id')
            if identifier in actual:error(frame,'Duplicate instance telemetry',identifier)
            actual[identifier]=source
        frame_index=frame['frame_id']%scene_frames if scene_loop else frame['frame_id']
        scene=Fraction(frame_index,1)/frame_rate
        while control_cursor<len(controls) and controls[control_cursor]['after_frame']<=frame['frame_id']:
            control=controls[control_cursor];control_cursor+=1
            pending[control['instance_id']]=control['representation_id']
        expected={}
        for instance in index['instances']:
            identifier=instance['id'];content=contents[instance['content_id']]
            elapsed=scene-_rational_time(instance.get('scene_start',0.))
            duration=instance.get('active_duration')
            if elapsed<0 or (duration is not None and elapsed>=_rational_time(duration)):continue
            media=_rational_time(instance.get('media_offset',0.))+elapsed
            if instance.get('loop',False):media%=_rational_time(content['duration'])
            elif not 0<=media<_rational_time(content['duration']):continue
            expected[identifier]=(content,media)
        if set(actual)!=set(expected):
            error(frame,'Active instance set differs',expected=sorted(expected),actual=sorted(actual,key=str))
        for identifier,(content,media) in expected.items():
            if identifier not in actual:continue
            source=actual[identifier];checked_objects+=1
            representation=next((r for r in content['representations'] if r['id']==source.get('representation_id')),None)
            if representation is None:
                error(frame,'Unknown representation for instance content',identifier,
                      actual=source.get('representation_id'));continue
            segment=next((s for s in representation['segments']
                          if _rational_time(s['start'])<=media<_rational_time(s['start'])+_rational_time(s['duration'])),None)
            if segment is None:
                error(frame,'Media time has no mapped GOP',identifier,media_time=float(media));continue
            segment_start=_rational_time(segment['start'])
            # Seed an initial manual selection from the first observed frame;
            # subsequent selections are constrained by requested intent rather
            # than inferred from whichever representation happened to render.
            active_reps.setdefault(identifier,source['representation_id'])
            changed=identifier in previous_segments and previous_segments[identifier]!=segment_start
            if identifier in pending and (media==segment_start or changed):
                active_reps[identifier]=pending.pop(identifier);checked_intents+=1
            previous_segments[identifier]=segment_start
            if source['representation_id']!=active_reps[identifier]:
                error(frame,'Requested representation did not activate at next GOP',identifier,
                      expected=active_reps[identifier],actual=source['representation_id'])
            candidates=[mapping for mapping in segment['frames'] if _rational_time(mapping['seconds'])<=media]
            if not candidates:
                error(frame,'Media time has no mapped source frame',identifier,media_time=float(media));continue
            mapping=candidates[-1]
            for field,expected_value in (('gop_id',segment['id']),('source_frame',mapping['source_frame'])):
                if source.get(field)!=expected_value:
                    error(frame,'Source mapping differs',identifier,field=field,
                          expected=expected_value,actual=source.get(field))
            for field,expected_value in (('model_time',mapping['model_time']),('media_time',float(media))):
                value=source.get(field)
                tolerance=1e-9 if field=='model_time' else max(8*math.ulp(expected_value),8*math.ulp(float(value))) if isinstance(value,(int,float)) else 0.
                if not isinstance(value,(int,float)) or not math.isclose(value,expected_value,rel_tol=0,abs_tol=tolerance):
                    error(frame,'Source time mapping differs',identifier,field=field,
                          expected=expected_value,actual=value)
    return dict(valid=error_count==0,scope='rational frame_id/fps timeline; independent raw index mappings and next-GOP quality intents; no seeks',
                checked_frames=len(frames),checked_objects=checked_objects,
                checked_quality_intents=checked_intents,pending_quality_intents=len(pending),
                error_count=error_count,errors=errors)


def summarize(report, events, timing, samples, launch, index=None, scene_loop=False):
    frames=[e for e in events if e.get('stage')=='scene_frame' and e.get('fresh', True)]
    fps=next((e.get('presentation_fps', 30.) for e in events if e.get('stage')=='runtime_configuration'),30.)
    first=frames[0]['monotonic_s'] if frames else None
    last=frames[-1]['monotonic_s'] if frames else None
    first_render=timing.get('first_render_monotonic_s',first)
    first_swap=timing.get('first_swap_monotonic_s')
    stages={}
    for row in timing.get('records',[]):
        stages.setdefault(row['stage'],[]).append(row['duration_s'])
    active={}; boundaries=[]; intervals=[]
    for event in events:
        if event.get('stage')=='active':
            identity=event.get('instance_id'); current=(event.get('gop_id'),event.get('representation_id'))
            if identity in active and active[identity]!=current:
                boundaries.append(dict(instance_id=identity,previous=active[identity],current=current,
                                       completed_monotonic_s=event['monotonic_s']))
            active[identity]=current
    for previous,current in zip(frames,frames[1:]):
        delta=current['monotonic_s']-previous['monotonic_s']
        changes=[b for b in boundaries if previous['monotonic_s']<b['completed_monotonic_s']<=current['monotonic_s']]
        if changes:
            intervals.append(dict(frame_id=current['frame_id'],scene_time=current['scene_time'],
                interval_s=delta,excess_over_frame_period_s=max(0.,delta-1/fps),objects=changes))
    readiness=[];stall_start=None
    by_frame={e['frame_id']:e for e in frames}
    for event in events:
        if event.get('stage')=='readiness_stall_start':
            stall_start=event
        elif event.get('stage')=='readiness_stall_end':
            frame_id=event.get('frame_id');completion=by_frame.get(frame_id,{}).get('monotonic_s',event['monotonic_s'])
            start=(stall_start or {}).get('monotonic_s',event['monotonic_s']-event['duration_s'])
            boundary=any(start<=b['completed_monotonic_s']<=completion for b in boundaries)
            kind='startup' if frame_id==0 else 'gop_boundary' if boundary else 'temporal_feature'
            readiness.append(dict(frame_id=frame_id,scene_time=event.get('scene_time'),
                duration_s=event['duration_s'],kind=kind,start_monotonic_s=start,
                completed_monotonic_s=event['monotonic_s']))
            stall_start=None
    # Incremental uploads publish stage telemetry when their completion event is
    # ready; they do not call the synchronous realize_gpu wrapper. Prefer the
    # production job events when available and retain wrappers for snapshots.
    identity_keys={'instance_id','gop_id','representation_id','generation'}
    envelope_keys=identity_keys|{'schema','run_id','stage','monotonic_s'}
    def job_stage(event, method, duration_key):
        duration=event.get(duration_key,0.)
        return dict(stage=method, completed_monotonic_s=event['monotonic_s'],
            start_monotonic_s=event['monotonic_s']-duration,duration_s=duration,
            stages={k:v for k,v in event.items() if k not in envelope_keys},
            **{k:event[k] for k in identity_keys if k in event})
    gpu_events=[e for e in events if e.get('stage')=='gop_gpu_stages']
    realized=([job_stage(e,'realize_gpu','realize_gpu_s') for e in gpu_events]
              if gpu_events else [r for r in timing.get('records',[]) if r['stage']=='realize_gpu'])
    # Progressive publication emits a canonical-only event before the producer
    # finishes. Aggregate final records to avoid doubling/incomplete timings.
    cpu_events=[e for e in events if e.get('stage')=='gop_cpu_stages' and 'prepare_cpu_s' in e]
    prepared=([job_stage(e,'prepare_cpu','prepare_cpu_s') for e in cpu_events]
              if cpu_events else [r for r in timing.get('records',[]) if r['stage']=='prepare_cpu'])
    detailed={}
    for row in realized:
        for key,value in row.get('stages',{}).items():
            if key.endswith('_s') and isinstance(value,(float,int)):
                detailed.setdefault(key,[]).append(value)
    cpu_detailed={}
    for row in prepared:
        for key,value in row.get('stages',{}).items():
            if key.endswith('_s') and isinstance(value,(float,int)):
                cpu_detailed.setdefault(key,[]).append(value)
    cpu=[s['cpu_percent'] for s in samples if s.get('cpu_percent') is not None]
    rss=[s['rss_bytes'] for s in samples if s.get('rss_bytes') is not None]
    gpu=[s['gpu_utilization_percent'] for s in samples if s.get('gpu_utilization_percent') is not None]
    vram=[s['device_memory_mib'] for s in samples if s.get('device_memory_mib') is not None]
    contiguous_ids=[e['frame_id'] for e in frames]==list(range(len(frames)))
    duration=index.get('duration') if index is not None else None
    loop_frames=math.ceil(duration*fps-1e-9) if duration is not None and duration>0 else None
    contiguous_timestamps=bool(not scene_loop or loop_frames) and all(
        abs(e['scene_time']-((i%loop_frames)/fps if scene_loop else i/fps))<1e-6
        for i,e in enumerate(frames))
    gaps=[b['monotonic_s']-a['monotonic_s'] for a,b in zip(frames,frames[1:])]
    completion_lateness=[max(0.,gap-1/fps) for gap in gaps]
    completion_misses=sum(gap>1/fps for gap in gaps)
    completion_misses_1ms=sum(gap>1/fps+.001 for gap in gaps)
    mapping_validation=validate_source_mappings(frames,index,fps=fps,scene_loop=scene_loop,
        quality_controls=[event for event in events if event.get('stage')=='manual_control'])
    return dict(status=report.get('status'),frames=len(frames),presentation_fps=fps,
        launch_to_first_rendered_s=first_render-launch if first_render else None,
        launch_to_first_swap_s=first_swap-launch if first_swap else None,
        reported_startup_s=report.get('startup_to_first_frame_s'),
        elapsed_s=report.get('elapsed_s'),cold_fps=report.get('end_to_end_fresh_scene_fps'),
        post_first_frame_fps=(len(frames)-1)/(last-first) if first is not None and last>first else None,
        frame_latency_s=distribution(e['duration_s'] for e in frames),
        frame_interval_s=distribution(b['monotonic_s']-a['monotonic_s'] for a,b in zip(frames,frames[1:])),
        frame_completion_miss_count_strict=completion_misses,
        frame_completion_miss_count_over_1ms=completion_misses_1ms,
        frame_completion_miss_fraction_strict=completion_misses/len(gaps) if gaps else None,
        frame_completion_miss_fraction_over_1ms=completion_misses_1ms/len(gaps) if gaps else None,
        frame_completion_lateness_s=distribution(completion_lateness),
        frame_completion_missed_lateness_s=distribution(value for value in completion_lateness if value>0),
        frame_completion_deadline_scope='successive fresh render completion gaps; first frame excluded; strict and 1ms tolerance; independent of clock deadline resynchronization',
        deadline_misses=sum(bool(e.get('deadline_missed')) for e in frames),
        reported_deadline_miss_scope='player clock flag; baseline and optimized scheduled_deadline/rebuffer semantics may differ; use completion gaps for common comparison',
        compute_deadline_misses=sum(bool(e.get('compute_deadline_missed')) for e in frames),
        contiguous_frame_ids=contiguous_ids,contiguous_timestamps=contiguous_timestamps,
        scene_loop=scene_loop,scene_duration_s=duration,
        timestamp_continuity_scope='fixed presentation steps, modulo scene frame count when --loop is enabled; no seeks',
        source_mapping_valid=mapping_validation['valid'],source_mapping_validation=mapping_validation,
        rebuffer_count=report.get('rebuffer_count'),rebuffer_s=report.get('rebuffer_seconds'),
        boundary_intervals=intervals,
        boundary_excess_s=distribution(r['excess_over_frame_period_s'] for r in intervals),
        readiness_stalls=readiness,
        readiness_stall_s=distribution(r['duration_s'] for r in readiness),
        gop_readiness_stall_s=distribution(r['duration_s'] for r in readiness if r['kind']=='gop_boundary'),
        feature_readiness_stall_s=distribution(r['duration_s'] for r in readiness if r['kind']=='temporal_feature'),
        instrumented_method_s={k:distribution(v) for k,v in stages.items()},
        realized_stage_s={k:distribution(v) for k,v in detailed.items()},
        gop_cpu_stage_s={k:distribution(v) for k,v in cpu_detailed.items()},
        realized_gops=realized,prepared_gops=prepared,
        stage_record_source='runtime job telemetry' if gpu_events else 'method wrappers',
        cpu_preparation_shared_jobs=sum(e.get('stage')=='cpu_preparation_shared' for e in events),
        physical_prepare_cpu_calls=sum(r['stage']=='prepare_cpu' for r in timing.get('records',[])),
        cpu_percent=distribution(cpu),gpu_utilization_percent=distribution(gpu),
        peak_tree_rss_bytes=max(rss) if rss else None,device_peak_mib=max(vram) if vram else None,
        torch_peak_allocated_bytes=report.get('torch_process_peak_allocated_bytes'),
        torch_peak_reserved_bytes=report.get('torch_process_peak_reserved_bytes'))


def sample_process(process, known):
    import psutil
    rows=[]
    try:
        parent=psutil.Process(process.pid)
        processes=[parent,*parent.children(recursive=True)]
    except psutil.Error:
        processes=[]
    for entry in processes:
        try:
            prior=known.setdefault(entry.pid,entry)
            rows.append((prior.cpu_percent(None),prior.memory_info().rss))
        except psutil.Error:
            pass
    return dict(cpu_percent=sum(r[0] for r in rows),rss_bytes=sum(r[1] for r in rows),process_count=len(rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',action='append',required=True,metavar='NAME=MPD')
    p.add_argument('--case-args',action='append',default=[],metavar='NAME=JSON_ARRAY',
                   help='Per-case additional player argv, e.g. n2=["--ffmpeg-threads","2"]')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2]/'src_final')
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--prefetch',type=int,default=1)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--height',type=int,default=768)
    p.add_argument('--mode',choices=('native','headless'),default='native')
    p.add_argument('--sample-ms',type=int,default=250)
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--display',default=os.environ.get('DISPLAY',':1'))
    p.add_argument('--xauthority',default=os.environ.get('XAUTHORITY','/run/user/1000/gdm/Xauthority'))
    args=p.parse_args(argv)
    if args.sample_ms<50 or args.frames<1 or args.timeout<=0:
        p.error('Require frames>0, timeout>0 and sample-ms>=50')
    source=args.source_root.resolve()
    if not (source/'multiobj3dgs').is_dir() and (source/'src_final/multiobj3dgs').is_dir():
        source=source/'src_final'
    if not (source/'multiobj3dgs').is_dir():
        p.error('--source-root must contain multiobj3dgs or src_final/multiobj3dgs')
    cases={}
    for text in args.case:
        name,sep,mpd=text.partition('=')
        if not sep or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            p.error('--case requires a safe NAME=MPD')
        if name in cases:p.error(f'Duplicate case {name}')
        cases[name]=mpd if urlsplit(mpd).scheme in ('http','https') else str(Path(mpd).resolve())
    extras={name:[] for name in cases}
    for text in args.case_args:
        name,sep,value=text.partition('=')
        try: parsed=json.loads(value)
        except json.JSONDecodeError:p.error('--case-args requires NAME=JSON_ARRAY')
        if not sep or name not in cases or not isinstance(parsed,list) or not all(isinstance(v,str) for v in parsed):
            p.error('--case-args requires an existing case and a JSON string array')
        extras[name].extend(parsed)
    # Fail before launching GPU work if resource sampling cannot run.
    import psutil
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output.resolve()
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONUNBUFFERED']='1'
    env['MULTIOBJ3DGS_VENDOR_ROOT']=str(Path(__file__).resolve().parents[2]/'third_party/streamstgs')
    if args.mode=='native':
        env.pop('PYOPENGL_PLATFORM',None)
        env.update(DISPLAY=args.display,XAUTHORITY=args.xauthority,
            __NV_PRIME_RENDER_OFFLOAD='1',__GLX_VENDOR_LIBRARY_NAME='nvidia')
    else:env['PYOPENGL_PLATFORM']='egl'
    safe_env={k:env[k] for k in ('PYTHONPATH','PYOPENGL_PLATFORM','DISPLAY','XAUTHORITY',
        '__NV_PRIME_RENDER_OFFLOAD','__GLX_VENDOR_LIBRARY_NAME','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH',
        'MULTIOBJ3DGS_VENDOR_ROOT') if k in env}
    matrix=[]
    for name,mpd in cases.items():
        case=output/name;case.mkdir()
        entry=Path(__file__).with_name('profile_entry.py').resolve()
        command=[args.python,str(entry),'--timing',str(case/'timing.json'),'--',
            '--mpd',mpd,'--mode',args.mode,'--frames',str(args.frames),
            '--workers',str(args.workers),'--prefetch',str(args.prefetch),
            '--width',str(args.width),'--height',str(args.height),
            '--cache',str(case/'cache'),'--report',str(case/'playback.json'),*extras[name]]
        (case/'invocation.json').write_text(json.dumps(dict(argv=command,env=safe_env,
            source_root=str(source),cache_policy='new independent application cache',
            sampling_ms=args.sample_ms),indent=2)+'\n')
        samples=[];known={};timed_out=False
        with (case/'player.log').open('x') as log:
            launch=time.monotonic()
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                start=time.monotonic()
                sample=dict(monotonic_s=start,**sample_process(process,known))
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                if gpu.returncode==0:
                    values=gpu.stdout.strip().splitlines()[0].split(',')
                    sample.update(gpu_utilization_percent=int(values[0]),device_memory_mib=int(values[1]))
                samples.append(sample)
                if time.monotonic()-launch>args.timeout:
                    timed_out=True;process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    break
                time.sleep(max(0.,args.sample_ms/1000-(time.monotonic()-start)))
            wall=time.monotonic()-launch
        (case/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
        load=lambda path:json.loads(path.read_text()) if path.exists() else {}
        report=load(case/'playback.json');timing=load(case/'timing.json')
        events=[json.loads(line) for line in (case/'playback.jsonl').read_text().splitlines()] if (case/'playback.jsonl').exists() else []
        index_path=Path(mpd).parent/'index.json' if urlsplit(mpd).scheme not in ('http','https') else None
        index=load(index_path) if index_path is not None and index_path.exists() else None
        summary=summarize(report,events,timing,samples,launch,index,scene_loop='--loop' in extras[name])
        summary.update(case=name,mpd=mpd,source_root=str(source),returncode=process.returncode,
            process_wall_s=wall,timed_out=timed_out,expected_frames=args.frames)
        (case/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        matrix.append(summary)
        (output/'summary.json').write_text(json.dumps(matrix,indent=2)+'\n')
        print(json.dumps({k:summary[k] for k in ('case','returncode','frames',
            'launch_to_first_rendered_s','launch_to_first_swap_s','post_first_frame_fps',
            'peak_tree_rss_bytes','device_peak_mib')}),flush=True)
    return int(any(r['returncode'] or r['frames']!=args.frames or r['source_mapping_valid'] is False or not r['contiguous_frame_ids']
                   or not r['contiguous_timestamps'] for r in matrix))


if __name__=='__main__':raise SystemExit(main())
