"""Requested 315-case real HTTP/2 evaluation using the existing two-phase runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

from experiments.adaptation_baselines.exhaustive import EXHAUSTIVE_SEARCH
from experiments.adaptation_baselines.methods import NAVA,VISIBILITY_PSNR,VISIBILITY_MSE,VISIBILITY_WEIGHTED_SSE
from experiments.visibility_weighted_sse.common import digest,file_digest,read_json,require
from experiments.visibility_weighted_sse.trace import TraceCameraProvider,camera_to_dict
from . import two_phase as existing
from .runtime_matrix import PROJECT,load_settings,validate_inputs,completed
from .runtime_native import ResourceLimit,NativeQualityProvider,commit
from .runtime_scenes import make_scene
from .two_phase import TwoPhaseRunner,verified_marker,quality_job,job_schedule
from .quality import FrameQualityContext

METHODS=(NAVA,VISIBILITY_PSNR,VISIBILITY_WEIGHTED_SSE,VISIBILITY_MSE,EXHAUSTIVE_SEARCH)
CONFIG=PROJECT/'experiments/streaming_evaluation/configs/streaming_http2_dp015.json'
ROOT=PROJECT/'experiments/streaming_evaluation/runs/streaming_http2_dp015_v1'
REPORT=PROJECT/'reports/STREAMING_HTTP2_DP015_EVALUATION.md'
UNAVAILABLE={'TIMEOUT','MEMORY_LIMIT','INFEASIBLE','ERROR'}


def validate_protocol(cfg):
    require(tuple(cfg['methods'])==METHODS,'This evaluation requires exactly the four existing utilities plus exhaustive PSNR')
    require(cfg['object_counts']==[3,5,7,10,12,15,20] and cfg['playback_durations_s']==[10,20,30] and
            cfg['scenarios']==['camera_rotation','long_line','random_spatial'],'Requested 315-case matrix differs')
    require(cfg['network']['transport']=='real_http2' and cfg['network']['max_concurrent_streams']==16,
            'Actual verified HTTP/2 with 16 streams required')
    require(cfg['adaptation']['solver_limits']['delta_bps']==150000 and
            cfg['adaptation']['solver_limits']['discretization']=='bitrate','DP step must be 150000 bps')
    original=read_json(PROJECT/'experiments/streaming_evaluation/configs/runtime_six_second.json')
    for name in ('sources','source_duration_s','gop_duration_s','fps','seed','profiles','baseline_metadata',
                 'packages_root','metadata_root','bounds_source','renderer','quality','playback'):
        require(cfg[name]==original[name],f'Original protocol changed: {name}')
    for name,value in original['network'].items():
        require(cfg['network'][name]==value,f'Original trace/bandwidth configuration changed: {name}')
    require(cfg['adaptation']['safety_factor']==original['adaptation']['safety_factor'] and
            cfg['adaptation']['resource_sharing']==original['adaptation']['resource_sharing'],
            'Original safety/sharing policy changed')


def null_metrics(value):
    if isinstance(value,dict):
        return {k:null_metrics(v) for k,v in value.items()}
    if isinstance(value,list):
        return []
    if isinstance(value,(int,float,bool)):
        return None
    return value


def display_records(contexts,trace,simulation):
    """First scheduled display wall time from the recorded continuous buffer."""
    poses={p.identity:p for g in trace.gops for p in trace.get_trajectory(g['gop_id'])}
    intervals=[r for r in simulation['buffer_intervals'] if r['playing']]
    cursor=0;records=[]
    for context in contexts:
        media=context.global_time_s
        while cursor<len(intervals) and intervals[cursor]['media_end_s']<=media+1e-10:
            cursor+=1
        require(cursor<len(intervals),'Displayed frame has no recorded playback interval')
        interval=intervals[cursor]
        require(interval['media_start_s']-1e-8<=media<interval['media_end_s']+1e-8,
                'Frame falls outside actual recorded playback')
        pose=poses[context.camera_sample_id]
        records.append(dict(context=context.to_dict(),display_wall_time_s=interval['start_s']+media-interval['media_start_s'],
            source_frame=round(context.content_time_s*30),source_gop_id=context.content_gop_id,
            camera=camera_to_dict(pose.camera),instances=[dict(r) for r in pose.instances]))
    return records


def audit_transport(stream,network):
    transport=stream['transport_model'];requests=stream['requests'];ledger=stream['network_service_ledger']
    responded=[r for r in requests if r.get('http_version') is not None]
    checks=dict(actual_h2=transport['mode']=='real_http2' and transport['http_version']=='HTTP/2' and transport['alpn']=='h2',
        one_connection=transport['connection_count']==1,
        no_fallback=all(r['http_version']=='HTTP/2' and r['alpn_protocol']=='h2' for r in responded),
        unique_odd_stream_ids=len({r['stream_id'] for r in responded})==len(responded) and
                              all(r['stream_id']%2 for r in responded),
        same_client_port=len({r['server_client_port'] for r in responded})<=1,
        stream_limit=transport['max_observed_server_streams']<=16,
        nonoverlapping_ticks=all(a['end_s']<=b['start_s']+1e-12 for a,b in zip(ledger,ledger[1:])),
        shared_capacity=all(t['delivered_bytes']*8<=network.integral_bits(t['start_s'],t['end_s'])+1e-5 for t in ledger),
        ledger_bytes=sum(t['delivered_bytes'] for t in ledger)==transport['server_body_bytes_released'],
        client_body_bytes=sum(r['delivered_bytes'] for r in requests)<=transport['server_body_bytes_released'],
        exact_solver_feasibility=all(not d['allocation']['feasible'] or d['total_required_bytes']<=d['budget_bytes']
                                     for d in stream['decisions']))
    if requests and all(r['status']=='complete' for r in requests):
        checks['complete_byte_reconciliation']=sum(r['delivered_bytes'] for r in requests)==transport['server_body_bytes_released']
        checks['actual_multiplexing']=transport['actual_multiplexing_verified']
    if stream['method']==EXHAUSTIVE_SEARCH:
        checks['no_dp_or_substitution']=all(d['allocation']['diagnostics'].get('dp_used') is False and
            d.get('delivery_policy')!='q0_fallback' for d in stream['decisions'])
    require(all(checks.values()),f'Real HTTP/2/solver audit failed: {checks}')
    return checks


class EvaluationRunner(TwoPhaseRunner):
    def __init__(self,root,cfg,*,smoke=False,seed_cache=None,report_path=REPORT):
        super().__init__(root,cfg)
        self.smoke,self.seed_cache,self.report_path=smoke,seed_cache,Path(report_path)
        self._report_lock=threading.RLock();self._stop_heartbeat=threading.Event()
        self._report_at=0.;self._science=None
        self.state.update(run_kind='bounded_smoke' if smoke else 'full_315',terminal_cases=0,
            timeout_cases=0,quality_error_cases=0)

    def validate_protocol(self):
        validate_protocol(self.cfg)

    def prepare_identity(self, identity):
        """Allow a trace-suite runner to pin additional orchestration sources."""
        return identity

    def seed_native_cache(self, identity):
        if self.seed_cache and not (self.root/'cache_seed.json').exists():
            require(read_json(self.seed_cache/'manifest.json')==identity,'Smoke cache scientific identity differs')
            if (self.seed_cache/'cache').exists():
                shutil.copytree(self.seed_cache/'cache',self.root/'cache',dirs_exist_ok=True)
            commit(self.root/'cache_seed.json',dict(source=str(self.seed_cache),source_manifest_sha256=file_digest(self.seed_cache/'manifest.json'),
                policy='Copies only exact-identity native caches; every payload is validated before reuse; no requests/decisions imported'))

    def progress(self,force=False):
        with self._report_lock:
            if not force and time.monotonic()-self.last_progress<10:
                return
            if self._science:
                changed=[p for p,h in self._science.items() if file_digest(PROJECT/p)!=h]
                require(not changed,f'Implementation changed during evaluation; resume requires reviewed new version: {changed}')
            super().progress(force=True)
            self.report()

    def report(self):
        with self._report_lock:
            from .evaluation_report import update_report
            update_report(self.root,self.state,self.report_path)

    def heartbeat(self):
        while not self._stop_heartbeat.wait(10):
            try:
                self.progress(force=True)
            except Exception:
                with (self.root/'logs/report_errors.log').open('a') as handle:
                    handle.write(traceback.format_exc())

    def prepare_streaming_export(self,phase,stream,performance,contexts):
        checks=audit_transport(stream,self.network)
        commit(phase/'transport_audit.json',checks)
        commit(phase/'utility_solver_summary.json',dict(records=[dict(
            global_segment_index=d['global_segment_index'],budget_bytes=d['budget_bytes'],
            utility=d.get('estimated_total_utility'),selected_quality=d.get('selected_quality'),
            solver_status=d['allocation']['status'],exact_byte_feasible=d['allocation']['feasible'],
            diagnostics=d['allocation']['diagnostics'],timings=t)
            for d,t in zip(stream['decisions'],performance['records'])]))
        solver_status=stream['decisions'][-1]['allocation']['status'] if stream['decisions'] else None
        if stream['method']==EXHAUSTIVE_SEARCH and solver_status in ('TIMEOUT','MEMORY_LIMIT','INFEASIBLE'):
            commit(phase/'aborted_streaming_diagnostics.json',dict(simulation=deepcopy(stream),
                note='Diagnostic prefix only; no incumbent or partial QoE/PSNR is published as this case result'))
            stream['status']=solver_status
            stream['metrics']=null_metrics(stream['metrics'])
            stream['metrics']['status']=solver_status
            stream['metrics']['unavailable_reason']='Exhaustive search did not certify a complete streaming run'
            stream['quality_frames']=[]

    def played_document(self,scene,trace,stream,contexts,parent):
        document=super().played_document(scene,trace,stream,contexts,parent)
        document.update(schema='streaming-played-frames.v2',displayed_frames=display_records(contexts,trace,stream),
            display_clock='first scheduled display in actual-wall-clock PlaybackBuffer; stall holds remain in buffer_intervals',
            phase_b_contract='Read persisted poses and selected CRFs; no trajectory regeneration or quality feedback')
        return document

    def extra_streaming_files(self,phase):
        return [name for name in ('transport_audit.json','utility_solver_summary.json','aborted_streaming_diagnostics.json')
                if (phase/name).exists()]

    def prepare_quality_export(self,folder,simulation,performance):
        records=read_json(folder/'phase_a/played_frames.json')['displayed_frames']
        by_time={r['context']['global_time_s']:r for r in records}
        for frame in simulation['quality_frames']:
            record=by_time[frame['global_time_s']]
            frame.update(display_wall_time_s=record['display_wall_time_s'],source_frame=record['source_frame'],
                exact_camera_and_instances_file='phase_a/played_frames.json')

    def load_run(self,folder):
        phase=folder/'phase_a';marker=verified_marker(phase,'completion.json')
        require(marker is not None,'Missing Phase A checkpoint')
        parent=folder.parent
        require(marker['trace_file_sha256']==file_digest(parent/'trace.json') and
                marker['scene_file_sha256']==file_digest(parent/'configuration.json'),'Recorded camera/scene changed')
        configuration=read_json(parent/'configuration.json')
        trace=TraceCameraProvider(read_json(parent/'trace.json'))
        document=read_json(phase/'played_frames.json')
        contexts=[FrameQualityContext(**r) for r in document['frames']]
        poses={p.identity:p for g in trace.gops for p in trace.get_trajectory(g['gop_id'])}
        require(len(document['displayed_frames'])==len(contexts),'Displayed frame record coverage differs')
        for context,record in zip(contexts,document['displayed_frames']):
            pose=poses[context.camera_sample_id]
            require(record['context']==context.to_dict() and record['camera']==camera_to_dict(pose.camera) and
                    record['instances']==[dict(r) for r in pose.instances],'Persisted display pose/selection differs')
        return configuration['scene'],trace,contexts,configuration['configuration_hash']

    def cases(self):
        if self.smoke:
            return [('camera_rotation',n,2,m) for n in (3,20) for m in self.cfg['methods']]
        return [(s,n,d,m) for n in self.cfg['object_counts'] for s in self.cfg['scenarios']
                for d in self.cfg['playback_durations_s'] for m in self.cfg['methods']]

    def terminal(self,folder):
        path=folder/'terminal.json'
        if not path.exists():return None
        record=read_json(path)
        if record.get('phase_a_sha256'):
            require(file_digest(folder/'phase_a/completion.json')==record['phase_a_sha256'],'Terminal checkpoint changed')
            verified_marker(folder/'phase_a','completion.json')
        return record

    def save_terminal(self,folder,status,**extra):
        record=dict(status=status,quality_eligible=status not in UNAVAILABLE,**extra)
        if (folder/'phase_a/completion.json').exists():
            record['phase_a_sha256']=file_digest(folder/'phase_a/completion.json')
        commit(folder/'terminal.json',record)
        return record

    def reset_engine(self):
        if self.engine:self.engine.close()
        self.engine=None
        self.engine_open(self.packages,self.times)

    def evaluate_case(self,profiles,metadata,scene,duration,method):
        folder=self.root/'matrix'/scene['kind']/f"N{scene['object_count']:02d}"/f'{duration:02d}s'/method
        terminal=self.terminal(folder)
        if terminal:
            self.state['streaming_completed_runs']+=1
            self.state['streaming_imported_runs']+=1
            return folder,terminal
        for attempt in range(1,3):
            try:
                folder=super().streaming(profiles,metadata,scene,duration,method,self.network)
                simulation=read_json(folder/'phase_a/simulation.json')
                status=simulation['status']
                terminal_status=('SUCCESS' if status=='completed' else status if status in UNAVAILABLE else
                                 'NETWORK_FAILURE' if status.startswith('network_') else 'ERROR')
                return folder,self.save_terminal(folder,terminal_status,simulation_status=status)
            except (ResourceLimit,KeyboardInterrupt):
                raise
            except Exception as exc:
                commit(folder/f'attempt_{attempt:02d}_error.json',dict(error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc(),phase='A'))
                if attempt==2:
                    return folder,self.save_terminal(folder,'ERROR',error=f'{type(exc).__name__}: {exc}')
                self.reset_engine()

    def offline_resilient(self,folders):
        self.state.update(phase='B',current=dict(stage='global_job_collection'))
        self.phase_started=time.monotonic();self.progress(force=True)
        plan=self.plan_quality(folders);jobs=plan['jobs']
        self.state.update(unique_quality_jobs=len(jobs),unique_images=plan['unique_images'],
            played_frame_occurrences=plan['frame_occurrences'],quality_jobs_processed=0,quality_cache_hits=0,
            quality_cache_misses=0)
        from collections import OrderedDict
        providers=OrderedDict();failures=[]
        for key in job_schedule(jobs):
            self.budget.check();job=jobs[key];relative=job['representative_run']
            self.state['current']=dict(stage='offline_native_joint_quality',job=key,run=relative,
                source_frame=job['source_frame'],source_gop=job['source_gop'])
            for attempt in range(1,3):
                try:
                    if relative not in providers:
                        scene,trace,_,_=self.load_run(self.root/relative)
                        providers[relative]=NativeQualityProvider(self.engine,trace,scene)
                        if len(providers)>4:providers.popitem(last=False)
                    provider=providers[relative];providers.move_to_end(relative)
                    context=FrameQualityContext(**job['context'])
                    require(quality_job(self.engine,provider,context)[0]==key,'Selected/reference render identity differs')
                    provider.measure(context);event=provider.records.pop()
                    self.state['quality_cache_hits' if event['cache_hit'] else 'quality_cache_misses']+=1
                    break
                except (ResourceLimit,KeyboardInterrupt):raise
                except Exception as exc:
                    commit(self.root/f'logs/phase_b/{key}-attempt-{attempt}.json',dict(
                        error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),job=key))
                    if attempt==2:failures.append(key)
                    else:self.reset_engine();providers.clear()
            self.state['quality_jobs_processed']+=1
            self.progress()
        self.engine.flush()
        commit(self.root/'phase_b_render_completion.json',dict(plan_sha256=file_digest(self.root/'phase_b_jobs.json'),
            unique_quality_jobs=len(jobs),frame_occurrences=plan['frame_occurrences'],failed_jobs=failures,workload=self.engine.stats))
        for folder in folders:
            self.budget.check()
            try:
                self.merge(folder)
                commit(folder/'phase_b_status.json',dict(status='COMPLETE',phase_a_sha256=file_digest(folder/'phase_a/completion.json')))
            except (ResourceLimit,KeyboardInterrupt):raise
            except Exception as exc:
                self.state['quality_error_cases']+=1
                commit(folder/'phase_b_status.json',dict(status='QUALITY_ERROR',error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc(),psnr_db=None))
            self.progress()

    def run(self,phase='all'):
        lock=(self.root/'writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        thread=None
        try:
            require(phase in ('A','B','all'),'Choose Phase A, B, or all')
            self.validate_protocol()
            self.state.update(status='validating',total_runs=len(self.cases()))
            plan=dict(schema='http2-dp015-case-plan.v1',cases=[dict(
                scenario=s,N=n,duration_s=d,method=m) for s,n,d,m in self.cases()],smoke=self.smoke)
            if (self.root/'run_plan.json').exists():
                require(read_json(self.root/'run_plan.json')==plan,'Existing case plan differs; choose a new output version')
            else:commit(self.root/'run_plan.json',plan)
            self.progress(force=True)
            profiles,metadata,self.packages,records,self.times,self.network,identity=validate_inputs(self.cfg)
            identity=self.prepare_identity(identity)
            self.profiles=profiles;self._science=identity['implementation_sha256']
            self.fingerprint=existing.pin_execution(self.root,identity)
            for relative in self._science:
                destination=self.root/'source_at_run'/relative
                if not destination.exists():destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/relative,destination)
            self.seed_native_cache(identity)
            self.engine_open(self.packages,self.times)
            self.state['status']='running';self.progress(force=True)
            thread=threading.Thread(target=self.heartbeat,name='evaluation-report-heartbeat',daemon=True);thread.start()
            eligible=[];outcomes={};scenes={}
            for family,count,duration,method in self.cases():
                if (family,count) not in scenes:scenes[family,count]=make_scene(family,count,records,seed=self.cfg['seed'])
                if phase=='B':
                    folder=self.root/'matrix'/family/f'N{count:02d}'/f'{duration:02d}s'/method
                    terminal=self.terminal(folder)
                    require(terminal is not None,'Phase B requires all persisted Phase A terminal cases')
                    self.state['streaming_completed_runs']+=1
                    self.state['streaming_imported_runs']+=1
                else:
                    folder,terminal=self.evaluate_case(profiles,metadata,scenes[family,count],duration,method)
                outcomes[str(folder.relative_to(self.root))]=file_digest(folder/'terminal.json')
                self.state['terminal_cases']+=1
                self.state['timeout_cases']+=terminal['status']=='TIMEOUT'
                self.state['failed_runs']+=terminal['status']=='ERROR'
                if terminal['quality_eligible']:eligible.append(folder)
                self.progress(force=True)
            commit(self.root/'phase_a_completion.json',dict(total_runs=len(outcomes),terminal_sha256=outcomes,
                eligible_quality_runs=[str(f.relative_to(self.root)) for f in eligible]))
            if phase=='A':
                self.state.update(status='phase_a_complete',current=None)
                return 0
            self.engine.close();self.engine=None;self.engine_open(self.packages,self.times)
            self.offline_resilient(eligible)
            self.state.update(status='complete',current=None)
            commit(self.root/'evaluation_completion.json',dict(cases=len(outcomes),quality_eligible=len(eligible),
                quality_complete=len(list((self.root/'matrix').glob('*/*/*/*/completion.json'))),
                quality_error_cases=self.state['quality_error_cases'],timeout_cases=self.state['timeout_cases'],
                failed_runs=self.state['failed_runs']))
            return 0
        except KeyboardInterrupt:
            self.state.update(status='interrupted',stop_reason='Checkpoints retained; rerun the same command to resume')
            return 130
        except ResourceLimit as exc:
            self.state.update(status='resource_limit',stop_reason=str(exc));return 3
        except Exception as exc:
            self.state.update(status='failed',stop_reason=f'{type(exc).__name__}: {exc}')
            commit(self.root/'runner_failure.json',dict(state=self.state,traceback=traceback.format_exc()))
            traceback.print_exc();return 1
        finally:
            self._stop_heartbeat.set()
            if thread:thread.join(timeout=30)
            if self.engine:self.engine.close()
            self.progress(force=True);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--smoke',action='store_true',help='Bounded two-second N3/N20, all five methods, both phases')
    parser.add_argument('--seed-cache-from',type=Path)
    parser.add_argument('--report-path',type=Path,default=REPORT)
    parser.add_argument('--max-wall-seconds',type=float)
    parser.add_argument('--max-native-seconds',type=float)
    parser.add_argument('--max-storage-gib',type=float)
    args=parser.parse_args();cfg=load_settings(args.config)
    for name,value in [('max_wall_seconds',args.max_wall_seconds),('max_native_seconds',args.max_native_seconds),
                       ('max_storage_bytes',args.max_storage_gib*1024**3 if args.max_storage_gib else None)]:
        if value is not None:require(value>0,'Positive resource limit required');cfg['limits'][name]=value
    return EvaluationRunner(args.root,cfg,smoke=args.smoke,
        seed_cache=args.seed_cache_from.resolve() if args.seed_cache_from else None,
        report_path=args.report_path).run()


if __name__=='__main__':
    raise SystemExit(main())
