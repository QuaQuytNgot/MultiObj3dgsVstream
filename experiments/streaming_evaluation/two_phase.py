"""Streaming first, globally deduplicated native quality second; one GPU owner."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import traceback

from experiments.adaptation_baselines.methods import METHODS
from experiments.adaptation_baselines.exhaustive import EXHAUSTIVE_SEARCH
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require
from .cache_upgrade import resolve
from .metrics import summarize_metrics
from .quality import FrameQualityContext, contexts_for_playback, collect_quality
from .runtime_adapter import RuntimePlanner, RuntimeVisibilityAdapter, ReadOnlyVisibilityEngine
from .runtime_matrix import (PROJECT, DEFAULT_CONFIG, DEFAULT_ROOT, load_settings, validate_inputs,
    completed, network_checkpoint, export_run)
from .runtime_native import NativeEngine, NativeQualityProvider, ExecutionBudget, ResourceLimit, commit
from .runtime_scenes import make_scene, make_trace, bind_profiles
from .simulator import simulate
from .timeline import Timeline

REVISION = 'two_phase_v1'


def pin_execution(root, identity):
    """Keep the validated pixel namespace; explicitly pin the new orchestration."""
    fingerprint, parent_revision = resolve(root, identity)
    files = [Path(__file__), Path(__file__).with_name('two_phase_report.py'),
             PROJECT/'scripts/run_streaming_experiments.sh']
    record = dict(revision=REVISION, parent_revision=parent_revision,
        original_artifact_fingerprint=fingerprint, validated_identity_sha256=digest(identity),
        orchestration_sha256={str(p.relative_to(PROJECT)):file_digest(p) for p in files},
        compatibility='Execution ordering and additional checkpoints only. All scientific inputs, '
          'native renderer, image protocol, adaptation utilities and simulator remain pinned by the parent revision.')
    path = root/'execution_revisions'/f'{REVISION}.json'
    if path.exists():
        require(read_json(path)==record, 'Two-phase execution revision changed; preserve/register a new revision')
    else:
        commit(path, record)
    return fingerprint


def verified_marker(folder, name, config_hash=None):
    path = folder/name
    if not path.exists():
        return None
    record = read_json(path)
    if config_hash is not None:
        require(record['configuration_hash']==config_hash, 'Phase checkpoint configuration changed')
    require(all((folder/p).is_file() and file_digest(folder/p)==sha
                for p,sha in record['outputs_sha256'].items()), 'Phase checkpoint checksum mismatch')
    return record


def write_marker(folder, name, config_hash, files, **extra):
    record = dict(configuration_hash=config_hash,
        outputs_sha256={p:file_digest(folder/p) for p in files}, **extra)
    commit(folder/name, record)
    return record


def frame_contexts(scene, trace, timeline, simulation, content_revision):
    indices = {r['global_segment_index'] for r in simulation['segment_downloads'] if r['completed']}
    return contexts_for_playback(scene_id=scene['scene_id'], content_revision=content_revision,
        trace_provider=trace, timeline=timeline,
        decisions=[r for r in simulation['decisions'] if r['global_segment_index'] in indices],
        played_duration_s=simulation['playback']['played_duration_s'],
        source_gop_ids=['0','1','2'], mode='frame_grid', fps=30)


def quality_job(engine, provider, context):
    """Use exactly the existing provider's scalar and physical image identities."""
    pose = provider.poses[context.camera_sample_id]
    frame = round(context.content_time_s*30)
    require(abs(context.content_time_s-frame/30)<1e-10, 'Unaligned source frame')
    allocation = {iid:int(str(q).removeprefix('CRF')) for iid,q in context.selected_quality.items()}
    candidate = engine.identity(pose, frame, allocation, 'joint')
    reference = engine.identity(pose, frame, {iid:20 for iid in allocation}, 'joint')
    identity = dict(candidate=candidate, reference=provider.reference_identity)
    return digest(identity), dict(identity=identity, candidate_image_key=digest(candidate),
        reference_image_key=digest(reference), source_gop=frame//60, source_frame=frame,
        scene_id=context.scene_id, quality_vector=[[iid,q] for iid,q in sorted(allocation.items())])


def job_schedule(jobs):
    # All scenes use the same two sources and recorded local eye. Eight resident
    # models cover both sources x four qualities for an entire source GOP.
    return sorted(jobs, key=lambda key:(jobs[key]['source_gop'], jobs[key]['scene_id'],
        jobs[key]['source_frame'], jobs[key]['quality_vector'], key))


class TwoPhaseRunner:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root).resolve(), cfg
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.budget = ExecutionBudget(self.root, cfg['limits'])
        self.started = time.monotonic()
        self.last_progress = 0.
        self.phase_started = time.monotonic()
        self.comparison_root = None
        self.state = dict(status='validating', phase='A', total_runs=len(cfg['object_counts'])*
            len(cfg['scenarios'])*len(cfg['playback_durations_s'])*len(cfg['methods']),
            streaming_completed_runs=0, streaming_imported_runs=0, streaming_computed_runs=0,
            completed_runs=0, failed_runs=0, smoke_completed_runs=0, current=None,
            execution_revision=REVISION, limits=cfg['limits'])

    def progress(self, force=False):
        if not force and time.monotonic()-self.last_progress<10:
            return
        self.last_progress=time.monotonic()
        self.state['completed_runs']=len(list((self.root/'matrix').glob('*/*/*/*/completion.json')))
        elapsed=time.monotonic()-self.phase_started
        phase=self.state['phase']
        if phase=='A':
            done=self.state['streaming_computed_runs']
            remaining=self.state['total_runs']-self.state['streaming_completed_runs']
        else:
            done=self.state.get('quality_jobs_processed',0)
            remaining=self.state.get('unique_quality_jobs',0)-done
        self.state.update(elapsed_seconds=time.monotonic()-self.started, phase_elapsed_seconds=elapsed,
            phase_eta_seconds=elapsed/done*remaining if done else None,
            invocation_native_seconds=self.budget.native_seconds, disk_bytes=self.budget.disk_bytes,
            workload=self.engine.stats if self.engine else self.state.get('workload',{}), updated_at=time.time())
        commit(self.root/'progress.json',self.state)
        commit(self.root/f'phase_{phase.lower()}_progress.json',self.state)
        print({k:self.state.get(k) for k in ('phase','status','streaming_completed_runs','total_runs',
            'quality_jobs_processed','unique_quality_jobs','completed_runs','current','phase_eta_seconds')},flush=True)

    def engine_open(self, packages, times):
        if self.cfg.get('visibility_cache_from'):
            self.engine=ReadOnlyVisibilityEngine(self.cfg['visibility_cache_from'],self.budget)
            return
        self.engine=NativeEngine(self.root,self.budget,packages,PROJECT/self.cfg['metadata_root'],
            PROJECT/self.cfg['bounds_source'],self.fingerprint,times,
            max_resident_models=self.cfg['renderer']['max_resident_models'],
            image_write_workers=self.cfg['renderer'].get('image_write_workers',1),
            max_pending_image_writes=self.cfg['renderer'].get('max_pending_image_writes',2),
            execution_revision=REVISION)
        self.engine.progress_callback=self.progress

    def streaming(self, profiles, metadata, scene, duration, method, network):
        self.budget.check()
        timeline=Timeline(6,duration,2)
        trace=make_trace(scene,timeline,profiles.content_revision)
        config_hash=digest(dict(matrix=self.fingerprint,scene=scene,duration=duration))
        parent=self.root/'matrix'/scene['kind']/f"N{scene['object_count']:02d}"/f'{duration:02d}s'
        folder=parent/method
        self.state['current']=dict(scenario=scene['kind'],N=scene['object_count'],duration_s=duration,
            method=method,stage='streaming_checkpoint_check')
        if verified_marker(folder/'phase_a','completion.json',config_hash):
            self.state['streaming_completed_runs']+=1
            self.state['streaming_imported_runs']+=1
            self.progress()
            return folder
        config=dict(experiment_id=f"runtime6-{scene['kind']}-N{scene['object_count']}-{duration}s",
            input_kind='prepared_experiment',adaptation={**self.cfg['adaptation'],'methods':list(self.cfg['methods'])},
            network=self.cfg['network'],playback=self.cfg['playback'],max_wall_time_s=102.,
            source_gop_ids=['0','1','2'],quality=self.cfg['quality'],configuration_hash=config_hash)
        configuration=dict(configuration_hash=config_hash,config=config,scene=scene,
            trace_sha256=digest(trace.trace),matrix_fingerprint=self.fingerprint)
        for path,value in ((parent/'configuration.json',configuration),(parent/'trace.json',trace.trace)):
            if path.exists():
                require(read_json(path)==value,'Existing scene/camera/configuration differs')
            else:
                commit(path,value)
        bound,bound_metadata=bind_profiles(profiles,metadata,scene,timeline)
        transfer=dict(bound.provenance['resource_transfer_aliases'])
        path=parent/'resource_transfer_map.json'
        if path.exists():
            require(read_json(path)==transfer,'Independent transfer identities changed')
        else:
            commit(path,transfer)
        imported=False
        if completed(folder,config_hash):
            simulation,performance=read_json(folder/'simulation.json'),read_json(folder/'performance.json')
            imported=True
        else:
            recovered=network_checkpoint(folder,config_hash,method)
            if recovered is not None:
                simulation,performance=recovered
                imported=True
            else:
                visibility=RuntimeVisibilityAdapter(self.engine,trace,bound)
                planner=RuntimePlanner(bound,trace,visibility,baseline_metadata=bound_metadata,
                    expected_object_count=scene['object_count'],cached_resource_ids=(),
                    solver_limits=self.cfg['adaptation']['solver_limits'],
                    exhaustive_limits=self.cfg['adaptation'].get('exhaustive_limits'))
                self.state['current']['stage']='runtime_visibility_adaptation_network_buffer'
                self.progress(force=True)
                initialization_ms=0. if method==EXHAUSTIVE_SEARCH else planner.initialize_solver()
                simulation,performance=simulate(config,timeline,scene,network,planner,method)
                performance['shared_solver_initialization_ms']=initialization_ms
                performance['runtime_visibility_events']=visibility.events
        contexts=frame_contexts(scene,trace,timeline,simulation,bound.content_revision)
        phase=folder/'phase_a'
        # A separate immutable streaming snapshot never overwrites historical
        # completed quality outputs or their hashes.
        stream=deepcopy(simulation)
        stream['status']=stream.get('network_status',stream['status'])
        quality=collect_quality(contexts)
        stream['quality_frames']=quality['frames']
        stream['metrics']=summarize_metrics(stream,quality)
        stream['metrics'].pop('performance',None)
        self.prepare_streaming_export(phase,stream,performance,contexts)
        export_run(phase,stream,performance)
        commit(phase/'played_frames.json',self.played_document(scene,trace,stream,contexts,parent))
        files=['simulation.json','metrics.json','performance.json','requests.csv','buffer.csv',
               'decisions.csv','frames.csv','played_frames.json']
        files.extend(self.extra_streaming_files(phase))
        write_marker(phase,'completion.json',config_hash,files,method=method,imported=imported,
            scene_file_sha256=file_digest(parent/'configuration.json'),
            trace_file_sha256=file_digest(parent/'trace.json'),frame_count=len(contexts),
            simulation_status=stream['status'],quality_pending=True)
        self.state['streaming_completed_runs']+=1
        self.state['streaming_imported_runs' if imported else 'streaming_computed_runs']+=1
        self.budget.disk_bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.progress(force=True)
        return folder

    def prepare_streaming_export(self,phase,stream,performance,contexts):
        """Extension hook for terminal diagnostic outcomes; old exports unchanged."""

    def extra_streaming_files(self,phase):
        return []

    def played_document(self,scene,trace,stream,contexts,parent):
        return dict(schema='streaming-played-frames.v1',fps=30,
            source_duration_s=6,frames=[c.to_dict() for c in contexts],
            exact_pose_and_instances_file='../../trace.json',
            trace_file_sha256=file_digest(parent/'trace.json'),
            scene_configuration_sha256=file_digest(parent/'configuration.json'),
            reconstruction='Resolve camera_sample_id in the parent trace; it includes the exact camera, '
              'all per-instance object transforms and source conditioning. content_time_s*30 identifies source frame. '
              'simulation.json carries continuous playback/download/buffer state; global segment and loop IDs remain distinct.')

    def load_run(self,folder):
        phase=folder/'phase_a'
        marker=verified_marker(phase,'completion.json')
        require(marker is not None,'Streaming phase not complete')
        parent=folder.parent
        require(marker['trace_file_sha256']==file_digest(parent/'trace.json') and
                marker['scene_file_sha256']==file_digest(parent/'configuration.json'),
                'Exact camera/scene artifacts changed after streaming')
        configuration=read_json(parent/'configuration.json')
        scene=configuration['scene']
        # Reconstruct from the frozen scene and verify every pose against the
        # persisted trace before any rendering is allowed.
        duration=int(parent.name[:-1])
        trace=make_trace(scene,Timeline(6,duration,2),self.profiles.content_revision)
        require(trace.trace==read_json(parent/'trace.json'),'Persisted played-frame trace cannot be reproduced exactly')
        contexts=[FrameQualityContext(**r) for r in read_json(phase/'played_frames.json')['frames']]
        return scene,trace,contexts,configuration['configuration_hash']

    def plan_quality(self,folders):
        jobs={};bindings={};images=set()
        for folder in folders:
            self.budget.check()
            scene,trace,contexts,_=self.load_run(folder)
            provider=NativeQualityProvider(self.engine,trace,scene,cache_only=True)
            relative=str(folder.relative_to(self.root));binding=[]
            for context in contexts:
                key,job=quality_job(self.engine,provider,context)
                if key not in jobs:
                    jobs[key]=dict(job,representative_run=relative,context=context.to_dict(),occurrences=0)
                jobs[key]['occurrences']+=1
                binding.append(key)
                images.update((job['candidate_image_key'],job['reference_image_key']))
            bindings[relative]=binding
            self.state['current']=dict(stage='global_quality_job_collection',run=relative)
            self.progress()
        document=dict(schema='streaming-global-native-jobs.v1',fingerprint=self.fingerprint,
            streaming_runs=len(folders),frame_occurrences=sum(map(len,bindings.values())),
            unique_quality_jobs=len(jobs),unique_images=len(images),jobs=jobs,run_bindings=bindings,
            schedule='source GOP, scene, source frame, selected quality vector; eight resident Gaussian models',
            source_checkpoints={str(f.relative_to(self.root)):file_digest(f/'phase_a/completion.json') for f in folders})
        path=self.root/'phase_b_jobs.json'
        if path.exists():
            require(read_json(path)==document,'Global job plan changed; preserve the prior plan before rebuilding')
        else:
            commit(path,document)
        self.budget.disk_bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        return document

    def merge(self,folder):
        scene,trace,contexts,config_hash=self.load_run(folder)
        if completed(folder,config_hash):
            # Original result files remain byte-for-byte unchanged.
            self.state['completed_runs']+=1
            return
        provider=NativeQualityProvider(self.engine,trace,scene,cache_only=True)
        quality=collect_quality(contexts,provider)
        require(quality['summary']['unavailable_frame_count']==0,'Native played-frame quality still missing')
        quality['summary'].update(sampling_mode='frame_grid',full_frame_grid=True)
        simulation=read_json(folder/'phase_a/simulation.json')
        performance=read_json(folder/'phase_a/performance.json')
        simulation['quality_frames']=quality['frames']
        simulation['metrics']=summarize_metrics(simulation,quality)
        simulation['metrics'].pop('performance',None)
        performance['quality_execution_revision']=REVISION
        performance['native_quality_events']=provider.records
        performance['quality_schedule']='global phase_b_jobs.json; per-run events are merge cache lookups, not render times'
        performance['global_quality_job_manifest_sha256']=file_digest(self.root/'phase_b_jobs.json')
        self.prepare_quality_export(folder,simulation,performance)
        export_run(folder,simulation,performance)
        files=['simulation.json','metrics.json','performance.json','requests.csv','buffer.csv','decisions.csv','frames.csv']
        write_marker(folder,'completion.json',config_hash,files,
            simulation_status=simulation['status'],measured_frames=len(quality['frames']))
        self.state['completed_runs']+=1

    def prepare_quality_export(self,folder,simulation,performance):
        """Extension hook for additional recorded display metadata."""

    def offline(self,folders):
        self.state.update(phase='B',current=dict(stage='global_job_collection'))
        self.phase_started=time.monotonic()
        self.progress(force=True)
        plan=self.plan_quality(folders)
        jobs=plan['jobs']
        self.state.update(unique_quality_jobs=len(jobs),unique_images=plan['unique_images'],
            played_frame_occurrences=plan['frame_occurrences'],quality_jobs_processed=0,
            quality_cache_hits=0,quality_cache_misses=0)
        providers=OrderedDict()
        for key in job_schedule(jobs):
            self.budget.check()
            job=jobs[key];relative=job['representative_run']
            if relative not in providers:
                scene,trace,_,_=self.load_run(self.root/relative)
                providers[relative]=NativeQualityProvider(self.engine,trace,scene)
                if len(providers)>4:
                    providers.popitem(last=False)
            provider=providers[relative];providers.move_to_end(relative)
            self.state['current']=dict(stage='offline_native_joint_quality',job=key,run=relative,
                source_frame=job['source_frame'],source_gop=job['source_gop'])
            context=FrameQualityContext(**job['context'])
            require(quality_job(self.engine,provider,context)[0]==key,'Global quality job identity differs')
            provider.measure(context)
            event=provider.records.pop()
            self.state['quality_cache_hits' if event['cache_hit'] else 'quality_cache_misses']+=1
            self.state['quality_jobs_processed']+=1
            self.progress()
        self.engine.flush()
        commit(self.root/'phase_b_render_completion.json',dict(
            plan_sha256=file_digest(self.root/'phase_b_jobs.json'),unique_quality_jobs=len(jobs),
            frame_occurrences=plan['frame_occurrences'],workload=self.engine.stats))
        self.state['current']=dict(stage='merge_actual_quality_into_streaming_results')
        self.progress(force=True)
        for folder in folders:
            self.budget.check()
            self.merge(folder)
            self.progress()

    def report(self):
        from .two_phase_report import update_report
        update_report(self.root,self.state)
        if self.comparison_root is not None and (self.root/'manifest.json').exists():
            from .analyze_runtime_results import compare_transport_runs
            compare_transport_runs(self.root,self.comparison_root)

    def run(self,phase='all'):
        require(not self.cfg.get('visibility_cache_from') or phase=='A',
                'Read-only visibility input is for transport-only Phase A validation')
        lock=(self.root/'writer.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profiles,metadata,packages,records,times,network,identity=validate_inputs(self.cfg)
            self.profiles=profiles
            self.fingerprint=pin_execution(self.root,identity)
            self.engine_open(packages,times)
            self.state['status']='running'
            folders=[]
            for count in sorted(self.cfg['object_counts']):
                for family in self.cfg['scenarios']:
                    scene=make_scene(family,count,records,seed=self.cfg['seed'])
                    for duration in sorted(self.cfg['playback_durations_s']):
                        for method in self.cfg['methods']:
                            if phase=='B':
                                folder=self.root/'matrix'/family/f'N{count:02d}'/f'{duration:02d}s'/method
                                require(verified_marker(folder/'phase_a','completion.json') is not None,
                                    'Phase B requires the complete streaming matrix; run Phase A first')
                                self.state['streaming_completed_runs']+=1
                            else:
                                folder=self.streaming(profiles,metadata,scene,duration,method,network)
                            folders.append(folder)
                    self.report()
            commit(self.root/'phase_a_completion.json',dict(total_runs=len(folders),
                checkpoints={str(f.relative_to(self.root)):file_digest(f/'phase_a/completion.json') for f in folders}))
            self.progress(force=True)
            self.report()
            if phase!='A':
                # Explicit phase boundary: release proxy models before batched
                # joint rendering. No second CUDA process is launched.
                self.engine.close();self.engine=None
                self.engine_open(packages,times)
                self.offline(folders)
            self.state.update(status='complete' if phase!='A' else 'streaming_complete',current=None)
            return 0
        except KeyboardInterrupt:
            self.state.update(status='interrupted',stop_reason='Graceful SIGINT; immutable per-run and per-image checkpoints preserved')
            return 130
        except ResourceLimit as exc:
            self.state.update(status='resource_limit',stop_reason=str(exc))
            return 3
        except Exception as exc:
            self.state.update(status='failed',failed_runs=self.state['failed_runs']+1,
                stop_reason=f'{type(exc).__name__}: {exc}')
            commit(self.root/'two_phase_failure.json',dict(state=self.state,traceback=traceback.format_exc()))
            traceback.print_exc()
            return 1
        finally:
            if self.engine:
                self.engine.close()
            self.progress(force=True)
            self.report()
            lock.close()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--phase',choices=('all','A','B'),default='all')
    parser.add_argument('--object-counts',type=int,nargs='+',help='Restrict this invocation to a subset of configured N')
    parser.add_argument('--playback-durations',type=int,nargs='+',help='Restrict configured playback durations')
    parser.add_argument('--scenarios',nargs='+',help='Restrict configured scene families')
    parser.add_argument('--max-dp-cells',type=int,help='Explicit solver memory/work limit; solver unchanged')
    parser.add_argument('--delta-bps',type=int,help='Conservative bitrate DP quantum; default 250000 bps')
    parser.add_argument('--transport',choices=('virtual','real_http2'),help='Virtual control or actual TLS HTTP/2 IO')
    parser.add_argument('--methods',nargs='+',choices=METHODS,help='Restrict unchanged methods for isolated validation')
    parser.add_argument('--visibility-cache-from',type=Path,help='Read-only exact native visibility cache for Phase A')
    parser.add_argument('--compare-against',type=Path,help='Read-only baseline root for Phase A transport comparison')
    parser.add_argument('--max-wall-seconds',type=float)
    parser.add_argument('--max-native-seconds',type=float)
    parser.add_argument('--max-storage-gib',type=float)
    args=parser.parse_args();cfg=load_settings(args.config)
    for key, values in [('object_counts',args.object_counts), ('playback_durations_s',args.playback_durations),
                        ('scenarios',args.scenarios)]:
        if values is not None:
            require(set(values) <= set(cfg[key]) and len(values)==len(set(values)), f'Choose unique configured {key}')
            cfg[key]=values
    if args.max_dp_cells is not None:
        require(args.max_dp_cells>0, 'Positive max_dp_cells required')
        cfg['adaptation']['solver_limits']['max_dp_cells']=args.max_dp_cells
    if args.delta_bps is not None:
        require(args.delta_bps>0,'Positive bitrate discretization required')
        cfg['adaptation']['solver_limits'].update(discretization='bitrate',delta_bps=args.delta_bps)
    if args.transport is not None:
        cfg['network']['transport']=args.transport
    if args.methods is not None:
        require(len(set(args.methods))==len(args.methods),'Choose unique unchanged methods')
        cfg['methods']=args.methods
    if args.visibility_cache_from is not None:
        require(args.phase=='A','Read-only visibility source is for Phase A only')
        require(args.root.resolve()!=args.visibility_cache_from.resolve(),'Visibility source root is read-only')
        cfg['visibility_cache_from']=str(args.visibility_cache_from.resolve())
        cfg['network']['prepare_visibility_before_clock']=True
    if args.compare_against is not None:
        require(args.phase=='A', 'Transport comparison is a streaming-only Phase A validation')
        require(args.root.resolve()!=args.compare_against.resolve(), 'Baseline output root is read-only')
    for key,value in [('max_wall_seconds',args.max_wall_seconds),('max_native_seconds',args.max_native_seconds),
        ('max_storage_bytes',args.max_storage_gib*1024**3 if args.max_storage_gib else None)]:
        if value is not None:
            require(value>0,'Positive execution limits required');cfg['limits'][key]=value
    runner=TwoPhaseRunner(args.root,cfg)
    runner.comparison_root=args.compare_against
    return runner.run(args.phase)


if __name__=='__main__':
    raise SystemExit(main())
