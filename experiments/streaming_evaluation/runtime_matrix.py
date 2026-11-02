"""Sequential, resumable native Task-4 experiment execution."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
import time
import traceback

from experiments.adaptation_baselines.metadata import load_metadata
from experiments.adaptation_baselines.methods import METHODS
from experiments.adaptation_baselines.exhaustive import EXHAUSTIVE_SEARCH, settings as exhaustive_settings
from experiments.nava_mse.geometry import select_camera_record
from experiments.nava_mse.rendering import StreamSTGSRenderer
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require
from experiments.visibility_weighted_sse.profiles import load_profiles
from .metrics import summarize_metrics
from .network import NetworkTrace, NetworkLink
from .quality import contexts_for_playback, collect_quality
from .runtime_adapter import RuntimePlanner, RuntimeVisibilityAdapter
from .runtime_native import NativeEngine, NativeQualityProvider, ExecutionBudget, ResourceLimit, commit
from .runtime_scenes import make_scene, make_trace, bind_profiles
from .simulator import simulate, _csv
from .timeline import Timeline
from .cache_upgrade import resolve as resolve_execution

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT/'experiments/streaming_evaluation/configs/runtime_discretized_http2.json'
DEFAULT_ROOT = PROJECT/'experiments/streaming_evaluation/runs/runtime_discretized_virtual_v1'


def load_settings(path):
    cfg = read_json(path)
    require(cfg['schema'] == 'streaming-evaluation.runtime-matrix.v1', 'Unknown matrix configuration')
    require(cfg['source_duration_s'] == 6 and cfg['gop_duration_s'] == 2 and cfg['fps'] == 30,
            'This protocol uses 180 source frames in three two-second GOPs')
    require(len(cfg['sources']) == 2 and len(set(cfg['sources'])) == 2, 'Two unique sources required')
    require(cfg['methods'] and len(cfg['methods'])==len(set(cfg['methods'])) and set(cfg['methods'])<=set((*METHODS,EXHAUSTIVE_SEARCH)),
            'Choose unique unchanged methods')
    require(cfg['adaptation']['resource_sharing'] == 'per_instance', 'Inter-instance deduplication forbidden')
    require(cfg['quality'] == dict(mode='frame_grid', fps=30, mask_mode='full_viewport',
        reference='joint_CRF20', normalization='clip'), 'Declare one exact joint quality protocol')
    require(all(type(n) is int and n >= 3 for n in cfg['object_counts']) and
        len(set(cfg['object_counts'])) == len(cfg['object_counts']), 'Unique matrix N >= 3 required')
    require(set(cfg['scenarios']) <= {'camera_rotation', 'long_line', 'random_spatial'}, 'Unknown scenario')
    require(set(cfg['playback_durations_s']) <= {10,20,30}, 'Choose requested playback durations')
    require(cfg['renderer']['width'] == 640 and cfg['renderer']['height'] == 480 and
            cfg['renderer']['matmul_allow_tf32'] is False and cfg['renderer']['cudnn_allow_tf32'] is False,
            'Native/profile calibration and TF32 protocol must match')
    for key in ('max_wall_seconds', 'max_native_seconds', 'max_storage_bytes', 'minimum_free_bytes'):
        require(cfg['limits'][key] > 0, f'Explicit positive {key} required')
    NetworkLink(NetworkTrace([(0., 1.)]), latency_s=cfg['network'].get('request_latency_s', 0.),
                max_concurrent_streams=cfg['network'].get('max_concurrent_streams', 16))
    require(cfg['network'].get('transport','virtual') in ('virtual','real_http2'), 'Unknown transport mode')
    from experiments.visibility_weighted_sse.allocation import solver_settings
    solver_settings(cfg['adaptation'].get('solver_limits'))
    if EXHAUSTIVE_SEARCH in cfg['methods']:
        exhaustive_settings(cfg['adaptation'].get('exhaustive_limits'))
    return cfg


def validate_inputs(cfg):
    profile_path, metadata_path = PROJECT/cfg['profiles'], PROJECT/cfg['baseline_metadata']
    profiles = load_profiles(profile_path)
    metadata = load_metadata(read_json(metadata_path), profiles, file_digest(profile_path),
                             'validated_cached_artifacts', METHODS)
    packages = {(oid,g,q):PROJECT/cfg['packages_root']/oid/f'gop_{g:03d}/encoded/crf_{q}'
                for oid in cfg['sources'] for g in range(3) for q in (40,36,28,20)}
    validator = StreamSTGSRenderer(packages, metadata_root=PROJECT/cfg['metadata_root'])
    representations, times, records = {}, {}, {}
    try:
        for key in packages:
            row = validator._package_metadata(key)  # existing complete component byte/SHA checks; CPU only
            oid, g, q = key
            require(any(p.object_id == oid for p in profiles.for_gop(str(g))), 'Missing offline source/GOP')
            stamps = row['reconstruction']['segment']['timestamps']
            require(len(stamps) == 60, 'Incomplete source presentation frames')
            for j, stamp in enumerate(stamps):
                frame = g*60+j
                require(stamp['seconds'] == frame/30, 'Encoded presentation timestamp mismatch')
                require(frame not in times or times[frame] == stamp['seconds'], 'Incompatible source timelines')
                times[frame] = stamp['seconds']
            records[oid] = select_camera_record(row['cameras'], 0)
            profile = next(p for p in profiles.for_gop(str(g)) if p.object_id == oid)
            quality = next(p for p in profile.qualities if p.quality_id == f'CRF{q}')
            physical = {profiles.resources[r].sha256:profiles.resources[r].bytes for r in quality.resource_ids}
            actual = {v['sha256']:v['bytes'] for v in row['manifest']['files'].values()}
            actual[row['manifest_sha256']] = packages[key].joinpath('manifest.json').stat().st_size
            require(physical == actual, 'Offline costs differ from actual prepared component union')
            representations[f'{oid}/{g}/{q}'] = dict(manifest_sha256=row['manifest_sha256'],
                reconstruction_sha256=row['reconstruction_sha256'], frame_count=len(stamps),
                physical_bytes=sum(physical.values()))
    finally:
        validator.close()
    require(len(times) == 180, 'Six-second source coverage incomplete')
    source = PROJECT/cfg['network']['path']
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    tkey = next(k for k in ('time_seconds','time_s','at_s','time') if k in rows[0])
    vkey = next(k for k in ('mbps','bandwidth_mbps') if k in rows[0])
    points = [(float(r[tkey]),float(r[vkey])) for r in rows]
    offset = cfg['network']['offset_s']
    rate = next(v for t,v in reversed(points) if t <= offset)
    rebased = [(0.,rate)] + [(t-offset,v) for t,v in points if t > offset]
    network = NetworkTrace(rebased, duration_s=cfg['network']['source_duration_s']-offset, end_policy='stop')
    code = [Path(__file__), *Path(__file__).parent.glob('runtime_*.py'),
        PROJECT/'experiments/adaptation_baselines/interface.py', PROJECT/'experiments/adaptation_baselines/methods.py',
        PROJECT/'experiments/visibility_weighted_sse/allocation.py', PROJECT/'experiments/visibility_weighted_sse/visibility.py',
        PROJECT/'experiments/nava_mse/rendering.py', PROJECT/'src_final/multiobj3dgs/client/visibility/analyzer.py',
        Path(__file__).parent/'simulator.py', Path(__file__).parent/'inputs.py', Path(__file__).parent/'quality.py',
        Path(__file__).parent/'image_cache.py',Path(__file__).parent/'cache_upgrade.py',
        *[Path(__file__).parent/f'{name}.py' for name in ('network','playback','metrics')],
        PROJECT/'experiments/abr_network_evaluation/network.py',
        PROJECT/'experiments/abr_network_evaluation/predictor.py']
    code += [p for p in (Path(__file__).parent/'real_http2.py', Path(__file__).parent/'http2_server.py') if p.exists()]
    if EXHAUSTIVE_SEARCH in cfg['methods']:
        code += [PROJECT/'experiments/adaptation_baselines/exhaustive.py',Path(__file__).parent/'two_phase.py',
            Path(__file__).parent/'http2_dp015_evaluation.py',Path(__file__).parent/'evaluation_report.py']
    science = {k:v for k,v in cfg.items() if k != 'limits'}
    identity = dict(config=science, profiles_sha256=file_digest(profile_path),
        baseline_metadata_sha256=file_digest(metadata_path), network_sha256=file_digest(source),
        representations=representations,
        implementation_sha256={str(p.relative_to(PROJECT)):file_digest(p) for p in sorted(set(code))})
    if cfg.get('visibility_cache_from'):
        cache_root=Path(cfg['visibility_cache_from'])
        previous=read_json(cache_root/'manifest.json')
        for key in ('profiles_sha256','baseline_metadata_sha256','representations'):
            require(previous[key]==identity[key],f'Read-only visibility source changed: {key}')
        # Exact lookup below also binds camera, frame, complete instance list,
        # source conditioning, and every transform.
        identity['visibility_cache_manifest_sha256']=file_digest(cache_root/'manifest.json')
    return profiles, metadata, packages, records, times, network, identity


def export_run(destination, simulation, performance):
    commit(destination/'simulation.json', simulation)
    commit(destination/'metrics.json', simulation['metrics'])
    commit(destination/'performance.json', performance)
    for name, rows in (('requests',simulation['requests']), ('buffer',simulation['buffer_samples']),
        ('decisions',simulation['metrics']['adaptation']['selected_quality']), ('frames',simulation['quality_frames'])):
        _csv(destination/f'{name}.csv', rows)


def completed(destination, identity):
    marker = destination/'completion.json'
    if not marker.exists():
        return False
    row = read_json(marker)
    require(row['configuration_hash'] == identity, 'Completed run has incompatible configuration')
    require(all((destination/p).is_file() and file_digest(destination/p) == sha
                for p,sha in row['outputs_sha256'].items()), 'Completed result checksum mismatch')
    return True


def quality_schedule(contexts):
    """Batch selected offline jobs by source GOP/frame; never add quality vectors."""
    return sorted(contexts,key=lambda c:(int(c.content_gop_id),c.content_time_s,
        tuple(sorted(c.selected_quality.items())),c.global_time_s))


def network_checkpoint(destination,config_hash,method):
    destination=Path(destination)
    marker=destination/'network_checkpoint.sha256.json'
    if not marker.exists():
        return None
    checkpoint=destination/'network_checkpoint.json';performance=destination/'performance.json'
    saved=read_json(marker)
    require(saved['configuration_hash']==config_hash and saved['method']==method and
            saved['checkpoint_sha256']==file_digest(checkpoint) and
            saved['performance_sha256']==file_digest(performance),
            'Network checkpoint checksum/configuration mismatch')
    return read_json(checkpoint),read_json(performance)


class MatrixRunner:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root).resolve(), cfg
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = ExecutionBudget(self.root, cfg['limits'])
        self.state = dict(status='validating', completed_runs=0, total_runs=len(cfg['object_counts'])*
            len(cfg['scenarios'])*len(cfg['playback_durations_s'])*len(cfg['methods']), smoke_completed_runs=0,
            failed_runs=0, current=None, limits=cfg['limits'], started_at=time.time())
        self.last_progress, self.engine = 0., None

    def progress(self, force=False):
        if not force and time.monotonic()-self.last_progress < 10:
            return
        self.last_progress = time.monotonic()
        elapsed = time.monotonic()-self.budget.started
        stats = dict(self.engine.stats) if self.engine else {}
        smoke = self.root/'smoke_workload.json'
        eta = None
        if smoke.exists() and stats.get('native_object_instances_processed', 0):
            initial = read_json(smoke)
            predicted_units = initial['native_object_instances_processed']/2 * len(self.cfg['scenarios']) * sum(self.cfg['object_counts'])
            unit_seconds = stats['native_wall_seconds']/stats['native_object_instances_processed']
            eta = max(0.,predicted_units-stats['native_object_instances_processed'])*unit_seconds
        self.state.update(elapsed_seconds=elapsed, estimated_remaining_native_seconds=eta,
            invocation_native_seconds=self.budget.native_seconds, disk_bytes=self.budget.disk_bytes,
            workload=stats, updated_at=time.time())
        commit(self.root/'progress.json', self.state)
        print(json.dumps({k:self.state[k] for k in ('status','completed_runs','total_runs','smoke_completed_runs',
            'failed_runs','current','elapsed_seconds','estimated_remaining_native_seconds','disk_bytes')},sort_keys=True), flush=True)

    def execute(self, profiles, metadata, scene, duration, method, network, *, smoke=False):
        self.budget.check()
        timeline = Timeline(6, duration, 2)
        trace = make_trace(scene, timeline, profiles.content_revision)
        config = dict(experiment_id=f"runtime6-{scene['kind']}-N{scene['object_count']}-{duration}s",
            input_kind='prepared_experiment', adaptation={**self.cfg['adaptation'],'methods':list(self.cfg['methods'])},
            network=self.cfg['network'], playback=self.cfg['playback'], max_wall_time_s=102.,
            source_gop_ids=['0','1','2'], quality=self.cfg['quality'])
        config_hash = digest(dict(matrix=self.fingerprint, scene=scene, duration=duration))
        config['configuration_hash'] = config_hash
        parent = self.root/('smoke' if smoke else 'matrix')/scene['kind']/f"N{scene['object_count']:02d}"/f'{duration:02d}s'
        destination = parent/method
        self.state['current'] = dict(scenario=scene['kind'], N=scene['object_count'], duration_s=duration,
                                      method=method, stage='resume_check')
        if completed(destination, config_hash):
            self.state['smoke_completed_runs' if smoke else 'completed_runs'] += 1
            self.progress(force=True)
            return
        commit(destination/'execution.json',dict(execution_revision=self.execution_revision,
            original_artifact_fingerprint=self.fingerprint,renderer=self.cfg['renderer']))
        commit(parent/'configuration.json', dict(configuration_hash=config_hash, config=config, scene=scene,
            trace_sha256=digest(trace.trace), matrix_fingerprint=self.fingerprint))
        if not (parent/'trace.json').exists():
            commit(parent/'trace.json', trace.trace)
        bound, bound_metadata = bind_profiles(profiles, metadata, scene, timeline)
        commit(parent/'resource_transfer_map.json', dict(bound.provenance['resource_transfer_aliases']))
        visibility = RuntimeVisibilityAdapter(self.engine, trace, bound)
        planner = RuntimePlanner(bound, trace, visibility, baseline_metadata=bound_metadata,
            expected_object_count=scene['object_count'], cached_resource_ids=(),
            solver_limits=self.cfg['adaptation']['solver_limits'])
        self.state['current']['stage'] = 'adaptation_live_visibility_network_buffer'
        self.progress(force=True)
        checkpoint=destination/'network_checkpoint.json'
        checkpoint_marker=destination/'network_checkpoint.sha256.json'
        recovered=network_checkpoint(destination,config_hash,method)
        if recovered is not None:
            simulation,performance=recovered
            self.state['current']['network_checkpoint_reused']=True
        else:
            initialization_ms=planner.initialize_solver()
            simulation, performance = simulate(config,timeline,scene,network,planner,method)
            performance['shared_solver_initialization_ms'] = initialization_ms
            performance['runtime_visibility_events'] = visibility.events
            commit(checkpoint, simulation)
            commit(destination/'performance.json', performance)
        # Network and original decision/visibility times remain valid across cache-only updates.
        commit(checkpoint_marker,dict(configuration_hash=config_hash,method=method,
            checkpoint_sha256=file_digest(checkpoint),performance_sha256=file_digest(destination/'performance.json')))
        indices = {r['global_segment_index'] for r in simulation['segment_downloads'] if r['completed']}
        decisions = [r for r in simulation['decisions'] if r['global_segment_index'] in indices]
        contexts = contexts_for_playback(scene_id=scene['scene_id'],content_revision=bound.content_revision,
            trace_provider=trace,timeline=timeline,decisions=decisions,
            played_duration_s=simulation['playback']['played_duration_s'],source_gop_ids=['0','1','2'],mode='frame_grid',fps=30)
        provider = NativeQualityProvider(self.engine,trace,scene)
        self.state['current']['stage'] = 'actual_30fps_joint_quality'
        self.progress(force=True)
        interruption = None
        try:
            quality = collect_quality(quality_schedule(contexts),provider)
            self.engine.flush()
        except Exception as exc:
            interruption = exc
            partial = NativeQualityProvider(self.engine,trace,scene,cache_only=True)
            quality = collect_quality(contexts,partial)
        quality['summary'].update(sampling_mode='frame_grid',full_frame_grid=True)
        quality['frames'].sort(key=lambda row:row['global_time_s'])
        simulation['quality_frames'] = quality['frames']
        metrics = summarize_metrics(simulation, quality)
        metrics.pop('performance')
        simulation['metrics'] = metrics
        previous_events=performance.pop('native_quality_events',None)
        if previous_events:
            performance.setdefault('previous_quality_attempts',[]).append(dict(
                events=previous_events,note='Earlier validated partial quality attempt; original timings preserved'))
        performance['native_quality_events'] = provider.records
        performance['quality_execution_revision']=self.execution_revision
        performance['quality_schedule']='source_GOP_then_source_frame_then_selected_vector; exported frames chronological'
        if interruption is not None:
            simulation['network_status'] = simulation['status']
            simulation['status'] = ('quality_incomplete_resource_limit' if isinstance(interruption, ResourceLimit)
                                    else 'quality_measurement_failed')
            simulation['metrics']['status'] = simulation['status']
        export_run(destination,simulation,performance)
        commit(checkpoint_marker,dict(configuration_hash=config_hash,method=method,
            checkpoint_sha256=file_digest(checkpoint),performance_sha256=file_digest(destination/'performance.json')))
        self.engine._save()
        self.budget.disk_bytes = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if interruption is not None:
            raise interruption
        require(quality['summary']['unavailable_frame_count'] == 0, 'Actual frame measurement missing')
        files = ['simulation.json','metrics.json','performance.json','requests.csv','buffer.csv','decisions.csv','frames.csv']
        commit(destination/'completion.json', dict(configuration_hash=config_hash,
            simulation_status=simulation['status'], measured_frames=len(simulation['quality_frames']),
            outputs_sha256={name:file_digest(destination/name) for name in files}))
        self.state['smoke_completed_runs' if smoke else 'completed_runs'] += 1
        self.progress(force=True)

    def run(self):
        from .runtime_results import make_report
        lock = (self.root/'writer.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            profiles,metadata,packages,records,times,network,identity = validate_inputs(self.cfg)
            self.fingerprint,self.execution_revision=resolve_execution(self.root,identity)
            commit(self.root/'current_execution.json',dict(revision=self.execution_revision,
                original_artifact_fingerprint=self.fingerprint,identity=identity))
            self.state['execution_revision']=self.execution_revision
            self.engine = NativeEngine(self.root,self.budget,packages,PROJECT/self.cfg['metadata_root'],
                PROJECT/self.cfg['bounds_source'],self.fingerprint,times,
                max_resident_models=self.cfg['renderer']['max_resident_models'],
                image_write_workers=self.cfg['renderer'].get('image_write_workers',1),
                max_pending_image_writes=self.cfg['renderer'].get('max_pending_image_writes',2),
                execution_revision=self.execution_revision)
            self.engine.progress_callback = self.progress
            self.state['status'] = 'running'
            smoke = make_scene('camera_rotation',2,records,seed=self.cfg['seed'])
            for method in self.cfg['methods']:
                self.execute(profiles,metadata,smoke,6,method,network,smoke=True)
            if not (self.root/'smoke_workload.json').exists():
                commit(self.root/'smoke_workload.json',self.engine.stats)
            make_report(self.root,PROJECT,self.state)
            for count in sorted(self.cfg['object_counts']):
                for family in self.cfg['scenarios']:
                    scene = make_scene(family,count,records,seed=self.cfg['seed'])
                    for duration in sorted(self.cfg['playback_durations_s']):
                        for method in self.cfg['methods']:
                            self.execute(profiles,metadata,scene,duration,method,network)
                    make_report(self.root,PROJECT,self.state)
            self.state.update(status='complete',current=None)
            return 0
        except ResourceLimit as exc:
            self.state.update(status='resource_limit',stop_reason=str(exc))
            return 3
        except Exception as exc:
            self.state.update(status='failed',stop_reason=f'{type(exc).__name__}: {exc}',
                              failed_runs=self.state['failed_runs']+1)
            commit(self.root/'failure.json',dict(current=self.state['current'],error=self.state['stop_reason'],
                                                traceback=traceback.format_exc()))
            traceback.print_exc()
            return 1
        finally:
            if self.engine:
                self.engine.close()
            self.progress(force=True)
            make_report(self.root,PROJECT,self.state)
            lock.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--max-wall-seconds',type=float)
    parser.add_argument('--max-native-seconds',type=float)
    parser.add_argument('--max-storage-gib',type=float)
    args = parser.parse_args()
    cfg = load_settings(args.config)
    for key,value in [('max_wall_seconds',args.max_wall_seconds),('max_native_seconds',args.max_native_seconds),
                      ('max_storage_bytes',args.max_storage_gib*1024**3 if args.max_storage_gib else None)]:
        if value is not None:
            require(value > 0,'Positive execution limits required')
            cfg['limits'][key] = value
    return MatrixRunner(args.root,cfg).run()


if __name__ == '__main__':
    raise SystemExit(main())
