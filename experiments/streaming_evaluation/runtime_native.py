"""Bounded native computation and exact physical-input experiment caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.nava_mse.metrics import MetricStats, foreground_metrics
from experiments.nava_mse.rendering import validate_render_alignment
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require
from experiments.visibility_weighted_sse.trace import camera_to_dict
from .image_cache import ImageWriter, image_metrics


def commit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


class ResourceLimit(RuntimeError):
    pass


class ExecutionBudget:
    def __init__(self, root, limits):
        self.root, self.limits = Path(root), limits
        self.started = time.monotonic()
        self.native_seconds = 0.
        self.disk_bytes = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check(self, reserve_bytes=0):
        if time.monotonic()-self.started >= self.limits['max_wall_seconds']:
            raise ResourceLimit('max_wall_seconds reached')
        if self.native_seconds >= self.limits['max_native_seconds']:
            raise ResourceLimit('max_native_seconds reached')
        if self.disk_bytes+reserve_bytes >= self.limits['max_storage_bytes']:
            raise ResourceLimit('max_storage_bytes reached')
        disk = os.statvfs(self.root)
        if disk.f_bavail*disk.f_frsize < self.limits.get('minimum_free_bytes', 5*1024**3):
            raise ResourceLimit('minimum_free_bytes reached')


def physical_inputs(pose, source_frame):
    require(0 <= source_frame < 180, 'Source frame must be in six-second source')
    fields = ('instance_id', 'object_id', 'object_to_world', 'source_camera', 'conditioning_mode')
    return dict(camera=camera_to_dict(pose.camera), source_frame=source_frame,
        source_gop=source_frame//60,
        instances=[{k: row[k] for k in fields} for row in pose.instances])


class NativeEngine:
    def __init__(self, root, budget, packages, metadata_root, bounds_source, fingerprint,
                 source_times, *, max_resident_models=4, renderer_factory=None,
                 image_write_workers=1,max_pending_image_writes=2,execution_revision='original_v1'):
        self.root, self.budget, self.packages = Path(root), budget, packages
        self.metadata_root, self.bounds_source = metadata_root, bounds_source
        self.fingerprint, self.source_times = fingerprint, source_times
        self.max_resident_models, self.renderer_factory = max_resident_models, renderer_factory
        self.renderer = self.backend = None
        self.path = self.root/'workload.json'
        self.stats = read_json(self.path) if self.path.exists() else dict(
            native_jobs_completed=0, native_jobs_failed=0, image_jobs=0, visibility_jobs=0,
            image_cache_hits=0, visibility_cache_hits=0, metric_cache_hits=0,
            native_wall_seconds=0., visibility_analysis_ms=0., measured_metric_jobs=0)
        self.events = []
        self.progress_callback = None
        self.stats.setdefault('native_object_instances_processed', 0)
        self.execution_revision=execution_revision
        self.writer=ImageWriter(image_write_workers,max_pending_image_writes)
        self.verified_images={}
        self.annotated_images=set()
        self.stats.setdefault('execution_revisions',{})
        self.revision_stats=self.stats['execution_revisions'].setdefault(execution_revision,dict(
            image_jobs=0,visibility_jobs=0,image_native_wall_seconds=0.,visibility_native_wall_seconds=0.,
            image_write_wall_seconds=0.,image_files_written=0,image_cache_hits=0,metric_cache_hits=0,
            metric_cache_misses=0,visibility_cache_hits=0,image_write_timing_available=True))

    def _save(self):
        commit(self.path, self.stats)
        if self.progress_callback is not None:
            self.progress_callback()

    def _ensure(self):
        if self.renderer is not None:
            return
        self.budget.check()
        if self.renderer_factory is not None:
            self.renderer, self.backend = self.renderer_factory()
            return
        import torch
        from experiments.visibility_guided_sse.backend import VisibilityBackend, native_cuda_smoke
        from experiments.nava_mse.rendering import StreamSTGSRenderer
        smoke = native_cuda_smoke()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 4500*1024**2/memory), 0)
        self.backend = VisibilityBackend(device='cuda', ffmpeg_threads=1)
        self.renderer = StreamSTGSRenderer(self.packages, metadata_root=self.metadata_root,
            bounds_source=self.bounds_source, max_resident_models=self.max_resident_models,
            device='cuda', ffmpeg_threads=1, backend=self.backend)
        commit(self.root/'native_runtime.json', dict(smoke=smoke,
            runtime=self.backend.runtime_metadata(), max_resident_models=self.max_resident_models,
            gpu_memory_limit_mb=4500,execution_revision=self.execution_revision,
            lossless_image_writers=self.writer.executor._max_workers,max_pending_images=self.writer.max_pending))

    def identity(self, pose, frame, allocation, kind):
        return dict(schema='streaming-native-job.v1', fingerprint=self.fingerprint,
                    kind=kind, **physical_inputs(pose, frame), allocation=dict(sorted(allocation.items())))

    def _render(self, identity, pose):
        self.budget.check(reserve_bytes=8*1024**2*(1+len(self.writer.pending)))
        self._ensure()
        frame = identity['source_frame']
        request = {**identity, 'timestamp_s': self.source_times[frame]}
        self.backend.configure(request)
        started = time.perf_counter()
        try:
            result = self.renderer.render(pose.camera, identity['instances'], self.source_times[frame],
                                          identity['source_gop'], identity['allocation'])
        except Exception:
            self.stats['native_jobs_failed'] += 1
            self._save()
            raise
        elapsed = time.perf_counter()-started
        self.budget.native_seconds += elapsed
        self.stats['native_wall_seconds'] += elapsed
        self.stats['native_jobs_completed'] += 1
        self.stats['native_object_instances_processed'] += len(identity['instances'])
        self.stats['visibility_jobs' if identity['kind'] == 'proxy' else 'image_jobs'] += 1
        kind='visibility' if identity['kind']=='proxy' else 'image'
        self.revision_stats[f'{kind}_jobs']+=1
        self.revision_stats[f'{kind}_native_wall_seconds']+=elapsed
        self._save()
        return result, elapsed

    def visibility(self, pose, frame):
        self.budget.check()
        identity = self.identity(pose, frame, {r['instance_id']:40 for r in pose.instances}, 'proxy')
        key = digest(identity)
        path = self.root/'cache/visibility'/f'{key}.json'
        started = time.perf_counter()
        if path.exists():
            record = read_json(path)
            require(record['identity'] == identity and record['payload_sha256'] == digest(record['payload']),
                    'Corrupt or incompatible runtime visibility cache')
            lookup_ms = (time.perf_counter()-started)*1000
            self.stats['visibility_cache_hits'] += 1
            self.revision_stats['visibility_cache_hits']+=1
            event = dict(cache_hit=True, cache_lookup_ms=lookup_ms, native_wall_ms=0., analysis_ms=0.,
                original_analysis_ms=record['analysis_ms'], original_native_wall_ms=record['native_wall_ms'])
        else:
            _, elapsed = self._render(identity, pose)
            sample = self.backend.last_visibility
            payload = {iid: dict(A_solo=row.A_solo, A_visible=row.A_vis)
                       for iid, row in sample.per_object.items()}
            analysis_ms = float(sample.metadata['analysis_time_ms'])
            record = dict(identity=identity, payload=payload, payload_sha256=digest(payload),
                          native_wall_ms=elapsed*1000, analysis_ms=analysis_ms,
                          analyzer_metadata=dict(sample.metadata))
            commit(path, record)
            self.budget.disk_bytes += path.stat().st_size
            self.stats['visibility_analysis_ms'] += analysis_ms
            event = dict(cache_hit=False, cache_lookup_ms=0., native_wall_ms=elapsed*1000,
                analysis_ms=analysis_ms, original_analysis_ms=analysis_ms, original_native_wall_ms=elapsed*1000)
        self.events.append(dict(kind='visibility', key=key, source_frame=frame, **event))
        return record['payload'], event

    def image(self, pose, frame, allocation):
        identity = self.identity(pose, frame, allocation, 'joint')
        key = digest(identity)
        path = self.root/'cache/images'/f'{key}.npz'
        marker_path = path.with_suffix('.json')
        self.finish_image(key)
        if marker_path.exists():
            record = self.validate_image(key)
            require(record['identity']==identity,'Incompatible exact native image cache')
            with np.load(path, allow_pickle=False) as arrays:
                result = {k: arrays[k].copy() for k in arrays.files}
            self.stats['image_cache_hits'] += 1
            self.revision_stats['image_cache_hits']+=1
            return {**result, 'metadata':record['metadata'], 'key':key}
        if len(self.writer.pending)>=self.writer.max_pending:
            self.finish_image(next(iter(self.writer.pending)))
        result, elapsed = self._render(identity, pose)
        record=dict(identity=identity,metadata=result.metadata,native_wall_seconds=elapsed,
                    execution_revision=self.execution_revision)
        self.writer.submit(key,path,result.rgb,result.alpha,record)
        return dict(rgb=result.rgb, alpha=result.alpha, metadata=result.metadata, key=key)

    def finish_image(self,key):
        result=self.writer.finish(key)
        if result is not None:
            self.budget.disk_bytes+=result['bytes']
            self.revision_stats['image_write_wall_seconds']+=result['write_wall_seconds']
            self.revision_stats['image_files_written']+=1

    def flush(self):
        for key in list(self.writer.pending):
            self.finish_image(key)
        self.revision_stats['peak_pending_images']=max(self.writer.peak_pending,
            self.revision_stats.get('peak_pending_images',0))

    def validate_image(self,key):
        self.finish_image(key)
        path=self.root/'cache/images'/f'{key}.npz';marker=path.with_suffix('.json')
        if not marker.exists():
            return None
        signature=tuple((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in (path,marker))
        known=self.verified_images.get(key)
        if known is not None and known[0]==signature:
            return known[1]
        record=read_json(marker)
        require(digest(record['identity'])==key and file_digest(path)==record['arrays_sha256'],
                'Corrupt or incompatible exact native image cache')
        self.verified_images[key]=(signature,record)
        return record

    def annotate_quality(self,record):
        payload=record['payload'];identity=record['identity']['candidate'];provenance=payload['provenance']
        key,reference_key=provenance['candidate_frame_id'],provenance['reference_frame_id']
        pair=(key,reference_key)
        if pair in self.annotated_images:
            return
        stats=MetricStats.from_dict(payload['metrics'])
        reference_identity={**identity,'allocation':{iid:20 for iid in identity['allocation']}}
        require(digest(identity)==key and digest(reference_identity)==reference_key,'Wrong joint image/reference link')
        image_metrics(self.root,key,identity,stats,reference_key,role='selected_joint_reconstruction')
        image_metrics(self.root,reference_key,reference_identity,
            MetricStats(0.,stats.foreground_pixels,normalization='clip'),reference_key,role='joint_CRF20_reference')
        self.annotated_images.add(pair)

    def close(self):
        try:
            self.flush()
        finally:
            self.writer.close()
            if self.renderer is not None:
                self.renderer.close()
            self._save()


class NativeQualityProvider:
    def __init__(self, engine, trace, scene, *, cache_only=False):
        self.engine, self.scene, self.cache_only = engine, scene, cache_only
        self.poses = {pose.identity:pose for gop in trace.gops for pose in trace.get_trajectory(gop['gop_id'])}
        self.reference_identity = digest(dict(reference='joint_CRF20', scene_id=scene['scene_id'],
            native_fingerprint=engine.fingerprint, mask_mode='full_viewport', normalization='clip'))
        self.records = []

    def measure(self, context):
        from .quality import FrameQualityMeasurement
        if not self.cache_only:
            self.engine.budget.check()
        pose = self.poses[context.camera_sample_id]
        frame = round(context.content_time_s*30)
        require(abs(context.content_time_s-frame/30) < 1e-10, 'Quality source time is not frame aligned')
        allocation = {iid:int(str(q).removeprefix('CRF')) for iid,q in context.selected_quality.items()}
        identity = dict(candidate=self.engine.identity(pose, frame, allocation, 'joint'),
                        reference=self.reference_identity)
        path = self.engine.root/'cache/quality'/f'{digest(identity)}.json'
        started = time.perf_counter()
        valid_record=None
        if path.exists():
            probe=read_json(path)
            require(probe['identity']==identity and probe['payload_sha256']==digest(probe['payload']),
                    'Corrupt actual frame-quality cache')
            provenance=probe['payload']['provenance']
            if all(self.engine.validate_image(provenance[k]) is not None
                   for k in ('candidate_frame_id','reference_frame_id')):
                valid_record=probe
        if valid_record is not None:
            record = valid_record
            require(record['identity'] == identity and record['payload_sha256'] == digest(record['payload']),
                    'Corrupt actual frame-quality cache')
            self.engine.stats['metric_cache_hits'] += 1
            self.engine.revision_stats['metric_cache_hits']+=1
            self.records.append(dict(global_time_s=context.global_time_s, cache_hit=True,
                lookup_ms=(time.perf_counter()-started)*1000, native_wall_ms=0.))
        elif self.cache_only:
            return None
        else:
            self.engine.budget.check()
            before = self.engine.stats['native_wall_seconds']
            reference = self.engine.image(pose, frame, {iid:20 for iid in allocation})
            candidate = reference if set(allocation.values()) == {20} else self.engine.image(pose, frame, allocation)
            validate_render_alignment(reference['metadata'], candidate['metadata'])
            difference = np.asarray(candidate['rgb'], np.float64)-np.asarray(reference['rgb'], np.float64)
            stats = MetricStats(float(np.sum(difference*difference, dtype=np.float64)),
                difference.shape[0]*difference.shape[1], normalization='clip')
            masked = None
            if (reference['alpha'] > .01).any():
                masked = foreground_metrics(reference['rgb'], candidate['rgb'], reference['alpha'],
                    normalization='clip').to_dict()
            payload = dict(metrics=stats.to_dict(), reference_foreground_metrics=masked,
                provenance=dict(reference_frame_id=reference['key'], candidate_frame_id=candidate['key'],
                                measurement_source='actual_native_joint_viewport_RGB'))
            record = dict(identity=identity, payload=payload, payload_sha256=digest(payload))
            commit(path, record)
            self.engine.budget.disk_bytes += path.stat().st_size
            self.engine.stats['measured_metric_jobs'] += 1
            self.engine.revision_stats['metric_cache_misses']+=1
            self.records.append(dict(global_time_s=context.global_time_s, cache_hit=False,
                lookup_ms=0., native_wall_ms=(self.engine.stats['native_wall_seconds']-before)*1000))
        payload = record['payload']
        self.engine.annotate_quality(record)
        return FrameQualityMeasurement(context, MetricStats.from_dict(payload['metrics']),
            self.reference_identity, 'full_viewport', payload['provenance'])
