"""Bounded network → CPU decode-ahead → owner-thread GPU playback runtime.

No decoder, GL or CUDA call is made on the network/CPU preparation threads.
Independent instances own handles even when their verified bytes are shared.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import time
import numpy as np

from ...shared.geometry import Camera, similarity_matrix
from ..adaptation import ManualSelector, Candidate, PlannerInput
from ..scene import media_time
from .clock import SceneClock
from .memory import ByteBudget, MemoryLimits, ResourceLimitError
from .scheduler import PreparationScheduler


class GopState(str, Enum):
    REQUESTED = 'requested'
    FETCHING = 'fetching'
    VERIFIED = 'verified'
    CPU_PREPARING = 'cpu_preparing'
    CPU_READY = 'cpu_ready'
    UPLOADING = 'uploading'
    DECODER_READY = 'decoder_ready'
    ACTIVE = 'active'
    RETIRING = 'retiring'
    EVICTED = 'evicted'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class GopJob:
    key: tuple
    instance_id: str
    representation_id: str
    segment: object
    generation: int
    deadline: float
    sequence: int
    created: float = field(default_factory=time.monotonic)
    visibility: float = 1.0
    state: GopState = GopState.REQUESTED
    cancel: threading.Event = field(default_factory=threading.Event)
    future: object = None
    producer: object = None
    package: Path | None = None
    prepared: object = None
    handle: object = None
    upload: object = None
    preparation_members: object = None
    host_key: object = None
    error: Exception | None = None
    estimates: dict = field(default_factory=dict)
    stage_started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    scene_time: float
    receipt: object
    duration_s: float
    deadline_missed: bool
    compute_deadline_missed: bool = False
    lateness_s: float = 0.0


class PlaybackRuntime:
    def __init__(self, catalog, adapter, cache, io, *, limits=None, workers=1,
                 prefetch=1, loop=False, selector=None, telemetry=None, fps=None,
                 upload_budget_s=.002):
        if type(workers) is not int or workers < 1 or type(prefetch) is not int or prefetch < 0:
            raise ValueError('workers must be positive; prefetch must be nonnegative')
        self.owner = threading.get_ident()
        self.catalog, self.adapter, self.cache, self.io = catalog, adapter, cache, io
        self.limits = limits or MemoryLimits()
        self.host = ByteBudget(self.limits.host_bytes, 'prepared host')
        self._host_consumers={}
        self.raw = ByteBudget(self.limits.raw_bytes, 'raw decode')
        self.gpu = ByteBudget(self.limits.gpu_bytes-self.limits.transient_gpu_bytes, 'persistent GPU')
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='gop-cpu')
        self.workers, self.prefetch = workers, prefetch
        if upload_budget_s<0:raise ValueError('upload budget must be nonnegative')
        self.upload_budget_s=upload_budget_s
        rate = fps or max(c.fps for c in catalog.contents.values())
        self.clock = SceneClock(catalog.duration, rate, loop=loop)
        self.selector = selector or ManualSelector(catalog)
        self.active_representations = dict(self.selector.selected)
        self.pending_representations = {}
        self.bandwidth_estimate = None
        self.planner_input = None
        self.last_selection_plan = None
        self._last_plan_signature = None
        self.telemetry = telemetry
        self.jobs, self.retired = {}, []
        self.required = {}
        self.sequence, self.frame_id = 0, 0
        self.closed = False
        self.scheduler = PreparationScheduler()
        self._clock_generation = self.clock.generation
        self.last_states = ()
        self.visibility = {}
        self.frame_completions = deque(maxlen=120)
        self.displayed_time = None
        self._readiness_stall_started=None
        self._last_frame_sources=()

    def _assert_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('GPU runtime must be called on its owner thread')

    def emit(self, stage, job=None, **fields):
        if self.telemetry:
            if job:
                fields.update(instance_id=job.instance_id, gop_id=job.segment.id,
                              representation_id=job.representation_id, generation=job.generation)
            self.telemetry.emit(stage, **fields)

    def _transition(self, job, state):
        now = time.monotonic()
        self.emit(job.state.value+'_complete',job,duration_s=now-job.stage_started)
        job.state = state
        job.stage_started = now
        self.emit(state.value, job)

    def seek(self, seconds):
        self._assert_owner()
        self.clock.seek(seconds)
        self._invalidate()
        self.emit('seek', scene_time=self.clock.position, generation=self.clock.generation)

    def request_quality(self, instance_id, representation_id):
        self._assert_owner()
        self.selector.set(instance_id, representation_id)
        self.pending_representations[instance_id] = representation_id
        self.emit('quality_requested', instance_id=instance_id, representation_id=representation_id)

    def _invalidate(self):
        for job in self.jobs.values():
            job.cancel.set()
            self.retired.append(job)
        self.jobs.clear()
        self.required.clear()
        self.last_states = ()
        self._clock_generation = self.clock.generation

    def _desired(self):
        desired = {}
        required = {}
        boundary_instances = set()
        for instance in self.catalog.instances:
            content = self.catalog.contents[instance.content_id]
            timestamp = media_time(instance, content, self.clock.position)
            # Fetch an upcoming instance before its activation, when possible.
            upcoming = timestamp is None and self.clock.position < instance.scene_start
            if timestamp is None and not upcoming:
                continue
            timestamp = media_time(instance,content,instance.scene_start) if upcoming else timestamp
            if timestamp is None:
                continue
            representation = self.active_representations[instance.id]
            segment = content.segment_at(representation, timestamp)
            previous = self.jobs.get(self.required.get(instance.id))
            # Presentation rate/seek phase need not land exactly on GOP start.
            # Commit manual quality on the first sample of a new local GOP.
            at_boundary = abs(timestamp-segment.start) < 1e-8 or (
                previous is not None and previous.segment.gop_index != segment.gop_index)
            if at_boundary:boundary_instances.add(instance.id)
            pending = self.pending_representations.get(instance.id)
            if pending and at_boundary:
                representation = pending
                segment = content.segment_at(representation, timestamp)
                self.active_representations[instance.id] = pending
                del self.pending_representations[instance.id]
            segments = content.representations[representation].segments
            index = next(i for i,s in enumerate(segments) if s.id == segment.id)
            for offset in range(min(self.prefetch+1,len(segments))):
                if index+offset >= len(segments) and not instance.loop:
                    break
                rep = self.pending_representations.get(instance.id, representation) if offset else representation
                candidate = content.representations[rep].segments[(index+offset)%len(segments)]
                key = (instance.id, candidate.id, rep, self.clock.generation)
                distance=candidate.start-timestamp+(content.duration if index+offset>=len(segments) else 0.)
                deadline = time.monotonic() + max(0.0,distance) + (instance.scene_start-self.clock.position if upcoming else 0.0)
                desired[key] = (instance, rep, candidate, deadline)
                if offset == 0 and not upcoming:
                    required[instance.id] = key
        self.required = required
        candidates = {}
        buffers = {}
        for instance,_,segment,deadline in desired.values():
            content = self.catalog.contents[instance.content_id]
            for rep in content.representations.values():
                alternative = next(s for s in rep.segments if s.start == segment.start)
                missing = self.cache.missing_bytes(alternative) if hasattr(self.cache,'missing_bytes') else alternative.cold_bytes
                candidate = Candidate(instance.id,alternative.id,rep.id,missing,alternative.duration,deadline)
                candidates[(instance.id,alternative.id,rep.id)] = candidate
        for identifier,key in required.items():
            job = self.jobs.get(key)
            instance = next(i for i in self.catalog.instances if i.id == identifier)
            stamp = media_time(instance,self.catalog.contents[instance.content_id],self.clock.position)
            buffers[identifier] = max(0.,job.segment.end-stamp) if job and job.handle is not None else 0.
        self.planner_input = PlannerInput(self.catalog.revision,tuple(candidates.values()),
            self.bandwidth_estimate.bps if self.bandwidth_estimate is not None else None,
            dict(self.visibility),buffers,
            {'bandwidth':self.bandwidth_estimate,'selection_policy':'manual; changes activate at GOP boundaries'})
        self.last_selection_plan = self.selector.select(self.planner_input)
        choices = {(s.instance_id,s.gop_id):s.representation_id for s in self.last_selection_plan.selections}
        # Apply the manual SelectionPlan to prefetch and current access boundaries.
        for key,value in list(desired.items()):
            instance,rep,segment,deadline = value
            target = choices.get((instance.id,segment.id),rep)
            is_current = required.get(instance.id) == key
            if target != rep and (not is_current or instance.id in boundary_instances):
                alternative = next(s for s in self.catalog.contents[instance.content_id].representations[target].segments if s.start == segment.start)
                updated = (instance.id,alternative.id,target,self.clock.generation)
                desired.pop(key)
                desired[updated] = (instance,target,alternative,deadline)
                if is_current:
                    self.required[instance.id] = updated
                    self.active_representations[instance.id] = target
        signature = tuple((s.instance_id,s.gop_id,s.representation_id) for s in self.last_selection_plan.selections)
        if signature != self._last_plan_signature:
            self.emit('selection_plan',reason=self.last_selection_plan.reason,
                      selections=[vars(s) for s in self.last_selection_plan.selections],
                      bandwidth_bps=self.planner_input.predicted_bandwidth_bps)
            self._last_plan_signature = signature
        return desired

    def _new_job(self, key, value):
        instance, representation, segment, deadline = value
        self.sequence += 1
        job = GopJob(key, instance.id, representation, segment, self.clock.generation,
                     deadline, self.sequence, visibility=self.visibility.get(instance.id, 1.0))
        self._transition(job, GopState.FETCHING)
        job.future = self.io.submit(self.cache.materialize(segment))
        self.jobs[key] = job

    def _estimates(self, package):
        # Adapter provides conservative peak raw/prepared/persistent estimates.
        if not hasattr(self.adapter, 'estimate_memory'):
            raise TypeError('codec adapter must implement estimate_memory(package_dir)')
        estimate = self.adapter.estimate_memory(package)
        return {name: int(estimate[name]) for name in ('raw_bytes','host_bytes','gpu_bytes')}

    def _reserve_host(self,job,nbytes,key=None):
        key=job.host_key if job.host_key is not None else (job.key if key is None else key)
        # A shared immutable preparation occupies physical host bytes once.
        nbytes=max(nbytes,self.host.reservations.get(key,0))
        if not self.host.reserve(key,nbytes):return False
        if job.host_key is None:
            self._host_consumers[key]=self._host_consumers.get(key,0)+1
            job.host_key=key
        return True

    def _release_host(self,job):
        key=job.host_key
        if key is None:return
        count=self._host_consumers[key]-1
        if count:self._host_consumers[key]=count
        else:
            self._host_consumers.pop(key);self.host.release(key)
        job.host_key=None

    def _dispose(self, job):
        # Stop GPU work promptly, but keep the source package and CPU byte
        # reservations until the streaming producer has really stopped.
        if job.cancel.is_set() and job.upload is not None:
            if job.handle is not None:
                self.adapter.release(job.handle);job.handle=None
            self.adapter.cancel_gpu(job.upload);job.upload=None
            self.gpu.release(job.key)
        if job.producer is not None and not job.producer.done():return False
        if job.producer is not None:
            try:job.producer.result()
            except BaseException:pass
            job.producer=None
            if job.future is not None and not job.future.done():job.future.cancel()
        if job.future is not None and not job.future.done():
            return False
        if job.future is not None:
            try:
                value = job.future.result()
                if job.state == GopState.FETCHING:
                    job.package = Path(value)
            except BaseException:
                pass
            job.future = None
        if job.handle is not None:
            self.adapter.release(job.handle)
            job.handle = None
        if job.upload is not None:
            self.adapter.cancel_gpu(job.upload)
            job.upload=None
        job.prepared = None
        self._release_host(job)
        self.raw.release(job.key)
        self.gpu.release(job.key)
        if job.package is not None:
            self.cache.release_package(job.package)
            job.package = None
        self._transition(job, GopState.EVICTED)
        return True

    def pump(self):
        self._assert_owner()
        if self.closed:
            raise RuntimeError('runtime is closed')
        if self.clock.generation != self._clock_generation:
            self._invalidate()
        desired = self._desired()
        for key in list(self.jobs):
            if key not in desired:
                job = self.jobs.pop(key)
                job.cancel.set()
                self.retired.append(job)
        self.retired = [j for j in self.retired if not self._dispose(j)]
        for key,value in desired.items():
            if key not in self.jobs:
                self._new_job(key,value)
        for job in self.jobs.values():
            if job.producer is not None and job.producer.done():
                try:
                    value=job.producer.result()
                    self.emit('gop_cpu_stages',job,**getattr(value,'stages',{}))
                    job.producer=None
                    self.raw.release(job.key)
                except Exception as error:
                    job.producer=None
                    job.error=error;self._transition(job,GopState.FAILED)
            if job.future is None or not job.future.done():
                continue
            try:
                value = job.future.result()
                job.future = None
                if job.state == GopState.FETCHING:
                    job.package = Path(value)
                    job.estimates = self._estimates(job.package)
                    self._transition(job, GopState.VERIFIED)
                elif job.state == GopState.CPU_PREPARING:
                    job.prepared = value
                    actual = int(value.nbytes)
                    if actual > job.estimates['host_bytes']:
                        raise ResourceLimitError(f'codec estimate underflow: prepared {actual} > reserved {job.estimates["host_bytes"]}')
                    self._reserve_host(job, actual)
                    if job.producer is None:self.raw.release(job.key)
                    self._transition(job, GopState.CPU_READY)
                    self.emit('gop_cpu_stages',job,**getattr(value,'stages',{}))
            except Exception as error:
                job.error = error
                self._transition(job, GopState.FAILED)
        required_keys = set(self.required.values())
        # Due work takes precedence over prefetch even if future downloads finish first.
        ordered = self.scheduler.order(self.jobs.values())
        ordered.sort(key=lambda j: j.key not in required_keys)
        running = len({id(j.producer or j.future) for j in ordered
            if (j.producer is not None and not j.producer.done()) or j.state==GopState.CPU_PREPARING})
        for job in ordered:
            if job.state != GopState.VERIFIED:
                continue
            if job.key not in required_keys and not self.ready:
                continue
            try:
                peer=None
                if hasattr(self.adapter,'preparation_key'):
                    signature=self.adapter.preparation_key(job.segment)
                    peer=next((p for p in ordered if p is not job and not p.cancel.is_set()
                        and (p.state==GopState.CPU_PREPARING or p.prepared is not None)
                        and self.adapter.preparation_key(p.segment)==signature),None)
                if peer is None and running>=self.workers:continue
                host_key=peer.host_key if peer is not None else job.key
                if not self._reserve_host(job, job.estimates['host_bytes'],host_key):
                    if job.key in required_keys:
                        self._reclaim_prefetch(required_keys, gpu=False)
                    if not self._reserve_host(job, job.estimates['host_bytes'],host_key):
                        continue
                if peer is not None:
                    self.emit('cpu_preparation_shared',job,source_instance_id=peer.instance_id)
                    if peer.preparation_members is not None:
                        peer.preparation_members.append(job)
                        job.preparation_members=peer.preparation_members
                    if peer.state==GopState.CPU_PREPARING:
                        job.future=peer.future
                        job.producer=peer.producer
                        self._transition(job,GopState.CPU_PREPARING)
                    else:
                        job.prepared=peer.prepared
                        job.producer=peer.producer
                        self._reserve_host(job,int(job.prepared.nbytes))
                        self._transition(job,GopState.CPU_READY)
                    continue
                if not self.raw.reserve(job.key, job.estimates['raw_bytes']):
                    self._release_host(job)
                    continue
                self._transition(job, GopState.CPU_PREPARING)
                if hasattr(self.adapter,'preparation_key'):
                    members=[job];job.preparation_members=members
                    cancel=lambda members=members:all(j.cancel.is_set() for j in members)
                else:cancel=job.cancel
                if self.upload_budget_s and hasattr(self.adapter,'start_cpu'):
                    job.future,job.producer=self.adapter.start_cpu(job.package,cancel,self.pool)
                else:job.future = self.pool.submit(self.adapter.prepare_cpu, job.package, cancel)
                running += 1
            except Exception as error:
                job.error = error
                self._transition(job, GopState.FAILED)
        # Only owner thread enters CUDA; do not upload prefetch while any due CPU work remains.
        # Each active instance advances one temporal feature per source frame.
        # Scale the bounded copy allowance with scene cardinality so N=4 cannot
        # starve behind a one-object transfer rate during sustained rendering.
        allowance=self.upload_budget_s*min(4,max(1,len(required_keys)))*(2 if not self.ready else 1)
        deadline=time.perf_counter()+allowance
        for job in ordered:
            if job.state != GopState.CPU_READY:
                continue
            if job.key not in required_keys and not self.ready:
                continue
            if self.upload_budget_s and time.perf_counter()>=deadline:break
            try:
                if self.upload_budget_s and hasattr(self.adapter,'estimate_upload_bytes'):
                    job.estimates['gpu_bytes']=self.adapter.estimate_upload_bytes(job.prepared)
                if not self.gpu.reserve(job.key, job.estimates['gpu_bytes']):
                    if job.key in required_keys:
                        self._reclaim_prefetch(required_keys, gpu=True)
                    if not self.gpu.reserve(job.key, job.estimates['gpu_bytes']):
                        continue
                self._transition(job, GopState.UPLOADING)
                if self.upload_budget_s and hasattr(self.adapter,'begin_gpu'):
                    if not self.raw.reserve('upload-staging',16*1024**2):
                        self._transition(job,GopState.CPU_READY);self.gpu.release(job.key);continue
                    job.upload=self.adapter.begin_gpu(job.prepared)
                    continue
                job.handle = self.adapter.realize_gpu(job.prepared)
                if int(job.handle.nbytes) > job.estimates['gpu_bytes']:
                    raise ResourceLimitError('codec GPU estimate smaller than realized handle')
                self.gpu.reserve(job.key, int(job.handle.nbytes))
                job.prepared = None
                self._release_host(job)
                self._transition(job, GopState.DECODER_READY)
                self.emit('gop_gpu_stages',job,**getattr(job.handle,'stages',{}))
            except Exception as error:
                job.error = error
                self._transition(job, GopState.FAILED)
        # Bulk H2D work advances within a small owner-thread budget. Copies run
        # on a dedicated stream; READY is published only after its final event.
        for job in ordered:
            if job.upload is None:continue
            remaining=deadline-time.perf_counter()
            if remaining<=0:break
            try:
                handle=self.adapter.poll_gpu(job.upload,budget_s=remaining)
                if handle is None:continue
                if int(handle.nbytes)>job.estimates['gpu_bytes']:
                    raise ResourceLimitError('codec GPU estimate smaller than realized handle')
                if job.handle is None:
                    job.handle=handle
                    self._transition(job,GopState.DECODER_READY)
                    self.emit('first_window_gpu_ready',job,duration_s=getattr(handle,'stages',{}).get('first_window_gpu_ready_s',0.))
                if getattr(job.upload,'done',True):
                    job.upload=None;job.prepared=None
                    self.gpu.reserve(job.key,int(handle.nbytes));self._release_host(job)
                    self.emit('gop_gpu_stages',job,**getattr(handle,'stages',{}))
            except Exception as error:
                job.error=error;self._transition(job,GopState.FAILED)
        for key in required_keys:
            if self.jobs[key].error:
                raise RuntimeError(f'GOP preparation failed for {key}: {self.jobs[key].error}') from self.jobs[key].error
        # A required working set that cannot coexist is a resource error, not an eternal rebuffer.
        if required_keys and all(self.jobs[k].state in (GopState.CPU_READY,GopState.DECODER_READY,GopState.ACTIVE)
                                 for k in required_keys):
            need = sum(int(self.jobs[k].handle.nbytes) if self.jobs[k].handle is not None
                       else self.jobs[k].estimates['gpu_bytes'] for k in required_keys)
            if need > self.gpu.capacity:
                raise ResourceLimitError(f'active scene requires estimated {need} GPU bytes, budget {self.gpu.capacity}')

    def _reclaim_prefetch(self, required_keys, *, gpu):
        for other in self.jobs.values():
            if other.key in required_keys:
                continue
            if gpu and other.upload is not None:
                if other.handle is not None:self.adapter.release(other.handle)
                self.adapter.cancel_gpu(other.upload)
                other.handle=None;other.upload=None
                self.gpu.release(other.key)
                self._transition(other,GopState.CPU_READY)
                continue
            if other.producer is not None:
                # The admitted CPU producer owns its host/raw reservation and
                # source lease. Let it drain; do not release bytes it still uses.
                continue
            if other.state == GopState.CPU_READY:
                other.prepared = None
                self._release_host(other)
                self.raw.release(other.key)
                self._transition(other, GopState.VERIFIED)
            elif gpu and other.state in (GopState.DECODER_READY,GopState.ACTIVE):
                self.adapter.release(other.handle)
                other.handle = None
                self.gpu.release(other.key)
                self._transition(other, GopState.VERIFIED)

    @property
    def ready(self):
        for instance in self.catalog.instances:
            key=self.required.get(instance.id)
            if key is None:continue
            job=self.jobs[key]
            if job.state not in (GopState.DECODER_READY,GopState.ACTIVE):return False
            if hasattr(self.adapter,'frame_ready'):
                content=self.catalog.contents[instance.content_id]
                stamp=media_time(instance,content,self.clock.position)
                if not self.adapter.frame_ready(job.handle,job.segment.frame_at(stamp).model_time):return False
        return True

    def reconstruct(self, camera, scene_time=None):
        self._assert_owner()
        states = []
        sources=[]
        for instance in self.catalog.instances:
            key = self.required.get(instance.id)
            if key is None:
                continue
            job = self.jobs[key]
            content = self.catalog.contents[instance.content_id]
            timestamp = media_time(instance, content, self.clock.position if scene_time is None else scene_time)
            mapping = job.segment.frame_at(timestamp)
            sources.append(dict(instance_id=instance.id,gop_id=job.segment.id,representation_id=job.representation_id,
                                source_frame=mapping.source_frame,model_time=mapping.model_time,media_time=timestamp))
            matrix = similarity_matrix(instance.translation, instance.rotation, instance.scale, content.pivot)
            # Local camera center and rigid orientation; scale belongs to geometry.
            rotation = matrix[:3,:3] / instance.scale
            center = np.linalg.solve(matrix[:3,:3], camera.center-matrix[:3,3])
            local_rotation = camera.world_to_camera[:3,:3] @ rotation
            view = np.eye(4)
            view[:3,:3], view[:3,3] = local_rotation, -local_rotation@center
            local_camera = Camera(camera.width,camera.height,view,camera.K,camera.near,camera.far)
            state = self.adapter.reconstruct(job.handle, mapping.model_time, local_camera)
            states.append(state.transformed(matrix))
            self._transition(job, GopState.ACTIVE) if job.state != GopState.ACTIVE else None
        self.last_states = tuple(states)
        self._last_frame_sources=tuple(sources)
        return self.last_states

    def render(self, camera, renderer, *, refresh=False):
        self._assert_owner()
        if not self.ready:
            if self.clock.playing and self._readiness_stall_started is None:
                self._readiness_stall_started=time.monotonic()
                self.emit('readiness_stall_start',scene_time=self.clock.position,frame_id=self.frame_id)
            self.clock.stall()
            return None
        if not refresh and not self.clock.due():
            return None
        from ...shared.gaussian import concatenate
        started = time.monotonic()
        timestamp = self.clock.position
        self.clock.unstall()
        scheduled_deadline = self.clock.deadline + 1/self.clock.fps
        if self._readiness_stall_started is not None:
            self.emit('readiness_stall_end',scene_time=timestamp,frame_id=self.frame_id,
                      duration_s=time.monotonic()-self._readiness_stall_started)
            self._readiness_stall_started=None
        states = self.reconstruct(camera)
        if not states:
            # An intentionally empty interval still advances presentation time.
            receipt = renderer.clear(camera,frame_id=self.frame_id) if hasattr(renderer,'clear') else None
        else:
            receipt = renderer.draw(concatenate(states), camera, frame_id=self.frame_id)
        duration = time.monotonic()-started
        lateness = max(0.0,time.monotonic()-scheduled_deadline)
        result = FrameResult(self.frame_id,timestamp,receipt,duration,lateness > 0,
                             duration > 1/self.clock.fps,lateness)
        self.emit('scene_frame', frame_id=self.frame_id,scene_time=timestamp,duration_s=duration,
                  deadline_missed=result.deadline_missed,
                  compute_deadline_missed=result.compute_deadline_missed,lateness_s=lateness,fresh=not refresh,
                  source_frames=self._last_frame_sources)
        if not refresh:
            self.frame_id += 1
            self.frame_completions.append(time.monotonic())
            self.displayed_time = timestamp
            self.clock.commit_frame()
        return result

    def refresh(self, camera, renderer):
        """Camera-only redraw; never advances the clock or records a fresh frame.

        While the displayed GOP is still resident, view-conditioned heads are
        rerun at exactly its displayed timestamp. At a GOP rebuffer boundary the
        last reconstructed state remains drawable after old models are released.
        """
        self._assert_owner()
        if not self.last_states:
            return None
        can_reconstruct = self.displayed_time is not None
        if can_reconstruct:
            for instance in self.catalog.instances:
                key = self.required.get(instance.id)
                if key is None:
                    continue
                if self.jobs[key].handle is None:
                    can_reconstruct=False;break
                stamp = media_time(instance,self.catalog.contents[instance.content_id],self.displayed_time)
                if stamp is None or not self.jobs[key].segment.start <= stamp < self.jobs[key].segment.end:
                    can_reconstruct = False
                    break
                if hasattr(self.adapter,'frame_ready') and not self.adapter.frame_ready(self.jobs[key].handle,self.jobs[key].segment.frame_at(stamp).model_time):
                    can_reconstruct=False;break
        if can_reconstruct:
            self.reconstruct(camera,scene_time=self.displayed_time)
        from ...shared.gaussian import concatenate
        return renderer.draw(concatenate(self.last_states),camera,frame_id=self.frame_id)

    def diagnostics(self):
        times = self.frame_completions
        fps = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None
        torch = sys.modules.get('torch')
        allocated = reserved = peak_allocated = peak_reserved = device_free = device_total = None
        if torch is not None and torch.cuda.is_available():
            allocated,reserved = torch.cuda.memory_allocated(),torch.cuda.memory_reserved()
            peak_allocated,peak_reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
            device_free,device_total = torch.cuda.mem_get_info()
        return {'scene_time':self.clock.position,'playing':self.clock.playing,'stalled':self.clock.stalled,
                'frames':self.frame_id,'fresh_wall_fps':fps,'torch_allocated_bytes':allocated,
                'torch_reserved_bytes':reserved,'torch_process_peak_allocated_bytes':peak_allocated,
                'torch_process_peak_reserved_bytes':peak_reserved,'device_free_bytes':device_free,
                'device_total_bytes':device_total,'rebuffer_count':self.clock.rebuffer_count,
                'rebuffer_seconds':self.clock.total_rebuffer_seconds,
                'host_bytes':self.host.used,'host_peak_bytes':self.host.peak,
                'raw_bytes':self.raw.used,'raw_peak_bytes':self.raw.peak,
                'gpu_bytes':self.gpu.used,'gpu_peak_bytes':self.gpu.peak,
                'objects':[{'instance_id':j.instance_id,'gop_id':j.segment.id,
                            'representation_id':j.representation_id,'state':j.state.value}
                           for j in self.jobs.values()]}

    def close(self):
        self._assert_owner()
        if self.closed:
            return
        self.closed = True
        for job in self.jobs.values():
            job.cancel.set()
            self.retired.append(job)
        self.jobs.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for job in self.retired:
            if job.future is not None and not job.future.done():
                try:
                    job.future.result(timeout=60)
                except Exception:
                    job.future.cancel()
            self._dispose(job)
        self.retired.clear()
        self.raw.release('upload-staging')
