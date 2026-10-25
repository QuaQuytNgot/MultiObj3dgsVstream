"""Progressive CPU/upload ownership contracts without CUDA or timing thresholds."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from multiobj3dgs.client.runtime import IOLoop, PlaybackRuntime, MemoryLimits, GopState
from multiobj3dgs.shared.geometry import Camera
from tests_final.unit.test_runtime import catalog, Cache, Adapter


class ProgressiveAdapter(Adapter):
    """Controlled producer/completion futures model asynchronous resource owners."""
    def __init__(self):
        super().__init__()
        self.productions=[]
        self.uploads=[]
        self.cancelled_uploads=[]
        self.defer_publication=False

    def preparation_key(self,segment):
        return segment.resource.id

    def start_cpu(self,path,cancel,pool):
        host=SimpleNamespace(path=path,nbytes=8,visible_features=0,complete=False,
                             error=None,stages={'canonical_weights_ready_s':.01})
        published=Future();producer=Future()
        if not self.defer_publication:published.set_result(host)
        self.productions.append(SimpleNamespace(path=path,host=host,published=published,
            producer=producer,cancel=cancel))
        return published,producer

    def begin_gpu(self,prepared):
        pending=SimpleNamespace(prepared=prepared,handle=None,done=False,
                                owner=threading.get_ident())
        self.uploads.append(pending)
        return pending

    def poll_gpu(self,pending,budget_s):
        host=pending.prepared
        if host.error is not None:raise host.error
        if host.visible_features>=3 and pending.handle is None:
            pending.handle=SimpleNamespace(path=host.path,nbytes=16,available_features=0,
                stages={'first_window_gpu_ready_s':.02},released=False)
        if pending.handle is not None:
            pending.handle.available_features=host.visible_features
        pending.done=host.complete and host.visible_features>=4
        return pending.handle

    def frame_ready(self,handle,model_time):
        return int(model_time*2)+3<=handle.available_features

    def cancel_gpu(self,pending):
        self.cancelled_uploads.append((pending,threading.get_ident()))

    def release(self,handle):
        assert not handle.released
        handle.released=True
        super().release(handle)

    def finish(self,production,error=None):
        production.host.complete=True
        if error is None:
            production.host.stages['prepare_cpu_s']=.1
            production.producer.set_result(production.host)
        else:
            production.host.error=error
            production.producer.set_exception(error)


def pump_until(runtime,predicate,timeout=2):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        runtime.pump()
        if predicate():return
        # This only yields to the real I/O loop; readiness is event-controlled.
        time.sleep(.001)
    raise AssertionError(runtime.diagnostics())


@pytest.fixture
def progressive(tmp_path):
    environments=[]
    def make(count=1,offset=False):
        cat=catalog(count)
        if offset:
            cat=replace(cat,instances=(cat.instances[0],replace(cat.instances[1],media_offset=.5)))
        io=IOLoop();cache=Cache(tmp_path);adapter=ProgressiveAdapter()
        runtime=PlaybackRuntime(cat,adapter,cache,io,prefetch=0,workers=1,
            limits=MemoryLimits(host_bytes=100,raw_bytes=16*1024**2+100,
                                gpu_bytes=100,transient_gpu_bytes=1))
        environments.append((runtime,adapter,cache,io))
        return runtime,adapter,cache,io
    yield make
    for runtime,adapter,_,io in environments:
        for production in adapter.productions:
            if not production.producer.done():adapter.finish(production)
        runtime.close();io.close()


def camera_and_renderer(monkeypatch):
    from multiobj3dgs.shared import gaussian
    monkeypatch.setattr(gaussian,'concatenate',lambda states:tuple(states))
    camera=Camera(64,64,np.eye(4),np.array([[50,0,32],[0,50,32],[0,0,1.]]))
    renderer=SimpleNamespace(draw=lambda states,camera,frame_id=None:SimpleNamespace(frame_id=frame_id))
    return camera,renderer


def test_partial_gpu_window_gates_frame_and_keeps_reservations(progressive,monkeypatch):
    runtime,adapter,_,_=progressive()
    pump_until(runtime,lambda:bool(adapter.uploads))
    production=adapter.productions[0];host=production.host
    job=next(iter(runtime.jobs.values()))
    assert job.state==GopState.UPLOADING and not runtime.ready
    host.visible_features=3;runtime.pump()
    assert runtime.ready and job.handle is not None and job.upload is not None
    assert not production.producer.done()
    assert job.key in runtime.host.reservations and job.key in runtime.raw.reservations
    camera,renderer=camera_and_renderer(monkeypatch)
    assert runtime.render(camera,renderer).scene_time==0
    assert runtime.frame_id==1 and runtime.clock.position==.5
    assert not runtime.ready
    assert runtime.render(camera,renderer) is None
    assert runtime.frame_id==1 and runtime.clock.position==.5
    host.visible_features=4;adapter.finish(production);runtime.pump()
    assert runtime.ready and job.upload is None and job.prepared is None
    assert runtime.host.used==0 and job.key not in runtime.raw.reservations
    assert runtime.gpu.used==16
    runtime.clock.make_due()
    assert runtime.render(camera,renderer).scene_time==.5
    for budget in (runtime.host,runtime.raw,runtime.gpu):assert budget.peak<=budget.capacity


def test_seek_cancels_upload_but_holds_source_until_producer_stops(progressive):
    runtime,adapter,cache,_=progressive()
    pump_until(runtime,lambda:bool(adapter.uploads))
    production=adapter.productions[0]
    production.host.visible_features=3;runtime.pump()
    old=next(iter(runtime.jobs.values()));old_package=old.package;old_handle=old.handle
    runtime.seek(1.);runtime.pump()
    assert old.cancel.is_set() and old in runtime.retired
    assert old_handle.released and adapter.cancelled_uploads
    assert old.key not in runtime.gpu.reservations
    assert old.key in runtime.host.reservations and old.key in runtime.raw.reservations
    assert old_package not in cache.released
    assert production.cancel() and not production.producer.done()
    adapter.finish(production);runtime.pump()
    assert old not in runtime.retired and old_package in cache.released
    assert old.key not in runtime.host.reservations and old.key not in runtime.raw.reservations
    assert all(key[-1]==1 for key in runtime.jobs)
    assert all(thread==runtime.owner for _,thread in adapter.cancelled_uploads)


@pytest.mark.parametrize('removed_index',[0,1])
def test_shared_cpu_producer_survives_one_consumer_cancellation(progressive,removed_index):
    runtime,adapter,cache,_=progressive(2)
    runtime.host.capacity=10 # Exactly one physical preparation, not two copies.
    pump_until(runtime,lambda:len(adapter.uploads)==2)
    assert len(adapter.productions)==1
    assert runtime.host.used==runtime.host.peak==10
    production=adapter.productions[0]
    removed=runtime.catalog.instances[removed_index]
    retained=runtime.catalog.instances[1-removed_index]
    old=next(j for j in runtime.jobs.values() if j.instance_id==removed.id)
    old_package=old.package
    runtime.catalog=replace(runtime.catalog,instances=(retained,))
    runtime.pump()
    assert old.cancel.is_set() and old in runtime.retired
    assert not production.cancel() and not production.producer.cancelled()
    assert old_package not in cache.released
    production.host.visible_features=4;adapter.finish(production);runtime.pump()
    assert runtime.ready and len(adapter.productions)==1
    assert old_package in cache.released
    assert len(runtime.jobs)==1 and next(iter(runtime.jobs.values())).instance_id==retained.id
    assert runtime.host.used==0 and runtime.gpu.used==16


@pytest.mark.parametrize('first_new_gop_sample',[1.,1.2])
def test_progressive_manual_quality_switch_uses_new_independent_gop_handle(progressive,first_new_gop_sample):
    runtime,adapter,_,_=progressive()
    pump_until(runtime,lambda:bool(adapter.uploads))
    first=adapter.productions[0]
    first.host.visible_features=4;adapter.finish(first);runtime.pump()
    old_handle=next(iter(runtime.jobs.values())).handle
    runtime.clock.position=.5
    runtime.request_quality('i0','crf40');runtime.pump()
    assert runtime.active_representations['i0']=='crf20'
    runtime.clock.position=first_new_gop_sample
    pump_until(runtime,lambda:len(adapter.productions)==2 and len(adapter.uploads)==2)
    assert old_handle.released
    target=next(j for j in runtime.jobs.values() if j.key==runtime.required['i0'])
    assert target.representation_id=='crf40' and target.segment.gop_index==1
    assert not runtime.ready
    adapter.productions[1].host.visible_features=3;runtime.pump()
    assert runtime.ready and target.handle is not old_handle


def test_independent_offsets_require_each_objects_actual_feature_window(progressive,monkeypatch):
    runtime,adapter,_,_=progressive(2,offset=True)
    pump_until(runtime,lambda:len(adapter.uploads)==2)
    host=adapter.productions[0].host
    host.visible_features=3;runtime.pump()
    assert all(j.handle is not None for j in runtime.jobs.values())
    assert not runtime.ready # Second instance starts at model_time=.5.
    camera,renderer=camera_and_renderer(monkeypatch)
    assert runtime.render(camera,renderer) is None and runtime.frame_id==0
    host.visible_features=4;runtime.pump()
    assert runtime.ready
    result=runtime.render(camera,renderer)
    assert result.scene_time==0 and len(runtime.last_states)==2
    assert [t for t,_ in adapter.calls]==[0.,.5]


def test_natural_scene_loop_reuses_prefetched_gop_without_seek_invalidation(progressive,monkeypatch):
    runtime,adapter,_,_=progressive()
    runtime.catalog=replace(runtime.catalog,instances=(replace(runtime.catalog.instances[0],loop=True),))
    runtime.prefetch=1;runtime.clock.loop=True
    pump_until(runtime,lambda:len(adapter.uploads)==1)
    first=adapter.productions[0]
    first.host.visible_features=4;adapter.finish(first)
    pump_until(runtime,lambda:len(adapter.uploads)==2)
    second=adapter.productions[1]
    second.host.visible_features=4;adapter.finish(second);runtime.pump()
    original=next(j.handle for j in runtime.jobs.values() if j.segment.gop_index==0)
    runtime.clock.position=1.5;runtime.clock.make_due();runtime.pump()
    assert runtime.jobs[runtime.required['i0']].segment.gop_index==1
    camera,renderer=camera_and_renderer(monkeypatch)
    assert runtime.render(camera,renderer).scene_time==1.5
    assert runtime.clock.position==0 and runtime.clock.generation==0
    runtime.pump()
    assert runtime.ready and runtime.jobs[runtime.required['i0']].handle is original
    assert len(adapter.productions)==2 and not original.released


def test_producer_failure_after_early_publication_is_not_silently_rendered(progressive):
    runtime,adapter,_,_=progressive()
    pump_until(runtime,lambda:bool(adapter.uploads))
    production=adapter.productions[0]
    production.host.visible_features=3;runtime.pump()
    assert runtime.ready
    adapter.finish(production,ValueError('truncated remaining feature video'))
    with pytest.raises(RuntimeError,match='truncated remaining feature video'):
        runtime.pump()
    assert not runtime.ready


def test_cancelled_unstarted_producer_does_not_leave_unresolved_publication_lease(progressive):
    runtime,adapter,cache,_=progressive()
    adapter.defer_publication=True
    pump_until(runtime,lambda:bool(adapter.productions))
    production=adapter.productions[0]
    assert production.producer.cancel()
    job=next(iter(runtime.jobs.values()))
    try:
        assert runtime._dispose(job)
        assert production.published.done()
        assert cache.fetched==cache.released
        assert runtime.host.used==runtime.raw.used==runtime.gpu.used==0
    finally:
        # Keep a failing regression independent of close's 60-second timeout.
        if not production.published.done():production.published.cancel()


def test_completed_transfer_becomes_frame_ready_without_an_extra_pump():
    import weakref
    from multiobj3dgs.codecs.streamstgs.adapter import StreamSTGSAdapter,ReadyGopHandle,PendingGopUpload
    model=SimpleNamespace(time_line=10,time_batch=3,_point_feats=[None]*6)
    handle=ReadyGopHandle(model,{},[],0,'cuda',available_features=3)
    pending=PendingGopUpload(None,model,(),0.,{},available_features=3,handle=handle)
    handle.transfer=weakref.ref(pending)
    pending.feature_events.append((3,SimpleNamespace(query=lambda:True)))
    assert StreamSTGSAdapter().frame_ready(handle,.1)
    assert handle.available_features==4
