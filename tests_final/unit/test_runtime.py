"""Runtime contracts use fake codec resources, exercising real I/O/CPU boundaries."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
import pytest

from multiobj3dgs.shared.metadata import AssetRef, FrameMapping, GopSegment, RepresentationInfo, ObjectContent, SceneInstance, ContentCatalog
from multiobj3dgs.shared.geometry import Camera
from multiobj3dgs.client.runtime import SceneClock, ByteBudget, MemoryLimits, ResourceLimitError, IOLoop, PlaybackRuntime, GopState
from multiobj3dgs.client.scene import media_time
from multiobj3dgs.client.runtime.scheduler import PreparationScheduler
from multiobj3dgs.client.adaptation import ManualSelector, Candidate, PlannerInput


def catalog(count=1):
    reps={}
    for rank,rep in enumerate(('crf20','crf40')):
        segments=[]
        for i in range(2):
            asset=AssetRef(f'asset-{rep}-{i}',f'/tmp/{rep}-{i}',1,'a'*64)
            segments.append(GopSegment(f'gop{i}',i,float(i),1.,asset,{'reconstruction.json':''},(),
                                       (FrameMapping(i*2,float(i),0.),FrameMapping(i*2+1,i+.5,.5))))
        reps[rep]=RepresentationInfo(rep,20 if rank==0 else 40,rank,1000,tuple(segments))
    content=ObjectContent('content',2.,2.,reps)
    return ContentCatalog('a'*64,{'content':content},tuple(SceneInstance(f'i{i}','content') for i in range(count)),2.)


class Cache:
    def __init__(self,path):
        self.path=path;self.released=[];self.fetched=[]
    async def materialize(self,segment):
        folder=self.path/f'{segment.resource.id}-{len(self.fetched)}'
        folder.mkdir();self.fetched.append(folder)
        return folder
    def release_package(self,path):
        self.released.append(path)


class Adapter:
    def __init__(self,gate=None):
        self.gate=gate;self.prepared=[];self.realized=[];self.released=[];self.calls=[]
    def estimate_memory(self,package):
        return dict(raw_bytes=5,host_bytes=10,gpu_bytes=20)
    def prepare_cpu(self,path,cancel):
        self.prepared.append((path,threading.get_ident()))
        if self.gate:self.gate.wait(timeout=5)
        if cancel.is_set():raise RuntimeError('cancelled')
        return SimpleNamespace(nbytes=8,path=path)
    def realize_gpu(self,prepared):
        self.realized.append((prepared.path,threading.get_ident()))
        return SimpleNamespace(nbytes=16,path=prepared.path)
    def reconstruct(self,handle,model_time,camera):
        self.calls.append((model_time,camera.center))
        return SimpleNamespace(transformed=lambda matrix:SimpleNamespace(matrix=matrix))
    def release(self,handle):
        self.released.append((handle.path,threading.get_ident()))


@pytest.fixture
def environment(tmp_path):
    io=IOLoop();cache=Cache(tmp_path);adapter=Adapter()
    runtime=PlaybackRuntime(catalog(),adapter,cache,io,prefetch=0,
                             limits=MemoryLimits(host_bytes=100,raw_bytes=100,gpu_bytes=100,transient_gpu_bytes=1))
    yield runtime,adapter,cache,io
    runtime.close();io.close()


def wait_ready(runtime,timeout=3):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        runtime.pump()
        if runtime.required and runtime.ready:return
        time.sleep(.005)
    raise AssertionError(runtime.diagnostics())


def test_byte_budget_reservations_replace_release_and_never_exceed():
    budget=ByteBudget(10,'test')
    assert budget.reserve('one',7)
    assert not budget.reserve('two',4)
    assert budget.used==7
    assert budget.reserve('one',3)
    assert budget.reserve('two',7)
    assert budget.used==budget.peak==10
    budget.release('one');budget.release('one')
    assert budget.used==7
    with pytest.raises(ResourceLimitError):budget.reserve('three',11)


def test_clock_no_frame_skip_and_rebuffer_accounting():
    now=[0.]
    clock=SceneClock(2.,2.,clock=lambda:now[0])
    clock.stall();now[0]=20
    assert clock.position==0
    clock.unstall()
    assert clock.rebuffer_seconds==20 and clock.rebuffer_count==1
    clock.commit_frame()
    assert clock.position==.5
    now[0]=100
    clock.commit_frame()
    assert clock.position==1. # late by 80s still exactly one frame
    clock.pause();assert not clock.due()
    clock.seek(0);assert clock.generation==1
    clock.loop=True;clock.position=1.5;clock.play();clock.commit_frame()
    assert clock.position==0 and clock.generation==1


def test_timeline_offset_active_interval_and_local_loop():
    content=catalog().contents['content']
    instance=SceneInstance('i','content',scene_start=3,media_offset=.5,active_duration=3,loop=True)
    assert media_time(instance,content,2.99) is None
    assert media_time(instance,content,3)==.5
    assert media_time(instance,content,5)==.5
    assert media_time(instance,content,6) is None


def test_priority_starvation_then_deadline_then_visibility():
    scheduler=PreparationScheduler(starvation_s=1,clock=lambda:10.)
    old=SimpleNamespace(created=8.,deadline=100,visibility=0,sequence=3)
    sooner=SimpleNamespace(created=9.8,deadline=11,visibility=0,sequence=2)
    later=SimpleNamespace(created=9.8,deadline=12,visibility=1,sequence=1)
    assert scheduler.order([later,sooner,old])==[old,sooner,later]


def test_cpu_worker_and_gpu_owner_separate_and_memory_released(environment):
    runtime,adapter,cache,_=environment
    owner=threading.get_ident();wait_ready(runtime)
    assert adapter.prepared[0][1]!=owner and adapter.realized[0][1]==owner
    assert runtime.host.used==0 and runtime.raw.used==0 and runtime.gpu.used==16
    runtime.close()
    assert runtime.gpu.used==0 and cache.fetched==cache.released
    assert adapter.released[0][1]==owner


def test_seek_cancels_stale_cpu_completion_before_gpu_publication(environment):
    runtime,adapter,cache,_=environment
    gate=threading.Event();adapter.gate=gate
    end=time.monotonic()+3
    while not adapter.prepared and time.monotonic()<end:
        runtime.pump();time.sleep(.005)
    assert adapter.prepared
    old=adapter.prepared[0][0]
    runtime.seek(1.);runtime.pump();gate.set()
    wait_ready(runtime)
    assert all(path!=old for path,_ in adapter.realized)
    assert old in cache.released
    assert all(key[-1]==1 for key in runtime.jobs)


def test_quality_switch_is_pending_until_next_gop(environment):
    runtime,adapter,cache,_=environment
    wait_ready(runtime)
    runtime.clock.position=.5
    runtime.request_quality('i0','crf40');runtime.pump()
    assert runtime.active_representations['i0']=='crf20'
    runtime.clock.position=1.
    wait_ready(runtime)
    assert runtime.active_representations['i0']=='crf40'
    assert runtime.jobs[runtime.required['i0']].representation_id=='crf40'


def test_owner_thread_enforced(environment):
    runtime,*_=environment
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(runtime.pump)
        with pytest.raises(RuntimeError,match='owner thread'):future.result()


def test_oversized_active_working_set_is_error_not_forever_rebuffer(tmp_path):
    io=IOLoop();runtime=PlaybackRuntime(catalog(2),Adapter(),Cache(tmp_path),io,prefetch=0,
        limits=MemoryLimits(host_bytes=100,raw_bytes=100,gpu_bytes=31,transient_gpu_bytes=1))
    try:
        with pytest.raises(ResourceLimitError,match='active scene'):
            wait_ready(runtime)
    finally:
        runtime.close();io.close()


def test_manual_planner_contract_opaque_representations():
    cat=catalog();selector=ManualSelector(cat,{'i0':'crf40'})
    inputs=PlannerInput('revision',(Candidate('i0','gop0','crf20',100,1,2),
                                    Candidate('i0','gop0','crf40',50,1,2)),10000)
    plan=selector.select(inputs)
    assert plan.reason=='manual' and len(plan.selections)==1
    assert plan.selections[0].representation_id=='crf40'
    with pytest.raises(KeyError):selector.set('i0','Q3')


def test_joint_frame_advances_only_when_all_due_objects_ready(environment,monkeypatch):
    from multiobj3dgs.shared import gaussian
    monkeypatch.setattr(gaussian,'concatenate',lambda states:tuple(states))
    runtime,adapter,cache,_=environment
    camera=Camera(64,64,np.eye(4),np.array([[50,0,32],[0,50,32],[0,0,1.]]))
    renderer=SimpleNamespace(draw=lambda states,camera,frame_id=None:SimpleNamespace(frame_id=frame_id))
    runtime.pump()
    assert runtime.render(camera,renderer) is None
    assert runtime.clock.position==0
    wait_ready(runtime)
    result=runtime.render(camera,renderer)
    assert result.scene_time==0 and runtime.clock.position==.5
    assert runtime.frame_id==1 and len(runtime.last_states)==1


def test_required_gpu_admission_reclaims_prefetch_handle(environment):
    runtime,adapter,cache,_=environment
    runtime.prefetch=1
    wait_ready(runtime)
    # With both handles resident, constrain remaining budget to force reclaim.
    end=time.monotonic()+3
    while len(adapter.realized)<2 and time.monotonic()<end:
        runtime.pump();time.sleep(.005)
    assert len(adapter.realized)==2
    runtime.gpu.capacity=36
    runtime.request_quality('i0','crf40')
    # Seeking causes old-generation handles to retire before new working set.
    runtime.seek(1.)
    wait_ready(runtime)
    assert runtime.ready and runtime.gpu.used<=36


def test_object_local_camera_center_uses_inverse_similarity(environment):
    from dataclasses import replace
    runtime,adapter,_,_=environment
    runtime.catalog=replace(runtime.catalog,instances=(SceneInstance('i0','content',translation=(2,0,0),scale=2),))
    wait_ready(runtime)
    camera=Camera(64,64,np.eye(4),np.array([[50,0,32],[0,50,32],[0,0,1.]]))
    states=runtime.reconstruct(camera)
    assert np.allclose(adapter.calls[-1][1],[-1,0,0])
    assert np.allclose(states[0].matrix[:3,3],[2,0,0])


def test_rebuffered_frame_reports_wall_deadline_not_only_compute(environment,monkeypatch):
    from multiobj3dgs.shared import gaussian
    monkeypatch.setattr(gaussian,'concatenate',lambda states:tuple(states))
    runtime,*_=environment
    wait_ready(runtime)
    runtime.clock._deadline=time.monotonic()-10
    camera=Camera(64,64,np.eye(4),np.array([[50,0,32],[0,50,32],[0,0,1.]]))
    renderer=SimpleNamespace(draw=lambda *a,**kw:None)
    result=runtime.render(camera,renderer)
    assert result.deadline_missed and not result.compute_deadline_missed
    assert result.lateness_s>9


def test_runtime_passes_missing_costs_through_manual_selection_plan(environment):
    runtime,_,cache,_=environment
    cache.missing_bytes=lambda segment:123
    wait_ready(runtime)
    assert runtime.planner_input.revision==runtime.catalog.revision
    assert all(c.missing_bytes==123 for c in runtime.planner_input.candidates)
    assert runtime.last_selection_plan.reason=='manual'
    assert runtime.last_selection_plan.selections[0].representation_id=='crf20'


def test_empty_scene_interval_clears_and_advances_instead_of_reusing_old_image(environment):
    from dataclasses import replace
    runtime,*_=environment
    runtime.catalog=replace(runtime.catalog,instances=(SceneInstance('i0','content',scene_start=1),),duration=3.)
    runtime.pump()
    clears=[]
    renderer=SimpleNamespace(clear=lambda *a,**kw:clears.append(kw))
    result=runtime.render(None,renderer)
    assert result is not None and runtime.clock.position==.5
    assert clears==[{'frame_id':0}]
