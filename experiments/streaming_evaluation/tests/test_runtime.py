"""CPU correctness fixtures for live integration, identity reuse and safe stops."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.adaptation_baselines.methods import METHODS
from experiments.adaptation_baselines.runner import load_experiment
from experiments.adaptation_baselines.tests.test_pipeline import frozen_fixture
from experiments.nava_mse.geometry import validate_source_conditioning
from experiments.streaming_evaluation.network import NetworkTrace
from experiments.streaming_evaluation.runtime_adapter import RuntimeVisibilityAdapter, RuntimePlanner
from experiments.streaming_evaluation.runtime_matrix import completed, load_settings, DEFAULT_CONFIG
from experiments.streaming_evaluation.runtime_native import (
    ExecutionBudget, NativeEngine, NativeQualityProvider, ResourceLimit, commit, physical_inputs)
from experiments.streaming_evaluation.quality import FrameQualityContext
from experiments.streaming_evaluation.runtime_scenes import make_scene, make_trace, bind_profiles, camera_at
from experiments.streaming_evaluation.simulator import simulate
from experiments.streaming_evaluation.timeline import Timeline
from experiments.visibility_weighted_sse.common import digest, file_digest


def cameras(names=('longdress','soldier')):
    record=dict(id=0,split='test',K=[[512.,0.,255.5],[0.,512.,255.5],[0.,0.,1.]],
        w2c=[[1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,-1.,4.],[0.,0.,0.,1.]])
    return {name:record for name in names}


@pytest.mark.parametrize('kind',['camera_rotation','long_line','random_spatial'])
@pytest.mark.parametrize('count',[2,3,8,15])
def test_distinct_spatial_origins_and_exact_source_conditioning(kind,count):
    scene=make_scene(kind,count,cameras())
    assert scene==make_scene(kind,count,cameras())
    transforms=[row['object_to_world'] for row in scene['instances']]
    assert len({digest(t) for t in transforms})==count
    assert len({tuple(np.asarray(t)[:3,3]) for t in transforms})==count
    assert len({row['instance_id'] for row in scene['instances']})==count
    for frame in (0,30,90,179):
        for row in scene['instances']:
            validate_source_conditioning(camera_at(scene,frame),row['object_to_world'],
                                         row['source_camera'],mode='recorded_center')
    if kind=='long_line':
        assert all(p[1]==0 and p[2]==0 for p in scene['positions'])


def test_six_second_loops_reuse_only_physical_camera_and_frame_identity():
    scene=make_scene('random_spatial',5,cameras())
    timeline=Timeline(6,30,2)
    trace=make_trace(scene,timeline,'test-content')
    poses={round(p.timestamp*30):p for g in trace.gops for p in trace.get_trajectory(g['gop_id'])}
    first=poses[30]
    for second in (7,13,25):
        pose=poses[second*30]
        assert timeline.content_time(second)==1
        assert pose.identity!=first.identity
        assert physical_inputs(pose,30)==physical_inputs(first,30)
    assert len({s.request_id for s in timeline})==15
    assert len(poses)==900


class FakeEngine:
    def __init__(self):
        self.budget=SimpleNamespace(check=lambda:None)
        self.cache={}
        self.computed=0

    def visibility(self,pose,frame):
        key=digest(physical_inputs(pose,frame))
        cached=key in self.cache
        if not cached:
            self.cache[key]={r['instance_id']:dict(A_solo=100.,A_visible=20.+10*i)
                             for i,r in enumerate(pose.instances)}
            self.computed+=1
        return self.cache[key],dict(cache_hit=cached,cache_lookup_ms=.01 if cached else 0.,
            native_wall_ms=0. if cached else 2.,analysis_ms=0. if cached else 1.,
            original_analysis_ms=1.,original_native_wall_ms=2.)


@pytest.mark.parametrize('method',METHODS)
def test_live_visibility_common_planner_and_independent_loop_transmission(tmp_path,method):
    source=load_experiment(frozen_fixture(tmp_path/'source'))['planner']
    width,height=source.profiles.compatibility['resolution']
    scene=make_scene('long_line',3,cameras(('object-0','object-1')),width=width,height=height)
    timeline=Timeline(6,10,2)
    trace=make_trace(scene,timeline,source.profiles.content_revision)
    profiles,metadata=bind_profiles(source.profiles,source.metadata,scene,timeline,('g0','g1','g0'))
    engine=FakeEngine()
    visibility=RuntimeVisibilityAdapter(engine,trace,profiles)
    planner=RuntimePlanner(profiles,trace,visibility,baseline_metadata=metadata,expected_object_count=3)
    planner.initialize_solver()
    config=dict(experiment_id='synthetic-live-correctness-only',input_kind='synthetic_fixture',
        adaptation=dict(methods=list(METHODS),safety_factor=1.),
        network=dict(estimator=dict(startup_bps=2000.,window=5,stale_s=30.)),
        playback=dict(startup_buffer_s=2.,max_buffer_s=4.),max_wall_time_s=100.,
        source_gop_ids=['g0','g1','g0'],quality=dict(mode='frame_grid',fps=30))
    sim,performance=simulate(config,timeline,scene,NetworkTrace([(0.,.002)],duration_s=100.),planner,method)
    assert sim['status']=='completed'
    assert len(sim['decisions'])==5
    assert len(sim['quality_frames'])==300
    assert engine.computed==180
    assert visibility.timings['3']['visibility_computed_sample_count']==0
    assert visibility.timings['3']['visibility_cached_sample_count']==60
    assert visibility.timings['0']['visibility_analysis_ms']==60
    assert visibility.timings['3']['visibility_analysis_ms']==0
    assert visibility.timings['3']['visibility_original_analysis_ms']==60
    assert performance['records'][0]['visibility_analysis_ms']==60
    assert all(r['decision_compute_excluding_preparation_ms']>=0 for r in performance['records'])
    requests=sim['requests']
    assert len({r['request_id'] for r in requests})==len(requests)
    assert {r['instance_id'] for r in requests}=={r['instance_id'] for r in scene['instances']}
    assert all(r['transfer_identity_sha256']!=r['source_resource_sha256'] for r in requests)
    assert sim['metrics']['bandwidth']['total_requested_bytes']==sum(d['total_required_bytes'] for d in sim['decisions'])
    hashes=[{r['resource_sha256'] for r in requests if r['global_segment_index']==i} for i in (0,3)]
    assert hashes[0]==hashes[1]  # identical payload, still fresh charged requests
    assert [d['bandwidth_estimate']['sample_count'] for d in sim['decisions']]==list(range(5))


def budget(root):
    root.mkdir(parents=True,exist_ok=True)
    return ExecutionBudget(root,dict(max_wall_seconds=100,max_native_seconds=100,
                                   max_storage_bytes=1024**3,minimum_free_bytes=1))


def test_exact_native_image_cache_reuses_loop_but_rejects_corruption(tmp_path):
    scene=make_scene('camera_rotation',2,cameras(),width=16,height=12)
    trace=make_trace(scene,Timeline(6,8,2),'test')
    p0=trace.get_trajectory('0')[0];p6=trace.get_trajectory('3')[0]
    backend=SimpleNamespace(configure=lambda identity:None)
    class Renderer:
        calls=0
        def render(self,*args):
            self.calls+=1
            return SimpleNamespace(rgb=np.zeros((12,16,3),np.float32),alpha=np.ones((12,16),np.float32),metadata={})
        def close(self):
            pass
    renderer=Renderer()
    engine=NativeEngine(tmp_path,budget(tmp_path),{},None,None,'test-fingerprint',
        {i:i/30 for i in range(180)},renderer_factory=lambda:(renderer,backend))
    allocation={r['instance_id']:28 for r in scene['instances']}
    first=engine.image(p0,0,allocation)
    repeated=engine.image(p6,0,allocation)
    assert first['key']==repeated['key'] and renderer.calls==1
    altered={**allocation,next(iter(allocation)):36}
    assert engine.image(p6,0,altered)['key']!=first['key'] and renderer.calls==2
    path=tmp_path/'cache/images'/f"{first['key']}.npz"
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):
        engine.image(p0,0,allocation)


def test_time_native_and_storage_limits_are_explicit_and_safe(tmp_path):
    b=budget(tmp_path)
    b.check()
    b.native_seconds=101
    with pytest.raises(ResourceLimit,match='max_native'):
        b.check()
    b.native_seconds=0;b.disk_bytes=b.limits['max_storage_bytes']
    with pytest.raises(ResourceLimit,match='max_storage'):
        b.check()
    b.disk_bytes=0;b.started-=101
    with pytest.raises(ResourceLimit,match='max_wall'):
        b.check()


def test_resume_requires_valid_results_and_frozen_configuration(tmp_path):
    commit(tmp_path/'metrics.json',dict(status='completed'))
    commit(tmp_path/'completion.json',dict(configuration_hash='science',
        outputs_sha256={'metrics.json':file_digest(tmp_path/'metrics.json')}))
    assert completed(tmp_path,'science')
    with pytest.raises(ValueError,match='incompatible'):
        completed(tmp_path,'changed-science')
    commit(tmp_path/'metrics.json',dict(status='edited'))
    with pytest.raises(ValueError,match='checksum'):
        completed(tmp_path,'science')


def test_default_protocol_has_finite_operational_limits():
    cfg=load_settings(DEFAULT_CONFIG)
    assert cfg['source_duration_s']==6 and len(cfg['sources'])==2
    assert cfg['limits']['max_wall_seconds']<=7200
    assert cfg['limits']['max_native_seconds']<=cfg['limits']['max_wall_seconds']


def test_actual_quality_cache_preserves_scalar_measurement_and_new_loop_context(tmp_path):
    """Fixture RGB is for metric correctness only; no experimental image values."""
    scene=make_scene('camera_rotation',2,cameras(),width=16,height=12)
    trace=make_trace(scene,Timeline(6,8,2),'test')
    object_fields=('instance_id object_id gop frame_index timestamp_s model_time training_identity '
        'checkpoint_sha256 object_to_world object_transform_sha256 source_identity_sha256 '
        'cameras_sha256 source_camera_id source_camera_split conditioning_mode '
        'source_camera_pose_sha256 source_local_camera source_local_camera_pose_sha256 '
        'source_local_camera_center').split()
    class Renderer:
        calls=0
        def render(self,camera,instances,timestamp,gop,allocation):
            self.calls+=1
            metadata={k:None for k in ('camera camera_pose_sha256 timestamp_s gop background '
                'normalization layout data_range renderer').split()}
            metadata.update(schema='nava-mse-render.v1',allocation=allocation,
                objects={iid:{**dict.fromkeys(object_fields), 'crf':q} for iid,q in allocation.items()})
            rgb=np.full((12,16,3),0. if set(allocation.values())=={20} else .5,np.float32)
            return SimpleNamespace(rgb=rgb,alpha=np.ones((12,16),np.float32),metadata=metadata)
        def close(self):
            pass
    renderer=Renderer()
    engine=NativeEngine(tmp_path,budget(tmp_path),{},None,None,'test',
        {i:i/30 for i in range(180)},renderer_factory=lambda:(renderer,SimpleNamespace(configure=lambda r:None)))
    provider=NativeQualityProvider(engine,trace,scene)
    allocation={r['instance_id']:'CRF28' for r in scene['instances']}
    def context(global_time,selected):
        g=str(global_time//2)
        return FrameQualityContext(scene['scene_id'],'test',trace.trace_id,trace.get_trajectory(g)[0].identity,
            'global-'+g,'0',global_time//6,float(global_time),0.,selected,1/30)
    first=provider.measure(context(0,allocation))
    repeated=provider.measure(context(6,allocation))
    assert first.metrics.mse==.25
    assert first.metrics.psnr==pytest.approx(6.020599913279624)
    assert first.metrics.sample_count==12*16*3
    assert repeated.metrics==first.metrics and repeated.context.global_time_s==6.
    assert repeated.context.global_segment_id!=first.context.global_segment_id
    assert repeated.provenance==first.provenance and renderer.calls==2
    exact=provider.measure(context(0,{iid:'CRF20' for iid in allocation}))
    assert exact.metrics.sse==0 and np.isinf(exact.metrics.psnr) and renderer.calls==2
    cache_only=NativeQualityProvider(engine,trace,scene,cache_only=True)
    assert cache_only.measure(context(6,allocation)).metrics==first.metrics
    assert cache_only.measure(context(0,{iid:'CRF36' for iid in allocation})) is None
    engine.close()
