"""Small CPU fixtures for phase ordering, exact deduplication and resume."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest

from experiments.adaptation_baselines.methods import METHODS
from experiments.adaptation_baselines.runner import load_experiment
from experiments.adaptation_baselines.tests.test_pipeline import frozen_fixture
from experiments.streaming_evaluation import two_phase as phase
from experiments.streaming_evaluation.network import NetworkTrace
from experiments.streaming_evaluation.runtime_matrix import load_settings,DEFAULT_CONFIG
from experiments.streaming_evaluation.runtime_native import NativeEngine,NativeQualityProvider,commit
from experiments.streaming_evaluation.runtime_scenes import make_scene,make_trace,bind_profiles
from experiments.streaming_evaluation.tests.test_runtime import cameras,FakeEngine,budget
from experiments.streaming_evaluation.timeline import Timeline
from experiments.streaming_evaluation.quality import FrameQualityContext
from experiments.streaming_evaluation.quality import FrameQualityMeasurement
from experiments.nava_mse.metrics import MetricStats
from experiments.visibility_weighted_sse.common import digest,file_digest,read_json


@pytest.fixture
def streaming_fixture(tmp_path,monkeypatch):
    source=load_experiment(frozen_fixture(tmp_path/'source'))['planner']
    cfg=load_settings(DEFAULT_CONFIG)
    cfg['object_counts']=[3];cfg['scenarios']=['long_line'];cfg['playback_durations_s']=[10]
    runner=phase.TwoPhaseRunner(tmp_path/'run',cfg)
    runner.fingerprint='fixture';runner.profiles=source.profiles
    engine=FakeEngine();engine.stats={}
    engine.image=lambda *a:pytest.fail('Phase A invoked a joint image')
    runner.engine=engine
    width,height=source.profiles.compatibility['resolution']
    scene=make_scene('long_line',3,cameras(('object-0','object-1')),width=width,height=height)
    original=bind_profiles
    monkeypatch.setattr(phase,'bind_profiles',lambda p,m,s,t:original(p,m,s,t,('g0','g1','g0')))
    return runner,source,scene


@pytest.mark.parametrize('method',METHODS)
def test_phase_a_all_methods_full_grid_without_joint_rendering_and_resume(streaming_fixture,method,monkeypatch):
    runner,source,scene=streaming_fixture
    folder=runner.streaming(source.profiles,source.metadata,scene,10,method,
        NetworkTrace([(0.,1000.)],duration_s=102.))
    phase_a=folder/'phase_a';document=read_json(phase_a/'played_frames.json')
    assert len(document['frames'])==300
    assert document['frames'][180]['loop_index']==1
    assert document['frames'][180]['content_time_s']==0
    assert document['exact_pose_and_instances_file']=='../../trace.json'
    assert (phase_a/document['exact_pose_and_instances_file']).is_file()
    assert not (folder/'completion.json').exists()
    assert not (runner.root/'cache/images').exists()
    sim=read_json(phase_a/'simulation.json')
    assert sim['metrics']['quality']['measured_frame_count']==0
    assert sim['metrics']['bandwidth']['total_requested_bytes']==sum(r['requested_bytes'] for r in sim['requests'])
    assert len({r['request_id'] for r in sim['requests']})==len(sim['requests'])
    before=file_digest(phase_a/'completion.json')
    monkeypatch.setattr(phase,'simulate',lambda *a:pytest.fail('Repeated completed streaming'))
    runner.streaming(source.profiles,source.metadata,scene,10,method,None)
    assert file_digest(phase_a/'completion.json')==before
    loaded_scene,trace,contexts,_=runner.load_run(folder)
    assert loaded_scene==scene and len(contexts)==300
    assert all(c.camera_sample_id in {p.identity for p in trace.get_trajectory(str(int(c.global_time_s)//2))}
               for c in contexts)


@pytest.mark.parametrize('completed_quality',[False,True])
def test_import_verified_existing_network_or_completed_result_without_simulating(streaming_fixture,monkeypatch,completed_quality):
    runner,source,scene=streaming_fixture
    folder=runner.streaming(source.profiles,source.metadata,scene,10,METHODS[0],
        NetworkTrace([(0.,1000.)],duration_s=102.))
    pa=folder/'phase_a'
    if completed_quality:
        files=['simulation.json','metrics.json','performance.json','requests.csv','buffer.csv','decisions.csv','frames.csv']
        for name in files:shutil.copyfile(pa/name,folder/name)
        phase.write_marker(folder,'completion.json',read_json(pa/'completion.json')['configuration_hash'],files)
        before={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}
    else:
        shutil.copyfile(pa/'simulation.json',folder/'network_checkpoint.json')
        shutil.copyfile(pa/'performance.json',folder/'performance.json')
        commit(folder/'network_checkpoint.sha256.json',dict(
            configuration_hash=read_json(pa/'completion.json')['configuration_hash'],method=METHODS[0],
            checkpoint_sha256=file_digest(folder/'network_checkpoint.json'),
            performance_sha256=file_digest(folder/'performance.json')))
    shutil.rmtree(pa)
    monkeypatch.setattr(phase,'simulate',lambda *a:pytest.fail('Reran imported network simulation'))
    runner.streaming(source.profiles,source.metadata,scene,10,METHODS[0],None)
    assert read_json(pa/'completion.json')['imported']
    if completed_quality:
        runner.merge(folder)
        assert all(file_digest(folder/name)==sha for name,sha in before.items())


def test_exact_job_dedup_across_loops_durations_and_camera_quality_changes(tmp_path):
    scene=make_scene('camera_rotation',3,cameras(),width=16,height=12)
    short=make_trace(scene,Timeline(6,10,2),'fixture')
    long=make_trace(scene,Timeline(6,30,2),'fixture')
    engine=NativeEngine(tmp_path,budget(tmp_path),{},None,None,'namespace',{i:i/30 for i in range(180)})
    p1,p2=NativeQualityProvider(engine,short,scene),NativeQualityProvider(engine,long,scene)
    allocation={r['instance_id']:'CRF28' for r in scene['instances']}
    def context(trace,t):
        g=str(t//2);pose=trace.get_trajectory(g)[0]
        return FrameQualityContext(scene['scene_id'],'fixture',trace.trace_id,pose.identity,
            'segment-'+g,str((t//2)%3),t//6,float(t),float(t%6),allocation,1/30)
    c0=context(short,0);c6=context(short,6);c24=context(long,24)
    keys=[phase.quality_job(engine,p,c)[0] for p,c in ((p1,c0),(p1,c6),(p2,c24))]
    assert len(set(keys))==1 and c0.key!=c6.key
    assert phase.quality_job(engine,p1,replace(c0,selected_quality={**allocation,'instance-0000':'CRF36'}))[0]!=keys[0]
    assert phase.quality_job(engine,p1,context(short,2))[0]!=keys[0]
    altered=deepcopy(scene);altered['instances'][0]['object_to_world'][0][3]+=.01
    altered['scene_id']='changed-transform'
    p3=NativeQualityProvider(engine,make_trace(altered,Timeline(6,10,2),'fixture'),altered)
    assert phase.quality_job(engine,p3,context(make_trace(altered,Timeline(6,10,2),'fixture'),0))[0]!=keys[0]
    engine.close()


def test_global_schedule_keeps_source_gop_models_together():
    jobs={str(i):dict(source_gop=g,scene_id=s,source_frame=f,quality_vector=[('a',q)])
          for i,(g,s,f,q) in enumerate([(2,'s',121,28),(0,'z',1,36),(1,'a',60,40),(0,'a',0,28)])}
    assert [jobs[k]['source_gop'] for k in phase.job_schedule(jobs)]==[0,0,1,2]


def test_global_plan_reuses_exact_frames_across_methods_and_loops(streaming_fixture,monkeypatch):
    runner,source,scene=streaming_fixture
    folders=[runner.streaming(source.profiles,source.metadata,scene,10,method,
        NetworkTrace([(0.,1000.)],duration_s=102.)) for method in METHODS]
    engine=NativeEngine(runner.root,runner.budget,{},None,None,runner.fingerprint,
        {i:i/30 for i in range(180)})
    runner.engine=engine
    # Normalize this synthetic Q0/Q1 fixture to an explicit common native CRF20
    # vector to test global job binding independently of utility preferences.
    original_load=runner.load_run
    def native_contexts(folder):
        scene,trace,contexts,config_hash=original_load(folder)
        return scene,trace,[replace(c,selected_quality={iid:'CRF20' for iid in c.selected_quality})
                             for c in contexts],config_hash
    monkeypatch.setattr(runner,'load_run',native_contexts)
    plan=runner.plan_quality(folders)
    assert plan['streaming_runs']==5 and plan['frame_occurrences']==1500
    assert plan['unique_quality_jobs']==180
    assert sum(j['occurrences'] for j in plan['jobs'].values())==1500
    assert all(len(binding)==300 and binding[0]==binding[180] for binding in plan['run_bindings'].values())
    assert plan['unique_images']==180
    assert runner.plan_quality(folders)==plan
    assert engine.renderer is None  # job planning never initializes native GPU work
    engine.close()


def test_phase_marker_rejects_corruption_and_configuration_change(tmp_path):
    commit(tmp_path/'frames.json',{'frames':[1]})
    phase.write_marker(tmp_path,'completion.json','science',['frames.json'])
    with pytest.raises(ValueError,match='configuration'):
        phase.verified_marker(tmp_path,'completion.json','other')
    commit(tmp_path/'frames.json',{'frames':[2]})
    with pytest.raises(ValueError,match='checksum'):
        phase.verified_marker(tmp_path,'completion.json','science')


def test_merge_actual_sse_keeps_streaming_bytes_times_and_full_frame_coverage(streaming_fixture,monkeypatch):
    runner,source,scene=streaming_fixture
    folder=runner.streaming(source.profiles,source.metadata,scene,10,METHODS[0],
        NetworkTrace([(0.,1000.)],duration_s=102.))
    original=read_json(folder/'phase_a/simulation.json')
    pinned=file_digest(folder/'phase_a/completion.json')
    commit(runner.root/'phase_b_jobs.json',{'synthetic_fixture':True})
    class SuppliedMeasurements:
        def __init__(self,*a,**kw):self.records=[]
        def measure(self,context):
            self.records.append(dict(cache_hit=True,lookup_ms=.01,native_wall_ms=0.))
            return FrameQualityMeasurement(context,MetricStats(.75,1,normalization='clip'),
                'fixture-joint-reference','full_viewport',dict(reference_frame_id='fixture-reference',
                candidate_frame_id='fixture-joint',measurement_source='CPU metric fixture, not experiment'))
    monkeypatch.setattr(phase,'NativeQualityProvider',SuppliedMeasurements)
    runner.merge(folder)
    merged=read_json(folder/'simulation.json')
    assert merged['requests']==original['requests'] and merged['playback']==original['playback']
    assert merged['buffer_samples']==original['buffer_samples']
    assert merged['metrics']['bandwidth']==original['metrics']['bandwidth']
    assert merged['metrics']['quality']['pooled_sse']==225.
    assert merged['metrics']['quality']['pooled_rgb_sample_count']==900
    assert merged['metrics']['quality']['measurement_coverage']==1
    assert len(merged['quality_frames'])==300
    assert [f['global_time_s'] for f in merged['quality_frames']]==[k/30 for k in range(300)]
    assert file_digest(folder/'phase_a/completion.json')==pinned
    assert phase.verified_marker(folder,'completion.json') is not None
    monkeypatch.setattr(phase,'NativeQualityProvider',lambda *a,**kw:pytest.fail('Re-merged completed quality'))
    runner.merge(folder)


def test_phase_b_requires_complete_streaming_matrix(tmp_path,monkeypatch):
    runner=phase.TwoPhaseRunner(tmp_path,load_settings(DEFAULT_CONFIG))
    monkeypatch.setattr(phase,'validate_inputs',lambda cfg:(None,None,{},cameras(),{},None,{}))
    monkeypatch.setattr(phase,'pin_execution',lambda *a:'fixture')
    monkeypatch.setattr(runner,'engine_open',lambda *a:None)
    monkeypatch.setattr(runner,'report',lambda:None)
    assert runner.run('B')==1
    assert 'complete streaming matrix' in runner.state['stop_reason']
