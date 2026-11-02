"""Numeric image preservation, exact metadata and explicit cache-only upgrades."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from experiments.nava_mse.metrics import MetricStats
from experiments.streaming_evaluation.cache_upgrade import compatible,resolve,register
from experiments.streaming_evaluation.image_cache import ImageWriter,image_metrics,backfill,atomic_json
from experiments.visibility_weighted_sse.common import digest,file_digest,read_json
from experiments.streaming_evaluation.runtime_matrix import network_checkpoint,quality_schedule


def identity(allocation=None):
    return dict(fingerprint='fixture',kind='joint',source_frame=0,source_gop=0,
        camera={'fixture':True},instances=[dict(instance_id='a',object_id='source')],
        allocation=allocation or {'a':20})


def test_lossless_float_bits_and_immutable_bounded_writer(tmp_path):
    rgb=np.array([0.,np.nextafter(np.float32(.5),np.float32(1.)),1.,.123456789,.9999999,.01],np.float32).reshape(1,2,3)
    alpha=np.array([[.2,.9]],np.float32);expected_rgb=rgb.copy();expected_alpha=alpha.copy()
    writer=ImageWriter(1,1);native=identity();key=digest(native);path=tmp_path/f'{key}.npz'
    writer.submit(key,path,rgb,alpha,dict(identity=native,metadata={},native_wall_seconds=0.))
    rgb.fill(0);alpha.fill(0)
    with pytest.raises(ValueError,match='Drain'):
        writer.submit('other',tmp_path/'other.npz',rgb,alpha,{})
    result=writer.finish(key);writer.close()
    with np.load(path,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['rgb'].view(np.uint32),expected_rgb.view(np.uint32))
        np.testing.assert_array_equal(arrays['alpha'].view(np.uint32),expected_alpha.view(np.uint32))
    assert result['bytes']>0 and writer.peak_pending==1
    assert read_json(path.with_suffix('.json'))['arrays']['rgb']['dtype']=='<f4'


def test_writer_error_is_propagated_and_failed_future_removed(tmp_path):
    blocker=tmp_path/'file';blocker.write_text('not a directory')
    writer=ImageWriter(1,1)
    writer.submit('bad',blocker/'bad.npz',np.zeros((1,1,3),np.float32),np.ones((1,1),np.float32),{})
    with pytest.raises(FileExistsError):
        writer.finish('bad')
    assert not writer.pending
    writer.close()


def test_sidecar_has_actual_sse_psnr_samples_identity_and_joint_reference(tmp_path):
    native=identity({'a':28});key=digest(native);ref=digest(identity())
    stats=MetricStats(.75,1,normalization='clip')
    image_metrics(tmp_path,key,native,stats,ref,role='selected_joint_reconstruction')
    sidecar=read_json(tmp_path/'cache/images'/f'{key}.metrics.json')
    measured=sidecar['measurements'][ref]
    assert sidecar['rendering_identity']==native
    assert measured['sse']==.75 and measured['rgb_sample_count']==3
    assert measured['psnr_db']==pytest.approx(6.020599913279624)
    assert measured['computed_from_actual_joint_RGB']
    with pytest.raises(ValueError,match='Conflicting'):
        image_metrics(tmp_path,key,native,MetricStats(.5,1,normalization='clip'),ref,role='selected_joint_reconstruction')


def test_backfill_preserves_original_pixel_bytes_and_scalar_metrics(tmp_path):
    writer=ImageWriter(1,2);keys=[]
    for q,value in ((20,0.),(28,.5)):
        native=identity({'a':q});key=digest(native);keys.append(key)
        path=tmp_path/'cache/images'/f'{key}.npz'
        writer.submit(key,path,np.full((2,2,3),value,np.float32),np.ones((2,2),np.float32),
            dict(identity=native,metadata={},native_wall_seconds=.01))
    for key in keys:writer.finish(key)
    writer.close()
    payload=dict(metrics=MetricStats(3.,4,normalization='clip').to_dict(),
        provenance=dict(candidate_frame_id=keys[1],reference_frame_id=keys[0]))
    atomic_json(tmp_path/'cache/quality/test.json',dict(payload=payload,payload_sha256=digest(payload)))
    before={key:file_digest(tmp_path/'cache/images'/f'{key}.npz') for key in keys}
    result=backfill(tmp_path)
    assert result['validated_images']==2 and result['old_pixel_payloads_rewritten']==0
    assert all(file_digest(tmp_path/'cache/images'/f'{key}.npz')==before[key] for key in keys)
    row=read_json(tmp_path/'cache/images'/f'{keys[1]}.metrics.json')
    assert row['measurements'][keys[0]]['sse']==3.


def manifest():
    return dict(config=dict(renderer=dict(max_resident_models=4,width=640),sources=['a','b']),
        implementation_sha256={'experiments/streaming_evaluation/runtime_native.py':'old',
                               'experiments/adaptation_baselines/methods.py':'unchanged'},profiles_sha256='profiles')


def revised():
    value=manifest();value['config']['renderer'].update(max_resident_models=8,image_write_workers=1)
    value['implementation_sha256']['experiments/streaming_evaluation/runtime_native.py']='updated'
    return value


@pytest.mark.parametrize('change',['utility','source','pixels','profiles'])
def test_cache_revision_rejects_scientific_changes(change):
    current=revised()
    if change=='utility':current['implementation_sha256']['experiments/adaptation_baselines/methods.py']='bad'
    elif change=='source':current['config']['sources']=['a','c']
    elif change=='pixels':current['config']['renderer']['width']=320
    else:current['profiles_sha256']='bad'
    with pytest.raises(ValueError):compatible(manifest(),current)


def test_explicit_compatible_revision_preserves_original_namespace_and_pins_code(tmp_path):
    old=manifest();current=revised();atomic_json(tmp_path/'manifest.json',old)
    proof=tmp_path/'proof.json';atomic_json(proof,dict(status='EXACT_PIXELS_PRESERVED'))
    with pytest.raises(ValueError,match='Register'):
        resolve(tmp_path,current)
    register(tmp_path,current,proof)
    namespace,revision=resolve(tmp_path,current)
    assert namespace==digest(old) and revision=='joint_cache_v2'
    assert read_json(tmp_path/'manifest.json')==old
    unknown=deepcopy(current);unknown['implementation_sha256']['experiments/streaming_evaluation/runtime_native.py']='later'
    with pytest.raises(ValueError,match='exact implementation'):
        resolve(tmp_path,unknown)


def test_grouped_selected_jobs_keep_the_exact_context_set_and_vectors():
    contexts=[SimpleNamespace(content_gop_id=str((t//2)%3),content_time_s=float(t%6),
        selected_quality={'a':'CRF28'},global_time_s=float(t)) for t in (0,2,4,6,8)]
    result=quality_schedule(contexts)
    assert [c.global_time_s for c in result]==[0,6,2,8,4]
    assert {id(c) for c in result}=={id(c) for c in contexts}


def test_network_checkpoint_reuses_original_timings_and_rejects_corruption(tmp_path):
    simulation=dict(status='completed',decisions=['original decision'])
    performance=dict(records=[dict(native_solver_ms=123.)])
    atomic_json(tmp_path/'network_checkpoint.json',simulation)
    atomic_json(tmp_path/'performance.json',performance)
    atomic_json(tmp_path/'network_checkpoint.sha256.json',dict(configuration_hash='same',method='NAVA',
        checkpoint_sha256=file_digest(tmp_path/'network_checkpoint.json'),
        performance_sha256=file_digest(tmp_path/'performance.json')))
    assert network_checkpoint(tmp_path,'same','NAVA')==(simulation,performance)
    with pytest.raises(ValueError):network_checkpoint(tmp_path,'changed','NAVA')
    atomic_json(tmp_path/'performance.json',dict(records=[]))
    with pytest.raises(ValueError):network_checkpoint(tmp_path,'same','NAVA')


def test_backfill_rejects_psnr_values_that_disagree_with_saved_joint_pixels(tmp_path):
    writer=ImageWriter(1,2);keys=[]
    for q,value in ((20,0.),(28,.5)):
        native=identity({'a':q});key=digest(native);keys.append(key)
        writer.submit(key,tmp_path/'cache/images'/f'{key}.npz',np.full((1,1,3),value,np.float32),
                      np.ones((1,1),np.float32),dict(identity=native,metadata={},native_wall_seconds=0.))
    for key in keys:writer.finish(key)
    writer.close()
    payload=dict(metrics=MetricStats(.1,1,normalization='clip').to_dict(),
        provenance=dict(candidate_frame_id=keys[1],reference_frame_id=keys[0]))
    atomic_json(tmp_path/'cache/quality/bad.json',dict(payload=payload,payload_sha256=digest(payload)))
    with pytest.raises(ValueError,match='differs from the actual'):
        backfill(tmp_path)
