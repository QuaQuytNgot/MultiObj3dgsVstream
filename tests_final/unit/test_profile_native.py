"""The benchmark oracle must reject mistakes its player can make at boundaries."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path

import pytest


_path=Path(__file__).resolve().parents[2]/'benchmarks/streaming/profile_native.py'
_spec=importlib.util.spec_from_file_location('native_profile_oracle',_path)
profile=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile)


def fixture(offsets=(0,),count=300,representations=None):
    reps=[]
    for quality in ('crf-20','crf-28','crf-40'):
        segments=[]
        for gop in range(5):
            segments.append(dict(id=f'{quality}:{gop}',start=gop*2,duration=2,
                frames=[dict(seconds=(gop*60+i)/30,source_frame=1000+gop*60+i,
                             model_time=i/62) for i in range(60)]))
        reps.append(dict(id=quality,segments=segments))
    index=dict(duration=10,contents=[dict(id='content',fps=30,duration=10,representations=reps)],
        instances=[dict(id=f'i{i}',content_id='content',media_offset=offset/30,loop=True)
                   for i,offset in enumerate(offsets)])
    frames=[]
    for frame in range(count):
        sources=[]
        for instance,offset in enumerate(offsets):
            tick=(frame+offset)%300;gop,local=divmod(tick,60)
            quality=(representations or {}).get((instance,frame),'crf-20')
            sources.append(dict(instance_id=f'i{instance}',gop_id=f'{quality}:{gop}',
                representation_id=quality,source_frame=1000+tick,
                model_time=local/62,media_time=tick/30))
        frames.append(dict(frame_id=frame,scene_time=(frame%300)/30,source_frames=sources))
    return index,frames


def test_rational_oracle_checks_600_frames_and_independent_phased_wraps():
    index,frames=fixture(offsets=(0,7,14,21),count=600)
    result=profile.validate_source_mappings(frames,index,fps=30,scene_loop=True)
    assert result['valid'] and result['checked_objects']==2400


def test_rational_oracle_rejects_accumulated_timestamp_duplicate_frame():
    index,frames=fixture()
    # An old player can report a mutually consistent timestamp/source pair
    # just before4s. Deriving expectations from scene_time would accept it.
    frames[120]['scene_time']=3.999999999999996
    frames[120]['source_frames'][0].update(gop_id='crf-20:1',source_frame=1119,
        model_time=59/62,media_time=3.999999999999996)
    result=profile.validate_source_mappings(frames,index,fps=30)
    assert not result['valid']
    assert any(e.get('field')=='source_frame' and e['expected']==1120 for e in result['errors'])


def test_quality_intent_must_activate_at_next_phased_local_boundary():
    schedule={(0,i):'crf-40' for i in range(53,150)}
    index,frames=fixture(offsets=(7,),count=150,representations=schedule)
    controls=[dict(action='quality',after_frame=40,instance_id='i0',representation_id='crf-40')]
    good=profile.validate_source_mappings(frames,index,fps=30,quality_controls=controls)
    assert good['valid'] and good['checked_quality_intents']==1
    wrong=deepcopy(frames)
    # Stale selection at frame53 is a valid representation/source mapping in
    # isolation, but violates the user's explicitly requested switch.
    wrong[53]['source_frames'][0].update(representation_id='crf-20',gop_id='crf-20:1')
    bad=profile.validate_source_mappings(wrong,index,fps=30,quality_controls=controls)
    assert not bad['valid']
    assert any('Requested representation' in e['message'] for e in bad['errors'])


def test_quality_intent_rejects_late_float_boundary_and_keeps_latest_request():
    schedule={(0,i):'crf-28' for i in range(120,180)}
    index,frames=fixture(count=180,representations=schedule)
    controls=[dict(action='quality',after_frame=95,instance_id='i0',representation_id='crf-40'),
              dict(action='quality',after_frame=110,instance_id='i0',representation_id='crf-28')]
    assert profile.validate_source_mappings(frames,index,fps=30,quality_controls=controls)['valid']
    wrong=deepcopy(frames)
    for frame in wrong[120:]:
        source=frame['source_frames'][0]
        source.update(representation_id='crf-20',gop_id='crf-20:2')
    wrong[120]['scene_time']=3.999999999999996
    result=profile.validate_source_mappings(wrong,index,fps=30,quality_controls=controls)
    assert not result['valid'] and result['checked_quality_intents']==1


@pytest.mark.parametrize('phase',[7/30-1e-10,7/30-5*math.ulp(7/30)])
def test_fractional_phase_below_grid_is_not_rounded_across_boundary(phase):
    index,frames=fixture(offsets=(7,),count=1)
    index['instances'][0]['media_offset']=phase
    frames[0]['source_frames'][0].update(source_frame=1006,model_time=6/62,media_time=phase)
    assert profile._rational_time(phase)<Fraction(7,30)
    assert profile.validate_source_mappings(frames,index,fps=30)['valid']
    frames[0]['source_frames'][0].update(source_frame=1007,model_time=7/62)
    assert not profile.validate_source_mappings(frames,index,fps=30)['valid']
