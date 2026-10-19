from types import SimpleNamespace
import asyncio

import pytest

from src.client.decode_ahead import DecodeAhead
from src.client.playback_buffer import plan_window, prime_window


class Source:
    def select_frame(self, obj, time, *, loop):
        return SimpleNamespace(frame=int(time*30+1e-8))

    async def get_frame(self, obj, frame, q):
        await asyncio.sleep(.001)
        return SimpleNamespace(frame_index=frame,quality=q,timestamp=frame/30)


def plan(source, seconds=.5, capacity=100, objects=None, clocks=None, active=None, quality=None, **kwargs):
    objects=objects or [{'id':'a','source':'x'}]
    return plan_window(source,objects,clocks or {o['id']:0. for o in objects},
                       quality or (lambda obj,offset:0),active or (lambda obj:True),
                       fps=30,duration_s=seconds,capacity=capacity,loop=True,**kwargs)


def test_actual_decoded_keys_and_holes_determine_contiguous_readiness():
    window=plan(Source())
    ahead=SimpleNamespace(ready={key:object() for key in window.requests})
    assert len(window.requests)==15
    assert window.availability(ahead)['decoded_duration_s']==pytest.approx(.5)
    del ahead.ready['x',4,0]
    availability=window.availability(ahead)
    assert not availability['ready']
    assert availability['decoded_duration_s']==pytest.approx(4/30)
    assert availability['ready_frames']==14


def test_memory_clipping_is_common_across_independent_objects():
    objects=[{'id':'a','source':'x'},{'id':'b','source':'y'}]
    window=plan(Source(),capacity=7,objects=objects)
    assert window.effective_s==pytest.approx(.1)
    assert len(window.requests)==6
    assert 'capacity' in window.reason
    assert len(window.tracks['a'])==len(window.tracks['b'])==3
    with pytest.raises(ValueError,match='every initial object'):
        plan(Source(),capacity=1,objects=objects)


def test_shared_instances_quality_offsets_and_pause_are_accounted_independently():
    objects=[{'id':'a','source':'x'},{'id':'b','source':'x'},{'id':'paused','source':'y'}]
    window=plan(Source(),objects=objects,clocks={'a':0.,'b':.2,'paused':.5},
                active=lambda obj:obj['id']!='paused',quality=lambda obj,step:1 if obj['id']=='b' else 0)
    assert ('x',6,1) in window.requests
    assert window.tracks['paused']==[('y',15,0)]*15
    ahead=SimpleNamespace(ready={key:1 for key in window.requests if key!=('x',10,1)})
    available=window.availability(ahead)
    assert available['per_object']['a']==pytest.approx(.5)
    assert available['per_object']['b']==pytest.approx(4/30)
    assert available['per_object']['paused']==0
    assert not available['ready']
    shared=plan(Source(),objects=objects[:2])
    assert len(shared.requests)==15


@pytest.mark.asyncio
async def test_priming_waits_for_decoded_outputs_and_propagates_corruption():
    source=Source()
    ahead=DecodeAhead(source,20)
    try:
        result=await prime_window(ahead,plan(source),timeout_s=1)
        assert result['status']=='ready' and result['decoded_duration_s']==pytest.approx(.5)
        assert len(ahead.ready)==15
        async def fail(*args): raise ValueError('corrupt Gaussian packet')
        source.get_frame=fail
        with pytest.raises(ValueError,match='corrupt'):
            await prime_window(ahead,plan(source,clocks={'a':2.}),timeout_s=1)
    finally:
        await ahead.close()


@pytest.mark.asyncio
async def test_timeout_can_accept_only_a_decoded_prefix_and_error_mode_fails():
    class Slow(Source):
        async def get_frame(self,obj,frame,q):
            await asyncio.sleep(.001 if frame<2 else .1)
            return SimpleNamespace(frame_index=frame)
    for fallback in ('shrink','error'):
        source=Slow();ahead=DecodeAhead(source,20)
        try:
            if fallback=='error':
                with pytest.raises(RuntimeError,match='timed out'):
                    await prime_window(ahead,plan(source),timeout_s=.02,fallback=fallback)
            else:
                result=await prime_window(ahead,plan(source),timeout_s=.02,fallback=fallback)
                assert result['status']=='timeout'
                assert result['decoded_duration_s']==pytest.approx(2/30)
        finally:
            await ahead.close()


@pytest.mark.asyncio
async def test_disabling_prefetch_keeps_primed_frames_and_admits_current_work():
    ahead=DecodeAhead(Source(),20)
    first=plan(ahead.source)
    try:
        await prime_window(ahead,first,timeout_s=1)
        future=plan(ahead.source,clocks={'a':1/30})
        await ahead.prefetch(future.requests,admit={('x',1,0)})
        assert ('x',14,0) in ahead.ready
        assert ('x',15,0) not in ahead.pending
        current=plan(ahead.source,clocks={'a':1.})
        await ahead.prefetch(current.requests,admit={('x',30,0)})
        assert ('x',30,0) in ahead.pending and len(ahead.pending)==1
    finally:
        await ahead.close()


def test_planning_bookkeeping_is_bounded_for_paused_or_repeated_frames():
    from src.client.playback_buffer import MAX_PLANNING_STEPS
    window=plan(Source(),seconds=1e308,active=lambda obj:False)
    assert window.reason=='presentation planning limit'
    assert len(window.tracks['a'])==MAX_PLANNING_STEPS
    assert len(window.requests)==1
