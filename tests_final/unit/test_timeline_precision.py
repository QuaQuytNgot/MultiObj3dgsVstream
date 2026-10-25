"""Source-frame continuity and GOP boundaries must survive binary64 rounding."""
import math
from types import SimpleNamespace

import pytest

from multiobj3dgs.client.runtime import PlaybackRuntime, SceneClock
from multiobj3dgs.client.scene import media_time
from multiobj3dgs.shared.metadata import (AssetRef, ContentCatalog, FrameMapping,
    GopSegment, ObjectContent, RepresentationInfo, SceneInstance)


def catalog(count):
    representations={}
    for rank,name in enumerate(('crf20','crf40')):
        segments=[]
        for gop in range(5):
            asset=AssetRef(f'{name}-{gop}',f'/tmp/{name}-{gop}',1,'a'*64)
            mappings=tuple(FrameMapping(1000+n,n/30,(n%60)/62)
                           for n in range(gop*60,(gop+1)*60))
            segments.append(GopSegment(f'{name}-gop{gop}',gop,gop*2.,2.,asset,
                {'reconstruction.json':''},(),mappings))
        representations[name]=RepresentationInfo(name,20 if rank==0 else 40,rank,1000,tuple(segments))
    content=ObjectContent('content',30.,10.,representations)
    instances=tuple(SceneInstance(f'i{i}','content',media_offset=i*7/30,loop=count>1)
                    for i in range(count))
    return ContentCatalog('a'*64,{'content':content},instances,10.)


@pytest.mark.parametrize('count',[1,4])
@pytest.mark.parametrize('cycles',[1,2])
def test_every_source_frame_occurs_once_with_independent_offset_wraps(count,cycles):
    cat=catalog(count);content=cat.contents['content']
    clock=SceneClock(10.,30.,loop=cycles>1,clock=lambda:0.)
    seen=[[] for _ in cat.instances]
    for n in range(300*cycles):
        assert clock.position==(n%300)/30
        for i,instance in enumerate(cat.instances):
            stamp=media_time(instance,content,clock.position)
            segment=content.segment_at('crf20',stamp)
            mapping=segment.frame_at(stamp)
            expected=1000+(n+i*7)%300
            assert mapping.source_frame==expected,(n,i,stamp,mapping.source_frame,expected)
            seen[i].append(mapping.source_frame)
        clock.commit_frame()
    for sequence in seen:
        assert all((b-a)%300==1 for a,b in zip(sequence,sequence[1:]))


def test_seek_and_direct_assignment_preserve_arbitrary_phase():
    clock=SceneClock(10.,30.,clock=lambda:0.)
    for phase in (.123456789,2.012345678):
        if phase<1:clock.seek(phase)
        else:clock.position=phase
        for step in range(100):
            assert clock.position==phase+step/30
            clock.commit_frame()


def test_media_snap_is_limited_to_floating_point_roundoff():
    content=catalog(1).contents['content'];instance=SceneInstance('i','content')
    boundary=4.
    assert media_time(instance,content,math.nextafter(boundary,-math.inf))==boundary
    # A real phase difference, even far below one frame, remains unchanged.
    for value in (boundary-1e-8,boundary-32*math.ulp(boundary)):
        assert media_time(instance,content,value)==value
    phase=7/30-5*math.ulp(7/30)
    looped=SceneInstance('i','content',media_offset=phase,loop=True)
    assert media_time(looped,content,0.)==phase


def test_exclusive_endpoints_and_loop_wrap_use_roundoff_bound():
    content=catalog(1).contents['content']
    before_end=math.nextafter(content.duration,-math.inf)
    assert media_time(SceneInstance('i','content'),content,before_end) is None
    assert media_time(SceneInstance('i','content',loop=True),content,before_end)==0.
    active=SceneInstance('i','content',scene_start=.1,active_duration=.2,loop=True)
    assert media_time(active,content,math.nextafter(.1,-math.inf))==0.
    assert media_time(active,content,math.nextafter(.3,-math.inf)) is None


def test_runtime_manual_quality_activates_at_each_later_gop_boundary():
    runtime=PlaybackRuntime(catalog(1),SimpleNamespace(),SimpleNamespace(),None,prefetch=1)
    requests={31:'crf40',91:'crf20',151:'crf40',211:'crf20'}
    expected='crf20'
    try:
        for n in range(300):
            if n in requests:runtime.request_quality('i0',requests[n])
            runtime._desired()
            if n in (60,120,180,240):expected=requests[n-29]
            assert runtime.active_representations['i0']==expected,(n,runtime.clock.position)
            assert runtime.required['i0'][2]==expected
            runtime.clock.commit_frame()
    finally:
        runtime.close()
