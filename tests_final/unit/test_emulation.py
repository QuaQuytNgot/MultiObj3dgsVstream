from multiobj3dgs.server.emulation import BodyPacer
import pytest

def test_trace_crossing_and_shared_request_budget():
    pacer=BodyPacer([{'at_s':0,'bps':8},{'at_s':1,'bps':16}],clock=lambda:0)
    assert pacer.reserve(3,0)==2
    assert pacer.reserve(2,0)==3
    assert pacer.reserve(1,5)==5.5

def test_invalid_trace():
    with pytest.raises(ValueError):BodyPacer([{'at_s':0,'bps':0}])
