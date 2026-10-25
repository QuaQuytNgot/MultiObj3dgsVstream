import pytest
from multiobj3dgs.client.visibility.analyzer import _object_visibility
from multiobj3dgs.client.visibility.prediction import CurrentPoseHold

def test_visibility_semantics_keep_zero_and_occlusion_distinct():
    result=_object_visibility('object',20,100,1e-6)
    assert result.occlusion_ratio==.8 and not result.out_of_view
    assert _object_visibility('empty',0,0,1e-6).occlusion_ratio is None
    with pytest.raises(ValueError):_object_visibility('bad',101,100,1e-6)

def test_prediction_is_explicit_hold_and_rejects_negative_horizon():
    camera=object();prediction=CurrentPoseHold().predict(camera,1,2)
    assert prediction.camera is camera and prediction.evidence=='current_pose_hold'
    with pytest.raises(ValueError):CurrentPoseHold().predict(camera,2,1)
