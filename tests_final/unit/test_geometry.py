import sys
import numpy as np
import pytest
from multiobj3dgs.shared.geometry import Camera, similarity_matrix, validate_transform
from multiobj3dgs.shared.gaussian import matrix_quaternion


def test_camera_projection_preserves_principal_point_and_resize():
    k=np.array([[500,0,201],[0,600,241],[0,0,1]])
    c=Camera(512,512,np.eye(4),k)
    np.testing.assert_allclose(c.projection[:2,2],np.array([201,241])/256-1)
    small=c.resized(256,128)
    np.testing.assert_allclose(small.K[[0,1],[0,1]],[250,150])
    np.testing.assert_allclose(small.K[[0,1],[2,2]],[(201+.5)/2-.5,(241+.5)/4-.5])
    assert not c.K.flags.writeable


def test_positive_uniform_similarity_and_rotation_at_pi():
    m=similarity_matrix((1,2,3),(0,0,1,0),2,pivot=(5,0,0))
    validate_transform(m)
    np.testing.assert_allclose(m@np.array([5,0,0,1]),[6,2,3,1])
    np.testing.assert_allclose(abs(matrix_quaternion(m[:3,:3]/2)),[0,0,1,0],atol=1e-9)
    with pytest.raises(ValueError):validate_transform(np.diag([-1,1,1,1]))
    with pytest.raises(ValueError):validate_transform(np.diag([2,1,1,1]))
    with pytest.raises(ValueError):similarity_matrix(scale=0)


def test_camera_center_and_invalid_extrinsics():
    m=np.eye(4);m[:3,3]=[1,2,3]
    c=Camera(100,100,m,np.diag([100,100,1]))
    np.testing.assert_allclose(c.center,[-1,-2,-3])
    m[0,0]=2
    with pytest.raises(ValueError):Camera(100,100,m,np.diag([100,100,1]))


def test_tensor_world_transform_preserves_activated_covariance():
    torch=pytest.importorskip('torch')
    from multiobj3dgs.shared.gaussian import GaussianTensorState, concatenate
    xyz=torch.tensor([[1.,0.,0.]])
    state=GaussianTensorState(xyz,torch.tensor([[1.,2.,3.]]),torch.tensor([[1.,0.,0.,0.]]),torch.tensor([[.5]]),torch.tensor([[.1,.2,.3]]))
    transform=similarity_matrix((5,0,0),(0,0,1,0),2)
    result=state.transformed(transform)
    torch.testing.assert_close(result.xyz,torch.tensor([[3.,0.,0.]]))
    torch.testing.assert_close(result.scales,torch.tensor([[2.,4.,6.]]))
    torch.testing.assert_close(result.rotations.abs(),torch.tensor([[0.,0.,1.,0.]]))
    assert result.opacities is state.opacities and result.colors is state.colors
    assert len(concatenate([state,result]).xyz)==2
