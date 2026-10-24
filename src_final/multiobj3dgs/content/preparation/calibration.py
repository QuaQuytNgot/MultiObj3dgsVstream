"""Explicit calibrated-camera checks and pinned StreamSTGS LLFF conversion.

OpenCV convention: +X right, +Y down, +Z forward; w2c acts on column vectors.
Blender transforms: c2w, +X right, +Y up, -Z forward; world axes stay unchanged.
The conversion below matches StreamSTGS ca860251's neural_3D_dataset_NDC loader.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class ReprojectionResult:
    points: int
    max_error_pixels: float
    rms_error_pixels: float
    positive_depth: bool


def validate_camera(camera):
    import numpy as np
    k=np.asarray(camera['K'],dtype=np.float64);w2c=np.asarray(camera['w2c'],dtype=np.float64)
    width,height=camera['width'],camera['height']
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (width,height)):raise ValueError('Positive integer image dimensions required')
    if camera.get('convention')!='opencv':raise ValueError('Explicit opencv camera convention required')
    if k.shape!=(3,3) or w2c.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(w2c).all():raise ValueError('Invalid camera matrix shape/values')
    if not np.allclose(k[2],[0,0,1],atol=1e-10,rtol=0) or k[0,0]<=0 or k[1,1]<=0 or abs(k[1,0])>1e-10:raise ValueError('Invalid pinhole intrinsics')
    if not np.allclose(w2c[3],[0,0,0,1],atol=1e-10,rtol=0):raise ValueError('Invalid homogeneous transform')
    rotation=w2c[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0) or not math.isclose(np.linalg.det(rotation),1.,abs_tol=1e-6):raise ValueError('Camera rotation must be rigid and right handed')
    return k,w2c


def blender_to_opencv(c2w,K,width,height):
    import numpy as np
    c2w=np.asarray(c2w,dtype=np.float64)
    if c2w.shape!=(4,4):raise ValueError('Blender c2w must be 4x4')
    camera={'convention':'opencv','K':np.asarray(K,dtype=np.float64).tolist(),'w2c':np.linalg.inv(c2w@np.diag([1.,-1.,-1.,1.])).tolist(),'width':width,'height':height}
    validate_camera(camera)
    return camera


def project_points(camera,world_points):
    import numpy as np
    k,w2c=validate_camera(camera)
    points=np.asarray(world_points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():raise ValueError('Nonempty finite Nx3 world points required')
    local=points@w2c[:3,:3].T+w2c[:3,3]
    if np.any(local[:,2]<=0):raise ValueError('Calibration points must lie in front of the camera')
    projected=local@k.T
    return projected[:,:2]/projected[:,2:3]


def validate_reprojection(camera,world_points,image_points, *, tolerance_pixels=.01):
    import numpy as np
    if not math.isfinite(tolerance_pixels) or tolerance_pixels<0:raise ValueError('Invalid reprojection tolerance')
    actual=project_points(camera,world_points);expected=np.asarray(image_points,dtype=np.float64)
    if expected.shape!=actual.shape or not np.isfinite(expected).all():raise ValueError('Image points must match world points')
    errors=np.linalg.norm(actual-expected,axis=1)
    report=ReprojectionResult(len(errors),float(errors.max()),float(np.sqrt(np.mean(errors**2))),True)
    if report.max_error_pixels>tolerance_pixels:raise ValueError(f'Reprojection error {report.max_error_pixels:.6g}px exceeds {tolerance_pixels}px')
    return report


def streamstgs_poses_bounds(cameras,near_fars):
    """Return Nx17 float64 LLFF rows; reject intrinsics the pinned loader loses.

    Caller preserves camera ordering when naming camNNN.mp4. This function does
    not generate video, choose bounds, normalize the scene or change world axes.
    """
    import numpy as np
    if not cameras:raise ValueError('At least one camera required')
    near_fars=np.asarray(near_fars,dtype=np.float64)
    if near_fars.shape!=(len(cameras),2) or not np.isfinite(near_fars).all() or np.any(near_fars[:,0]<=0) or np.any(near_fars[:,1]<=near_fars[:,0]):raise ValueError('Explicit positive near/far pair required per camera')
    rows=[];baseline=None
    for camera,bounds in zip(cameras,near_fars):
        k,w2c=validate_camera(camera);w,h=camera['width'],camera['height']
        if not np.allclose(k,[[k[0,0],0,w/2],[0,k[0,0],h/2],[0,0,1]],atol=1e-8,rtol=0):raise ValueError('Pinned StreamSTGS loader requires centered square-pixel intrinsics')
        identity=(w,h,float(k[0,0]))
        if baseline is not None and identity!=baseline:raise ValueError('Pinned StreamSTGS loader requires shared image size/focal length')
        baseline=identity;c2w=np.linalg.inv(w2c)
        # Loader: poses=[raw_y,-raw_x,raw_z,t]; R=poses.R*diag(1,-1,-1).
        # Its stored R is the transpose of the final OpenCV w2c rotation.
        pose=np.column_stack([c2w[:3,1],c2w[:3,0],-c2w[:3,2],c2w[:3,3],[h,w,k[0,0]]])
        rows.append(np.concatenate([pose.reshape(-1),bounds]))
    return np.asarray(rows,dtype=np.float64)


def export_poses_bounds(path,cameras,near_fars):
    import numpy as np
    values=streamstgs_poses_bounds(cameras,near_fars)
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
    return path
