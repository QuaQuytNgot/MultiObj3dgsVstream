"""Calibrated cameras and positive similarity transforms (CPU-only imports)."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('Expected a finite nonzero WXYZ quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def similarity_matrix(translation=(0,0,0), rotation=(1,0,0,0), scale=1., pivot=(0,0,0)):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')
    t, p = np.asarray(translation, dtype=float), np.asarray(pivot, dtype=float)
    if t.shape != (3,) or p.shape != (3,) or not np.isfinite([t,p]).all():
        raise ValueError('Translation and pivot must be finite 3-vectors')
    m = np.eye(4)
    m[:3,:3] = quaternion_matrix(rotation) * scale
    m[:3,3] = t + p - m[:3,:3] @ p
    return m


def validate_transform(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4,4) or not np.isfinite(m).all() or not np.allclose(m[3], [0,0,0,1]):
        raise ValueError('Expected finite homogeneous 4x4 transform')
    s = np.linalg.norm(m[:3,0])
    if s <= 0 or not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3)*s*s, atol=1e-6) or np.linalg.det(m[:3,:3]) <= 0:
        raise ValueError('Only positive uniform scale and proper rotations are supported')
    return m


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    world_to_camera: np.ndarray
    K: np.ndarray
    near: float = .01
    far: float = 100.

    def __post_init__(self):
        w = np.array(self.world_to_camera, dtype=np.float64, copy=True)
        k = np.array(self.K, dtype=np.float64, copy=True)
        if self.width <= 0 or self.height <= 0 or w.shape != (4,4) or k.shape != (3,3):
            raise ValueError('Invalid camera dimensions')
        if not np.isfinite(w).all() or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0 or not 0 < self.near < self.far:
            raise ValueError('Invalid camera calibration/clipping planes')
        if not np.allclose(w[3], [0,0,0,1]) or not np.allclose(w[:3,:3] @ w[:3,:3].T, np.eye(3), atol=1e-5) or np.linalg.det(w[:3,:3]) <= 0:
            raise ValueError('Camera extrinsic must be a proper rigid transform')
        if not np.allclose(k[2], [0,0,1]) or not np.isclose(k[0,1], 0):
            raise ValueError('Unsupported skew/projective intrinsics')
        w.setflags(write=False); k.setflags(write=False)
        object.__setattr__(self, 'world_to_camera', w)
        object.__setattr__(self, 'K', k)

    @property
    def center(self):
        return -self.world_to_camera[:3,:3].T @ self.world_to_camera[:3,3]

    @property
    def projection(self):
        # Positive-Z camera and column-vector convention; transpose at CUDA boundary.
        p = np.zeros((4,4), dtype=np.float64)
        p[0,0], p[1,1] = 2*self.K[0,0]/self.width, 2*self.K[1,1]/self.height
        p[0,2], p[1,2] = 2*self.K[0,2]/self.width-1, 2*self.K[1,2]/self.height-1
        p[2,2], p[2,3], p[3,2] = self.far/(self.far-self.near), -self.far*self.near/(self.far-self.near), 1
        return p

    def resized(self, width, height):
        k = self.K.copy(); sx, sy = width/self.width, height/self.height
        k[0,0] *= sx; k[1,1] *= sy
        k[0,2] = (k[0,2]+.5)*sx-.5; k[1,2] = (k[1,2]+.5)*sy-.5
        return Camera(width,height,self.world_to_camera,k,self.near,self.far)

    def to_torch(self, device='cuda'):
        import torch
        view = torch.tensor(self.world_to_camera.T.copy(),dtype=torch.float32,device=device)
        proj = torch.tensor(self.projection.T.copy(),dtype=torch.float32,device=device)
        return SimpleNamespace(image_width=self.width,image_height=self.height,
            FoVx=2*math.atan(self.width/(2*self.K[0,0])), FoVy=2*math.atan(self.height/(2*self.K[1,1])),
            world_view_transform=view,full_proj_transform=view@proj,
            camera_center=torch.linalg.inv(view)[3,:3],time=0.,fisheye_mapper=None)

    @classmethod
    def from_package(cls, row, width=None, height=None):
        original_width = int(row.get('width',row.get('image_width',512)))
        original_height = int(row.get('height',row.get('image_height',512)))
        camera = cls(original_width,original_height,np.asarray(row['w2c']),np.asarray(row['K']))
        return camera.resized(width or original_width,height or original_height)
