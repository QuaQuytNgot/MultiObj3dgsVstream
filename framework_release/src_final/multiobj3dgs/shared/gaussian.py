"""Activated Gaussian tensor contract. Torch is imported only for tensor operations."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .geometry import validate_transform


def quaternion_multiply(left, right):
    import torch
    a,b,c,d = left.unbind(-1); e,f,g,h = right.unbind(-1)
    return torch.stack((a*e-b*f-c*g-d*h,a*f+b*e+c*h-d*g,a*g-b*h+c*e+d*f,a*h+b*g-c*f+d*e),dim=-1)


def matrix_quaternion(r):
    # Robust CPU conversion, including rotations close to pi.
    from math import sqrt
    r = np.asarray(r,dtype=float); trace = np.trace(r)
    if trace > 0:
        s=sqrt(trace+1)*2; return np.array([s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s])
    i=int(np.argmax(np.diag(r))); j=(i+1)%3; k=(i+2)%3
    s=sqrt(1+r[i,i]-r[j,j]-r[k,k])*2
    q=np.zeros(4);q[0]=(r[k,j]-r[j,k])/s;q[i+1]=s/4;q[j+1]=(r[j,i]+r[i,j])/s;q[k+1]=(r[k,i]+r[i,k])/s
    return q


@dataclass(frozen=True)
class GaussianTensorState:
    xyz: Any
    scales: Any
    rotations: Any
    opacities: Any
    colors: Any
    provenance: str = ''

    def raster_args(self):
        import torch
        return dict(means3D=self.xyz,means2D=torch.zeros_like(self.xyz),shs=None,
            colors_precomp=self.colors,opacities=self.opacities,scales=self.scales,
            rotations=self.rotations,cov3D_precomp=None)

    def transformed(self, matrix):
        import torch
        m=validate_transform(matrix); scale=float(np.linalg.norm(m[:3,0]))
        rot=torch.as_tensor(m[:3,:3].copy(),device=self.xyz.device,dtype=self.xyz.dtype)
        trans=torch.as_tensor(m[:3,3].copy(),device=self.xyz.device,dtype=self.xyz.dtype)
        q=torch.as_tensor(matrix_quaternion(m[:3,:3]/scale),device=self.xyz.device,dtype=self.xyz.dtype)
        return GaussianTensorState(self.xyz@rot.T+trans,self.scales*scale,
            torch.nn.functional.normalize(quaternion_multiply(q.expand_as(self.rotations),self.rotations),dim=-1),
            self.opacities,self.colors,self.provenance)


def concatenate(states):
    import torch
    values=tuple(states)
    if not values:
        raise ValueError('Cannot concatenate zero Gaussian states')
    return GaussianTensorState(*(torch.cat([getattr(s,k) for s in values],dim=0) for k in ('xyz','scales','rotations','opacities','colors')),
        provenance='joint')
