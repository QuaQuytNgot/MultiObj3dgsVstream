"""Existing RGB-basis analyzer adapted to activated StreamSTGS snapshots.

Algorithm source: src/renderer/analysis.py and src/client/visibility.py. All
geometry and opacity remain in each joint pass; constant colors identify groups.
CUDA contribution is labeled separately from FAST presentation.
"""
from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol


@dataclass(frozen=True)
class ObjectVisibility:
    object_id: str
    A_visible: float
    A_solo: float
    occlusion_ratio: float | None
    out_of_view: bool


@dataclass(frozen=True)
class VisibilityResult:
    timestamp: float
    resolution: tuple[int,int]
    per_object: dict
    metadata: dict
    diagnostics: dict | None = None


class PreparedContributionScene(Protocol):
    object_ids: tuple
    counts: tuple
    def render_joint(self, start): ...
    def render_solo(self, slot): ...


def _object_visibility(object_id: str, visible: float, solo: float,
                       epsilon: float) -> ObjectVisibility:
    visible, solo = float(visible), float(solo)
    if not math.isfinite(visible) or not math.isfinite(solo):
        raise ValueError(f"{object_id}: contribution areas must be finite")

    # The absolute term covers near-zero reductions; the relative term scales
    # with the solo area. Only numerical excursions are corrected.
    tolerance = 1e-6 + 1e-5 * max(0.0, solo)
    if visible < -tolerance or solo < -tolerance:
        raise ValueError(f"{object_id}: contribution areas must be nonnegative")
    visible, solo = max(0.0, visible), max(0.0, solo)
    if visible > solo + tolerance:
        raise ValueError(f"{object_id}: A_visible exceeds A_solo")
    visible = min(visible, solo)

    out_of_view = solo <= epsilon
    occlusion = None if out_of_view else 1.0 - visible / solo
    return ObjectVisibility(object_id, visible, solo, occlusion, out_of_view)


class TensorContributionScene:
    def __init__(self, objects, camera):
        import torch
        from multiobj3dgs.shared.gaussian import concatenate
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self.torch=torch;self.object_ids=tuple(objects)
        self.states=tuple(objects.values());self.counts=tuple(len(s.xyz) for s in self.states)
        self.joint=concatenate(self.states);self.camera=camera
        self.slots=torch.repeat_interleave(torch.arange(len(self.states),device=self.joint.xyz.device),torch.tensor(self.counts,device=self.joint.xyz.device))
        view=camera.to_torch(device=self.joint.xyz.device)
        self.settings=GaussianRasterizationSettings(image_height=camera.height,image_width=camera.width,
            tanfovx=math.tan(view.FoVx/2),tanfovy=math.tan(view.FoVy/2),
            bg=torch.zeros(3,device=self.joint.xyz.device),scale_modifier=1.,
            viewmatrix=view.world_view_transform,projmatrix=view.full_proj_transform,
            sh_degree=0,campos=view.camera_center,prefiltered=False,debug=False)
        self.rasterizer=GaussianRasterizer(raster_settings=self.settings)

    def render_joint(self,start):
        args=self.joint.raster_args();colors=self.torch.zeros_like(self.joint.colors)
        for slot in range(start,min(start+3,len(self.states))):
            colors[self.slots==slot,slot-start]=1.
        args['colors_precomp']=colors
        return self.rasterizer(**args)[0][:min(3,len(self.states)-start)]

    def render_solo(self,slot):
        args=self.states[slot].raster_args();colors=self.torch.zeros_like(self.states[slot].colors)
        colors[:,0]=1.;args['colors_precomp']=colors
        return self.rasterizer(**args)[0][0]


class VisibilityAnalyzer:
    def __init__(self):
        self.owner=threading.get_ident()

    def analyze(self,objects,camera,timestamp,*,epsilon=1e-6,diagnostics=False,revision=''):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Visibility analysis must run on GPU owner')
        if not math.isfinite(timestamp) or isinstance(epsilon,bool) or not math.isfinite(epsilon) or epsilon<0:
            raise ValueError('Invalid visibility timestamp/epsilon')
        counts=tuple(len(s.xyz) for s in objects.values())
        metadata=dict(backend='streamstgs_cuda_contribution',revision=revision,
            area_units='alpha_weighted_pixels',joint_passes=0,solo_passes=0,
            exact_fast_contribution=False)
        maps={} if diagnostics else None
        if not sum(counts):
            return VisibilityResult(timestamp,(camera.width,camera.height),
                {k:_object_visibility(k,0.,0.,epsilon) for k in objects},metadata,maps)
        import torch
        start=time.monotonic()
        # Tensor references are held through reductions. No opacity reconstruction
        # occurs here: caller provides the frozen states for this same camera.
        with torch.no_grad():
            scene=TensorContributionScene(objects,camera)
            areas=torch.zeros((len(objects),2),dtype=torch.float64,device=scene.joint.xyz.device)
            for i in range(0,len(objects),3):
                image=scene.render_joint(i)
                areas[i:i+len(image),0]=image.sum(dim=(1,2),dtype=torch.float64)
                if maps is not None:
                    for j,key in enumerate(scene.object_ids[i:i+3]):maps[key]={'joint':image[j].cpu().numpy()}
                metadata['joint_passes']+=1
            for i,key in enumerate(scene.object_ids):
                if counts[i]:
                    image=scene.render_solo(i);areas[i,1]=image.sum(dtype=torch.float64)
                    metadata['solo_passes']+=1
                    if maps is not None:maps[key]['solo']=image.cpu().numpy()
            values=areas.cpu().tolist()
        metadata['duration_s']=time.monotonic()-start
        return VisibilityResult(timestamp,(camera.width,camera.height),
            {key:_object_visibility(key,*values[i],epsilon) for i,key in enumerate(objects)},metadata,maps)
