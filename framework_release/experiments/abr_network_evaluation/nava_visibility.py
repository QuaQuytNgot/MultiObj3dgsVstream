"""NAVA III-E, equations 3/4: front-facing projected AABB triangle areas.

Six homogeneous clip planes retain viewport-intersecting triangles. This is
an explicit clipping improvement over discarding a triangle with any exterior
vertex. No alpha, transmittance, occlusion or viewpoint prediction is used.
Inputs must bind GOP bounds to verified metadata; missing data is an error.
"""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
from pathlib import Path

from .visibility import VisibilityProvider


class MissingVisibilityInput(ValueError):
    pass


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def model_matrix(transform):
    """Experiment convention: translation, Euler XYZ radians, positive scale."""
    tx, ty, tz, rx, ry, rz, scale = transform
    x, y, z = (math.cos(a) for a in (rx, ry, rz))
    sx, sy, sz = (math.sin(a) for a in (rx, ry, rz))
    rotation = matmul([[z,-sz,0],[sz,z,0],[0,0,1]],
                     matmul([[y,0,sy],[0,1,0],[-sy,0,y]], [[1,0,0],[0,x,-sx],[0,sx,x]]))
    return [[*(scale*v for v in row), t] for row, t in zip(rotation, (tx,ty,tz))] + [[0,0,0,1]]


def _clip(poly):
    # Sutherland-Hodgman in clip space, before perspective division.
    for axis, sign in itertools.product(range(3), (-1, 1)):
        result = []
        if not poly:
            return []
        previous = poly[-1]
        dp = previous[3] + sign*previous[axis]
        for current in poly:
            dc = current[3] + sign*current[axis]
            if (dp >= 0) != (dc >= 0):
                t = dp/(dp-dc)
                result.append(tuple(a+t*(b-a) for a,b in zip(previous,current)))
            if dc >= 0:
                result.append(current)
            previous, dp = current, dc
        poly = result
    return poly


def projected_area(bounds, mvp, width=512, height=512):
    """Pixel area, outward winding, OpenGL clip convention (-w <= xyz <= w)."""
    if (len(bounds) != 2 or any(len(r) != 3 for r in bounds)
            or any(not math.isfinite(v) for row in bounds for v in row)
            or any(a >= b for a,b in zip(*bounds))):
        raise ValueError('Bounds must be finite, strictly ordered [min_xyz,max_xyz]')
    if len(mvp) != 4 or any(len(r) != 4 for r in mvp) or any(not math.isfinite(v) for r in mvp for v in r):
        raise ValueError('MVP must be a finite 4x4 matrix')
    if width <= 0 or height <= 0:
        raise ValueError('Positive viewport dimensions required')
    vertices = list(itertools.product(*zip(*bounds)))
    clips = [tuple(sum(row[j]*(*v,1)[j] for j in range(4)) for row in mvp) for v in vertices]
    area = 0.
    for axis, side in itertools.product(range(3), (0,1)):
        others = [d for d in range(3) if d != axis]
        corners = []
        for a,b in ((0,0),(1,0),(1,1),(0,1)):
            bits = [0,0,0]; bits[axis]=side; bits[others[0]]=a; bits[others[1]]=b
            corners.append(bits[0]*4+bits[1]*2+bits[2])
        # Cross of edges along the other two axes has parity (-1)^axis.
        if (-1 if axis%2 else 1) != (1 if side else -1):
            corners.reverse()
        for indices in ((0,1,2),(0,2,3)):
            polygon = _clip([clips[corners[j]] for j in indices])
            if len(polygon) < 3 or any(v[3] <= 1e-12 for v in polygon):
                continue
            xy = [(v[0]/v[3],v[1]/v[3]) for v in polygon]
            signed = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(xy,xy[1:]+xy[:1]))/2
            area += max(0.,signed)*width*height/4
    return area


class AABBVisibility(VisibilityProvider):
    """Precompute pose-hold observations from provenance-bound metadata.

    JSON schema: bounds[content][gop] = {bounds, source, source_sha256};
    samples = [{time_s, view_projection}]. Source paths are relative to JSON.
    Bounds are in the same source-local coordinates as instance transforms.
    Camera matrices use OpenGL clip coordinates. Every used GOP is required.
    """
    def __init__(self, path, instances, catalog):
        if path is None or not Path(path).is_file():
            raise MissingVisibilityInput('No verified per-GOP AABB + matching camera observations for the measured StreamSTGS catalog')
        path = Path(path)
        data = json.loads(path.read_text())
        if data.get('clip_convention') != 'opengl' or data.get('time_basis') != 'media':
            raise ValueError('NAVA inputs require OpenGL clip matrices and media observation timestamps')
        for instance in instances:
            for g in range(len(catalog.contents[instance.content_id].gops)):
                record = data['bounds'][instance.content_id][str(g)]
                source = (path.parent/record['source']).resolve()
                if hashlib.sha256(source.read_bytes()).hexdigest() != record['source_sha256']:
                    raise ValueError(f'Unverified bounds metadata: {source}')
                # Bind the numerical bounds themselves to a field in the source.
                value = json.loads(source.read_text())
                for field in record['source_field']:
                    value = value[field]
                if value != record['bounds']:
                    raise ValueError('Bounds disagree with verified source metadata field')
        self.mode, self.time_basis, self.seed = 'recorded', 'media', 0
        self.samples = []
        for observation in data['samples']:
            time = self._finite(observation['time_s'], 'time')
            areas = {}
            for i in instances:
                content = catalog.contents[i.content_id]
                g = math.floor((time+i.offset_s+1e-9)/content.gops[0][0].duration_s)%len(content.gops)
                bounds = data['bounds'][i.content_id][str(g)]['bounds']
                areas[i.id] = projected_area(bounds,matmul(observation['view_projection'],model_matrix(i.transform)),
                                             *data.get('resolution',[512,512]))
            self.samples.append((time,areas))
        self.samples.sort()
        self._times = [t for t,_ in self.samples]
        if not self._times or self._times[0] != 0 or len(set(self._times)) != len(self._times):
            raise ValueError('NAVA camera trace needs distinct timestamps starting at zero')
        self.input_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
