"""Deterministic layouts preserving each decoder's recorded local eye position.

Layout targets are requests, never measurements. In particular, a fixed source
camera domain cannot make arbitrarily many human-sized objects disjoint within
the same field of view. The geometric diagnostics deliberately expose that
limitation instead of shrinking the calibrated source views or moving the eye.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from experiments.nava_mse.geometry import (
    camera_from_record, camera_metadata, select_camera_record,
    validate_source_conditioning,
)
from src_final.multiobj3dgs.shared.geometry import Camera


def camera_from_metadata(value):
    return Camera(value['width'], value['height'], np.asarray(value['world_to_camera']),
                  np.asarray(value['K']), value['near'], value['far'])


def _rotation(horizontal, vertical):
    h, v = math.radians(horizontal), math.radians(vertical)
    yaw = np.array([[math.cos(h), 0., math.sin(h)], [0., 1., 0.],
                    [-math.sin(h), 0., math.cos(h)]])
    pitch = np.array([[1., 0., 0.], [0., math.cos(v), math.sin(v)],
                      [0., -math.sin(v), math.cos(v)]])
    return yaw @ pitch


def _corners(bounds):
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError('Bounds must be finite ordered 2x3 XYZ')
    return np.asarray(list(itertools.product(*zip(*box))))


def instance_geometry(bounds, camera, matrix):
    corners = _corners(bounds)
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    xyz = world @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    near, far = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    # All native prepared-content corners are in front; keep invalid cases
    # explicit rather than divide by an invalid z or declare them invisible.
    in_front = near > max(camera.near, .2)
    pixel_bounds = None
    if near > 0:
        xy = xyz[:, :2] / xyz[:, 2:]
        uv = xy * np.diag(camera.K)[:2] + camera.K[:2, 2]
        pixel_bounds = [uv.min(axis=0).tolist(), uv.max(axis=0).tolist()]
    overlap = bool(pixel_bounds and pixel_bounds[1][0] > 0 and pixel_bounds[0][0] < camera.width
                   and pixel_bounds[1][1] > 0 and pixel_bounds[0][1] < camera.height)
    fully_inside = bool(in_front and pixel_bounds and min(pixel_bounds[0]) >= 0
                        and pixel_bounds[1][0] <= camera.width and pixel_bounds[1][1] <= camera.height
                        and far < camera.far)
    return dict(world_bounds=[world.min(axis=0).tolist(), world.max(axis=0).tolist()],
                camera_space_corners=xyz.tolist(), camera_depth_interval=[near, far],
                depth_center=float(xyz[:, 2].mean()),
                image_edge_bounds=pixel_bounds, positive_renderer_depth=in_front,
                aabb_intersects_viewport=overlap, aabb_fully_inside_viewport=fully_inside,
                bounds_definition='full-GOP CRF20 decoded-mean AABB; excludes covariance tails')


def _separated(a, b):
    if a is None or b is None:
        return False
    return a[1][0] <= b[0][0] or b[1][0] <= a[0][0] or a[1][1] <= b[0][1] or b[1][1] <= a[0][1]


def _budget_rows(costs, config):
    minimum, maximum = sum(row[-1] for row in costs), sum(row[0] for row in costs)
    if any(row[-1] != min(row) or row[0] != max(row) for row in costs):
        raise ValueError('Prepared CRF40/CRF20 must bound actual physical package costs')
    values = []
    from fractions import Fraction
    for spec in config['budgets']:
        fraction = Fraction(str(spec['fraction']))
        amount = minimum + ((maximum-minimum) * fraction.numerator // fraction.denominator)
        values.append(dict(name=spec['name'], bytes=amount, cost_range_fraction=spec['fraction'],
                           min_bytes=minimum, max_bytes=maximum,
                           mbps=amount*8/(1_000_000*config['gop_duration_s'])))
    return values


def build_scene(config, inputs, bounds, *, n, category, target, camera_id):
    """Build a requested arrangement; alpha measurement later establishes class.

    X_world = C_world + s R (X_source - C_source). Thus the exact source-local
    eye is C_source for every positive scale s. A scale changes physical depth,
    not projected size. R includes camera-space steering, never Rx(pi).
    """
    sources = config['source_objects']
    chosen = [sources[i % len(sources)] for i in range(n)]
    records = {obj: select_camera_record(inputs['sources'][obj]['cameras'], camera_id)
               for obj in set(chosen)}
    source_base = camera_from_record(records[chosen[0]])
    width, height = config['width'], config['height']
    k = source_base.K.copy() * (width/source_base.width)
    k[2] = [0., 0., 1.]
    k[0, 2], k[1, 2] = (width-1)/2, (height-1)/2
    camera = Camera(width, height, source_base.world_to_camera, k, .01, 1000.)
    span_x, span_y, angular_centers = [], [], {}
    for obj in chosen:
        source = camera_from_record(records[obj])
        xyz = _corners(bounds[obj]) @ source.world_to_camera[:3, :3].T + source.world_to_camera[:3, 3]
        if xyz[:, 2].min() <= .2:
            raise ValueError('Prepared source mean bounds reach invalid rasterizer depth')
        angles = np.degrees(np.arctan2(xyz[:, :2], xyz[:, 2:]))
        span_x.append(float(np.ptp(angles[:, 0])))
        span_y.append(float(np.ptp(angles[:, 1])))
        angular_centers[obj] = (angles.min(axis=0)+angles.max(axis=0))/2
    columns = n if n <= 3 else math.ceil(math.sqrt(n))
    rows = math.ceil(n/columns)
    # Targets parameterize angular center spacing only. Box overlap is not the
    # occlusion denominator, and native alpha can differ markedly from boxes.
    gap = 1.08 if category == 'A' else max(.03, 1.-target)
    step_x, step_y = max(span_x)*gap, max(span_y)*gap
    half_x = math.degrees(math.atan((width/2)/k[0, 0]))
    half_y = math.degrees(math.atan((height/2)/k[1, 1]))
    step_x = min(step_x, 1.7*half_x/max(columns-1, 1))
    step_y = min(step_y, 1.7*half_y/max(rows-1, 1))
    instances = []
    source_positions = {obj: i for i, obj in enumerate(sources)}
    for i, obj in enumerate(chosen):
        horizontal = (i % columns - (columns-1)/2)*step_x - angular_centers[obj][0]
        vertical = (i // columns - (rows-1)/2)*step_y - angular_centers[obj][1]
        source = camera_from_record(records[obj])
        # Separated depth layers are an approximation for transmittance, not a
        # per-pixel Gaussian ordering guarantee. AABB intervals are exported.
        scale = 1. + .35*i if category in ('B', 'C', 'D') else 1.
        rotation = camera.world_to_camera[:3, :3].T @ _rotation(horizontal, vertical) @ source.world_to_camera[:3, :3]
        matrix = np.eye(4)
        matrix[:3, :3] = scale*rotation
        matrix[:3, 3] = camera.center - scale*rotation @ source.center
        diagnostic = validate_source_conditioning(camera, matrix, records[obj], mode='recorded_center')
        geometry = instance_geometry(bounds[obj], camera, matrix)
        instances.append(dict(instance_id=f'{obj}_{i:02d}', object_id=obj,
            duplicate_source_instance=chosen.index(obj) != i,
            source_instance_ordinal=sum(old == obj for old in chosen[:i]),
            object_to_world=matrix.tolist(), source_camera=records[obj],
            conditioning_mode='recorded_center', scale=scale,
            camera_steering_degrees=[horizontal, vertical],
            costs_bytes=list(inputs['costs_bytes'][source_positions[obj]]),
            source_local_camera_diagnostics=diagnostic, geometry=geometry))
    pairs = []
    for a, b in itertools.combinations(instances, 2):
        da, db = a['geometry']['camera_depth_interval'], b['geometry']['camera_depth_interval']
        pairs.append(dict(instances=[a['instance_id'], b['instance_id']],
            projected_bounds_separated=_separated(a['geometry']['image_edge_bounds'], b['geometry']['image_edge_bounds']),
            disjoint_depth_intervals=da[1] < db[0] or db[1] < da[0]))
    separated = all(p['projected_bounds_separated'] for p in pairs)
    all_intersect = all(i['geometry']['aabb_intersects_viewport'] for i in instances)
    limitations = []
    if category == 'A' and not separated:
        limitations.append('Supported fixed camera/FOV does not certify a disjoint Scene A layout at this N; exclude from primary no-occlusion class until measured alpha establishes separation.')
    if not all(i['geometry']['aabb_fully_inside_viewport'] for i in instances):
        limitations.append('Some decoded-mean bounds are viewport-clipped; assess actual reference-alpha coverage before scientific classification.')
    if any(not p['disjoint_depth_intervals'] for p in pairs):
        limitations.append('Mean-bound depth intervals overlap: object-level proxy ordering is approximate and cannot prove per-pixel Gaussian depth order.')
    if category == 'D':
        limitations.append('Native alpha/transmittance interaction is unverified until actual pixel measurements; no artificial transparency is applied.')
    identifier = f'N{n}_{category}_target{int(round(target*100)):02d}_cam{camera_id}'
    return dict(scene_id=identifier, n=n, category=category, target_occlusion=target,
                target_is_measurement=False, camera_id=camera_id, camera=camera_metadata(camera),
                instances=instances, evaluation_mode='candidate_set',
                budgets=_budget_rows([i['costs_bytes'] for i in instances], config),
                validity=dict(camera_supported=True, source_conditioning='exact recorded local center; declared steering/crop',
                    primary_quality_eligible=all_intersect and all(i['geometry']['positive_renderer_depth'] for i in instances),
                    measured_category='UNMEASURED', primary_scene_A_confirmed=False,
                    aabb_separation_sufficient=separated, all_mean_bounds_intersect_viewport=all_intersect,
                    per_pair_geometry=pairs, limitations=limitations),
                occlusion_denominator='per object/time: sum solo proxy alpha; loss = 1 - sum(T_i * alpha_i)/sum(alpha_i); zero-alpha undefined',
                depth_order_approximation='native Gaussian per-pixel order through RGB-basis contribution passes; contribution/solo-alpha gives effective object-pixel attenuation; this is an object-level approximation when Gaussians interleave')


def build_cases(config, inputs, bounds):
    cases = []
    targets = config['occlusion_targets']
    for n in config['object_counts']:
        for category in config['categories']:
            selected = ([t for t in targets if t == 0] if category == 'A' else
                        [t for t in targets if 0 < t <= .5] if category == 'B' else
                        [t for t in targets if t > .5] if category == 'C' else targets)
            for target, camera_id in itertools.product(selected, config['camera_ids']):
                cases.append(build_scene(config, inputs, bounds, n=n, category=category,
                                         target=target, camera_id=camera_id))
    return cases
