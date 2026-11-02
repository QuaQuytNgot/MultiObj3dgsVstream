"""Frozen six-second scenes, with exact recorded-center decoder support."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import math
from types import MappingProxyType

import numpy as np

from experiments.nava_mse.geometry import camera_from_record, validate_source_conditioning
from experiments.visibility_weighted_sse.common import digest, require
from experiments.visibility_weighted_sse.trace import TraceCameraProvider, camera_to_dict
from src_final.multiobj3dgs.shared.geometry import Camera
from .inputs import independent_instance_transfer_profiles


def rotation_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-12:
        axis = np.cross(a, [1., 0., 0.])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + k + k @ k / (1 + c)


def make_scene(kind, count, camera_records, *, seed=238019, width=640, height=480):
    require(kind in ('camera_rotation', 'long_line', 'random_spatial'), 'Unknown scene family')
    require(type(count) is int and count >= 2, 'At least two instances required')
    source_order = list(camera_records)
    base = camera_from_record(camera_records[source_order[0]])
    intrinsic = base.K.copy() * (width / base.width)
    intrinsic[2] = [0., 0., 1.]
    intrinsic[0, 2], intrinsic[1, 2] = (width-1)/2, (height-1)/2
    camera = Camera(width, height, base.world_to_camera, intrinsic, .01, 1000.)
    rng = np.random.Generator(np.random.PCG64(seed))
    if kind == 'long_line':
        positions = [[(i-(count-1)/2)*1.25, 0., 0.] for i in range(count)]
    elif kind == 'camera_rotation':
        positions = [[2.0*math.cos(2*math.pi*i/count), 0.,
                      1.2*math.sin(2*math.pi*i/count)] for i in range(count)]
    else:
        positions = []
        while len(positions) < count:
            candidate = rng.uniform([-3.5, -.9, -1.2], [3.5, .9, 1.2])
            if all(np.linalg.norm(candidate-np.asarray(p)) >= .8 for p in positions):
                positions.append(candidate.tolist())
    instances = []
    for i, position in enumerate(positions):
        oid = source_order[i % len(source_order)]
        record = camera_records[oid]
        source = camera_from_record(record)
        delta = camera.center - position
        scale = float(np.linalg.norm(delta) / np.linalg.norm(source.center))
        rotation = rotation_between(source.center, delta)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = scale*rotation, position
        validate_source_conditioning(camera, matrix, record, mode='recorded_center')
        instances.append(dict(instance_id=f'instance-{i:04d}', object_id=oid,
            source_camera_id=f"{record['split']}:{record['id']}", source_camera=record,
            conditioning_mode='recorded_center', object_to_world=matrix.tolist()))
    result = dict(schema='streaming-evaluation.runtime-scene.v1', kind=kind, object_count=count,
        seed=seed, source_order=source_order, instances=instances, base_camera=camera_to_dict(camera),
        source_duration_s=6., fps=30, positions=positions,
        layout_semantics='distinct origins; scale/rotation preserve exact recorded source-local eye')
    result['scene_id'] = kind + '-' + digest(result)[:20]
    return result


def camera_at(scene, source_frame):
    from experiments.visibility_weighted_sse.trace import camera_from_dict
    base = camera_from_dict(scene['base_camera'])
    phase = 2*math.pi*source_frame/180
    amplitude = {'camera_rotation': 40., 'long_line': 65., 'random_spatial': 48.}[scene['kind']]
    yaw = math.radians(amplitude) * math.sin(phase)
    pitch = math.radians(5.)*math.sin(2*phase) if scene['kind'] == 'random_spatial' else 0.
    ry = np.array([[math.cos(yaw), 0., math.sin(yaw)], [0., 1., 0.],
                   [-math.sin(yaw), 0., math.cos(yaw)]])
    rx = np.array([[1., 0., 0.], [0., math.cos(pitch), -math.sin(pitch)],
                   [0., math.sin(pitch), math.cos(pitch)]])
    rotation = base.world_to_camera[:3, :3] @ ry @ rx
    view = np.eye(4)
    view[:3, :3], view[:3, 3] = rotation, -rotation @ base.center
    return Camera(base.width, base.height, view, base.K, base.near, base.far)


def make_trace(scene, timeline, content_revision):
    gops = []
    for segment in timeline:
        samples = []
        start_tick = round(segment.global_start_s*30)
        for tick in range(start_tick, round(segment.global_end_s*30)):
            frame = tick % 180
            instances = [{**row, 'source_state_id': digest(dict(content_revision=content_revision,
                object_id=row['object_id'], source_frame=frame))} for row in scene['instances']]
            samples.append(dict(timestamp=float(Fraction(tick, 30)),
                camera=camera_to_dict(camera_at(scene, frame)), instances=instances))
        gops.append(dict(gop_id=segment.gop_id, start_s=segment.global_start_s,
                         end_s=segment.global_end_s, samples=samples))
    trace = dict(schema='visibility-weighted-sse.trace.v1',
        trace_id=scene['scene_id'] + '-periodic6s',
        coordinate_convention='opencv_positive_z_column_vectors',
        prediction_assumption='ORACLE_CAMERA_TRACE', predictor_enabled=False, gops=gops)
    return TraceCameraProvider(trace)


def bind_profiles(profiles, metadata, scene, timeline, source_gop_ids=('0', '1', '2')):
    objects, bound_metadata = {}, {}
    for segment in timeline:
        source_gid = source_gop_ids[segment.source_gop_index]
        originals = {row.object_id: row for row in profiles.for_gop(source_gid)}
        for instance in scene['instances']:
            original = originals[instance['object_id']]
            row = replace(original, instance_id=instance['instance_id'], gop_id=segment.gop_id)
            objects[row.instance_id, row.gop_id] = row
            bound_metadata[row.instance_id, row.gop_id] = deepcopy(
                metadata['objects'][original.instance_id, source_gid])
    bound = independent_instance_transfer_profiles(replace(profiles, objects=MappingProxyType(objects)))
    return bound, {**metadata, 'objects': bound_metadata}
