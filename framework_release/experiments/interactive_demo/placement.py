"""Startup-only layouts adapted from streaming_evaluation/runtime_scenes.py.

The experiment's local PCG64, sampling domain, rejection placement and angular
ordering are retained. This native demo uses true circles and complete measured
all-GOP AABBs, rather than the experiment's center-only clearance and calibrated
source-camera similarity transforms. Existing explicit scenes are preserved.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from multiobj3dgs.client.scene import transform_matrix
from multiobj3dgs.shared.metadata import SceneInstance

PADDING = .15
MIN_ORIGIN_SPACING = .8
KINDS = ('random_spatial', 'camera_rotation', 'long_line')


def _path(value):
    from .abr.config import path
    return path(value)


def _bank(config):
    location = _path(config['mse_profiles'])
    return location.parent if location.is_file() else location


def normalize_scene_config(config):
    """Validate startup inputs before publication; explicit scene owns contents.

    This copies the config, never mutates a scene file, and resolves all content
    names from a supplied scene before archive discovery. Generated-scene count,
    contents and placement flags have no effect on an explicit scene's instances.
    """
    result = dict(config)
    if result.get('scene'):
        scene = result['scene']
        if not isinstance(scene, dict):
            scene = json.loads(_path(scene).read_text())
        if not isinstance(scene, dict) or not isinstance(scene.get('instances'), list) or not scene['instances']:
            raise ValueError('scene requires a nonempty instances list')
        duration = scene.get('duration')
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('scene duration must be finite and positive')
        instances = tuple(SceneInstance(**row) for row in scene['instances'])
        if len({row.id for row in instances}) != len(instances):
            raise ValueError('scene instance IDs must be unique')
        result['scene'] = scene
        result['contents'] = list(dict.fromkeys(row.content_id for row in instances))
    # Unknown/unprofiled content fails before creating a publication or origin.
    bounds = json.loads((_bank(result) / 'bounds.json').read_text())
    known = {key.rsplit('/', 1)[0] for key in bounds}
    missing = set(result['contents']) - known
    if missing:
        raise ValueError('No compatible MSE/bounds bank for content: ' + ', '.join(sorted(missing)))
    return result


def load_envelopes(catalog, config):
    """Union every measured source GOP, preserving outliers and source binding.

    Bounds describe CRF20 decoded Gaussian means, not covariance support. The
    clearance is therefore a conservative mean-envelope policy with an explicit
    fixed padding, not a mathematical guarantee for infinite Gaussian tails.
    """
    document = json.loads((_bank(config) / 'bounds.json').read_text())
    metadata = _path(config['quality_metadata'])
    if metadata.is_file():
        metadata = metadata.parent
    if not (metadata / 'quality-index.json').is_file() and (metadata / 'streamstgs_8i_quality_metadata').is_dir():
        metadata = metadata / 'streamstgs_8i_quality_metadata'
    envelopes, sources = {}, {}
    for name in dict.fromkeys(instance.content_id for instance in catalog.instances):
        content = catalog.contents[name]
        reference = next((rep for rep in content.representations.values() if rep.crf == 20), None)
        if reference is None:
            raise ValueError(f'Placement requires the measured CRF20 reference: {name}')
        rows, provenance = [], []
        for segment in reference.segments:
            key = f'{name}/{segment.gop_index}'
            if key not in document:
                raise ValueError(f'No all-source placement bounds for {key}')
            row = document[key]
            bounds = np.asarray(row['bounds'], dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
                raise ValueError(f'Invalid placement bounds: {key}')
            manifest = metadata / name / 'content' / f'gop_{segment.gop_index:03d}_crf_20' / 'manifest.json'
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if row.get('reference_crf') != 20 or row.get('manifest_sha256') != digest or row.get('frames') != len(segment.frames):
                raise ValueError(f'Placement bounds/source manifest mismatch: {key}')
            rows.append(bounds)
            provenance.append(dict(gop_index=segment.gop_index, manifest_sha256=digest,
                                   reconstruction_sha256=row.get('reconstruction_sha256'),
                                   source_sha256=row.get('source_sha256')))
        envelopes[name] = np.stack([np.min([row[0] for row in rows], axis=0),
                                    np.max([row[1] for row in rows], axis=0)])
        sources[name] = provenance
    return envelopes, sources


def corners(bounds):
    return np.asarray(list(product(*zip(bounds[0], bounds[1]))), dtype=float)


def world_bounds(instance, content, envelope, padding=PADDING):
    matrix = transform_matrix(instance, content.pivot)
    transformed = corners(envelope) @ matrix[:3, :3].T + matrix[:3, 3]
    return np.stack([transformed.min(axis=0)-padding, transformed.max(axis=0)+padding])


def overlaps(a, b):
    """Strict padded AABB intersection; touching faces are permitted."""
    return bool(np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-10))


def generate_positions(kind, envelopes, *, seed=238019, padding=PADDING, attempts=256, expansions=10):
    """Return deterministic translations plus bounded-placement diagnostics."""
    if kind not in KINDS:
        raise ValueError('Unknown placement mode: ' + str(kind))
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    count = len(envelopes)
    if not count:
        raise ValueError('At least one object required')
    bounds = [np.asarray(item, dtype=float) + np.asarray([[-padding]*3, [padding]*3]) for item in envelopes]
    if any(row.shape != (2, 3) or not np.isfinite(row).all() or not np.all(row[1] > row[0]) for row in bounds):
        raise ValueError('Invalid local envelopes')
    if count == 1:
        return [(0., 0., 0.)], dict(mode=kind, seed=seed, expansions=0)
    if kind == 'long_line':
        return [((i-(count-1)/2)*1.8, 0., 0.) for i in range(count)], dict(mode=kind, seed=seed, spacing=1.8, expansions=0)
    if kind == 'camera_rotation':
        radius = 2.
        for expansion in range(expansions+1):
            positions = np.asarray([[radius*math.cos(2*math.pi*i/count), 0.,
                                     radius*math.sin(2*math.pi*i/count)] for i in range(count)])
            placed = [row+point for row, point in zip(bounds, positions)]
            clear = not any(overlaps(a, b) for i, a in enumerate(placed) for b in placed[i+1:])
            spaced = all(np.linalg.norm(a-b) >= MIN_ORIGIN_SPACING for i, a in enumerate(positions) for b in positions[i+1:])
            if clear and spaced:
                return [tuple(row) for row in positions], dict(mode=kind, seed=seed, radius=radius, expansions=expansion)
            radius *= 1.5
        raise ValueError(f'Circular placement could not clear {count} full padded envelopes after {expansions+1} radii')
    # Copied domain/RNG/rejection structure from validated runtime_scenes.make_scene.
    # Expand X/Z deterministically rather than allow an unbounded rejection loop.
    rng = np.random.Generator(np.random.PCG64(seed))
    for expansion in range(expansions+1):
        factor = 1.5**expansion
        extent = np.asarray([3.5*factor, .9, 1.2*factor])
        positions, placed = [], []
        for envelope in bounds:
            for _ in range(attempts):
                candidate = rng.uniform(-extent, extent)
                trial = envelope + candidate
                if all(not overlaps(trial, previous) for previous in placed) and all(
                        np.linalg.norm(candidate-np.asarray(previous)) >= MIN_ORIGIN_SPACING for previous in positions):
                    positions.append(tuple(candidate))
                    placed.append(trial)
                    break
            else:
                break
        if len(positions) == count:
            return positions, dict(mode=kind, seed=seed, expansions=expansion, domain=[(-extent).tolist(), extent.tolist()], attempts_per_object=attempts)
    raise ValueError(f'Random placement could not clear {count} full padded envelopes after {expansions+1} bounded domains')


def build_scene(catalog, config):
    """Apply a fixed explicit/generated instance list to an immutable catalog."""
    if config.get('scene'):
        scene = config['scene']
        if not isinstance(scene, dict):
            scene = json.loads(_path(scene).read_text())
        result = replace(catalog, instances=tuple(SceneInstance(**item) for item in scene['instances']), duration=float(scene['duration']))
        # Do not alter user-provided geometry even when envelopes intersect.
        config['_placement'] = dict(mode='explicit_scene', seed=config['seed'], generated_flags_ignored=True)
        load_envelopes(result, config)
        return result
    names = config['contents']
    count = config.get('objects') or len(names)
    if type(count) is not int or count <= 0:
        raise ValueError('objects must be a positive integer')
    if set(names) - set(catalog.contents):
        raise ValueError('Requested content missing from publication: ' + ', '.join(sorted(set(names)-set(catalog.contents))))
    instances = tuple(SceneInstance(f'{names[i%len(names)]}-{i}', names[i%len(names)],
                     active_duration=catalog.contents[names[i%len(names)]].duration,
                     loop=bool(config.get('loop', False))) for i in range(count))
    preliminary = replace(catalog, instances=instances, duration=max(catalog.contents[row.content_id].duration for row in instances))
    envelopes, _ = load_envelopes(preliminary, config)
    positions, report = generate_positions(config.get('placement', 'random_spatial'),
                          [envelopes[row.content_id] for row in instances], seed=config.get('seed', 238019))
    config['_placement'] = report
    return replace(preliminary, instances=tuple(replace(row, translation=position) for row, position in zip(instances, positions)))


def placement_report(catalog, config):
    envelopes, sources = load_envelopes(catalog, config)
    placed = [(row, world_bounds(row, catalog.contents[row.content_id], envelopes[row.content_id])) for row in catalog.instances]
    conflicts = [[a.id, b.id] for i, (a, aa) in enumerate(placed) for b, bb in placed[i+1:] if overlaps(aa, bb)]
    union = np.stack([np.min([row[0] for _, row in placed], axis=0), np.max([row[1] for _, row in placed], axis=0)])
    return {**config.get('_placement', dict(mode=config.get('placement', 'random_spatial'), seed=config.get('seed', 238019))),
            'padding_world_units':PADDING,
            'minimum_origin_spacing':MIN_ORIGIN_SPACING,
            'envelope_policy':'union of every source GOP CRF20 decoded Gaussian mean AABB; fixed padding; no outlier trimming; not covariance support',
            'local_bounds':{key:row.tolist() for key, row in envelopes.items()},
            'world_bounds':{row.id:bounds.tolist() for row, bounds in placed},
            'scene_bounds':union.tolist(), 'overlap_pairs':conflicts,
            'source_binding':sources,
            'outlier_contents':[name for name, bounds in envelopes.items() if np.max(bounds[1]-bounds[0])>5.],
            'positions':{row.id:list(row.translation) for row in catalog.instances}}


def frame_scene_camera(controller, catalog, config):
    """Frame complete padded all-GOP envelopes using the calibrated intrinsics."""
    envelopes, _ = load_envelopes(catalog, config)
    vertices = np.concatenate([corners(world_bounds(row, catalog.contents[row.content_id], envelopes[row.content_id])) for row in catalog.instances])
    controller.target = (vertices.min(axis=0)+vertices.max(axis=0))/2
    controller._invalidate()
    camera = controller.camera()
    offsets = (vertices-controller.target) @ camera.world_to_camera[:3, :3].T
    tanx = max(1e-6, min(camera.K[0, 2]+.5, camera.width-camera.K[0, 2]-.5)/camera.K[0, 0])
    tany = max(1e-6, min(camera.K[1, 2]+.5, camera.height-camera.K[1, 2]-.5)/camera.K[1, 1])
    required = np.max(np.maximum(np.abs(offsets[:, 0])/tanx, np.abs(offsets[:, 1])/tany)-offsets[:, 2])
    controller.radius = max(controller.radius, float(required)*1.05, float(camera.near-offsets[:, 2].min())+.05)
    controller._invalidate()
    framed = controller.camera()
    depths = offsets[:, 2]+controller.radius
    if depths.max() >= framed.far:
        raise ValueError(f'Complete placement framing exceeds existing native camera far plane ({framed.far}); reduce N or use a smaller explicit scene')
    return dict(radius=controller.radius, target=controller.target.tolist(), envelope_vertices=len(vertices), max_depth=float(depths.max()))
