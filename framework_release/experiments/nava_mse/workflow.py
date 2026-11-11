"""Resumable offline stages; GPU construction is confined to explicit render stages."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
from .common import PROJECT, canonical, digest, read_json, sha_file, write_json
from .storage import ArtifactStore, save_preview


def run_root(frozen):
    return PROJECT / frozen['config']['output_dir']


def packages(frozen):
    c, root = frozen['config'], run_root(frozen)
    return {(obj, c['gop'], crf): root / 'assets' / obj / f'gop_{c["gop"]:03d}/encoded/crf_{crf}'
            for obj in c['objects'] for crf in c['crfs']}


def prepare_assets(frozen):
    """Task 2 only: reuse the repository's safe, hash-verified archive importer."""
    sys.path.insert(0, str(PROJECT / 'src_final'))
    from multiobj3dgs.content.streamstgs.archive import extract_verified
    root, c = run_root(frozen), frozen['config']
    missing = [str(PROJECT / c['archive_root'] / s['archive']) for s in frozen['inputs']['sources'].values()
               if not (PROJECT / c['archive_root'] / s['archive']).is_file()]
    if missing:
        raise ValueError('Required encoded archives unavailable: ' + ', '.join(missing))
    for obj in c['objects']:
        source = frozen['inputs']['sources'][obj]
        destination = root / 'assets' / obj
        marker = root / 'assets' / (obj + '.verified.json')
        if marker.is_file():
            saved = read_json(marker)
            if saved['archive_sha256'] != source['archive_sha256']:
                raise ValueError('Prepared source archive differs from frozen identity')
            continue
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f'Incomplete asset extraction: remove only this staging directory and retry: {destination}')
        archive = PROJECT / c['archive_root'] / source['archive']
        if sha_file(archive) != source['archive_sha256']:
            raise ValueError(f'Encoded archive identity mismatch: {archive}')
        manifest, audit = extract_verified(archive, destination)
        if manifest['object'] != obj or sha_file(destination / 'stream_package_manifest.json') != source['package_manifest_sha256']:
            raise ValueError('Extracted package identity mismatch')
        write_json(marker, dict(archive_sha256=audit.sha256, package_sha256=source['package_manifest_sha256']))


def make_renderer(frozen):
    from .rendering import StreamSTGSRenderer
    c = frozen['config']
    for path in packages(frozen).values():
        if not path.is_dir():
            raise ValueError(f'Missing encoded package: {path}; run prepare-assets after supplying required archives')
    return StreamSTGSRenderer(packages(frozen), metadata_root=PROJECT / c['metadata_root'],
        max_resident_models=c['max_resident_models'], bounds_source=PROJECT / c['bounds_source'])


def camera_for(scene):
    sys.path.insert(0, str(PROJECT / 'src_final'))
    from multiobj3dgs.shared.geometry import Camera
    row = scene['camera']
    return Camera(row['width'], row['height'], np.asarray(row['world_to_camera']), np.asarray(row['K']), row['near'], row['far'])


def image_job(scene, timestamp, allocation, *, object_id=None):
    return dict(kind='object' if object_id else 'joint', camera_id=scene['camera_id'],
                timestamp_s=timestamp, allocation=allocation, object_id=object_id)


def render_cached(frozen, renderer, store, scene, timestamp, allocation, *, object_id=None):
    from .rendering import validate_render_job
    job = image_job(scene, timestamp, allocation, object_id=object_id)
    instances = [i for i in scene['instances'] if object_id is None or i['object_id'] == object_id]
    existing = store.load(job)
    if existing is not None:
        validate_render_job(existing['metadata'], job, frozen['config']['gop'],
                            expected_camera=scene['camera'], expected_instances=instances)
        validate_frozen_image(frozen, existing)
        return existing
    result = renderer.render(camera_for(scene), instances, timestamp, frozen['config']['gop'], allocation)
    validate_render_job(result.metadata, job, frozen['config']['gop'],
                        expected_camera=scene['camera'], expected_instances=instances)
    validate_frozen_image(frozen, {'metadata': result.metadata, 'rgb': result.rgb,
                                  'raw_rgb': result.raw_rgb, 'alpha': result.alpha})
    image = store.save(job, result)
    if frozen['config']['save_png']:
        path, _ = store.paths(job)
        save_preview(path.with_suffix('.png'), image['rgb'])
    return image


def load_image(frozen, store, scene, timestamp, allocation, *, object_id=None):
    """Read a committed render only after checking its exact requested context."""
    from .rendering import validate_render_job
    job = image_job(scene, timestamp, allocation, object_id=object_id)
    image = store.load(job)
    if image is None:
        return None
    instances = [i for i in scene['instances'] if object_id is None or i['object_id'] == object_id]
    validate_render_job(image['metadata'], job, frozen['config']['gop'], expected_camera=scene['camera'],
                        expected_instances=instances)
    validate_frozen_image(frozen, image)
    return image


def validate_frozen_image(frozen, image):
    """Bind actual decoded frame and physical package to authoritative inputs."""
    metadata = image['metadata']
    if (metadata['normalization'] != 'clip' or metadata['data_range'] != 1
        or metadata['layout'] != 'HWC' or metadata['background'] != frozen['config']['background']):
        raise ValueError('Rendered RGB normalization/background differs from frozen protocol')
    if 'rgb' in image:
        shape = (metadata['camera']['height'], metadata['camera']['width'], 3)
        if (image['rgb'].shape != shape or image['raw_rgb'].shape != shape
            or image['alpha'].shape != shape[:2]):
            raise ValueError('Rendered array dimensions differ from frozen camera')
    for row in metadata['objects'].values():
        obj, gop, crf = row['object_id'], row['gop'], row['crf']
        expected = frozen['inputs']['representations'][f'{obj}/{gop}/{crf}']
        source = frozen['inputs']['sources'][obj]
        frame, = [f for f in expected['samples'] if f['seconds'] == row['timestamp_s']]
        if (row['frame_index'] != frame['frame'] or row['model_time'] != frame['model_time']
            or row['manifest_sha256'] != expected['manifest_sha256']
            or row['reconstruction_sha256'] != expected['components']['reconstruction.json']['sha256']
            or row['cameras_sha256'] != expected['components']['cameras.json']['sha256']
            or row['source_identity_sha256'] != source['package_manifest_sha256']):
            raise ValueError('Rendered source frame or physical representation differs from frozen inputs')


def validate_geometry(frozen, value):
    if (value['freeze_sha256'] != frozen['freeze_sha256']
        or value.get('content_sha256') != digest({k:v for k,v in value.items() if k != 'content_sha256'})):
        raise ValueError('Geometry signature mismatch')


def ensure_geometry(frozen, renderer):
    from .geometry import aabb_area
    root, c = run_root(frozen), frozen['config']
    path = root / 'geometry.json'
    if path.exists():
        value = read_json(path)
        validate_geometry(frozen, value)
        return value
    bound_records = {obj: renderer.full_gop_bounds(obj, c['gop']) for obj in c['objects']}
    bounds = {obj: record['bounds'] for obj, record in bound_records.items()}
    areas = {str(scene['camera_id']): [float(aabb_area(bounds[i['object_id']], camera_for(scene),
                 np.asarray(i['object_to_world']), method=c['area_method'])) for i in scene['instances']]
             for scene in frozen['scenes']}
    value = dict(freeze_sha256=frozen['freeze_sha256'], bounds={k: np.asarray(v).tolist() for k, v in bounds.items()},
                 bound_provenance=bound_records,
                 projected_aabb_pixels=areas, definition='fixed CRF20 full-GOP Gaussian-mean AABB; paper projected camera-facing triangle area')
    value['content_sha256'] = digest(value)
    write_json(path, value)
    return value


def render_objects(frozen, *, renderer=None, limit=None):
    root, c = run_root(frozen), frozen['config']
    store = ArtifactStore(root, frozen['freeze_sha256'])
    own = renderer is None
    renderer = make_renderer(frozen) if own else renderer
    count = 0
    try:
        ensure_geometry(frozen, renderer)
        for scene in frozen['scenes']:
            for timestamp in c['timestamps_s']:
                for obj in c['objects']:
                    for crf in c['crfs']:
                        allocation = {obj: crf}
                        job = image_job(scene, timestamp, allocation, object_id=obj)
                        if load_image(frozen, store, scene, timestamp, allocation, object_id=obj) is not None:
                            continue
                        if limit is not None and count >= limit:
                            return dict(new_renders=count, complete=False)
                        render_cached(frozen, renderer, store, scene, timestamp, allocation, object_id=obj)
                        count += 1
        return dict(new_renders=count, complete=True)
    finally:
        if own:
            renderer.close()


def profile_objects(frozen):
    from .rendering import validate_render_alignment, validate_render_job
    from .geometry import relative_camera
    from .metrics import foreground_metrics, rgb_range_statistics, reference_foreground_mask
    from .profiles import DistortionSample, build_profiles
    c, root = frozen['config'], run_root(frozen)
    store = ArtifactStore(root, frozen['freeze_sha256'])
    geometry = read_json(root / 'geometry.json')
    validate_geometry(frozen, geometry)
    samples, diagnostics = [], []
    for scene in frozen['scenes']:
        for timestamp in c['timestamps_s']:
            for i, instance in enumerate(scene['instances']):
                obj = instance['object_id']
                def image(crf):
                    job = image_job(scene, timestamp, {obj: crf}, object_id=obj)
                    value = store.load(job)
                    if value is None:
                        raise ValueError(f'Missing object render: camera={scene["camera_id"]}/{timestamp}/{obj}/CRF{crf}')
                    validate_render_job(value['metadata'], job, c['gop'], expected_camera=scene['camera'],
                                        expected_instances=[instance])
                    validate_frozen_image(frozen, value)
                    return value
                ref = image(20)
                local = relative_camera(camera_for(scene), np.asarray(instance['object_to_world']))
                source = frozen['inputs']['representations'][f'{obj}/{c["gop"]}/20']
                frame, = [s for s in source['samples'] if s['seconds'] == timestamp]
                area = geometry['projected_aabb_pixels'][str(scene['camera_id'])][i]
                mask = reference_foreground_mask(ref['alpha'], ref['rgb'].shape[:2], alpha_threshold=c['alpha_threshold'])
                base_provenance = dict(config_sha256=frozen['config_sha256'], freeze_sha256=frozen['freeze_sha256'],
                    comparison_signature=digest(dict(scene=scene, timestamp=timestamp, object_id=obj)),
                    source_identity=frozen['inputs']['sources'][obj]['archive_sha256'], object_transform=instance['object_to_world'],
                    camera_intrinsics=scene['camera']['K'], source_camera_conditioning=instance['source_camera'],
                    reference_identity=ref['provenance'], reference_mask_sha256=digest(mask.astype(np.uint8).tolist()),
                    background=c['background'], projected_aabb_area_pixels=area, alpha_threshold=c['alpha_threshold'],
                    reference_rgb_range=rgb_range_statistics(ref['raw_rgb']),
                    reference_foreground_rgb_range=rgb_range_statistics(ref['raw_rgb'][mask].reshape(-1, 1, 3)))
                frame_metrics = {}
                for crf in c['crfs']:
                    candidate = image(crf)
                    validate_render_alignment(ref['metadata'], candidate['metadata'])
                    metric = foreground_metrics(ref['rgb'], candidate['rgb'], ref['alpha'],
                        alpha_threshold=c['alpha_threshold'], normalization='clip')
                    frame_metrics[crf] = metric
                    provenance = dict(base_provenance, candidate_identity=candidate['provenance'],
                        candidate_rgb_range=rgb_range_statistics(candidate['raw_rgb']),
                        candidate_foreground_rgb_range=rgb_range_statistics(candidate['raw_rgb'][mask].reshape(-1, 1, 3)))
                    samples.append(DistortionSample(obj, c['gop'], str(scene['camera_id']),
                        tuple(local.world_to_camera.reshape(-1)), timestamp, frame['frame'], crf, metric,
                        projected_scale=area/(c['width']*c['height']), camera_relative_position=tuple(local.center),
                        camera_relative_orientation=tuple(local.world_to_camera[:3, :3].reshape(-1)),
                        normalized_media_time=(timestamp-c['gop']*c['gop_duration_s'])/c['gop_duration_s'], provenance=provenance))
                diagnostics.extend(dict(object_id=obj, camera_id=scene['camera_id'], timestamp_s=timestamp, crf=crf,
                    **metric.to_dict(), distortion_reduction_mse=frame_metrics[40].mse-metric.mse,
                    projected_aabb_area_pixels=area) for crf, metric in frame_metrics.items())
    average, exact = build_profiles(samples)
    write_json(root / 'profiles/samples.json', dict(freeze_sha256=frozen['freeze_sha256'], dataset_sha256=exact.dataset_sha256,
                                                   samples=[s.to_dict() for s in samples]))
    write_json(root / 'profiles/average.json', average.to_dict())
    write_json(root / 'profiles/exact.json', exact.to_dict())
    write_json(root / 'profiles/diagnostics.json', diagnostics)
    return dict(samples=len(samples), dataset_sha256=exact.dataset_sha256)


def load_samples(frozen):
    from .profiles import DistortionSample, dataset_fingerprint
    saved = read_json(run_root(frozen) / 'profiles/samples.json')
    if saved['freeze_sha256'] != frozen['freeze_sha256']:
        raise ValueError('Profile belongs to a different frozen input set')
    samples = [DistortionSample.from_dict(s) for s in saved['samples']]
    if dataset_fingerprint(samples) != saved['dataset_sha256']:
        raise ValueError('Distortion dataset fingerprint mismatch')
    return samples


def allocate_all(frozen):
    """Policy decisions are saved before independent joint rendering/oracle stages."""
    from .policies import PolicyInputs, POLICIES, allocate
    from .profiles import build_profiles
    c, root = frozen['config'], run_root(frozen)
    samples = load_samples(frozen)
    average, exact = build_profiles(samples)
    geometry = read_json(root / 'geometry.json')
    validate_geometry(frozen, geometry)
    decisions = []
    for scene in frozen['scenes']:
        area = tuple(geometry['projected_aabb_pixels'][str(scene['camera_id'])])
        exact_mse = []
        for timestamp in c['timestamps_s']:
            rows = []
            for instance in scene['instances']:
                from .geometry import relative_camera
                obj = instance['object_id']
                local = relative_camera(camera_for(scene), np.asarray(instance['object_to_world']))
                scale = area[c['objects'].index(obj)]/(c['width']*c['height'])
                rows.append(tuple(exact.lookup(object_id=obj, gop=c['gop'], camera_id=str(scene['camera_id']),
                    camera_pose=local.world_to_camera, timestamp=timestamp, crf=q, projected_scale=scale).require().mse for q in c['crfs']))
            exact_mse.append(tuple(rows))
        for budget in frozen['budgets']:
            inputs = PolicyInputs(tuple(c['objects']), tuple(c['crfs']), tuple(tuple(r) for r in frozen['inputs']['costs_bytes']),
                budget['bytes'], (area,) * len(c['timestamps_s']), (1.,) * len(c['timestamps_s']),
                average_mse=tuple(tuple(average.lookup(obj, c['gop'], q).require().mse for q in c['crfs']) for obj in c['objects']),
                exact_mse=tuple(exact_mse), nava_psnr=tuple(tuple(r) for r in frozen['inputs']['nava_psnr']), gop=c['gop'])
            for policy in POLICIES:
                decision = allocate(policy, inputs)
                if decision.status != 'optimal':
                    raise ValueError(f'Policy cannot complete: {policy}/{decision.status}')
                decisions.append(dict(camera_id=scene['camera_id'], budget=budget, **decision.to_dict()))
    payload = dict(freeze_sha256=frozen['freeze_sha256'], dataset_sha256=exact.dataset_sha256,
                   decisions=decisions, decision_sha256=digest(decisions))
    path = root / 'allocations/decisions.json'
    if path.exists() and canonical(read_json(path)) != canonical(payload):
        raise ValueError('Existing decisions changed; use an isolated new run')
    write_json(path, payload)
    return dict(decisions=len(decisions), decision_sha256=payload['decision_sha256'])


def load_decisions(frozen):
    from .profiles import dataset_fingerprint
    saved = read_json(run_root(frozen) / 'allocations/decisions.json')
    if saved['freeze_sha256'] != frozen['freeze_sha256'] or digest(saved['decisions']) != saved['decision_sha256']:
        raise ValueError('Allocation input signature mismatch')
    if saved['dataset_sha256'] != dataset_fingerprint(load_samples(frozen)):
        raise ValueError('Allocation dataset differs from current profile dataset')
    c = frozen['config']
    for d in saved['decisions']:
        if set(d['allocation']) != set(c['objects']):
            raise ValueError('Allocation object mismatch')
        cost = sum(frozen['inputs']['costs_bytes'][i][c['crfs'].index(d['allocation'][obj])] for i, obj in enumerate(c['objects']))
        if cost != d['solver']['cost_bytes'] or cost > d['budget']['bytes']:
            raise ValueError('Policy physical budget inconsistency')
    return saved


def evaluate_all(frozen, *, renderer=None, limit=None):
    from .evaluation import enumerate_allocations, joint_quality, exact_joint_oracle, oracle_gap
    from .metrics import pool_metrics
    c, root = frozen['config'], run_root(frozen)
    selected = load_decisions(frozen)  # Selection is irrevocably bound before oracle evaluation.
    store = ArtifactStore(root, frozen['freeze_sha256'])
    own = renderer is None
    renderer = make_renderer(frozen) if own else renderer
    count, rows, all_scores = 0, [], []
    max_budget = max(b['bytes'] for b in frozen['budgets'])
    candidates = [r for r in enumerate_allocations(c['objects'], c['crfs'], frozen['inputs']['costs_bytes']) if r['cost_bytes'] <= max_budget]
    try:
        for scene in frozen['scenes']:
            pooled, frames = {}, {}
            reference_allocation = {obj: 20 for obj in c['objects']}
            allocations = [r['allocation'] for r in candidates]
            if reference_allocation not in allocations:
                allocations.append(reference_allocation)
            for allocation in allocations:
                metrics = []
                for timestamp in c['timestamps_s']:
                    for a in (reference_allocation, allocation):
                        job = image_job(scene, timestamp, a)
                        if load_image(frozen, store, scene, timestamp, a) is None:
                            if limit is not None and count >= limit:
                                return dict(new_renders=count, complete=False)
                            render_cached(frozen, renderer, store, scene, timestamp, a)
                            count += 1
                    ref = load_image(frozen, store, scene, timestamp, reference_allocation)
                    image = load_image(frozen, store, scene, timestamp, allocation)
                    metric = joint_quality(ref, image, alpha_threshold=c['alpha_threshold'])
                    metrics.append(metric)
                    frame = dict(camera_id=scene['camera_id'], timestamp_s=timestamp, allocation=allocation, metrics=metric.to_dict(),
                                 reference=ref['provenance'], candidate=image['provenance'],
                                 render_metadata=image['metadata'])
                    frames.setdefault(digest(allocation), []).append(frame)
                pooled[digest(allocation)] = pool_metrics(metrics)
                all_scores.append(dict(camera_id=scene['camera_id'], allocation=allocation, metrics=pooled[digest(allocation)].to_dict()))
            for budget in frozen['budgets']:
                oracle = exact_joint_oracle(c['objects'], c['crfs'], frozen['inputs']['costs_bytes'], budget['bytes'],
                                           lambda allocation: pooled[digest(allocation)])
                rows.append(dict(camera_id=scene['camera_id'], budget=budget, **oracle.to_dict()))
                for d in selected['decisions']:
                    if d['camera_id'] != scene['camera_id'] or d['budget'] != budget:
                        continue
                    allocation = d['allocation']
                    metric = pooled[digest(allocation)]
                    gaps = oracle_gap(metric, oracle)
                    gaps['allocation_agrees_with_oracle'] = allocation == oracle.allocation
                    rows.append(dict(camera_id=scene['camera_id'], budget=budget, policy=d['policy'], allocation=allocation,
                        cost_bytes=d['solver']['cost_bytes'], metrics=metric.to_dict(), **gaps,
                        quality_information=d['quality_information'], frames=frames[digest(allocation)]))
                    export_diagnostics(frozen, store, scene, allocation, oracle.allocation, budget, d['policy'])
        write_json(root / 'evaluation/results.json', dict(freeze_sha256=frozen['freeze_sha256'],
            decision_sha256=selected['decision_sha256'], results=rows, candidate_scores=all_scores,
            interpretation='Exact view profiles are in-sample. Joint SSE oracle is evaluation-only.'))
        # Verify immutable decisions were not changed by the downstream evaluator.
        if load_decisions(frozen) != selected:
            raise ValueError('Policy decisions changed during evaluation')
        return dict(new_renders=count, complete=True, result_rows=len(rows))
    finally:
        if own:
            renderer.close()


def export_diagnostics(frozen, store, scene, allocation, oracle_allocation, budget, policy):
    c = frozen['config']
    folder = run_root(frozen) / f'diagnostics/camera_{scene["camera_id"]}/{budget["name"]}/{policy}'
    rows = []
    for timestamp in c['timestamps_s']:
        ref = load_image(frozen, store, scene, timestamp, {obj: 20 for obj in c['objects']})
        candidate = load_image(frozen, store, scene, timestamp, allocation)
        error = np.abs(candidate['rgb'].astype(np.float64)-ref['rgb'].astype(np.float64))
        from .metrics import rgb_range_statistics
        rows.append(dict(timestamp_s=timestamp, reference=ref['provenance'], candidate=candidate['provenance'],
            reference_raw_range=rgb_range_statistics(ref['raw_rgb']), candidate_raw_range=rgb_range_statistics(candidate['raw_rgb'])))
        folder.mkdir(parents=True, exist_ok=True)
        # Raw scalar absolute-error heatmap has fixed scale [0,1], no per-image rescaling.
        np.save(folder / f't{timestamp:g}_absolute_error.npy', error.mean(axis=2), allow_pickle=False)
        if c['save_png']:
            save_preview(folder / f't{timestamp:g}_reference.png', ref['rgb'])
            save_preview(folder / f't{timestamp:g}_selected.png', candidate['rgb'])
            heat = error.mean(axis=2)
            save_preview(folder / f't{timestamp:g}_error.png', np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=2))
    write_json(folder / 'diagnostics.json', dict(allocation=allocation, oracle_allocation=oracle_allocation,
        disagrees_with_oracle=allocation != oracle_allocation, camera=scene['camera'], instances=scene['instances'],
        mask='fixed all-CRF20 joint alpha', error_scale=[0, 1], frames=rows,
        approximation='Solo foreground MSE times projected AABB ignores joint occlusion and compositing cross terms.'))
