"""Small native equivalence check against saved joint images, plus explicit upgrade."""
import argparse
import fcntl
from pathlib import Path
import shutil

import numpy as np

from experiments.visibility_weighted_sse.common import digest,file_digest,read_json,require
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from .cache_upgrade import register
from .image_cache import backfill
from .runtime_matrix import validate_inputs,load_settings,DEFAULT_CONFIG,completed
from .runtime_native import NativeEngine,ExecutionBudget,commit


def verify_pixels(root,cfg):
    root=Path(root).resolve()
    _,_,packages,_,times,_,_=validate_inputs(cfg)
    old_manifest=read_json(root/'manifest.json');fingerprint=digest(old_manifest)
    parent=root/'matrix/long_line/N05/10s'
    simulation=read_json(parent/'NAVA/simulation.json')
    trace=TraceCameraProvider(read_json(parent/'trace.json'))
    validation=root/'execution_revisions/pixel_validation_v2'
    validation.mkdir(parents=True,exist_ok=True)
    budget=ExecutionBudget(validation,dict(max_wall_seconds=120,max_native_seconds=90,
        max_storage_bytes=1024**3,minimum_free_bytes=5*1024**3))
    engine=NativeEngine(validation,budget,packages,Path(__file__).resolve().parents[2]/cfg['metadata_root'],
        Path(__file__).resolve().parents[2]/cfg['bounds_source'],fingerprint,times,
        max_resident_models=cfg['renderer']['max_resident_models'],execution_revision='joint_cache_v2_validation')
    rows=[]
    source_map={i['instance_id']:i['object_id'] for i in read_json(parent/'configuration.json')['scene']['instances']}
    chosen=None
    for frame in range(179):
        context=simulation['quality_frames'][frame]
        if frame%60==59 or not context['sse']:
            continue
        pairs={(source_map[iid],q) for iid,q in context['selected_quality'].items()}
        pairs.update((oid,'CRF20') for oid in source_map.values())
        if len(pairs)<=4:
            continue
        with np.load(root/'cache/images'/f"{context['provenance']['reference_frame_id']}.npz",allow_pickle=False) as saved:
            if np.count_nonzero(saved['alpha']>.01)>1000:
                chosen=frame
                break
    require(chosen is not None,'Need a nonempty visible mixed-quality pair exercising the old residency bottleneck')
    try:
        for frame in (chosen,chosen+1):
            pose=trace.get_trajectory(str(frame//60))[frame%60]
            context=simulation['quality_frames'][frame]
            allocation={iid:int(q.removeprefix('CRF')) for iid,q in context['selected_quality'].items()}
            for role,qualities,source_key in (
                ('joint_CRF20_reference',{iid:20 for iid in allocation},context['provenance']['reference_frame_id']),
                ('selected_joint_reconstruction',allocation,context['provenance']['candidate_frame_id'])):
                identity=engine.identity(pose,frame,qualities,'joint')
                require(digest(identity)==source_key,'Validation job differs from saved actual input')
                original=read_json(root/'cache/images'/f'{source_key}.json')
                path=root/'cache/images'/f'{source_key}.npz'
                require(file_digest(path)==original['arrays_sha256'],'Original pixels failed checksum')
                result,seconds=engine._render(identity,pose)
                with np.load(path,allow_pickle=False) as saved:
                    require(result.rgb.dtype==saved['rgb'].dtype and result.alpha.dtype==saved['alpha'].dtype,
                            'Native image precision changed')
                    require(np.array_equal(result.rgb,saved['rgb']) and np.array_equal(result.alpha,saved['alpha']),
                            'Updated cache residency changed actual native pixels')
                    foreground_pixels=int(np.count_nonzero(saved['alpha']>.01))
                rows.append(dict(source_frame=frame,role=role,image_key=source_key,
                    source_image_sha256=file_digest(path),pixel_and_alpha_bitwise_equal=True,
                    original_native_wall_seconds=original['native_wall_seconds'],updated_native_wall_seconds=seconds,
                    nonempty_reference_foreground_pixels=foreground_pixels,
                    conditioning='cold first frame / resident models second frame'))
    finally:
        engine.close()
    proof=dict(status='EXACT_PIXELS_PRESERVED',max_resident_models=cfg['renderer']['max_resident_models'],
        original_artifact_fingerprint=fingerprint,checked_joint_images=rows,
        measurement_scope='Observed native GPU pipeline wall time includes CPU decode/reconstruction; not pure kernel time')
    path=root/'execution_revisions/pixel_validation_v2.json'
    commit(path,proof)
    return path


def prepare(root,cfg,proof):
    root=Path(root).resolve()
    validated=[]
    for marker in sorted(root.glob('matrix/*/*/*/*/completion.json')):
        parent=read_json(marker.parent.parent/'configuration.json')
        require(completed(marker.parent,parent['configuration_hash']),'Invalid completed result')
        validated.append(dict(path=str(marker.relative_to(root)),sha256=file_digest(marker),
                              outputs_sha256=read_json(marker)['outputs_sha256']))
    commit(root/'execution_revisions/preserved_results_v2.json',dict(
        completed_runs=len(validated),unchanged_completion_markers=validated))
    # Register old partial network checkpoints only after binding them to frozen configuration.
    for checkpoint in root.glob('matrix/*/*/*/*/network_checkpoint.json'):
        folder=checkpoint.parent
        if (folder/'completion.json').exists():
            continue
        simulation=read_json(checkpoint);frozen=read_json(folder.parent/'configuration.json')
        require(simulation['configuration_hash']==frozen['configuration_hash'] and simulation['method']==folder.name
                and simulation['source_duration_s']==6,'Invalid legacy partial checkpoint')
        commit(folder/'network_checkpoint.sha256.json',dict(configuration_hash=frozen['configuration_hash'],
            method=folder.name,checkpoint_sha256=file_digest(checkpoint),
            performance_sha256=file_digest(folder/'performance.json')))
    result=backfill(root)
    workload_path=root/'workload.json';workload=read_json(workload_path)
    revisions=workload.setdefault('execution_revisions',{})
    if 'original_v1' not in revisions:
        images=[read_json(p) for p in (root/'cache/images').glob('*.json') if len(p.stem)==64]
        visibility=[read_json(p) for p in (root/'cache/visibility').glob('*.json')]
        revisions['original_v1']=dict(image_jobs=len(images),visibility_jobs=len(visibility),
            image_native_wall_seconds=sum(r['native_wall_seconds'] for r in images),
            visibility_native_wall_seconds=sum(r['native_wall_ms']/1000 for r in visibility),
            image_write_wall_seconds=0.,image_write_timing_available=False,image_files_written=len(images),
            image_cache_hits=workload['image_cache_hits'],metric_cache_hits=workload['metric_cache_hits'],peak_pending_images=0)
        commit(workload_path,workload)
    _,_,_,_,_,_,current=validate_inputs(cfg)
    revision=register(root,current,proof)
    return dict(preserved_completed_runs=len(validated),backfill=result,revision=revision['revision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--verify-pixels',action='store_true')
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args();cfg=load_settings(args.config)
    proof=args.root/'execution_revisions/pixel_validation_v2.json'
    if args.verify_pixels:
        proof=verify_pixels(args.root,cfg)
        print(read_json(proof))
    if args.register:
        with (args.root/'writer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(prepare(args.root,cfg,proof))
