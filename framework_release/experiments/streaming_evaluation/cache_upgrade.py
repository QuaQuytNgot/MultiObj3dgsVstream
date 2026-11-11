"""Explicit, pinned compatibility record for the requested cache-only revision."""
from copy import deepcopy
from pathlib import Path

from experiments.visibility_weighted_sse.common import digest,file_digest,read_json,require
from .image_cache import atomic_json

REVISION='joint_cache_v2'
ALLOWED_CODE={f'experiments/streaming_evaluation/{name}.py' for name in
              ('runtime_native','runtime_matrix','runtime_results','image_cache','cache_upgrade')}
OPERATIONAL_RENDERER_KEYS=('max_resident_models','image_write_workers','max_pending_image_writes')


def compatible(old,new):
    a,b=deepcopy(old),deepcopy(new)
    for value in (a,b):
        for key in OPERATIONAL_RENDERER_KEYS:
            value['config']['renderer'].pop(key,None)
    old_code,new_code=a.pop('implementation_sha256'),b.pop('implementation_sha256')
    require(a==b,'Scientific inputs changed; old image/results cannot be adopted')
    require(set(old_code)<=set(new_code),'Previously pinned implementation disappeared')
    changed={p for p in new_code if old_code.get(p)!=new_code[p]}
    require(changed<=ALLOWED_CODE,f'Unreviewed adaptation/network/scene implementation change: {changed-ALLOWED_CODE}')
    return sorted(changed)


def resolve(root,current):
    root=Path(root);path=root/'manifest.json'
    if not path.exists():
        atomic_json(path,current)
        return digest(current),'original_v1'
    original=read_json(path)
    if original==current:
        return digest(original),'original_v1'
    compatible(original,current)
    revision_path=root/'execution_revisions'/f'{REVISION}.json'
    require(revision_path.exists(),'Register and validate the explicit cache revision before resuming')
    revision=read_json(revision_path)
    require(revision['original_artifact_fingerprint']==digest(original) and revision['updated_identity']==current,
            'Cache revision does not match this exact implementation/input set')
    proof=root/revision['pixel_validation_file']
    require(file_digest(proof)==revision['pixel_validation_sha256'],'Cache pixel-validation proof changed')
    require(read_json(proof)['status']=='EXACT_PIXELS_PRESERVED','Pixel validation did not pass')
    # Keep the physical render namespace and old completed-result hashes intact.
    return digest(original),REVISION


def register(root,current,proof):
    root,proof=Path(root).resolve(),Path(proof).resolve()
    old=read_json(root/'manifest.json')
    changes=compatible(old,current)
    require(read_json(proof)['status']=='EXACT_PIXELS_PRESERVED','Actual native pixel-equivalence proof required')
    revision=dict(schema='streaming-cache-execution-revision.v1',revision=REVISION,
        original_artifact_fingerprint=digest(old),original_manifest_file_sha256=file_digest(root/'manifest.json'),
        updated_identity=current,reviewed_changed_code=changes,
        pixel_validation_file=str(proof.relative_to(root)),pixel_validation_sha256=file_digest(proof),
        compatibility_scope='Only model residency, image IO, cache sidecars and checkpoint/report handling; '
          'source/scene/camera/quality protocol, utilities, solver, network and pixel kernels unchanged',
        authorization='User requested lossless joint-image caches, safe throughput optimization and preservation of validated results')
    destination=root/'execution_revisions'/f'{REVISION}.json'
    if destination.exists():
        require(read_json(destination)==revision,'Registered revision already exists with different contents')
    else:
        atomic_json(destination,revision)
    return revision
