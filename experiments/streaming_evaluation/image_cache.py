"""Lossless native joint-image storage and bounded CPU compression pipelining."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.nava_mse.metrics import MetricStats
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def write_image(path,rgb,alpha,record):
    started=time.perf_counter()
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,rgb=rgb,alpha=alpha)
    os.replace(temporary,path)
    record={**record,'arrays_sha256':file_digest(path),
        'image_format':'NPZ_DEFLATE_lossless_original_floating_dtype',
        'arrays':{name:dict(dtype=a.dtype.str,shape=list(a.shape)) for name,a in (('rgb',rgb),('alpha',alpha))}}
    atomic_json(path.with_suffix('.json'),record)
    return dict(bytes=path.stat().st_size+path.with_suffix('.json').stat().st_size,
                write_wall_seconds=time.perf_counter()-started)


class ImageWriter:
    """One native GPU caller; at most a bounded number of immutable CPU copies."""
    def __init__(self,workers=1,max_pending=2):
        require(type(workers) is int and 1<=workers<=2,'Choose one or two CPU image writers')
        require(type(max_pending) is int and 1<=max_pending<=4,'Bound pending images to at most four')
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='joint-image-writer')
        self.pending={};self.max_pending=max_pending;self.peak_pending=0

    def submit(self,key,path,rgb,alpha,record):
        require(key not in self.pending,'Duplicate pending image')
        require(len(self.pending)<self.max_pending,'Drain the bounded image queue before submission')
        # Do not let mutable renderer buffers or later decoder calls change queued pixels.
        rgb,alpha=np.array(rgb,copy=True,order='C'),np.array(alpha,copy=True,order='C')
        require(rgb.dtype.kind=='f' and alpha.dtype.kind=='f','Store original floating images')
        rgb.setflags(write=False);alpha.setflags(write=False)
        self.pending[key]=self.executor.submit(write_image,Path(path),rgb,alpha,record)
        self.peak_pending=max(self.peak_pending,len(self.pending))

    def finish(self,key):
        if key not in self.pending:
            return None
        future=self.pending.pop(key)
        return future.result()

    def close(self):
        self.executor.shutdown(wait=True)


def image_metrics(root,key,identity,stats,reference_key,*,role):
    """Keep numerical image-quality data beside, rather than inside, the NPZ."""
    path=Path(root)/'cache/images'/f'{key}.metrics.json'
    require(digest(identity)==key,'Image metrics must identify the exact native render')
    value=read_json(path) if path.exists() else dict(schema='streaming-joint-image-metrics.v1',
        image_key=key,image_file=f'{key}.npz',rendering_identity=identity,
        scene_geometry_sha256=digest(identity['instances']),roles=[],measurements={})
    require(value['rendering_identity']==identity,'Image sidecar identity mismatch')
    measurement=dict(reference_image_key=reference_key,mask_mode='full_viewport',
        computed_from_actual_joint_RGB=True,rgb_sample_count=stats.sample_count,**stats.to_dict())
    previous=value['measurements'].get(reference_key)
    if previous is not None:
        require(previous==measurement,'Conflicting quality values for one exact image/reference pair')
    value['measurements'][reference_key]=measurement
    value['roles']=sorted(set(value['roles'])|{role})
    if not path.exists() or read_json(path)!=value:
        atomic_json(path,value)


def backfill(root):
    """Validate and annotate existing images without rendering or altering pixels."""
    root=Path(root)
    metrics={}
    for p in (root/'cache/quality').glob('*.json'):
        row=read_json(p)
        require(row['payload_sha256']==digest(row['payload']),'Corrupt scalar-quality cache')
        payload=row['payload'];provenance=payload['provenance']
        metrics[provenance['candidate_frame_id'],provenance['reference_frame_id']]=MetricStats.from_dict(payload['metrics'])
    count=0;bytes_checked=0;verified=set()
    def validate(key):
        p=root/'cache/images'/f'{key}.npz';marker=read_json(p.with_suffix('.json'))
        require(digest(marker['identity'])==key,'Invalid native image key')
        if key not in verified:
            require(file_digest(p)==marker['arrays_sha256'],'Corrupt lossless image')
            verified.add(key)
        return p,marker
    for p in sorted((root/'cache/images').glob('*.json')):
        if len(p.stem)!=64:
            continue
        key=p.stem;array_path,row=validate(key)
        identity=row['identity'];reference_identity={**identity,'allocation':{iid:20 for iid in identity['allocation']}}
        reference_key=digest(reference_identity);reference_path,_=validate(reference_key)
        with np.load(array_path,allow_pickle=False) as arrays:
            rgb=arrays['rgb']
            require(rgb.dtype.kind=='f' and rgb.ndim==3 and rgb.shape[2]==3,'Invalid native RGB image')
            sample_pixels=rgb.shape[0]*rgb.shape[1]
            if key==reference_key:
                stats=MetricStats(0.,sample_pixels,normalization='clip')
            elif (key,reference_key) in metrics:
                stats=metrics[key,reference_key]
                with np.load(reference_path,allow_pickle=False) as reference:
                    delta=np.asarray(rgb,np.float64)-np.asarray(reference['rgb'],np.float64)
                require(math.isclose(stats.sse,float(np.sum(delta*delta,dtype=np.float64)),rel_tol=1e-12,abs_tol=1e-9),
                        'Saved scalar SSE differs from the actual lossless images')
            else:
                with np.load(reference_path,allow_pickle=False) as reference:
                    delta=np.asarray(rgb,np.float64)-np.asarray(reference['rgb'],np.float64)
                stats=MetricStats(float(np.sum(delta*delta,dtype=np.float64)),sample_pixels,normalization='clip')
        require(stats.foreground_pixels==sample_pixels,'Saved quality/image sample count mismatch')
        image_metrics(root,key,identity,stats,reference_key,
                      role='joint_CRF20_reference' if key==reference_key else 'selected_joint_reconstruction')
        if key==reference_key and (key,key) in metrics:
            image_metrics(root,key,identity,stats,reference_key,role='selected_joint_reconstruction')
        count+=1;bytes_checked+=array_path.stat().st_size
    result=dict(validated_images=count,lossless_image_bytes=bytes_checked,
        old_pixel_payloads_rewritten=0,metric_records_reused=len(metrics))
    atomic_json(root/'image_cache_backfill.json',result)
    return result
