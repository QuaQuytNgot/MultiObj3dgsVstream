"""Non-learned correction-coding baselines for existing, verified Q0/Q1 checkpoints.

The standalone decoder reads only payload files (including transmitted Q0), never
checkpoints. Training and rendering implementations are not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.content_preparation.paths import UPSTREAM_ROOT, configure_upstream_imports
from tools.content_preparation.compatibility import store_point_cloud
from tools.check_progressive_composability import (
    digest, dump, groups, matrix, read_vertices, verify_lineage, write_vertices,
)

ZSTD = shutil.which('zstd')
ZSTD_FLAGS = ['-6', '--long=27', '-T1', '-q', '--no-progress', '-c']


def bit_rows_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return np.all(np.ascontiguousarray(a).view('u1').reshape(len(a), -1) ==
                  np.ascontiguousarray(b).view('u1').reshape(len(b), -1), axis=1)


def static_rows(arrays):
    return np.logical_and.reduce([bit_rows_equal(arrays[0], a) for a in arrays])


def rotation(q):
    # Official PLY uses wxyz; SciPy's default is xyzw. Normalization is intrinsic.
    return Rotation.from_quat(np.asarray(q, dtype='f8')[:, [1, 2, 3, 0]])


def residual(a, b, attribute):
    if attribute == 'rotation':
        result = (rotation(b) * rotation(a).inv()).as_rotvec()
        result[bit_rows_equal(a, b)] = 0  # exact identity, no numerical noise
        return result
    return b.astype('f8') - a.astype('f8')


def apply_residual(a, r, attribute):
    if attribute != 'rotation':
        return (a.astype('f8') + r).astype('<f4')
    result = a.copy()
    active = np.any(r != 0, axis=1)
    if active.any():
        q = (Rotation.from_rotvec(r[active]) * rotation(a[active])).as_quat()
        result[active] = q[:, [3, 0, 1, 2]].astype('<f4')
    return result


def pack_signed(values, bits):
    flat = (np.asarray(values, dtype='i8').ravel() & ((1 << bits)-1)).astype('u4')
    planes = (flat[:, None] >> np.arange(bits, dtype='u4')) & 1
    return np.packbits(planes.astype('u1').ravel(), bitorder='little').tobytes()


def unpack_signed(data, count, bits):
    planes = np.unpackbits(np.frombuffer(data, dtype='u1'), bitorder='little')[:count*bits]
    planes = planes.reshape(count, bits).astype('i8')
    result = (planes * (1 << np.arange(bits, dtype='i8'))).sum(1)
    result[result >= (1 << (bits-1))] -= 1 << bits
    return result


def encode_values(values, precision, maximum):
    a = np.asarray(values)
    meta = dict(shape=list(a.shape), precision=precision)
    if precision in ['f32', 'f16']:
        encoded = a.astype('<f4' if precision == 'f32' else '<f2')
        assert np.isfinite(encoded).all()
        return encoded.tobytes(), meta
    bits = int(precision[1:]); limit = (1 << (bits-1))-1
    step = float(maximum/limit) if maximum else 1.0
    quantized = np.rint(a/step).astype('i8')
    assert np.all(np.abs(quantized) <= limit), 'Quantizer overflow; no silent clipping'
    meta.update(bits=bits, step=step, rounding='nearest-even', packing='little-bit-order signed twos-complement')
    return pack_signed(quantized, bits), meta


def decode_values(data, meta):
    shape = meta['shape']; count = math.prod(shape)
    precision = meta['precision']
    if precision in ['f32', 'f16']:
        a = np.frombuffer(data, dtype='<f4' if precision == 'f32' else '<f2').astype('f8')
    else:
        a = unpack_signed(data, count, meta['bits']).astype('f8') * meta['step']
    assert len(a) == count
    return a.reshape(shape)


def _require_zstd():
    if ZSTD is None:
        raise RuntimeError("The zstd executable is required for this codec experiment; install it and put it on PATH.")
    return ZSTD


def compress(data):
    return subprocess.run([_require_zstd(), *ZSTD_FLAGS], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def decompress(data):
    return subprocess.run([_require_zstd(), '-d', '--long=27', '-q', '-c'], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


class Packet:
    """Physical streams grouped by owner/category/attribute across the whole GoF.

    Index and value streams are separate even for B0. Reinterleaving them exactly
    recovers the legacy records; raw value+index byte counts are unchanged.
    Each stream has its own actual zstd frame. No compression-size estimation.
    """
    def __init__(self, path, pool):
        self.path, self.pool = path, pool
        self.path.mkdir(parents=True)
        self.pool.mkdir(parents=True, exist_ok=True)
        self.streams, self.labels = {}, {}

    def append(self, owner, category, attribute, data, meta):
        name = f'{owner}.{category}.{attribute}.bin'
        stream = self.streams.setdefault(name, bytearray())
        ref = dict(meta, file=name, offset=len(stream), nbytes=len(data))
        stream.extend(data)
        self.labels[name] = dict(owner=owner, category=category, attribute=attribute)
        return ref

    def values(self, owner, category, attribute, values, precision='f32', maximum=0):
        data, meta = encode_values(values, precision, maximum)
        return self.append(owner, category, attribute, data, meta)

    def indices(self, attribute, selected, force_ids=False):
        n = len(selected); indices = np.flatnonzero(selected).astype('<u4')
        if len(indices) == n and not force_ids:
            return dict(kind='dense', count=n)
        if force_ids or indices.nbytes < (n+7)//8:
            data, kind = indices.tobytes(), 'ids'
        else:
            data, kind = np.packbits(selected, bitorder='little').tobytes(), 'mask'
        ref = self.append('correction', 'index_mask', attribute, data, {})
        return dict(kind=kind, count=len(indices), total_rows=n, ref=ref)

    def finish(self, documents):
        for owner, document in documents.items():
            name = owner+'.json'
            self.streams[name] = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()+b'\n'
            self.labels[name] = dict(owner=owner, category='metadata', attribute='header')
        ledger = []
        for name, source in self.streams.items():
            data = bytes(source); key = hashlib.sha256(data).hexdigest()
            raw_pool, zip_pool = self.pool/key, self.pool/(key+'.zst')
            if not raw_pool.exists():
                encoded = compress(data)
                assert decompress(encoded) == data
                raw_pool.write_bytes(data); zip_pool.write_bytes(encoded)
            assert raw_pool.stat().st_size == len(data)
            os.link(raw_pool, self.path/name)
            os.link(zip_pool, self.path/(name+'.zst'))
            ledger.append(dict(file=name, **self.labels[name], raw_bytes=len(data),
                               compressed_bytes=zip_pool.stat().st_size, sha256=key))
        return ledger


class Decoder:
    """No source model, manifest or target checkpoint parameter is accepted."""
    def __init__(self, folder, compressed=True):
        self.folder, self.compressed, self.cache = folder, compressed, {}
        self.base = json.loads(self.read('base.json'))
        self.new = json.loads(self.read('new.json'))
        self.correction = json.loads(self.read('correction.json'))

    def read(self, name):
        assert Path(name).name == name, 'Payload references must stay in their directory'
        if name not in self.cache:
            data = (self.folder/(name+'.zst' if self.compressed else name)).read_bytes()
            self.cache[name] = decompress(data) if self.compressed else data
        return self.cache[name]

    def blob(self, ref):
        data = self.read(ref['file']); start = ref['offset']; end = start+ref['nbytes']
        assert 0 <= start <= end <= len(data)
        return data[start:end]

    def values(self, ref):
        return decode_values(self.blob(ref), ref)

    def indices(self, index, n):
        if index['kind'] == 'dense':
            ids = np.arange(n)
        elif index['kind'] == 'complement':
            chosen = self.indices(index['of'], n)
            mask = np.ones(n, bool); mask[chosen] = False; ids = np.flatnonzero(mask)
        elif index['kind'] == 'ids':
            ids = np.frombuffer(self.blob(index['ref']), dtype='<u4')
        elif index['kind'] == 'mask':
            mask = np.unpackbits(np.frombuffer(self.blob(index['ref']), dtype='u1'), bitorder='little')[:n]
            ids = np.flatnonzero(mask)
        else:
            raise ValueError(index['kind'])
        assert len(ids) == index['count'] and np.all(ids < n)
        assert len(ids) < 2 or np.all(np.diff(ids.astype('i8')) > 0)
        return ids

    @staticmethod
    def select(schedule, frame):
        return schedule['once'] if 'once' in schedule else schedule['frames'][frame]

    def frame(self, frame):
        n, k = self.base['shared_count'], self.base['new_count']
        result = np.zeros(n+k, dtype=[(name, '<f4') for name in self.base['ply_fields']])
        schema = self.base['groups']
        for owner, section in [('base', slice(0,n)), ('new', slice(n,n+k))]:
            document = self.base if owner == 'base' else self.new
            for attribute, schedule in document['arrays'].items():
                values = self.values(self.select(schedule, frame))
                for col, field in enumerate(schema[attribute]):
                    result[field][section] = values[:, col]
        for attribute, schedule in self.correction['arrays'].items():
            if 'row_split' in schedule:
                entries = [schedule['static'], schedule['dynamic']['frames'][frame]]
            else:
                entries = [self.select(schedule, frame)]
            for entry in entries:
                if entry is None:
                    continue
                ids = self.indices(entry['indices'], n)
                if 'copy_base_frame' in entry:
                    source = self.select(self.base['arrays'][attribute], entry['copy_base_frame'])
                    values = self.values(source)[ids]
                else:
                    values = self.values(entry['values'])
                assert len(values) == len(ids)
                fields = schema[attribute]
                if self.correction['mode'] == 'residual':
                    values = apply_residual(matrix(result[:n], fields)[ids], values, attribute)
                for col, field in enumerate(fields):
                    result[field][ids] = values[:, col]
        return result


def collect(manifest):
    n = verify_lineage(manifest)
    b = [read_vertices(r['Q0']) for r in manifest['frames']]
    q = [read_vertices(r['Q1']) for r in manifest['frames']]
    schema = groups(b[0].dtype.names)
    assert all(x.dtype == b[0].dtype and len(x) == len(q[0]) for x in q)
    seq = {owner: {a: [matrix(x, fields) for x in arrays] for a,fields in schema.items()}
           for owner,arrays in [('base',b),('shared',[x[:n] for x in q]),('new',[x[n:] for x in q])]}
    seq['residual'] = {a: [residual(x,y,a) for x,y in zip(seq['base'][a],seq['shared'][a])] for a in schema}
    lifetimes = {}
    for owner, attributes in seq.items():
        lifetimes[owner] = {}
        for a, arrays in attributes.items():
            fixed = static_rows(arrays)
            lifetimes[owner][a] = dict(rows=len(fixed), static_rows=int(fixed.sum()),
                static_fraction=float(fixed.mean()), whole_group_static=bool(fixed.all()),
                changed_rows_per_transition=[int((~bit_rows_equal(x,y)).sum()) for x,y in zip(arrays,arrays[1:])])
    return b, q, schema, seq, lifetimes


def configurations():
    result = [dict(name='B0_raw_replacement',mode='replacement',layout='ids',precision='f32',split=False),
              dict(name='B1_dense_replacement',mode='replacement',layout='dense',precision='f32',split=False)]
    for precision in ['f32','f16','q16','q12','q8']:
        result.append(dict(name='B2_dense_residual_'+precision,mode='residual',layout='dense',precision=precision,split=False))
    thresholds = {'mild':[.001,math.radians(.5),.001], 'medium':[.005,math.radians(2),.01],
                  'coarse':[.02,math.radians(5),.03], 'aggressive':[.05,math.radians(15),.1]}
    for label,threshold in thresholds.items():
        result.append(dict(name='B3_sparse_f16_'+label,mode='residual',layout='sparse',precision='f16',split=False,threshold=threshold))
    for layout in ['ids','dense','row_static','row_alias']:
        result.append(dict(name='B4_split_exact_'+layout,mode='replacement',layout=layout,precision='f32',split=True))
    for precision in ['f32','f16','q16','q12','q8']:
        result.append(dict(name='B4_split_dense_'+precision,mode='residual',layout='dense',precision=precision,split=True))
    for precision in ['f16','q12']:
        for label,threshold in thresholds.items():
            result.append(dict(name='B4_split_sparse_'+precision+'_'+label,mode='residual',layout='sparse',precision=precision,split=True,threshold=threshold))
    result.append(dict(name='naive_split',mode='none',layout='dense',precision='f32',split=True))
    return result


def threshold_mask(attribute, base, target, values, threshold):
    if attribute == 'xyz':
        return np.linalg.norm(values,axis=1) > threshold[0]
    if attribute == 'rotation':
        return np.linalg.norm(values,axis=1) > threshold[1]
    if attribute == 'opacity':
        alpha = lambda v: 1/(1+np.exp(-v.astype('f8')))
        return np.abs(alpha(target)-alpha(base)).ravel() > threshold[2]
    return np.any(values != 0, axis=1)


def serialize(out, cfg, manifest, b, q, schema, seq, life):
    packet = Packet(out/'payloads'/cfg['name'], out/'blob_pool')
    documents = {'base':dict(format='correction-coding-v1', frames=[r['frame'] for r in manifest['frames']],
                            shared_count=len(b[0]),new_count=len(q[0])-len(b[0]),
                            ply_fields=list(b[0].dtype.names),groups=schema,arrays={}),
                 'new':dict(arrays={}), 'correction':dict(mode=cfg['mode'],config=cfg,arrays={})}
    for owner in ['base','new']:
        for a, arrays in seq[owner].items():
            fixed = life[owner][a]['whole_group_static']
            category = 'static_values' if fixed else 'dynamic_values'
            refs = [packet.values(owner,category,a,x) for x in (arrays[:1] if cfg['split'] and fixed else arrays)]
            documents[owner]['arrays'][a] = {'once':refs[0]} if cfg['split'] and fixed else {'frames':refs}
    selected_rows = {}
    if cfg['mode'] != 'none':
        for a in schema:
            ba, ta = seq['base'][a], seq['shared'][a]
            changed = [~bit_rows_equal(x,y) for x,y in zip(ba,ta)]
            if not any(c.any() for c in changed):
                continue
            arrays = ta if cfg['mode'] == 'replacement' else seq['residual'][a]
            maximum = max(float(np.abs(v).max()) for v in arrays)
            def entry(values, chosen, category, index=None):
                if not chosen.any():
                    return None
                ids = packet.indices(a,chosen,force_ids=cfg['layout']=='ids') if index is None else index
                ref = packet.values('correction',category,a,values[chosen],cfg['precision'],maximum)
                return dict(indices=ids,values=ref)
            if cfg['layout'] in ['row_static','row_alias']:
                # Empirical target-state lifetime, not residual lifetime. Static
                # overrides are applied each frame, even when Q0 itself moves.
                fixed = static_rows(ta); dynamic = ~fixed
                if cfg['layout']=='row_alias' and fixed.any() and bit_rows_equal(ba[0][fixed],ta[0][fixed]).all():
                    # A byte-exact reference to state already carried by base;
                    # not a prediction and not a target/checkpoint side channel.
                    static_entry = dict(indices=packet.indices(a,fixed),copy_base_frame=0)
                else:
                    static_entry = entry(ta[0],fixed,'static_values')
                if fixed.any():
                    dynamic_index = dict(kind='complement',of=static_entry['indices'],count=int(dynamic.sum()))
                else:
                    dynamic_index = dict(kind='dense',count=len(dynamic))
                dynamic_entries = [entry(v,dynamic,'dynamic_values',dynamic_index) if (c & dynamic).any() else None for v,c in zip(ta,changed)]
                documents['correction']['arrays'][a] = dict(row_split=True,static=static_entry,dynamic={'frames':dynamic_entries})
                selected_rows[a] = [int(fixed.sum())+int(dynamic.sum() if x else 0) for x in dynamic_entries]
                continue
            if cfg['layout'] == 'dense':
                masks = [np.ones(len(x),bool) if c.any() else np.zeros(len(x),bool) for x,c in zip(arrays,changed)]
            elif cfg['layout'] == 'ids':
                masks = changed
            else:
                masks = [threshold_mask(a,x,y,v,cfg['threshold']) for x,y,v in zip(ba,ta,arrays)]
            fixed = static_rows(arrays).all() and all(np.array_equal(masks[0],m) for m in masks)
            category = 'static_values' if fixed else 'dynamic_values'
            entries = [entry(v,m,category) for v,m in zip(arrays[:1] if cfg['split'] and fixed else arrays,
                                                         masks[:1] if cfg['split'] and fixed else masks)]
            documents['correction']['arrays'][a] = {'once':entries[0]} if cfg['split'] and fixed else {'frames':entries}
            selected_rows[a] = [int(m.sum()) for m in masks]
    ledger = packet.finish(documents)
    return dict(config=cfg,streams=ledger,selected_rows=selected_rows)


def accounting(record):
    result = {}
    for prefix,key in [('raw','raw_bytes'),('zstd','compressed_bytes')]:
        total = lambda owner=None, category=None: sum(s[key] for s in record['streams']
                    if (owner is None or s['owner']==owner) and (category is None or s['category']==category))
        result.update({prefix+'_base_bytes':total('base'),prefix+'_new_bytes':total('new'),
                       prefix+'_correction_bytes':total('correction'),prefix+'_total_bytes':total()})
        for owner in ['base','new','correction']:
            for category in ['static_values','dynamic_values','metadata','index_mask']:
                result[f'{prefix}_{owner}_{category}_bytes'] = total(owner,category)
        result[prefix+'_correction_new_ratio'] = total('correction')/total('new')
        assert result[prefix+'_total_bytes'] == sum(result[prefix+'_'+o+'_bytes'] for o in ['base','new','correction'])
    return result


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def pareto(rows, multiple=False):
    def dominates(a,b):
        metrics = [('gt_psnr',1),('gt_ssim',1),('gt_lpips',-1)] if multiple else [('gt_psnr',1)]
        weak = a['zstd_total_bytes'] <= b['zstd_total_bytes'] and all(a[k]*s >= b[k]*s for k,s in metrics)
        strict = a['zstd_total_bytes'] < b['zstd_total_bytes'] or any(a[k]*s > b[k]*s for k,s in metrics)
        return weak and strict
    return {r['variant'] for r in rows if not any(dominates(a,r) for a in rows if a is not r)}


def evaluate(out, manifest, records, q, schema):
    import torch
    configure_upstream_imports()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    torch.set_num_threads(2)
    criterion = LPIPS(net_type='vgg').cuda().eval()
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    background=torch.zeros(3,device='cuda')
    degree=int(round(math.sqrt(len(schema['sh'])/3)-1))
    assert 3*(degree+1)**2 == len(schema['sh'])
    def compare(a,b):
        mse=(a-b).square().mean().item()
        return dict(psnr=float('inf') if mse==0 else -10*math.log10(mse),ssim=ssim(a,b).item(),
                    lpips=criterion(a[None],b[None]).item(),max_abs_pixel=(a-b).abs().max().item())
    per_view, frames, rows=[],[],[]
    references=[]
    with torch.no_grad():
        for i,r in enumerate(manifest['frames']):
            infos=readCamerasFromTransforms(r['eval_source'],'transforms_test.json',False)
            cams=cameraList_from_camInfos(infos,1.,SimpleNamespace(resolution=1,data_device='cuda'))
            model=GaussianModel(degree);model.load_ply(r['Q1'])
            images=[render(cam,model,pipe,background)['render'].clamp(0,1) for cam in cams]
            references.append((cams,images,[compare(image,cam.original_image) for cam,image in zip(cams,images)]))
        del model
        fields=sum(schema.values(),[])
        with tempfile.TemporaryDirectory(prefix='coding_render_',dir=out) as tmp:
            temp=Path(tmp)/'decoded.ply'
            for record in records:
                cfg=record['config']; name=cfg['name']; folder=out/'payloads'/name
                decoder,raw_decoder=Decoder(folder,True),Decoder(folder,False)
                exact=[];local=[]
                for fi,r in enumerate(manifest['frames']):
                    decoded=decoder.frame(fi)
                    assert decoded.tobytes() == raw_decoder.frame(fi).tobytes(), 'zstd/raw decoder mismatch'
                    same = matrix(decoded,fields).tobytes() == matrix(q[fi],fields).tobytes()
                    exact.append(same)
                    if cfg['mode']=='replacement':
                        assert same, f'Exact replacement not bitwise exact: {name}'
                    write_vertices(temp,decoded)
                    model=GaussianModel(degree);model.load_ply(str(temp))
                    cams,official,official_gt=references[fi]
                    local_frame=[]
                    for vi,(cam,ref,ref_gt) in enumerate(zip(cams,official,official_gt)):
                        image=render(cam,model,pipe,background)['render'].clamp(0,1)
                        identical=torch.equal(image,ref)
                        if cfg['mode']=='replacement':
                            assert identical, f'Exact render mismatch: {name}'
                        qmetric=dict(psnr=float('inf'),ssim=1.,lpips=0.,max_abs_pixel=0.) if identical else compare(image,ref)
                        gtmetric=ref_gt if identical else compare(image,cam.original_image)
                        entry=dict(variant=name,frame=r['frame'],view=vi,
                                   **{'q1_'+k:v for k,v in qmetric.items()},**{'gt_'+k:v for k,v in gtmetric.items()},
                                   gt_psnr_loss=ref_gt['psnr']-gtmetric['psnr'],
                                   gt_ssim_loss=ref_gt['ssim']-gtmetric['ssim'],gt_lpips_increase=gtmetric['lpips']-ref_gt['lpips'])
                        local.append(entry);per_view.append(entry);local_frame.append(entry)
                        if vi==0:
                            dest=out/'renders'/name/str(r['frame']);dest.mkdir(parents=True,exist_ok=True)
                            for label,im in [('decoded',image),('official',ref),('gt',cam.original_image)]:
                                Image.fromarray((im.permute(1,2,0).cpu().numpy()*255+.5).astype('u1')).save(dest/(label+'.png'))
                    average={k:float(np.mean([v[k] for v in local_frame])) for k in local_frame[0] if k not in ['variant','frame','view']}
                    frames.append(dict(variant=name,frame=r['frame'],bitwise_state_equal=same,**average))
                    del model
                average={k:float(np.mean([v[k] for v in local])) for k in local[0] if k not in ['variant','frame','view']}
                rows.append(dict(variant=name,mode=cfg['mode'],layout=cfg['layout'],precision=cfg['precision'],
                    static_reuse=cfg['split'],bitwise_state_equal=all(exact),**accounting(record),**average,
                    max_view_gt_psnr_loss=max(v['gt_psnr_loss'] for v in local),
                    p95_view_gt_psnr_loss=float(np.quantile([v['gt_psnr_loss'] for v in local],.95)),
                    global_max_abs_pixel=max(v['q1_max_abs_pixel'] for v in local)))
                print(f"{name}: zstd correction={rows[-1]['zstd_correction_bytes']:,} B; "
                      f"GT PSNR={rows[-1]['gt_psnr']:.6f}; loss={rows[-1]['gt_psnr_loss']:.6f} dB",flush=True)
                dump(out/'partial_summary.json',rows)
                del decoder,raw_decoder
                torch.cuda.empty_cache()
    front=pareto(rows);multi=pareto(rows,True)
    for row in rows:
        row['pareto_psnr']=row['variant'] in front;row['pareto_multi_metric']=row['variant'] in multi
    csv_write(out/'summary.csv',rows);csv_write(out/'per_frame.csv',frames);csv_write(out/'per_view.csv',per_view)
    official_metrics={k:float(np.mean([m[k] for _,_,ms in references for m in ms])) for k in references[0][2][0]}
    return dict(variants=rows,per_frame=frames,official_q1_vs_gt=official_metrics,
                pareto_psnr=sorted(front),pareto_multi_metric=sorted(multi),views=len(references[0][0])*len(references))


def self_test():
    rng=np.random.default_rng(17)
    for bits in [8,12,16]:
        limit=(1 << (bits-1))-1
        x=rng.integers(-limit,limit+1,size=123)
        np.testing.assert_array_equal(unpack_signed(pack_signed(x,bits),len(x),bits),x)
    a=Rotation.random(50,random_state=rng).as_quat()[:,[3,0,1,2]].astype('f4')*2
    b=Rotation.random(50,random_state=rng).as_quat()[:,[3,0,1,2]].astype('f4')*3
    delta=residual(a,b,'rotation');restored=apply_residual(a,delta,'rotation')
    assert np.max((rotation(restored)*rotation(b).inv()).magnitude())<2e-7
    sign=residual(a,-a,'rotation');assert np.max(np.linalg.norm(sign,axis=1))<1e-12
    np.testing.assert_array_equal(apply_residual(a,np.zeros((50,3)),'rotation'),a)
    for p in ['f32','f16','q16','q12','q8']:
        x=rng.uniform(-2,2,(17,3));payload,meta=encode_values(x,p,2)
        y=decode_values(payload,meta)
        if p.startswith('q'): assert np.max(np.abs(x-y))<=meta['step']/2+1e-14
    payload=b'correction baseline\x00'*1234
    assert decompress(compress(payload))==payload
    print('PASS signed packing 8/12/16, SO(3) composition/sign/identity, quantization bounds, zstd roundtrip',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'output/progressive_gap_real/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'output/correction_coding_baseline')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--render-only',action='store_true',help='Re-evaluate existing payloads, without encoding or overwriting them')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    vendor=args.manifest.resolve().parent/'vendor'
    configure_upstream_imports(vendor if vendor.is_dir() else None)
    manifest=json.loads(args.manifest.read_text())
    b,q,schema,seq,life=collect(manifest)
    source_hashes={name:digest(UPSTREAM_ROOT/name) for name in ['train.py','render.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']}
    if args.render_only:
        records=json.loads((out/'payload_index.json').read_text())
    else:
        if (out/'payloads').exists():
            raise FileExistsError('Payloads already exist; use --render-only or a new --output')
        dump(out/'lifetimes.json',life)
        records=[]
        for cfg in configurations():
            record=serialize(out,cfg,manifest,b,q,schema,seq,life);records.append(record)
            stats=accounting(record)
            print(f"Encoded {cfg['name']}: raw total={stats['raw_total_bytes']:,}, zstd total={stats['zstd_total_bytes']:,}",flush=True)
        dump(out/'payload_index.json',records)
        csv_write(out/'streams.csv',[dict(variant=r['config']['name'],**s) for r in records for s in r['streams']])
    result=evaluate(out,manifest,records,q,schema)
    assert all(digest(UPSTREAM_ROOT/name)==h for name,h in source_hashes.items())
    import torch,scipy,diff_gaussian_rasterization
    result.update(manifest=str(args.manifest.resolve()),manifest_sha256=digest(args.manifest),lifetimes=life,
        source_hashes=source_hashes,original_sources_unchanged=True,script_sha256=digest(__file__),
        configuration_count=len(records),compressor=dict(executable=ZSTD,flags=ZSTD_FLAGS,
        version=subprocess.check_output([ZSTD,'--version'],text=True).strip()),
        environment=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
                         gpu=torch.cuda.get_device_name(0),rasterizer=diff_gaussian_rasterization.__file__),
        accounting='All actual payload value/index/JSON bytes and zstd frame headers included. Q0/new are exact float32. Logical byte counts include hardlinked common streams once per candidate. No filesystem overhead, cameras, network headers or external report files.',
        metric_convention='Original float RGB renderer, clamp [0,1]; repo SSIM and LPIPS VGG [0,1]; equal per-view average including background; both vs GT and vs official Q1.',
        precision_note='Residual f32 is near-lossless, NOT bitwise lossless: arithmetic rounding and SO(3) normalization can differ from raw quaternion parameters. Only replacement baselines assert bitwise/pixel exactness.')
    dump(out/'summary.json',result)
    print(f'Completed {len(records)} variants: {out}/summary.json',flush=True)


if __name__=='__main__':
    main()
