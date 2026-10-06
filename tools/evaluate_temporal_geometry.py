"""Validate both quality prefixes, then measure closed-loop temporal coding.

Only existing decoded payload state is predicted. No checkpoint is available to
the decoder; correction bytes remain unchanged. Original renderer is reused.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.content_preparation.paths import UPSTREAM_ROOT, configure_upstream_imports
from tools.content_preparation.compatibility import store_point_cloud
from tools import evaluate_full_payload_coding as full
from tools import evaluate_correction_coding as old

SOURCE=ROOT/'output/full_payload_coding'
MANIFEST=ROOT/'output/progressive_gap_real/manifest.json'
PROTECTED=['train.py','render.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']


class Decoder(full.Decoder):
    def __init__(self,folder,compressed=True,base_only=False):
        self.folder,self.compressed,self.cache=folder,compressed,{}
        self.decoded_cache={}
        self.base=json.loads(self.read('base.json'))
        self.base_only=base_only
        if not base_only:
            self.new=json.loads(self.read('new.json'))
            self.correction=json.loads(self.read('correction.json'))

    def values(self,ref):
        key=(ref['file'],ref['offset'],ref['nbytes'])
        if key not in self.decoded_cache:
            value=super().values(ref)
            if 'prediction' in ref:
                value=old.apply_residual(self.values(ref['previous']).astype('<f4'),value,ref['prediction'])
            self.decoded_cache[key]=value
        return self.decoded_cache[key]

    def owner(self,owner,fi):
        doc=self.base if owner=='base' else self.new
        n=self.base['shared_count'] if owner=='base' else self.base['new_count']
        result=np.zeros(n,dtype=[(field,'<f4') for field in self.base['ply_fields']])
        for a,schedule in doc['arrays'].items():
            values=self.values(self.select(schedule,fi))
            for col,field in enumerate(self.base['groups'][a]): result[field]=values[:,col]
        return result

    def frame(self,fi):
        if self.base_only:
            result=self.owner('base',fi)
            assert all(name.startswith('base.') for name in self.cache), 'Q0 touched enhancement data'
            return result
        return super().frame(fi)


class BaseDecoder(Decoder):
    def __init__(self,folder,compressed=True):
        super().__init__(folder,compressed,True)


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def evaluate_prefix(out,label,records,manifest,models,schema,prefix):
    dest=out/label;dest.mkdir(parents=True,exist_ok=True)
    link=dest/'payloads'
    if not link.exists(): link.symlink_to(out/'payloads',target_is_directory=True)
    m=copy.deepcopy(manifest)
    if prefix==0:
        for r in m['frames']: r['Q1']=r['Q0']
    old.Decoder=BaseDecoder if prefix==0 else Decoder
    old.accounting=full.accounting
    result=old.evaluate(dest,m,records,models,schema)
    old.dump(dest/'summary.json',result)
    return result


def meets_budget(row):
    return row['gt_psnr_loss']<=.1 and row['max_view_gt_psnr_loss']<=.25


def prefix_phase(out,manifest,b,q,schema):
    names=['F0_exact_reuse','F1_all_f16']+['F1_all_'+p for p in ['q16','q12','q10','q8']]+[
        'F2_mixed_c'+p for p in ['q16','q12','q8']]+['F2_mixed_cq8_frame_access']
    records=[r for r in json.loads((SOURCE/'payload_index.json').read_text()) if r['config']['name'] in names]
    (out/'payloads').mkdir(parents=True,exist_ok=True)
    for r in records:
        name=r['config']['name'];path=out/'payloads'/name
        if not path.exists(): path.symlink_to(SOURCE/'payloads'/name,target_is_directory=True)
    result=evaluate_prefix(out,'prefix_q0',records,manifest,b,schema,0)
    q1={r['variant']:r for r in json.loads((SOURCE/'summary.json').read_text())['variants']}
    table=[]
    for row in result['variants']:
        target=q1[row['variant']]
        table.append(dict(variant=row['variant'],q0_bytes=row['zstd_base_bytes'],total_bytes=row['zstd_total_bytes'],
            q0_psnr=row['gt_psnr'],q0_loss=row['gt_psnr_loss'],q1_loss=target['gt_psnr_loss'],
            q0_worst=row['max_view_gt_psnr_loss'],q1_worst=target['max_view_gt_psnr_loss'],
            both_pass=meets_budget(row) and meets_budget(target)))
    chosen=min([r for r in table if r['both_pass']],key=lambda r:r['total_bytes'])
    old.dump(out/'prefix_validation.json',dict(table=table,selected=chosen,quality_budget=dict(mean_loss_db=.1,worst_loss_db=.25),
        q0_evaluation='Q0-only payload; same 16 res4/256 test views as Q1, against official Q0 rendered at those same views. No correction/new files read.',
        q1_metrics_reused_from=str(SOURCE/'summary.json')))
    old.csv_write(out/'prefix_quality.csv',table)
    print('BOTH-PREFIX BASELINE:',chosen,flush=True)


def temporal_configs(parent):
    choices=[('T1',[('base','xyz')]),('T2',[('base','rotation')]),
        ('T3',[('base','xyz'),('base','rotation')]),('T4',[('new','xyz')]),
        ('T5',[('new','rotation')]),('T6',[('new','xyz'),('new','rotation')]),
        ('T7',[('base','xyz'),('new','xyz')]),('T8',[('base','rotation'),('new','rotation')]),
        ('T9',[(o,a) for o in ['base','new'] for a in ['xyz','rotation']])]
    configs=[]
    for name,selected in choices:
        configs.append(dict(parent['config'],name=name+'_same_precision',selected=selected,prediction=True,override=None))
    for precision in ['q16','q12','q10','q8']:
        for predicted in [True,False]:
            configs.append(dict(parent['config'],name=('T9_' if predicted else 'A_')+precision,
                selected=choices[-1][1],prediction=predicted,override=precision))
    return configs


def encode_temporal(out,cfg,parent):
    source=out/'payloads'/parent['config']['name']
    decoder=Decoder(source)
    docs={o:copy.deepcopy(getattr(decoder,o)) for o in ['base','new','correction']}
    packet=full.Packet(out/'payloads'/cfg['name'],out/'blob_pool')
    selected=set(map(tuple,cfg['selected']))
    for stream in parent['streams']:
        if stream['category']=='metadata' or (stream['owner'],stream['attribute']) in selected: continue
        name=stream['file']
        packet.streams[name]=bytearray(decoder.read(name))
        packet.labels[name]={k:stream[k] for k in ['owner','category','attribute']}
    traces=[]
    for owner,a in cfg['selected']:
        old_schedule=docs[owner]['arrays'][a]
        assert 'frames' in old_schedule, 'Only empirically dynamic groups are predicted'
        refs=[];previous=None
        for fi,old_ref in enumerate(old_schedule['frames']):
            target=decoder.values(old_ref).astype('<f4')
            packet.segment=fi
            if fi==0:
                # Copy absolute initialization exactly, including quantizer scale.
                data=decoder.blob(old_ref)
                meta={k:v for k,v in old_ref.items() if k not in ['file','offset','nbytes']}
                ref=packet.append(owner,'dynamic_values',a,data,meta)
                current=target
                maximum=float(np.abs(old.rotation(target).as_rotvec() if a=='rotation' else target).max())
            else:
                precision=cfg['override'] or old_ref['precision']
                if cfg['prediction']:
                    values=old.residual(previous,target,a)
                else:
                    values=old.rotation(target).as_rotvec() if a=='rotation' else target
                maximum=float(np.abs(values).max())
                data,meta=old.encode_values(values,precision,maximum)
                decoded=old.decode_values(data,meta)
                if cfg['prediction']:
                    meta.update(prediction=a,previous=refs[-1])
                    current=old.apply_residual(previous,decoded,a)
                else:
                    if a=='rotation':
                        meta['transform']='absolute_rotvec'
                        current=full.Rotation.from_rotvec(decoded).as_quat()[:,[3,0,1,2]].astype('<f4')
                    else: current=decoded.astype('<f4')
                ref=packet.append(owner,'dynamic_values',a,data,meta)
            refs.append(ref)
            error=old.residual(current,target,a)
            traces.append(dict(variant=cfg['name'],owner=owner,attribute=a,frame_index=fi,
                predictor='decoded_previous' if cfg['prediction'] and fi else 'absolute',
                predictor_sha256=None if previous is None else old.hashlib.sha256(previous.tobytes()).hexdigest(),
                decoded_sha256=old.hashlib.sha256(current.tobytes()).hexdigest(),
                source_max_abs=maximum,max_error_norm=float(np.linalg.norm(error,axis=1).max())))
            previous=current
        docs[owner]['arrays'][a]={'frames':refs}
    packet.segment=None
    # Encoder configuration belongs in report/index, not in the correction header:
    # correction must remain byte-identical, including its metadata.
    record=dict(config=cfg,streams=packet.finish(docs),encoder_trace=traces)
    coded=Decoder(out/'payloads'/cfg['name'])
    for trace in traces:
        ref=coded.select(getattr(coded,trace['owner'])['arrays'][trace['attribute']],trace['frame_index'])
        state=coded.values(ref).astype('<f4')
        assert old.hashlib.sha256(state.tobytes()).hexdigest()==trace['decoded_sha256']
        if trace['predictor']=='decoded_previous':
            pred=coded.values(ref['previous']).astype('<f4')
            assert old.hashlib.sha256(pred.tobytes()).hexdigest()==trace['predictor_sha256']
    for s in record['streams']:
        if s['owner']=='correction':
            original=next(t for t in parent['streams'] if t['file']==s['file'])
            assert s['sha256']==original['sha256'] and s['compressed_bytes']==original['compressed_bytes']
    return record


def diagnostic_stats(out,parent,records):
    source=Decoder(out/'payloads'/parent['config']['name'])
    rows=[]
    for owner in ['base','new']:
        doc=getattr(source,owner)
        for a in ['xyz','rotation']:
            values=[source.values(source.select(doc['arrays'][a],fi)).astype('<f4') for fi in range(5)]
            for fi in range(1,5):
                difference=old.residual(values[fi-1],values[fi],a)
                magnitude=np.linalg.norm(difference,axis=1)
                rows.append(dict(owner=owner,attribute=a,frame_index=fi,unit='radians' if a=='rotation' else 'scene_units',
                    mean=float(magnitude.mean()),median=float(np.median(magnitude)),p95=float(np.quantile(magnitude,.95)),
                    maximum=float(magnitude.max()),zero_fraction=float((magnitude==0).mean())))
    old.csv_write(out/'temporal_stats.csv',rows)
    breakdown=[]
    for record in [parent]+records:
        total=full.accounting(record)['zstd_total_bytes']
        for owner in ['base','new','correction']:
            for a in ['xyz','rotation','sh','scale','opacity']:
                streams=[s for s in record['streams'] if s['owner']==owner and s['attribute']==a]
                raw=sum(s['raw_bytes'] for s in streams);compressed=sum(s['compressed_bytes'] for s in streams)
                breakdown.append(dict(variant=record['config']['name'],owner=owner,attribute=a,raw_bytes=raw,
                    zstd_bytes=compressed,percent_total=100*compressed/total,compression_ratio=raw/compressed if compressed else 0))
        streams=[s for s in record['streams'] if s['category']=='metadata']
        breakdown.append(dict(variant=record['config']['name'],owner='metadata',attribute='headers',
            raw_bytes=sum(s['raw_bytes'] for s in streams),zstd_bytes=sum(s['compressed_bytes'] for s in streams),
            percent_total=100*sum(s['compressed_bytes'] for s in streams)/total,compression_ratio=0))
    old.csv_write(out/'owner_attribute_breakdown.csv',breakdown)


def temporal_phase(out,manifest,b,q,schema):
    prefix=json.loads((out/'prefix_validation.json').read_text())
    selected=prefix['selected']['variant']
    parent=next(r for r in json.loads((SOURCE/'payload_index.json').read_text()) if r['config']['name']==selected)
    # Ensure adapters preserve all prior Q1 payload semantics before experimentation.
    for fi in range(5):
        assert Decoder(out/'payloads'/selected).frame(fi).tobytes()==full.Decoder(SOURCE/'payloads'/selected).frame(fi).tobytes()
    records=[]
    for cfg in temporal_configs(parent):
        record=encode_temporal(out,cfg,parent);records.append(record)
        print('ENCODED',cfg['name'],full.accounting(record)['zstd_total_bytes'],flush=True)
    old.dump(out/'payload_index.json',[parent]+records)
    old.csv_write(out/'streams.csv',[dict(variant=r['config']['name'],**s) for r in [parent]+records for s in r['streams']])
    diagnostic_stats(out,parent,records)
    evaluate_prefix(out,'temporal_q0',records,manifest,b,schema,0)
    evaluate_prefix(out,'temporal_q1',records,manifest,q,schema,1)
    report(out)


def report(out):
    original_hashes=json.loads((out/'source_validation.json').read_text())['hashes']
    assert all(old.digest(UPSTREAM_ROOT/p)==h for p,h in original_hashes.items())
    prefix=json.loads((out/'prefix_validation.json').read_text())
    parent_name=prefix['selected']['variant']
    baseline0=json.loads((out/'prefix_q0/summary.json').read_text())
    baseline1=json.loads((SOURCE/'summary.json').read_text())
    temporal0=json.loads((out/'temporal_q0/summary.json').read_text())
    temporal1=json.loads((out/'temporal_q1/summary.json').read_text())
    q0={r['variant']:r for r in baseline0['variants']+temporal0['variants']}
    q1={r['variant']:r for r in baseline1['variants']+temporal1['variants']}
    records=json.loads((out/'payload_index.json').read_text())
    configs={r['config']['name']:r['config'] for r in records}
    rows=[]
    metrics=['gt_psnr','gt_ssim','gt_lpips','gt_psnr_loss','gt_ssim_loss','gt_lpips_increase',
             'max_view_gt_psnr_loss','p95_view_gt_psnr_loss','global_max_abs_pixel','q1_psnr','q1_ssim','q1_lpips']
    for name,a in q0.items():
        b=q1[name];cfg=configs.get(name,{})
        row=dict(variant=name,kind='temporal' if cfg.get('prediction') else ('absolute_recode' if 'prediction' in cfg else 'original_absolute'),
            selected=json.dumps(cfg.get('selected',[])),residual_precision=cfg.get('override') or 'parent',
            **{k:v for k,v in b.items() if k.startswith(('raw_','zstd_'))},both_pass=meets_budget(a) and meets_budget(b))
        for p,source in [('q0',a),('q1',b)]:
            for k in metrics: row[p+'_'+k.replace('q1_','reference_')]=source[k]
        rows.append(row)
    by={r['variant']:r for r in rows};parent=by[parent_name]
    eligible=[r for r in rows if r['both_pass']]
    best=min(eligible,key=lambda r:r['zstd_total_bytes'])
    absbest=min([r for r in eligible if r['kind']!='temporal'],key=lambda r:r['zstd_total_bytes'])
    near=[r for r in rows if r['kind']=='temporal' and all(
        r[f'{p}_gt_psnr_loss']<=parent[f'{p}_gt_psnr_loss']+.01 and
        r[f'{p}_max_view_gt_psnr_loss']<=parent[f'{p}_max_view_gt_psnr_loss']+.02 for p in ['q0','q1'])]
    matched=min(near,key=lambda r:r['zstd_total_bytes']) if near else None
    # Two-prefix, two-tail rate-quality frontier; no single scalar hides bad Q0.
    dimensions=[p+'_'+k for p in ['q0','q1'] for k in ['gt_psnr_loss','max_view_gt_psnr_loss']]
    def dominates(a,b):
        keys=['zstd_total_bytes']+dimensions
        return all(a[k]<=b[k] for k in keys) and any(a[k]<b[k] for k in keys)
    for r in rows: r['pareto_two_prefix_psnr']=not any(dominates(a,r) for a in rows if a is not r)
    old.csv_write(out/'summary.csv',rows)
    # Consolidate all views; 'reference' is official Q0 or official Q1 respectively.
    for file in ['per_view.csv','per_frame.csv']:
        merged=[]
        for p,folder,names in [(0,out/'prefix_q0',q0.keys()),(0,out/'temporal_q0',q0.keys()),
                               (1,SOURCE,[r['variant'] for r in baseline0['variants']]),(1,out/'temporal_q1',q0.keys())]:
            for entry in read_csv(folder/file):
                if entry['variant'] not in names: continue
                merged.append(dict(prefix=p,**{k.replace('q1_','reference_'):v for k,v in entry.items()}))
        old.csv_write(out/file,merged)
    access=[];checks=[]
    parent_initial=Decoder(out/'payloads'/parent_name).frame(0)
    for record in records:
        name=record['config']['name'];folder=out/'payloads'/name
        for s in record['streams']:
            assert (folder/s['file']).stat().st_size==s['raw_bytes']
            assert (folder/(s['file']+'.zst')).stat().st_size==s['compressed_bytes']
            assert old.digest(folder/s['file'])==s['sha256']
            assert old.decompress((folder/(s['file']+'.zst')).read_bytes())==(folder/s['file']).read_bytes()
        for p in [0,1]:
            seqdecoder=BaseDecoder(folder) if p==0 else Decoder(folder)
            sequential=[]
            sizes={s['file']:s['compressed_bytes'] for s in record['streams']}
            for fi in range(5):
                previous_files=set(seqdecoder.cache)
                state=seqdecoder.frame(fi);sequential.append(state)
                fresh=BaseDecoder(folder) if p==0 else Decoder(folder)
                random=fresh.frame(fi)
                raw=BaseDecoder(folder,False) if p==0 else Decoder(folder,False)
                assert state.tobytes()==random.tobytes()==raw.frame(fi).tobytes()
                selected=record['config'].get('selected',[]) if record['config'].get('prediction') else []
                dependent=any(o=='base' or p==1 for o,a in selected)
                access.append(dict(variant=name,prefix=p,frame_index=fi,dependency_depth=fi if dependent else 0,
                    cold_access_bytes=sum(sizes[f] for f in fresh.cache),
                    sequential_cumulative_bytes=sum(sizes[f] for f in seqdecoder.cache),
                    sequential_increment_bytes=sum(sizes[f] for f in set(seqdecoder.cache)-previous_files),
                    startup_header_bytes=sum(sizes[f] for f in previous_files) if fi==0 else 0))
            if p==1: assert sum(sizes[f] for f in seqdecoder.cache)==full.accounting(record)['zstd_total_bytes']
            checks.append(dict(variant=name,prefix=p,sequential_random_raw_equal=True,base_only_no_enhancement_reads=p==0))
        d=Decoder(folder)
        assert d.frame(0).tobytes()==parent_initial.tobytes(), 'Initialization changed'
        for trace in record.get('encoder_trace',[]):
            ref=d.select(getattr(d,trace['owner'])['arrays'][trace['attribute']],trace['frame_index'])
            state=d.values(ref).astype('<f4')
            assert old.hashlib.sha256(state.tobytes()).hexdigest()==trace['decoded_sha256']
            if trace['frame_index']==0: assert 'prediction' not in ref
            if 'prediction' in ref:
                assert old.hashlib.sha256(d.values(ref['previous']).astype('<f4').tobytes()).hexdigest()==trace['predictor_sha256']
        print('VALIDATED',name,flush=True)
    import tempfile
    for name in [parent_name,best['variant']]:
        with tempfile.TemporaryDirectory(prefix='q0_only_',dir=out) as tmp:
            folder=Path(tmp)
            for file in (out/'payloads'/name).glob('base.*.zst'):
                os.link(file,folder/file.name)
            for fi in range(5):
                assert BaseDecoder(folder).frame(fi).tobytes()==BaseDecoder(out/'payloads'/name).frame(fi).tobytes()
    old.csv_write(out/'access_cost.csv',access)
    assert len(read_csv(out/'per_view.csv'))==len(rows)*160
    assert len(read_csv(out/'per_frame.csv'))==len(rows)*10
    old.dump(out/'validation.json',dict(passed=True,configs=len(rows),view_pairs=len(rows)*160,
        newly_rendered_view_pairs=(len(baseline0['variants'])+2*len(temporal0['variants']))*80,
        reused_q1_view_pairs=len(baseline0['variants'])*80,closed_loop_hash_checks=True,
        first_frame_absolute=True,correction_unchanged=True,all_bytes_accounted=True,
        original_sources_unchanged=True,protected_hashes=original_hashes,
        raw_zstd_roundtrip=True,so3_test='Existing helper randomized SO(3) composition/sign/identity self-test passed',
        compressed_only_base_directory_decode=True,
        decode_checks=checks))
    result=dict(variants=rows,parent=parent_name,best_budget=best['variant'],best_absolute_budget=absbest['variant'],
        near_parent_quality=None if matched is None else matched['variant'],
        quality_budget=dict(mean_loss_db=.1,worst_loss_db=.25,both_prefixes=True),
        matching_tolerance=dict(mean_loss_db=.01,worst_loss_db=.02,relative_to_parent=True),
        official_q0_vs_gt=baseline0['official_q1_vs_gt'],official_q1_vs_gt=baseline1['official_q1_vs_gt'],
        methodology='Predict existing decoded Base/New geometry. First frame copied exactly. Predictor always decoded previous frame; per-frame max-abs calibration is transmitted. Correction bitstreams frozen. Q0 and Q1 use the same 80 res4 views/GT, compared with their own official renders. A_* controls recode the same targets with per-frame absolute scales and same precision; initialization unchanged. No representation dimension reduction (xyz/rotvec are both 3D).',
        environment=baseline1['environment'],compressor=baseline1['compressor'],
        script_sha256=old.digest(__file__),manifest_sha256=old.digest(MANIFEST))
    old.dump(out/'summary.json',result)
    print('BEST BOTH-PREFIX BUDGET:',best['variant'],best['zstd_total_bytes'],flush=True)
    print('BEST ABSOLUTE:',absbest['variant'],absbest['zstd_total_bytes'],flush=True)
    print('NEAR PARENT:',None if matched is None else matched['variant'],flush=True)
    write_report(out)


def write_report(out):
    s=json.loads((out/'summary.json').read_text());rows=s['variants'];by={r['variant']:r for r in rows}
    parent=by[s['parent']];best=by[s['best_budget']];absolute=by[s['best_absolute_budget']]
    breakdown=read_csv(out/'owner_attribute_breakdown.csv');access=read_csv(out/'access_cost.csv')
    stats=read_csv(out/'temporal_stats.csv')
    value=lambda name,owner,a: int(next(r['zstd_bytes'] for r in breakdown if r['variant']==name and r['owner']==owner and r['attribute']==a))
    geom=lambda name,a: sum(value(name,o,a) for o in ['base','new','correction'])
    access_bytes=lambda name,p,fi: int(next(r['cold_access_bytes'] for r in access if r['variant']==name and int(r['prefix'])==p and int(r['frame_index'])==fi))
    gain=100*(1-best['zstd_total_bytes']/parent['zstd_total_bytes'])
    matched_gain=100*(1-best['zstd_total_bytes']/absolute['zstd_total_bytes'])
    remain=100*sum(geom(best['variant'],a) for a in ['xyz','rotation'])/best['zstd_total_bytes']
    base_new_geometry=sum(value(best['variant'],o,a) for o in ['base','new'] for a in ['xyz','rotation'])
    penalty=100*(access_bytes(best['variant'],1,4)/access_bytes(absolute['variant'],1,4)-1)
    recommendation='CONTINUE: investigate the measured temporal geometry rate/access trade-off; this is not evidence that a new codec or motion method will outperform stronger baselines.'
    s.update(recommendation=recommendation,conclusions=dict(total_saving_vs_parent_percent=gain,
        total_saving_vs_budget_matched_absolute_percent=matched_gain,remaining_geometry_percent=remain,
        frame4_q1_cold_access_penalty_vs_absolute_percent=penalty),script_sha256=old.digest(__file__))
    old.dump(out/'summary.json',s)
    lines=['# Temporal geometry baseline — Longdress 1051–1055','',
        '## Outcome','',
        f"Both prefixes pass the specified quality budget. The selected temporal operating point is **{best['variant']}**, **{best['zstd_total_bytes']/1e6:.6f} MB/GoF**: {gain:.2f}% smaller than the previous operating point and {matched_gain:.2f}% smaller than the best tested absolute configuration satisfying the same two-prefix budget. Cold access to the last Q1 frame is {penalty:.2f}% more expensive than that absolute comparator. Geometry still accounts for {remain:.2f}% of total coded bytes.", '',
        '**Recommendation: CONTINUE a bounded investigation of temporal geometry rate versus access cost.** The motivation is the measured RD/access trade-off after the simple baseline, not non-composability alone. These five frames do not establish that a new codec/motion method is warranted or competitive with stronger codecs.', '',
        '## Reproduce and scope','',
        '```bash\nconda activate Hoang\npython tools/evaluate_temporal_geometry.py --phase prefix\npython tools/evaluate_temporal_geometry.py --phase temporal\n# Existing measurements: validate and regenerate aggregates/report\npython tools/evaluate_temporal_geometry.py --phase report\n```','',
        'A fresh run requires a new output directory if temporal payloads already exist (`--output PATH`). The first stage must finish before temporal coding chooses its parent. No training/model/renderer source is changed. No learned codec, DASH or video codec is used. Existing Longdress checkpoints, prefix correspondence checks and exact upstream CUDA renderer are reused.', '',
        '27 configurations: 10 existing absolute controls and 17 new temporal/absolute-recoding variants. 3,520 new view-pair evaluations (800 Q0-prefix checks + 2,720 temporal/control Q0/Q1 checks), plus 800 previously measured Q1 view pairs reused from full_payload_coding. The aggregate contains 4,320 view pairs and 270 frame/prefix rows.', '',
        'Both Q0 and Q1 are rendered at the same **256×256 res4 test cameras/GT**, 16 views/frame. Q0 was trained at lower resolution; its official PSNR at these evaluation views is 29.612411 dB versus 39.647976 dB for official Q1. “Q0 quality preserved” means quantization adds little loss, not that Q0 has Q1-level absolute quality. No Q1 suffix/correction is loaded by the Q0 decoder.', '',
        'PSNR loss means `official-prefix vs GT PSNR − decoded-prefix vs GT PSNR`; it is not the PSNR between the two rendered images. Positive is degradation. Direct decoded-vs-official PSNR/SSIM/LPIPS and max pixel errors are retained as `*_reference_*` and `*_global_max_abs_pixel` columns. Equal-weight full-image averages include background; small improvements can be incidental smoothing/metric noise.', '',
        '## Phase 1: validate each prefix','',
        'Budget: mean GT PSNR loss ≤0.1 dB and worst-view loss ≤0.25 dB, **for each prefix**. SSIM/LPIPS are also measured, not implicitly assumed equal.', '',
        '| Existing configuration | Q0 bytes | Full bytes | Q0 GT PSNR | Q0 loss | Q1 loss | Q0 worst | Q1 worst | Both pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['kind']!='original_absolute': continue
        lines.append(f"| {r['variant']} | {r['zstd_base_bytes']:,} | {r['zstd_total_bytes']:,} | {r['q0_gt_psnr']:.6f} | {r['q0_gt_psnr_loss']:.6f} | {r['q1_gt_psnr_loss']:.6f} | {r['q0_max_view_gt_psnr_loss']:.6f} | {r['q1_max_view_gt_psnr_loss']:.6f} | {r['both_pass']} |")
    lines += ['', '**A1:** mixed cq8 passes both prefixes: Q0 mean loss 0.000959 dB, worst 0.004422 dB; Q1 mean 0.070788 dB, worst 0.144644 dB.', '',
        '**A2:** no tested configuration has good Q1 but materially damaged Q0 under the specified GT-PSNR budget. In fact uniform q8 improves Q0 GT PSNR while greatly damaging Q1. That does not make it a faithful reconstruction of official Q0; see direct-reference metrics. Mixed cq16/cq12/cq8 produce identical Q0 metrics, confirming correction precision does not enter Q0 rendering.', '',
        '**A3:** the existing frame-access mixed operating point is valid for both-prefix quality and the tested in-GoF access semantics. This does not validate an end-to-end adaptive streaming system, transport behavior, arbitrary GoF switching, or perceptual equivalence.', '',
        '## Coding method and fair accounting','',
        '- Targets are the **already decoded** Base/New geometry streams from the selected parent, not original floating checkpoints. This is a controlled lossy transcode. Correction payloads, including their header, stay byte-identical. Other attributes stay byte-identical. Additional Base error is therefore not secretly repaired by newly optimized correction.',
        '- First-frame absolute bytes and quantizer parameters are copied exactly and fully charged. For later frames, xyz residual is target minus decoded previous xyz. Rotation is `Log(R_target · inverse(R_decoded_previous))`, decoded by left multiplication `Exp(r) · R_previous`, with wxyz at the renderer boundary. No quaternion subtraction.',
        '- Predictor state is always the previous **decoded float32** state. Encoder/decoder state hashes and previous-state hashes are checked. Principal SO(3) rotation vectors are three-dimensional in both absolute and temporal variants; there is no dimension-reduction advantage.',
        '- T1–T9 use parent precision (xyz q16, rotation q12). T9_q16/q12/q10/q8 sweep the predicted streams after frame 0; static state, initialization and correction remain fixed. A_q* controls recode the same decoded targets with absolute values, the same bit depth, and the same per-frame max-abs scale policy, without prediction.',
        '- Per-frame signed fixed-point max-abs scale is transmitted, nearest-even rounding, no clipping. Every value block is a separate zstd level-6 frame (`--long=27 -T1`). Global JSON contains all reference/scale descriptors. Metadata includes recursively represented predecessor descriptors; their actual bytes are counted.',
        '- Full cost includes Base, New, Correction, frame-0/reference cost, static state once, scales, metadata and zstd headers. Existing cameras and transport/filesystem overhead are excluded consistently. MB is decimal. Owner-inclusive byte columns already contain each owner’s headers; component value bytes + separate metadata is the nonoverlapping breakdown. No cost is inferred from in-memory array sizes.', '',
        '## Owner × attribute breakdown','',
        '| Owner | Attribute | Parent raw B | Parent zstd B | Parent total % | T9_q10 zstd B |',
        '|---|---|---:|---:|---:|---:|']
    for r in breakdown:
        if r['variant']!=parent['variant']: continue
        match=next(x for x in breakdown if x['variant']==best['variant'] and x['owner']==r['owner'] and x['attribute']==r['attribute'])
        lines.append(f"| {r['owner']} | {r['attribute']} | {int(r['raw_bytes']):,} | {int(r['zstd_bytes']):,} | {float(r['percent_total']):.3f} | {int(match['zstd_bytes']):,} |")
    lines += ['', 'Before prediction: Base/New geometry is 3,849,182 B (58.515% total); correction geometry is 687,470 B (10.451%). Thus the old 68.97% geometry figure is mainly Base/New, not correction.', '',
        '## Temporal ablations and precision sweep','',
        'T1 Base xyz; T2 Base rotation; T3 Base both; T4 New xyz; T5 New rotation; T6 New both; T7 Base+New xyz; T8 Base+New rotation; T9 all four streams. A_* is the no-prediction control.', '',
        '| Configuration | Full MB | Q0 loss dB | Q0 worst | Q1 loss dB | Q1 p95 | Q1 worst | Both pass |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['kind']=='original_absolute' and r['variant']!=parent['variant']: continue
        lines.append(f"| {r['variant']} | {r['zstd_total_bytes']/1e6:.6f} | {r['q0_gt_psnr_loss']:.6f} | {r['q0_max_view_gt_psnr_loss']:.6f} | {r['q1_gt_psnr_loss']:.6f} | {r['q1_p95_view_gt_psnr_loss']:.6f} | {r['q1_max_view_gt_psnr_loss']:.6f} | {r['both_pass']} |")
    lines += ['', '| Operating point | Prefix | GT PSNR | SSIM | LPIPS | p95 loss | Worst loss |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in [parent,absolute,best]:
        for p in ['q0','q1']:
            lines.append(f"| {r['variant']} | {p} | {r[p+'_gt_psnr']:.6f} | {r[p+'_gt_ssim']:.8f} | {r[p+'_gt_lpips']:.8f} | {r[p+'_p95_view_gt_psnr_loss']:.6f} | {r[p+'_max_view_gt_psnr_loss']:.6f} |")
    lines += ['', '### Quality-matched interpretation','',
        f"At the original precision, T9 costs 6.404027 MB: only {100*(1-by['T9_same_precision']['zstd_total_bytes']/parent['zstd_total_bytes']):.2f}% less than the parent. With the same q12 per-frame calibration policy, T9_q12 versus A_q12 saves {100*(1-by['T9_q12']['zstd_total_bytes']/by['A_q12']['zstd_total_bytes']):.2f}% total. Prediction alone at fixed bit depth is therefore a modest coding gain.", '',
        f"The useful gain is that residuals tolerate q10 while A_q10 fails the Q1 budget. T9_q10 costs {best['zstd_total_bytes']:,} B versus A_q12 {absolute['zstd_total_bytes']:,} B, a **{matched_gain:.2f}%** saving at the same stated quality budget. This is not mathematically identical quality: Q1 mean/worst losses are 0.074038/0.146368 versus 0.077659/0.161986 dB; Q0 worst loss is 0.007494 versus 0.005406 dB, and Q0 LPIPS is slightly higher for temporal coding. No BD-rate is claimed from this sparse sweep.", '',
        'A separate near-parent quality filter (each prefix: mean loss no more than parent +0.01 dB; worst no more than parent +0.02 dB) also selects T9_q10. These tolerances are an explicit comparison convention, not proof of visual equivalence. T9_q8 fails both Q1 bounds (0.106991 mean, 0.258076 worst); do not select it because its total is smaller.', '',
        '## Temporal redundancy diagnostics','',
        'Measured on decoded parent states, before predictor quantization. xyz norms use the normalized scene coordinate system; rotation angles are radians.', '',
        '| Owner | Attribute | Mean range over transitions | Median range | p95 range |',
        '|---|---|---:|---:|---:|']
    for owner in ['base','new']:
        for a in ['xyz','rotation']:
            chosen=[r for r in stats if r['owner']==owner and r['attribute']==a]
            ranges=[f"{min(float(r[k]) for r in chosen):.5f}–{max(float(r[k]) for r in chosen):.5f}" for k in ['mean','median','p95']]
            lines.append(f"| {owner} | {a} | {' | '.join(ranges)} |")
    lines += ['', '| Owner / attribute | Parent raw/zstd | T9 same-precision raw/zstd |', '|---|---:|---:|']
    for owner in ['base','new']:
        for a in ['xyz','rotation']:
            items=[next(r for r in breakdown if r['variant']==n and r['owner']==owner and r['attribute']==a) for n in [parent['variant'],'T9_same_precision']]
            lines.append(f"| {owner} / {a} | {float(items[0]['compression_ratio']):.4f} | {float(items[1]['compression_ratio']):.4f} |")
    lines += ['', 'Same-precision value raw bytes are unchanged. Smaller residual magnitude does not itself reduce a fixed-width symbol’s byte count; it improves precision at a given bit depth. zstd benefits are modest; most additional saving at T9_q10 comes from fewer bits enabled by prediction. Static payload and correction are unchanged, and both xyz/rotation remain 3D. No learned entropy or motion model is used.', '',
        '## Sequential and cold random access','',
        'Cold access includes all required shared static/header bytes and the chosen frame’s enhancement. Temporal Base/New references recursively fetch the necessary frame0→t geometry; unrelated prior-frame correction blocks are not fetched. Depth is number of predecessor links. Sequential costs and startup headers are separately recorded in access_cost.csv; a fresh full-Q1 sequential decode accounts for every transmitted byte.', '',
        '| Frame index | Absolute Q0 B | Temporal Q0 B | Absolute Q1 B | Temporal Q1 B | Temporal depth |',
        '|---:|---:|---:|---:|---:|---:|']
    for fi in range(5):
        lines.append(f"| {fi} | {access_bytes(absolute['variant'],0,fi):,} | {access_bytes(best['variant'],0,fi):,} | {access_bytes(absolute['variant'],1,fi):,} | {access_bytes(best['variant'],1,fi):,} | {fi} |")
    lines += ['', f"Worst Q1 cold access is {access_bytes(best['variant'],1,4):,} B for temporal versus {access_bytes(absolute['variant'],1,4):,} B for A_q12 (+{penalty:.2f}%). Relative to the previous mixed parent’s 3,029,315 B it increases {100*(access_bytes(best['variant'],1,4)/3029315-1):.2f}%. Worst Q0 cold access is 2,635,437 B temporal versus 1,494,131 B absolute (absolute worst is frame0). Static state is not free. Cached/warm access will differ; no GoF switching latency or network throughput is simulated.", '',
        '## Answers B1–B10','',
        '**B1. Q0 quality:** preserved by the selected mixed parent and temporal q10 under the specified budget. Q0’s lower absolute GT quality remains a property of the original lower level.', '',
        '**B2. Valid point for both prefixes:** original parent remains valid; T9_q10 is the smallest tested full payload passing both mean/worst limits. A_q12 is the best tested absolute comparator under those limits.', '',
        '**B3. Owner geometry bytes:** see the exact owner×attribute table. Parent Base xyz/rotation = 1,216,446/902,258 B; New = 926,075/804,403 B; Correction = 329,851/357,619 B.', '',
        f"**B4. Geometry savings:** including unchanged correction, xyz {geom(parent['variant'],'xyz'):,} → {geom(best['variant'],'xyz'):,} B ({100*(1-geom(best['variant'],'xyz')/geom(parent['variant'],'xyz')):.2f}%); rotation {geom(parent['variant'],'rotation'):,} → {geom(best['variant'],'rotation'):,} B ({100*(1-geom(best['variant'],'rotation')/geom(parent['variant'],'rotation')):.2f}%); total {gain:.2f}%. These include both prediction and the successful lower precision, not prediction alone.", '',
        f"**B5. Matched-quality gain:** {matched_gain:.2f}% versus the budget-matched A_q12, with small nonidentical metric differences explicitly given above. Near-parent-tolerance saving is {gain:.2f}%. Same-bit q12 gain is only ~2.53%.", '',
        '**B6. Base versus New:** both benefit. At same precision, Base geometry saves 91,329 B (4.31%) and New 83,087 B (4.80%); Base saves slightly more absolute bytes, New slightly more proportionally. At T9_q10 the corresponding savings are 554,840 B (26.19%) and 493,596 B (28.52%).', '',
        '**B7. xyz versus rotation:** xyz has the larger measured coding benefit: same-precision Base+New xyz saves 116,569 B (~5.44%) versus rotation 57,847 B (~3.39%). This is a rate/quality observation, not a comparison of scene units to radians.', '',
        f"**B8. Access penalty:** significant for late cold access: +{penalty:.2f}% Q1 bytes versus A_q12, depth four at the last frame, despite lower sequential total. All dependencies and initialization are paid.", '',
        f"**B9. Remaining bottleneck:** xyz+rotation still occupy {remain:.2f}% of total; Base/New geometry alone is {base_new_geometry:,} B ({100*base_new_geometry/best['zstd_total_bytes']:.2f}%). Static SH is now a substantial secondary component. Remaining geometry share is not a measure of theoretically removable redundancy.", '',
        '**B10. CONTINUE, narrowly:** a material geometry rate share remains after a legitimate closed-loop baseline, and the measured quality-preserving reduction comes with a large cold-access penalty. That is enough to investigate the rate/access frontier further. It is **not** evidence to launch a complex new motion/learned codec or claim novelty: simple prediction already captures the demonstrated gain, zstd ratios at fixed precision are near one, and no remaining entropy bound or stronger codec comparison was measured. Next evidence should come from longer GoFs/more sequences and stronger conventional prediction/quantization controls before redesign.', '',
        '## Validation and limits','',
        'validation.json checks raw/zstd file hashes/roundtrip, encoder and decoder closed-loop state hashes, exact absolute initialization, unchanged correction bytes, sequential versus fresh random-access decoding, Q0-only reads, full byte accounting and protected source hashes. Randomized SO(3) composition/sign/identity tests are reused. Raw and compressed decoding agree for every rendered candidate. Prefix correspondence is rechecked against checkpoint provenance; no new matching is introduced.', '',
        'Only five consecutive frames of one sequence and 16 background-inclusive views/frame. No motion prediction beyond the previous state, no temporal coding of correction, no random-access points beyond frame0, no integer-domain lossless temporal codec, no foreground-specific metric or user study. Quantizer scales/headers are computed offline even though predictor reconstruction is closed-loop; this is not a real-time causal encoder claim. The fixed parent’s quantization errors cannot be undone by this decoded-target transcode.', '',
        'Artifacts: summary.json/summary.csv, prefix_quality.csv, per_frame.csv, per_view.csv, owner_attribute_breakdown.csv, temporal_stats.csv, streams.csv, access_cost.csv, payload_index.json (encoder traces), validation.json, source_validation.json and serialized raw/zstd payloads. Earlier experiments are unchanged.','']
    (ROOT/'docs/TEMPORAL_GEOMETRY_BASELINE.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/temporal_geometry_baseline')
    parser.add_argument('--phase',choices=['prefix','temporal','report','docs'],default='prefix')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    vendor=MANIFEST.parent/'vendor'
    configure_upstream_imports(vendor if vendor.is_dir() else None)
    manifest=json.loads(MANIFEST.read_text())
    hashes={p:old.digest(UPSTREAM_ROOT/p) for p in PROTECTED}
    old.self_test()
    b,q,schema,seq,life=old.collect(manifest)
    if args.phase=='prefix': prefix_phase(out,manifest,b,q,schema)
    elif args.phase=='temporal': temporal_phase(out,manifest,b,q,schema)
    elif args.phase=='report': report(out)
    else: write_report(out)
    assert all(old.digest(UPSTREAM_ROOT/p)==h for p,h in hashes.items())
    old.dump(out/'source_validation.json',dict(hashes=hashes,unchanged=True,manifest_sha256=old.digest(MANIFEST),
        script_sha256=old.digest(__file__),helpers={p:old.digest(ROOT/p) for p in ['tools/evaluate_correction_coding.py','tools/evaluate_full_payload_coding.py']}))


if __name__=='__main__':
    main()
