#!/usr/bin/env python3
"""Standalone decode correctness sweep of every independent GOP/representation.

Does not measure physical presentation or streaming FPS. Uses sequential GPU
ownership and bounded materialization. Run separately from timing benchmarks.
"""
import argparse, gc, json, tempfile, time
from pathlib import Path


def validate_scale_underflow(scales,preactivation):
    """Accept zero only with exact exp parity and float32 rounding evidence.

    This validates the unmodified model output. It never clamps scales or
    changes Gaussian data. Half the smallest float32 subnormal is the strict
    upper bound for a positive real value rounding to zero.
    """
    import torch
    threshold=2.**-150
    if scales.dtype!=torch.float32 or preactivation.dtype!=torch.float32:
        raise AssertionError('Expected float32 scale activation')
    if scales.shape!=preactivation.shape:
        raise AssertionError('Scale preactivation shape mismatch')
    if not torch.isfinite(scales).all() or not torch.isfinite(preactivation).all():
        raise AssertionError('Nonfinite scale or preactivation')
    if (scales<0).any():raise AssertionError('Negative activated scales')
    zero=scales==0;count=int(zero.sum().item())
    evidence=dict(zero_count=count,negative_count=0,float32_zero_rounding_upper_bound=threshold)
    if count:
        pre=preactivation[zero]
        reference=torch.exp(pre.double())
        if not (reference<threshold).all():
            raise AssertionError('Zero scales are not explained by float32 exp underflow')
        evidence.update(zero_preactivation_min=float(pre.min().item()),
            zero_preactivation_max=float(pre.max().item()),
            zero_exp_float64_max=float(reference.max().item()))
    if not torch.equal(scales,torch.exp(preactivation)):
        raise AssertionError('Activated scales differ from exact upstream torch.exp')
    evidence['exact_upstream_exp_verified']=True
    return evidence


def verify_activated_scales(handle,state,model_time):
    """Recompute the upstream temporal/covariance path only for observed zeros."""
    import torch
    if (state.scales<0).any():raise AssertionError('Negative activated scales')
    if not (state.scales==0).any():
        return dict(zero_count=0,negative_count=0,float32_zero_rounding_upper_bound=2.**-150,
                    exact_upstream_exp_verified=False)
    pc=handle.model;t=float(model_time)
    with torch.inference_mode():
        p=torch.ones((len(pc._xyz),1),device=pc._xyz.device)*t
        feature=torch.cat([pc.get_time_features(t,False,noise=0),pc.get_time_pe(p)],1)
        feature=pc.get_time_mlp(feature)
        cov=pc.get_cov_mlp(feature)
        preactivation=pc._scaling+cov[:,:3]
        return validate_scale_underflow(state.scales,preactivation)


def verify_feature_pictures(prepared):
    """Independent PNG decode + vectorized row-major tile mapping oracle.

    Do not compare to soldier_runtime's historical spatial transpose: it shares
    the bug this check guards against. Preserve pixel/Gaussian and channel IDs.
    """
    import csv
    import subprocess
    import imagecodecs
    import numpy as np
    m=prepared.reconstruction['model'];grid=int(m['grid_sidelen'])
    rows,columns=int(m['atlas_rows']),int(m['atlas_columns'])
    indices=sorted(set((0,len(prepared.features)//2,len(prepared.features)-1)))
    with (prepared.package/'compression_info.csv').open() as f:
        info=next(row for row in csv.DictReader(f) if 'point_feat' in row[''])
    config=next(a for a in prepared.reconstruction['experiment']['attributes'] if 'point_feat' in a['name'])
    with tempfile.TemporaryDirectory(prefix='streamstgs-picture-reference-') as tmp:
        expr='+'.join(f'eq(n,{i})' for i in indices)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',
            str(prepared.package/'_point_feats.mp4'),'-vf',f"select='{expr}'",'-vsync','0',
            '-pix_fmt','gray16be',str(Path(tmp)/'%02d.png')],check=True)
        for number,index in enumerate(indices,1):
            picture=imagecodecs.imread(Path(tmp)/f'{number:02d}.png')
            assert picture.shape==(rows*grid,columns*grid)
            values=picture/65535
            if config.get('normalize'):
                values=values*(float(info['max'])-float(info['min']))+float(info['min'])
            expected=values.reshape(rows,grid,columns,grid).transpose(1,3,0,2).reshape(grid*grid,rows*columns).astype(np.float32)
            np.testing.assert_array_equal(prepared.features[index],expected)
    return dict(indices=indices,exact=True,oracle='FFmpeg PNG + metadata row-major vectorized tile mapping')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mpd',required=True,type=Path)
    ap.add_argument('--report',required=True,type=Path)
    ap.add_argument('--render',action='store_true')
    args=ap.parse_args()
    import torch
    from multiobj3dgs.client.metadata import MPDParser
    from multiobj3dgs.content.streamstgs.materialize import materialize_gop
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.shared.geometry import Camera
    catalog=MPDParser.from_local(args.mpd)
    decoder=StreamSTGSAdapter();renderer=None
    if args.render:
        from multiobj3dgs.client.rendering.fast import FASTRenderer
        renderer=FASTRenderer()
    report={'catalog_revision':catalog.revision,'scope':'standalone independent GOP decoder correctness','physical_display_fps':None,'cases':[]}
    try:
        for content in catalog.contents.values():
            for rep in content.representations.values():
                for segment in rep.segments:
                    case={'content':content.id,'representation':rep.id,'gop':segment.gop_index};started=time.perf_counter();handle=None
                    try:
                        with tempfile.TemporaryDirectory(prefix='streamstgs-decode-validation-') as tmp:
                            package=materialize_gop(segment,{r.id:Path(r.url) for r in segment.resources},Path(tmp)/'package')
                            prepared=decoder.prepare_cpu(package);handle=decoder.realize_gpu(prepared);camera=Camera.from_package(prepared.cameras[0])
                            case['feature_picture_reference']=verify_feature_pictures(prepared)
                            case['host_bytes']=prepared.nbytes;case['stages']=handle.stages;case['frames']=[];case['scale_validation']=[]
                            del prepared
                            for index in sorted(set([0,len(segment.frames)//2,len(segment.frames)-1])):
                                mapping=segment.frames[index];state=decoder.reconstruct(handle,mapping.model_time,camera)
                                for name in ('xyz','scales','rotations','opacities','colors'):
                                    if not torch.isfinite(getattr(state,name)).all():raise AssertionError(f'Nonfinite {name}')
                                scale_evidence=verify_activated_scales(handle,state,mapping.model_time)
                                case['scale_validation'].append(dict(frame=index,model_time=mapping.model_time,**scale_evidence))
                                if renderer:renderer.draw(state,camera,frame_id=(content.id,rep.id,segment.gop_index,index))
                                case['frames'].append(index)
                            case['gaussians']=len(state.xyz);del state
                            case['status']='passed'
                    except Exception as exc:
                        case['status']='failed';case['error']=repr(exc)
                    finally:
                        if handle is not None:decoder.release(handle)
                        gc.collect();case['elapsed_s']=time.perf_counter()-started
                        report['cases'].append(case)
                        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
                    print(content.id,rep.id,segment.gop_index,case['status'],flush=True)
        if any(c['status']!='passed' for c in report['cases']):raise SystemExit(1)
    finally:
        if renderer:renderer.close()

if __name__=='__main__':main()
