"""Measure prefix composability without modifying Dynamic-LapisGS training.

--run-demo generates a moving 3-D fixture and invokes the unmodified train.py.
--manifest accepts the same JSON schema for an existing, known-lineage sequence.
--train-data trains two levels from a prepared real-data specification.
The decoder reads ONLY Q0 + the written enhancement payload, never Q1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.content_preparation.paths import UPSTREAM_ROOT, configure_upstream_imports
from tools.content_preparation.compatibility import store_point_cloud


def dump(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(names):
    ordered = lambda prefix: sorted((n for n in names if n.startswith(prefix)),
                                   key=lambda n: int(n.rsplit('_', 1)[1]))
    return {"xyz": ["x", "y", "z"], "rotation": ordered("rot_"),
            "scale": ordered("scale_"), "opacity": ["opacity"],
            "sh": ordered("f_dc_") + ordered("f_rest_")}


def matrix(vertices, fields):
    return np.column_stack([vertices[name] for name in fields]).astype('<f4')


def read_vertices(path):
    data = PlyData.read(path)["vertex"].data.copy()
    assert len(data), f"Empty PLY: {path}"
    for fields in groups(data.dtype.names).values():
        assert np.isfinite(matrix(data, fields)).all(), path
    return data


def write_vertices(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(path))


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def attribute_stats(a, b, schema, tolerance):
    stats, any_exact, any_tol = {}, np.zeros(len(a), bool), np.zeros(len(a), bool)
    for name, fields in schema.items():
        av, bv = matrix(a, fields).astype('f8'), matrix(b, fields).astype('f8')
        diff = bv - av
        exact = np.any(av != bv, axis=1)
        above_tol = np.any(np.abs(diff) > tolerance, axis=1)
        any_exact |= exact
        any_tol |= above_tol
        row = dict(changed_count=int(exact.sum()), changed_pct=float(exact.mean()*100),
                   above_tolerance_pct=float(above_tol.mean()*100),
                   raw_component_mae=float(np.abs(diff).mean()),
                   raw_component_rms=float(np.sqrt(np.mean(diff**2))),
                   raw_l2=summary(np.linalg.norm(diff, axis=1)))
        if name == 'rotation':
            aq = av / np.linalg.norm(av, axis=1, keepdims=True)
            bq = bv / np.linalg.norm(bv, axis=1, keepdims=True)
            angle = np.degrees(2*np.arccos(np.clip(np.abs((aq*bq).sum(1)), 0, 1)))
            angle[~exact] = 0
            row['physical_rotation_degrees'] = summary(angle)
            row['physical_changed_pct_gt_1e_4_degrees'] = float((angle > 1e-4).mean()*100)
        elif name == 'scale':
            row['physical_scale_l2'] = summary(np.linalg.norm(np.exp(bv)-np.exp(av), axis=1))
        elif name == 'opacity':
            row['physical_alpha_abs'] = summary(np.abs(1/(1+np.exp(-bv))-1/(1+np.exp(-av))).ravel())
        elif name == 'sh':
            row['dc_rgb_l2_unclamped'] = summary(np.linalg.norm(.28209479177387814*diff[:, :3], axis=1))
        stats[name] = row
    return stats, float(any_exact.mean()*100), float(any_tol.mean()*100)


def encode(base, target, folder):
    """Sparse exact replacement values, uint32 row index + float32 attribute group."""
    folder.mkdir(parents=True, exist_ok=True)
    n = len(base)
    assert len(target) >= n
    schema = groups(base.dtype.names)
    fields = sum(schema.values(), [])
    # Normals are ignored by the renderer and are zero in official saved checkpoints.
    for data in [base, target]:
        assert all(np.all(data[k] == 0) for k in ['nx', 'ny', 'nz'])
    matrix(target[n:], fields).tofile(folder / 'new.bin')
    meta = dict(format='attribute-replacement-v1', endian='little', index_dtype='uint32',
                value_dtype='float32', shared_count=n, new_count=len(target)-n,
                schema=schema, new_fields=fields, correction_counts={})
    sizes = {}
    for name, names in schema.items():
        a, b = matrix(base, names), matrix(target[:n], names)
        changed = np.any(a.view('<u4') != b.view('<u4'), axis=1)
        records = np.empty(changed.sum(), dtype=[('id', '<u4'), ('value', '<f4', (len(names),))])
        records['id'] = np.flatnonzero(changed)
        records['value'] = b[changed]
        records.tofile(folder / f'{name}.bin')
        meta['correction_counts'][name] = int(changed.sum())
        sizes[name] = int(records.nbytes)
        assert (folder / f'{name}.bin').stat().st_size == records.nbytes
    dump(folder / 'enhancement.json', meta)
    return dict(new_bytes=(folder/'new.bin').stat().st_size,
                correction_bytes=sum(sizes.values()), correction_bytes_by_attribute=sizes,
                metadata_bytes=(folder/'enhancement.json').stat().st_size,
                bytes_per_new_gaussian=len(fields)*4,
                full_q1_render_attribute_bytes=len(target)*len(fields)*4)


def decode(base, folder, corrected_attributes=()):
    """Decoder deliberately has no target/checkpoint argument."""
    meta = json.loads((folder/'enhancement.json').read_text())
    n = meta['shared_count']
    assert len(base) == n
    result = np.concatenate([base.copy(), np.zeros(meta['new_count'], dtype=base.dtype)])
    new = np.fromfile(folder/'new.bin', dtype='<f4').reshape(meta['new_count'], len(meta['new_fields']))
    for col, field in enumerate(meta['new_fields']):
        result[field][n:] = new[:, col]
    for name in corrected_attributes:
        fields = meta['schema'][name]
        rows = np.fromfile(folder/f'{name}.bin',
                           dtype=[('id', '<u4'), ('value', '<f4', (len(fields),))])
        assert len(rows) == meta['correction_counts'][name]
        assert np.all(rows['id'] < n)
        assert len(np.unique(rows['id'])) == len(rows)
        for col, field in enumerate(fields):
            result[field][rows['id']] = rows['value'][:, col]
    return result


def run_command(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open('w') as f:
        f.write('COMMAND: '+json.dumps(command)+'\n'); f.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed: {log}\n{log.read_text()[-5000:]}')
    print(f'Completed {log.stem} in {time.monotonic()-start:.1f}s', flush=True)


def camera_pose(angle):
    eye = np.array([3.0*np.sin(angle), .65, 3.0*np.cos(angle)])
    back = eye / np.linalg.norm(eye)
    right = np.cross([0, 1, 0], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pose = np.eye(4); pose[:3, :3] = np.column_stack([right, up, back]); pose[:3, 3] = eye
    return pose


def objects(frame):
    out = []
    for i, (axes, center) in enumerate([([.40,.62,.30],[-.36,0,0]), ([.28,.36,.45],[.43,-.12,.05])]):
        angle = (.15 if i == 0 else -.25) + frame*(.09 if i == 0 else -.12)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        center = np.array(center) + [frame*(.035 if i == 0 else -.025), .015*np.sin(frame), 0]
        out.append((np.array(axes), center, rot))
    return out


def colors(local, index):
    stripe = np.sin(18*local[..., 0] + 5*local[..., 1]) > 0
    palette = [([.95,.22,.1],[.12,.6,.9]), ([.25,.85,.18],[.9,.7,.15])][index]
    return np.where(stripe[..., None], palette[0], palette[1])


def raytrace(pose, frame, size=128, fov=.8):
    """Independent analytic opaque ellipsoid ground truth; NOT a GS renderer."""
    y, x = np.mgrid[:size, :size]
    rays = np.stack([(2*(x+.5)/size-1)*np.tan(fov/2),
                     (1-2*(y+.5)/size)*np.tan(fov/2), -np.ones_like(x)], -1) @ pose[:3,:3].T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    depth = np.full((size,size), np.inf)
    image = np.zeros((size,size,3))
    light = np.array([.4,.8,1.0]); light /= np.linalg.norm(light)
    for i, (axes, center, rot) in enumerate(objects(frame)):
        origin = ((pose[:3,3]-center) @ rot) / axes
        direction = (rays @ rot) / axes
        a = (direction**2).sum(-1); b = 2*(direction*origin).sum(-1)
        c = (origin**2).sum()-1
        discriminant = b*b-4*a*c
        t = (-b-np.sqrt(np.maximum(discriminant,0)))/(2*a)
        hit = (discriminant>=0)&(t>0)&(t<depth)
        local = origin + t[...,None]*direction
        normal = (local/axes) @ rot.T
        normal /= np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        shade = .45+.55*np.maximum((normal*light).sum(-1),0)
        rgb = colors(local,i)*shade[...,None]
        image[hit] = rgb[hit]; depth[hit]=t[hit]
    return Image.fromarray(np.round(np.clip(image,0,1)*255).astype('uint8'))


def make_demo_data(out, frames, seed):
    storePly = store_point_cloud
    rng = np.random.default_rng(seed)
    points, rgb = [], []
    for i, (axes, center, rot) in enumerate(objects(0)):
        unit = rng.normal(size=(256,3)); unit /= np.linalg.norm(unit,axis=1,keepdims=True)
        points.append((unit*axes)@rot.T+center)
        rgb.append(colors(unit,i)*180)
    points, rgb = np.concatenate(points), np.concatenate(rgb)
    splits = {'train': np.arange(12)*2*np.pi/12, 'test': (np.arange(4)+.5)*2*np.pi/4}
    for t in range(frames):
        for split, angles in splits.items():
            camera_frames = []
            for idx, angle in enumerate(angles):
                pose = camera_pose(angle)
                image = raytrace(pose,t)
                file_path = f'{split}/r_{idx}'
                camera_frames.append(dict(file_path=file_path, transform_matrix=pose.tolist()))
                for q, size in [('Q0',64),('Q1',128)]:
                    target = out/'data'/q/f'{t:04d}'/f'{file_path}.png'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    image.resize((size,size),Image.Resampling.LANCZOS).save(target)
            for q in ['Q0','Q1']:
                folder = out/'data'/q/f'{t:04d}'
                dump(folder/f'transforms_{split}.json',dict(camera_angle_x=.8,frames=camera_frames))
        for q in ['Q0','Q1']:
            # Identical 512-point supplied initialization. Later frames load prior GS.
            storePly(str(out/'data'/q/f'{t:04d}'/'points3d.ply'),points,rgb)


def train_demo(out, args):
    for name in ['data', 'models', 'manifest.json']:
        if (out/name).exists():
            raise FileExistsError(f'Refusing to overwrite {out/name}. Use --manifest to re-evaluate, or a new --output.')
    make_demo_data(out, args.frames, args.seed)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_MODE='offline',
               PYTHONPATH=str(out/'vendor')+os.pathsep+str(ROOT)+os.pathsep+str(UPSTREAM_ROOT))
    manifest = dict(sequence='analytic_moving_ellipsoids', synthetic=True, seed=args.seed,
                    repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    matching='verified inherited prefix: row i in Q0 maps to row i in Q1',
                    frames=[], commands=[])
    previous = {}
    for frame in range(args.frames):
        entry = {'frame':frame, 'eval_source':str(out/'data/Q1'/f'{frame:04d}')}
        for q in ['Q0','Q1']:
            steps = args.initial_iterations if frame==0 else args.dynamic_iterations
            model = out/'models'/q/f'{frame:04d}'
            if model.exists():
                raise FileExistsError(f'Refusing to overwrite {model}. Use a new --output directory.')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=str(sock.getsockname()[1])
            cmd = [sys.executable,str(ROOT/'tools/content_preparation/backend.py'),'--entry',str(UPSTREAM_ROOT/'train.py'),'-s',str(out/'data'/q/f'{frame:04d}'),
                   '-m',str(model),'--iterations',str(steps),'--lambda_dssim','0.8','--eval',
                   '--save_iterations',str(steps),'--test_iterations',str(steps),'--port',port]
            if frame==0:
                cmd += ['--densify_until_iter',str(max(501,steps-200))]
                if q=='Q1':
                    cmd += ['--dynamic_opacity','--foundation_gs_path',entry['Q0']]
            else:
                cmd += ['--dynamic_lapis','--initial_gs_path',previous[q],
                        '--densify_from_iter',str(steps+1),'--densify_until_iter','0',
                        '--opacity_reset_interval',str(steps+1)]
            log=out/'logs'/f'train_{q}_{frame:04d}.log'
            print(f'Training {q} frame {frame}: {steps} iterations; {log}',flush=True)
            run_command(cmd,log,env)
            checkpoint=model/'point_cloud'/f'iteration_{steps}'/'point_cloud.ply'
            assert checkpoint.is_file()
            entry[q]=str(checkpoint)
            previous[q]=str(checkpoint)
            manifest['commands'].append(dict(frame=frame,level=q,argv=cmd,log=str(log),sha256=digest(checkpoint)))
        manifest['frames'].append(entry)
        dump(out/'manifest.json',manifest)
    return manifest


def verify_lineage(manifest):
    """Fail closed on unknown/reordered inputs: require the original training argv."""
    rows=manifest['frames']; first=rows[0]
    assert rows and len({r['frame'] for r in rows}) == len(rows)
    for record in manifest['commands']:
        row = next(r for r in rows if r['frame'] == record['frame'])
        assert digest(row[record['level']]) == record['sha256'], 'Checkpoint SHA256 mismatch'
    command_map={(r['frame'],r['level']):r['argv'] for r in manifest['commands']}
    def option(cmd,name):
        return cmd[cmd.index(name)+1] if name in cmd else None
    q0=read_vertices(first['Q0']); q1=read_vertices(first['Q1']); n=len(q0)
    assert len(q1)>=n
    initial=command_map[(first['frame'],'Q1')]
    assert Path(option(initial,'--foundation_gs_path')).resolve()==Path(first['Q0']).resolve()
    assert '--dynamic_opacity' in initial
    schema=groups(q0.dtype.names)
    # These groups are frozen in the inherited prefix during first-frame Q1 training.
    for name in ['xyz','rotation','scale','sh']:
        np.testing.assert_array_equal(matrix(q0,schema[name]),matrix(q1[:n],schema[name]),err_msg=name)
    previous={'Q0':q0,'Q1':q1}
    for before,row in zip(rows,rows[1:]):
        for q in ['Q0','Q1']:
            cmd=command_map[(row['frame'],q)]
            assert Path(option(cmd,'--initial_gs_path')).resolve()==Path(before[q]).resolve()
            assert '--dynamic_lapis' in cmd and option(cmd,'--densify_until_iter')=='0'
            assert int(option(cmd,'--opacity_reset_interval'))>int(option(cmd,'--iterations'))
            current=read_vertices(row[q])
            assert len(current)==len(previous[q]), f'Point count changed: {q}'
            for name in ['scale','opacity','sh']:
                np.testing.assert_array_equal(matrix(current,schema[name]),matrix(previous[q],schema[name]),err_msg=f'{q}.{name}')
            previous[q]=current
    return n


def train_real(out, args):
    """Only orchestrate official train.py; validate correspondence after EVERY frame."""
    if (out/'models').exists() or (out/'manifest.json').exists():
        raise FileExistsError('Refusing to overwrite real training. Use a new output or --manifest.')
    spec = json.loads(args.train_data.read_text())
    rows = spec['frames']
    assert not spec['synthetic'] and 5 <= len(rows) <= 10
    assert [r['frame'] for r in rows] == list(range(rows[0]['frame'], rows[0]['frame']+len(rows)))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_MODE='offline',
               PYTHONPATH=str(out/'vendor')+os.pathsep+str(ROOT)+os.pathsep+str(UPSTREAM_ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    manifest = dict(sequence=spec['sequence'], synthetic=False, data_spec=str(args.train_data.resolve()),
                    data_spec_sha256=digest(args.train_data), training_rng_seed=0,
                    repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    matching='inherited row prefix; checked immediately after each trained frame',
                    frames=[], commands=[])
    previous = {}
    for index, source in enumerate(rows):
        frame = source['frame']; steps = args.initial_iterations if index == 0 else args.dynamic_iterations
        entry = dict(frame=frame, eval_source=source['Q1_source'])
        for q in ['Q0','Q1']:
            model = out/'models'/q/f'{frame:04d}'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=str(sock.getsockname()[1])
            cmd = [sys.executable,str(ROOT/'tools/content_preparation/backend.py'),'--entry',str(UPSTREAM_ROOT/'train.py'),'-s',source[q+'_source'],'-m',str(model),
                   '--iterations',str(steps),'--lambda_dssim','0.8','--eval',
                   '--save_iterations',str(steps),'--test_iterations',str(steps),'--port',port]
            if index == 0 and q == 'Q1':
                cmd += ['--dynamic_opacity','--foundation_gs_path',entry['Q0']]
            elif index > 0:
                cmd += ['--dynamic_lapis','--initial_gs_path',previous[q],
                        '--densify_from_iter',str(steps+1),'--densify_until_iter','0',
                        '--opacity_reset_interval',str(steps+1)]
            log=out/'logs'/f'train_{q}_{frame:04d}.log'
            print(f'Training REAL {q} frame {frame}: {steps} iterations',flush=True)
            run_command(cmd,log,env)
            checkpoint=model/'point_cloud'/f'iteration_{steps}'/'point_cloud.ply'
            assert checkpoint.is_file()
            entry[q]=previous[q]=str(checkpoint)
            manifest['commands'].append(dict(frame=frame,level=q,argv=cmd,log=str(log),sha256=digest(checkpoint)))
        manifest['frames'].append(entry)
        dump(out/'manifest.json',manifest)
        try:
            shared = verify_lineage(manifest)
        except Exception as error:
            dump(out/'CORRESPONDENCE_FAILED.json',dict(frame=frame,error=str(error),action='STOP; no alternative matching attempted'))
            raise
        print(f'PASS prefix correspondence through frame {frame}: {shared} shared rows',flush=True)
    return manifest


def evaluate(manifest,out,tolerance):
    import torch
    configure_upstream_imports()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    import diff_gaussian_rasterization
    torch.set_num_threads(2)
    criterion=LPIPS(net_type='vgg').cuda().eval()
    n=verify_lineage(manifest)
    all_frames, per_view, attribute_rows=[],[],[]
    bg=torch.zeros(3,device='cuda')
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    def compare(a,b):
        mse=(a-b).square().mean().item()
        return {'psnr':float('inf') if mse==0 else -10*math.log10(mse),
                'ssim':ssim(a,b).item(), 'lpips':criterion(a[None],b[None]).item(),
                'max_abs_pixel':(a-b).abs().max().item()}
    for row in manifest['frames']:
        t=row['frame']; folder=out/'analysis'/f'{t:04d}'; folder.mkdir(parents=True,exist_ok=True)
        base,target=read_vertices(row['Q0']),read_vertices(row['Q1'])
        assert len(base)==n and base.dtype==target.dtype
        schema=groups(base.dtype.names)
        stats,changed_pct,tol_pct=attribute_stats(base,target[:n],schema,tolerance)
        sizes=encode(base,target,folder/'payload')
        sizes['correction_to_new_ratio'] = sizes['correction_bytes']/sizes['new_bytes'] if sizes['new_bytes'] else None
        variants={'Q0':base,'official':target,
                  'naive':decode(base,folder/'payload'),
                  'correct':decode(base,folder/'payload',schema),
                  'opacity_only':decode(base,folder/'payload',['opacity']),
                  'xyz_only':decode(base,folder/'payload',['xyz']),
                  'rotation_only':decode(base,folder/'payload',['rotation']),
                  'without_xyz':decode(base,folder/'payload',['opacity','rotation','scale','sh']),
                  'without_rotation':decode(base,folder/'payload',['opacity','xyz','scale','sh']),
                  'motion_only':decode(base,folder/'payload',['xyz','rotation'])}
        fields=sum(schema.values(),[])
        assert matrix(variants['correct'],fields).tobytes()==matrix(target,fields).tobytes()
        for name,values in variants.items():
            write_vertices(folder/f'{name}.ply',values)
        names=base.dtype.names
        degree=int(round(math.sqrt((len(schema['sh']))/3)-1))
        assert 3*(degree+1)**2==len(schema['sh'])
        models={}
        for name in variants:
            model=GaussianModel(degree); model.load_ply(str(folder/f'{name}.ply')); models[name]=model
        infos=readCamerasFromTransforms(row['eval_source'],'transforms_test.json',False)
        assert infos, 'Need held-out test cameras'
        cams=cameraList_from_camInfos(infos,1.0,SimpleNamespace(resolution=1,data_device='cuda'))
        local_views=[]
        with torch.no_grad():
            for i,cam in enumerate(cams):
                images={name:render(cam,model,pipe,bg)['render'].clamp(0,1) for name,model in models.items()}
                image_dir=folder/'renders'/f'{i:03d}'; image_dir.mkdir(parents=True,exist_ok=True)
                for name,image in dict(images,gt=cam.original_image).items():
                    Image.fromarray((image.permute(1,2,0).cpu().numpy()*255+.5).astype('uint8')).save(image_dir/f'{name}.png')
                for name,image in images.items():
                    metrics=dict(frame=t,view=i,variant=name)
                    for ref,reference in [('q1',images['official']),('gt',cam.original_image)]:
                        metrics.update({f'{ref}_{k}':v for k,v in compare(image,reference).items()})
                    per_view.append(metrics); local_views.append(metrics)
        average={}
        for name in variants:
            selected=[v for v in local_views if v['variant']==name]
            average[name]={k:float(np.mean([v[k] for v in selected])) for k in selected[0] if k not in ['frame','view','variant']}
        assert average['correct']['q1_max_abs_pixel']==0, 'Corrected rendering is not exact'
        for name,stat in stats.items():
            attribute_rows.append(dict(frame=t,attribute=name,changed_pct=stat['changed_pct'],
                above_tolerance_pct=stat['above_tolerance_pct'],raw_component_mae=stat['raw_component_mae'],
                raw_l2_mean=stat['raw_l2']['mean'],raw_l2_p95=stat['raw_l2']['p95'],raw_l2_max=stat['raw_l2']['max'],
                correction_bytes=sizes['correction_bytes_by_attribute'][name]))
        result=dict(frame=t,shared_count=n,new_count=len(target)-n,q1_count=len(target),
                    shared_changed_pct=changed_pct,shared_above_tolerance_pct=tol_pct,
                    attributes=stats,bytes=sizes,metrics=average,bitwise_correct_attributes=True)
        all_frames.append(result); dump(folder/'summary.json',result)
        print(f"Frame {t}: shared={n}, new={len(target)-n}, changed={changed_pct:.2f}%, "
              f"correction={sizes['correction_bytes']} B, naive-vs-Q1 PSNR={average['naive']['q1_psnr']:.3f}",flush=True)
        del models, images
        torch.cuda.empty_cache()
    result=dict(manifest=str(out/'manifest.json'),matching_verified=True,raw_absolute_tolerance=tolerance,
                metric_convention='float RGB [0,1], repo SSIM and lpipsPyTorch VGG on [0,1] as metrics.py; per-view dB averaged; common Q1 test cameras',
                byte_convention='uncompressed float32 new renderer attributes; uint32 ID + replacement float32 group for each changed shared row; JSON metadata reported separately',
                environment=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
                                 measurement_script_sha256=digest(__file__),
                                 gpu=torch.cuda.get_device_name(0),rasterizer=diff_gaussian_rasterization.__file__),
                frames=all_frames)
    dump(out/'summary.json',result)
    def csv_write(path,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    csv_write(out/'per_view.csv',per_view)
    csv_write(out/'attributes.csv',attribute_rows)
    flat=[]
    for row in all_frames:
        for name,metrics in row['metrics'].items():
            flat.append(dict(frame=row['frame'],variant=name,shared_count=row['shared_count'],new_count=row['new_count'],
                             shared_changed_pct=row['shared_changed_pct'],new_bytes=row['bytes']['new_bytes'],
                             correction_bytes=row['bytes']['correction_bytes'],
                             correction_to_new_ratio=row['bytes']['correction_to_new_ratio'],**metrics))
    csv_write(out/'summary.csv',flat)
    return result


def self_test():
    import tempfile
    fields=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
            'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    a=np.zeros(3,dtype=[(n,'<f4') for n in fields]); a['rot_0']=1
    b=np.concatenate([a.copy(),a[:1].copy()]); b['opacity'][1]=.3; b['x'][2]=.05
    b['f_dc_2'][3]=.7
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp); sizes=encode(a,b,path)
        assert sizes['new_bytes']==56 and sizes['correction_bytes']==24
        assert decode(a,path,groups(a.dtype.names)).tobytes()==b.tobytes()
        naive=decode(a,path)
        assert naive[:3].tobytes()==a.tobytes() and naive[3:].tobytes()==b[3:].tobytes()
        identity=path/'identity'; sizes=encode(a,np.concatenate([a,a[:1]]),identity)
        assert sizes['correction_bytes']==0
        no_new=path/'no_new'; sizes=encode(a,a,no_new)
        assert sizes['new_bytes']==sizes['correction_bytes']==0
        assert decode(a,no_new,groups(a.dtype.names)).tobytes()==a.tobytes()
        sign=a.copy(); sign['rot_0']=-1
        stats, _, _=attribute_stats(a,sign,groups(a.dtype.names),1e-6)
        assert stats['rotation']['changed_pct']==100
        assert stats['rotation']['physical_rotation_degrees']['max']==0
    print('PASS exact roundtrip, sparse bytes, identity/zero-new controls, naive prefix, quaternion sign equivalence',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/progressive_gap')
    p.add_argument('--run-demo',action='store_true')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--train-data',type=Path,help='Prepared real-data specification from prepare_progressive_real.py')
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--initial-iterations',type=int,default=3000)
    p.add_argument('--dynamic-iterations',type=int,default=800)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test(); return
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    vendor=out/'vendor'
    if vendor.is_dir():
        sys.path.insert(0,str(vendor))
    if args.run_demo:
        assert args.frames>=2
        manifest=train_demo(out,args)
    elif args.train_data:
        manifest=train_real(out,args)
    elif args.manifest:
        manifest=json.loads(args.manifest.read_text())
        if args.manifest.resolve() != (out/'manifest.json').resolve():
            dump(out/'manifest.json',manifest)
    else:
        p.error('Provide --run-demo, --train-data, or --manifest (or --self-test)')
    evaluate(manifest,out,args.tolerance)
    print(f'Experiment complete: {out}/summary.json',flush=True)


if __name__=='__main__':
    main()
