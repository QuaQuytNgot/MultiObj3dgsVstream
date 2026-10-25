#!/usr/bin/env python3
"""Bounded real-package validation. Run legacy and new backends in separate processes.

Legacy mode is explicitly a regression harness, never imported by production.
Historical Soldier/Longdress references contain the atlas transpose bug and are
expected to fail temporal/state parity after its correction. Use
validate_publication.py's independent picture oracle and validate_upstream.py
for correctness; do not regenerate this reference with the same broken trial.
"""
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np


def fingerprints(pc):
    names=['_xyz','_features_dc','_scaling','_rotation','_opacity']
    return {name:hashlib.sha256(getattr(pc,name).detach().cpu().numpy().tobytes()).hexdigest() for name in names}|{'temporal_features':hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in pc._point_feats)).hexdigest()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package',required=True,type=Path)
    ap.add_argument('--legacy-scripts',type=Path)
    ap.add_argument('--reference',required=True,type=Path)
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    from multiobj3dgs.shared.geometry import Camera
    if args.legacy_scripts:
        sys.path.insert(0,str(args.legacy_scripts))
        from soldier_runtime import load_gop, bench, camera_from_package
        meta=json.loads((args.package/'reconstruction.json').read_text())
        loaded=load_gop(meta['segment']['gop_index'],meta['crf'],args.package.parents[2])
        pc=loaded['model'];camera=camera_from_package(loaded['camera_list'][0]);camera.time=0
        state=bench.state(pc,camera)
        import torch
        image=bench.raster(pc,camera,state,torch.ones(3,device='cuda'))
        args.reference.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.reference,**{k:v.detach().cpu().numpy() for k,v in state.items() if v is not None},image=image.detach().cpu().numpy())
        args.reference.with_name(args.reference.stem+'.fingerprints.json').write_text(json.dumps(fingerprints(pc),indent=2))
        return
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    decoder=StreamSTGSAdapter();host=decoder.prepare_cpu(args.package);handle=decoder.realize_gpu(host)
    assert fingerprints(handle.model)==json.loads(args.reference.with_name(args.reference.stem+'.fingerprints.json').read_text()),'CPU/GPU decoded tensors differ from legacy'
    camera=Camera.from_package(host.cameras[0]);state=decoder.reconstruct(handle,0,camera)
    renderer=FASTRenderer();receipt=renderer.draw(state,camera,readback=True)
    ref=np.load(args.reference)
    errors={k:float(np.abs(v.detach().cpu().numpy()-ref[k]).max()) for k,v in state.raster_args().items() if v is not None}
    errors['image']=float(np.abs(receipt.image.cpu().numpy()-ref['image']).max())
    if max(v for k,v in errors.items() if k!='image')>1e-6 or errors['image']>1/255+1e-6:raise AssertionError(errors)
    report={'package':str(args.package),'tensors_exact_sha256_match':True,'state_tolerance':1e-6,'image_tolerance':1/255+1e-6,'image_precision':'FAST uint8 framebuffer; calibrated camera float32 rounding may move a quantization boundary','max_abs_errors':errors,'stages':handle.stages,'host_bytes':host.nbytes,'memory_reservations':decoder.estimate_memory(args.package),'headless_draw_s':receipt.elapsed_s,'physical_display_fps':None}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));renderer.close();decoder.release(handle)

if __name__=='__main__':main()
