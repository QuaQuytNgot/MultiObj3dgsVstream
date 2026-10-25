"""Compare reconstruction and pixels with the pinned upstream render function.

Extract only stream_render_eval from a local upstream checkout, avoiding its
training-only imports. The function body is executed unchanged.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--legacy-fingerprints',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    import torch
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.shared.geometry import Camera
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    source=subprocess.check_output(['git','-C',str(args.upstream),'show',
        'ca8602516352efd83e9e69e665b2e26ed03cb5cc:gaussian_renderer/__init__.py']).decode()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='stream_render_eval')
    module=ast.Module(body=[function],type_ignores=[])
    decoder=StreamSTGSAdapter();host=decoder.prepare_cpu(args.package);handle=decoder.realize_gpu(host)
    renderer=FASTRenderer();captured={}
    class CaptureRaster:
        def __init__(self,raster_settings):
            self.inner=renderer.fast.GaussianRasterizer(raster_settings)
        def __call__(self,**kwargs):
            captured.update(kwargs)
            return self.inner(**kwargs)
    namespace=dict(torch=torch,math=math,fastSettings=renderer.fast.GaussianRasterizationSettings,fastRaster=CaptureRaster)
    exec(compile(module,'pinned-upstream:stream_render_eval','exec'),namespace)
    old=json.loads(args.legacy_fingerprints.read_text())
    actual={k:hashlib.sha256(getattr(handle.model,k).cpu().numpy().tobytes()).hexdigest() for k in old if k!='temporal_features'}
    assert all(actual[k]==old[k] for k in actual), {k:(old[k],actual[k]) for k in actual if old[k]!=actual[k]}
    temporal=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in handle.model._point_feats)).hexdigest()
    cases=[]
    with torch.inference_mode():
        for index in (0,29,59):
            camera=Camera.from_package(host.cameras[index%len(host.cameras)])
            t=index/handle.model.time_line
            state=decoder.reconstruct(handle,t,camera)
            image=renderer.draw(state,camera,readback=True).image
            view=camera.to_torch();view.time=t
            reference=namespace['stream_render_eval'](view,handle.model,SimpleNamespace(debug=False),renderer.bg)
            errors={k:float((v-captured[k]).abs().max()) for k,v in state.raster_args().items() if v is not None and k!='means2D'}
            assert max(errors.values())==0,errors
            image_error=float((image-reference).abs().max())
            assert image_error<=1/255+1e-6,image_error
            cases.append(dict(frame=index,attribute_max_errors=errors,image_max_error=image_error))
    args.report.write_text(json.dumps(dict(status='passed',canonical_sha256_exact=actual,
        corrected_temporal_sha256=temporal,historical_temporal_sha256=old['temporal_features'],
        historical_temporal_match=temporal==old['temporal_features'],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),cases=cases),indent=2)+'\n')
    renderer.close();decoder.release(handle)


if __name__=='__main__':main()
