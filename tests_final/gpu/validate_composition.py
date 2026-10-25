"""Synthetic translucent depth-swap and >3-object contribution regression."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True,type=Path);args=p.parse_args()
    import torch
    from multiobj3dgs.shared.geometry import Camera
    from multiobj3dgs.shared.gaussian import GaussianTensorState,concatenate
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    from multiobj3dgs.client.visibility.analyzer import VisibilityAnalyzer
    renderer=FASTRenderer();analyzer=VisibilityAnalyzer()
    view=np.eye(4);view[2,3]=3
    camera=Camera(256,256,view,np.array([[220.,0,127.5],[0,220.,127.5],[0,0,1]]))
    def state(x,z,color,empty=False):
        n=0 if empty else 1
        return GaussianTensorState(torch.tensor([[x,0,z]],device='cuda').reshape(1,3)[:n],
            torch.full((n,3),.12,device='cuda'),torch.tensor([[1.,0,0,0]],device='cuda')[:n],
            torch.full((n,1),.8,device='cuda'),torch.tensor([color],device='cuda',dtype=torch.float32)[:n])
    objects={'red':state(0.,0.,[1,0,0]),'blue':state(0.,.4,[0,0,1])}
    with torch.inference_mode():
        measured=analyzer.analyze(objects,camera,0.)
        assert measured.per_object['red'].occlusion_ratio < measured.per_object['blue'].occlusion_ratio
        first=renderer.draw(concatenate(objects.values()),camera,readback=True).image
        assert float(first[0,128,128])>float(first[2,128,128])
        swapped={'red':state(0.,.4,[1,0,0]),'blue':state(0.,0.,[0,0,1])}
        second=renderer.draw(concatenate(swapped.values()),camera,readback=True).image
        assert float(second[2,128,128])>float(second[0,128,128])
        objects.update(extra=state(.6,0.,[0,1,0]),outside=state(100.,0.,[1,1,0]),empty=state(0.,0.,[1,1,1],True))
        groups=analyzer.analyze(objects,camera,0.)
        assert groups.metadata['joint_passes']==2 and groups.metadata['solo_passes']==4
        assert groups.per_object['outside'].out_of_view and groups.per_object['empty'].out_of_view
        for item in groups.per_object.values():assert 0<=item.A_visible<=item.A_solo+1e-4
    args.report.parent.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    Image.fromarray((first.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(args.report.with_suffix('.png'))
    args.report.write_text(json.dumps(dict(status='passed',scope='synthetic translucent joint FAST depth-swap; CUDA RGB-basis contribution groups',
        physical_display_fps=None,objects={k:vars(v) for k,v in groups.per_object.items()},metadata=groups.metadata),indent=2)+'\n')
    renderer.close()

if __name__=='__main__':main()
