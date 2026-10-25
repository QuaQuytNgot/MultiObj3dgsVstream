"""All aligned GOP-boundary CRF transitions with independent model handles.

Functional decoder regression, not a playback throughput benchmark. The separate
runtime/control-trace tests exercise scheduling and activation of these handles.
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mpd',required=True,type=Path);p.add_argument('--report',required=True,type=Path);args=p.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import torch
    from multiobj3dgs.client.metadata import MPDParser
    from multiobj3dgs.content.streamstgs.materialize import materialize_gop
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.shared.geometry import Camera
    catalog=MPDParser.from_local(args.mpd);decoder=StreamSTGSAdapter()
    report=dict(scope='all representation pairs at aligned GOP boundaries; standalone target reference',cases=[])
    def load(segment,directory):
        package=materialize_gop(segment,{r.id:Path(r.url) for r in segment.resources},directory)
        prepared=decoder.prepare_cpu(package);handle=decoder.realize_gpu(prepared)
        del prepared
        return handle
    for content in catalog.contents.values():
        reps=tuple(content.representations.values())
        for boundary in range(1,len(reps[0].segments)):
            targets={};expected={}
            with tempfile.TemporaryDirectory(prefix='streamstgs-switch-') as temporary:
                root=Path(temporary)
                try:
                    for rep in reps:
                        segment=rep.segments[boundary]
                        handle=load(segment,root/('target-'+rep.id));targets[rep.id]=handle
                        camera=Camera.from_package(handle.cameras[0])
                        expected[rep.id]=decoder.reconstruct(handle,segment.frames[0].model_time,camera)
                    for previous in reps:
                        old_segment=previous.segments[boundary-1]
                        old=load(old_segment,root/('source-'+previous.id))
                        try:
                            old_state=decoder.reconstruct(old,old_segment.frames[-1].model_time,Camera.from_package(old.cameras[0]));del old_state
                            for target in reps:
                                segment=target.segments[boundary]
                                actual=decoder.reconstruct(targets[target.id],segment.frames[0].model_time,Camera.from_package(targets[target.id].cameras[0]))
                                for name in ('xyz','scales','rotations','opacities','colors'):
                                    torch.testing.assert_close(getattr(actual,name),getattr(expected[target.id],name),rtol=1e-6,atol=1e-6)
                                report['cases'].append(dict(content=content.id,boundary=boundary,previous=previous.id,target=target.id,status='passed'))
                                del actual
                        finally:decoder.release(old)
                finally:
                    expected.clear()
                    for handle in targets.values():decoder.release(handle)
                    targets.clear();gc.collect()
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
            print(content.id,boundary,'all transitions passed',flush=True)
    report['status']='passed';args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
