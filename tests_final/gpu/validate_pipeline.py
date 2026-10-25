#!/usr/bin/env python3
"""Progressive CPU/H2D pipeline parity against independent complete-GOP decode.

Requires actual CUDA/EGL. Both paths preserve the current corrected atlas layout;
the historical Soldier trial is deliberately not used as a temporal oracle.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time


def fingerprints(prepared):
    canonical={name:hashlib.sha256(value.tobytes()).hexdigest()
               for name,value in prepared.attributes.items()}
    features=hashlib.sha256()
    for array in prepared.features:
        if array is None:raise AssertionError('Missing completed CPU temporal feature')
        features.update(array.tobytes())
    return dict(canonical=canonical,temporal=features.hexdigest(),features=len(prepared.features))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpd',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--crfs',type=int,nargs='+',default=[20,40])
    parser.add_argument('--gops',type=int,nargs='+',default=[0,4])
    parser.add_argument('--all-gops',action='store_true')
    parser.add_argument('--packages',type=int,help='Limit package count for a bounded smoke')
    parser.add_argument('--timeout',type=float,default=90)
    args=parser.parse_args(argv)
    if args.report.exists():raise FileExistsError(args.report)
    if args.packages is not None and args.packages<1:parser.error('--packages must be positive')
    import torch
    from multiobj3dgs.client.metadata import MPDParser
    from multiobj3dgs.content.streamstgs.materialize import materialize_gop
    from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
    from multiobj3dgs.shared.geometry import Camera
    from multiobj3dgs.client.rendering.fast import FASTRenderer
    catalog=MPDParser.from_local(args.mpd)
    decoder=StreamSTGSAdapter()
    renderer=FASTRenderer(mode='headless')
    report=dict(schema='multiobj3dgs.progressive_validation.v1',status='running',
        mpd=str(args.mpd),revision=catalog.revision,gpu=torch.cuda.get_device_name(),
        scope='independent complete-GOP versus progressive CPU/copy-stream decode; exact state and EGL image parity',
        physical_display_fps=None,cases=[])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
    selected=[]
    for content in catalog.contents.values():
        for representation in content.representations.values():
            if representation.crf not in args.crfs:continue
            for segment in representation.segments:
                if args.all_gops or segment.gop_index in args.gops:
                    selected.append((content,representation,segment))
    if args.packages is not None:selected=selected[:args.packages]
    if not selected:raise ValueError('No packages matched requested CRF/GOP selection')
    try:
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='pipeline-validation') as pool:
            for content,representation,segment in selected:
                case=dict(content=content.id,crf=representation.crf,gop=segment.gop_index,status='running')
                started=time.monotonic();oracle=None;pending=None;progressive=None;producer=None
                cancel=threading.Event();references=[];host=None;completed=None;temporary=None
                try:
                    temporary=tempfile.TemporaryDirectory(prefix='streamstgs-pipeline-parity-')
                    package=materialize_gop(segment,{r.id:Path(r.url) for r in segment.resources},Path(temporary.name)/'package')
                    torch.cuda.reset_peak_memory_stats()
                    host=decoder.prepare_cpu(package)
                    expected_cpu=fingerprints(host)
                    oracle=decoder.realize_gpu(host)
                    camera=Camera.from_package(host.cameras[0])
                    indices=sorted(set((0,len(segment.frames)//2,len(segment.frames)-1)))
                    with torch.inference_mode():
                        for index in indices:
                            mapping=segment.frames[index]
                            state=decoder.reconstruct(oracle,mapping.model_time,camera)
                            image=renderer.draw(state,camera,readback=True).image
                            references.append((index,mapping,state,image))
                    del host;host=None
                    published,producer=decoder.start_cpu(package,cancel,pool)
                    early=published.result(timeout=args.timeout)
                    pending=decoder.begin_gpu(early)
                    deadline=time.monotonic()+args.timeout
                    case['frames']=[];case['unavailable_frame_rejection_checked']=False
                    for index,mapping,expected_state,expected_image in references:
                        while progressive is None or not decoder.frame_ready(progressive,mapping.model_time):
                            if producer.done():producer.result()
                            if time.monotonic()>deadline:raise TimeoutError('Progressive frame readiness timed out')
                            value=decoder.poll_gpu(pending,budget_s=.004)
                            if value is not None:progressive=value
                            if progressive is None or not decoder.frame_ready(progressive,mapping.model_time):time.sleep(.0005)
                        if index==indices[0]:
                            case['first_window_before_producer_complete']=not producer.done()
                            case['available_features_at_first_frame']=progressive.available_features
                            assert progressive.model._xyz.data_ptr()!=oracle.model._xyz.data_ptr()
                            assert progressive.model._point_feats[0].data_ptr()!=oracle.model._point_feats[0].data_ptr()
                            # A tensor may be allocated before its copy event.
                            # Reconstruction must reject every unavailable window.
                            future=references[-1][1].model_time
                            if not decoder.frame_ready(progressive,future):
                                try:decoder.reconstruct(progressive,future,camera)
                                except (RuntimeError,ValueError):case['unavailable_frame_rejection_checked']=True
                                else:raise AssertionError('Read an unavailable temporal feature window')
                        with torch.inference_mode():
                            actual=decoder.reconstruct(progressive,mapping.model_time,camera)
                            errors={}
                            for attribute in ('xyz','scales','rotations','opacities','colors'):
                                value,expected=getattr(actual,attribute),getattr(expected_state,attribute)
                                torch.testing.assert_close(value,expected,rtol=0,atol=0)
                                errors[attribute]=float((value-expected).abs().max())
                            image=renderer.draw(actual,camera,readback=True).image
                            image_error=float((image-expected_image).abs().max())
                            assert image_error<=1/255+1e-6,image_error
                        case['frames'].append(dict(frame=index,source_frame=mapping.source_frame,
                            available_features=progressive.available_features,
                            attribute_max_errors=errors,image_max_error=image_error))
                    while not pending.done:
                        if producer.done():producer.result()
                        if time.monotonic()>deadline:raise TimeoutError('Complete progressive upload timed out')
                        decoder.poll_gpu(pending,budget_s=.004)
                        if not pending.done:time.sleep(.0005)
                    completed=producer.result(timeout=args.timeout)
                    actual_cpu=fingerprints(completed)
                    assert actual_cpu==expected_cpu,'CPU canonical/temporal fingerprints differ'
                    assert progressive.available_features==len(completed.features)
                    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(
                        progressive.model._point_feats,oracle.model._point_feats))
                    case.update(status='passed',cpu_fingerprints=actual_cpu,
                        independent_gpu_storage=True,producer_drained=True,upload_drained=True,
                        progressive_stages=dict(progressive.stages),
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                except BaseException as error:
                    case.update(status='failed',error=repr(error))
                    raise
                finally:
                    cancel.set()
                    if progressive is not None and not progressive.released:decoder.release(progressive)
                    if pending is not None:decoder.cancel_gpu(pending)
                    if producer is not None:
                        try:producer.result()
                        except BaseException:pass
                    if oracle is not None and not oracle.released:decoder.release(oracle)
                    references.clear();host=completed=early=None;pending=progressive=oracle=producer=published=None
                    actual=image=state=expected_state=expected_image=value=expected=None
                    if temporary is not None:temporary.cleanup()
                    gc.collect()
                    case['elapsed_s']=time.monotonic()-started
                    report['cases'].append(case);save()
                    print(content.id,representation.id,segment.gop_index,case['status'],flush=True)
        report['status']='passed';save()
    except BaseException:
        report['status']='failed';save()
        raise
    finally:renderer.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
