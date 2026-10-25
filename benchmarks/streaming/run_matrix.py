#!/usr/bin/env python3
"""Cold HTTP-to-EGL playback matrix with shared-body network emulation.

Use a CUDA-compatible Python environment. No physical display claims are made.
Existing trial outputs are never overwritten. Every case has its own cache.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from multiobj3dgs.content.publication import publish
from multiobj3dgs.server.origin import create_server
from multiobj3dgs.server.emulation import BodyPacer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',action='append',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--objects',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--mbps',nargs='+',type=float,default=[20,80])
    p.add_argument('--workers',nargs='+',type=int,default=[1,2])
    p.add_argument('--prefetch',nargs='+',type=int,default=[0,1])
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--latency-ms',type=float,default=20)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--height',type=int,default=512)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for count in args.objects:
        root=args.output/f'content-n{count}';publish(args.archive,root,instances=count)
        for rate in args.mbps:
            for workers in args.workers:
                for lookahead in args.prefetch:
                    tag=f'n{count}-{rate:g}mbps-w{workers}-ahead{lookahead}'
                    pacer=BodyPacer([dict(at_s=0,bps=rate*1e6,latency_s=args.latency_ms/1000)])
                    server=create_server(root,port=0,pacer=pacer)
                    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                    command=[args.python,'-m','multiobj3dgs.client','--mpd',f'http://127.0.0.1:{server.server_port}/manifest.mpd',
                        '--mode','headless','--frames',str(args.frames),'--workers',str(workers),'--prefetch',str(lookahead),
                        '--width',str(args.width),'--height',str(args.height),'--cache',str(args.output/(tag+'-cache')),
                        '--report',str(args.output/(tag+'.json'))]
                    try:
                        with (args.output/(tag+'.log')).open('x') as log:
                            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                        cases.append(dict(case=tag,returncode=completed.returncode,argv=command,objects=count,mbps=rate,
                                          cache='cold independent cache',presentation='EGL completion',workers=workers,prefetch=lookahead))
                    finally:
                        server.shutdown();thread.join();server.server_close()
                    (args.output/'matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
    return int(any(c['returncode'] for c in cases))

if __name__=='__main__':raise SystemExit(main())
