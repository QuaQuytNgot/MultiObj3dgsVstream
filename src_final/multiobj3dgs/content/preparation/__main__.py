"""Execute explicit external preparation stages or verify camera calibration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from ...shared.metadata.serialization import strict_json
from .runner import StageRunner,StageSpec


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('config');run.add_argument('--resume',action='store_true');run.add_argument('--dry-run',action='store_true')
    calibration=commands.add_parser('calibration');calibration.add_argument('config');calibration.add_argument('--poses-bounds')
    args=parser.parse_args(argv);path=Path(args.config).resolve();config=strict_json(path.read_bytes())
    if args.command=='run':
        if config.get('schema')!='multiobj3dgs.preparation.v1':raise ValueError('Unsupported stage config')
        runner=StageRunner(path.parent/config['journal'],resume=args.resume)
        names=[s['name'] for s in config['stages']]
        if len(set(names))!=len(names):raise ValueError('Duplicate stage names')
        results=[runner.run(StageSpec.from_dict(s,path.parent),dry_run=args.dry_run) for s in config['stages']]
    else:
        from .calibration import validate_camera,validate_reprojection,export_poses_bounds
        cameras=config['cameras']
        for camera in cameras:validate_camera(camera)
        results=[asdict(validate_reprojection(cameras[o['camera']],o['world_points'],o['image_points'],tolerance_pixels=config.get('tolerance_pixels',.01))) for o in config.get('observations',[])]
        if args.poses_bounds:
            if {o['camera'] for o in config.get('observations',[])}!=set(range(len(cameras))):raise ValueError('Measured reprojection observations required for every camera before exporting poses_bounds')
            export_poses_bounds(args.poses_bounds,cameras,config['near_fars'])
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
