#!/usr/bin/env python3
"""Prepare deliverable content; no online ABR/client/scheduler is implemented."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.content_preparation.config import STAGES,configure_runtime,read_config,resolve_path,validate_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--runtime-config",type=Path,help="Execution-only machine settings; research fields are rejected")
    parser.add_argument("--stage",choices=("all",)+STAGES,default="all")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--overwrite",action="store_true",help="Recompute conflicting pipeline-owned artifacts; never deletes arbitrary output directories")
    args=parser.parse_args(argv)
    try:
        raw=read_config(args.config,args.runtime_config)
        configure_runtime(raw)
        config=validate_config(raw)
        extension=resolve_path(config["runtime"]["extension_path"]) if config["runtime"].get("extension_path") else None
        if extension is not None:
            if not extension.is_dir():
                raise FileNotFoundError(f"Configured extension_path does not exist: {extension}")
            sys.path.insert(0,str(extension))
        from tools.content_preparation.pipeline import Pipeline
        result=Pipeline(config,args.config,args.resume,args.overwrite).run(args.stage,args.dry_run)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError) as error:
        print(f"content preparation: {error}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
