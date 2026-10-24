"""Publish or validate compressed StreamSTGS content without loading the codec."""
import argparse
from dataclasses import asdict
import json
from .publication import publish, validate_publication
from .streamstgs.archive import validate_archive
from ..shared.metadata.serialization import strict_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('publish')
    build.add_argument('--archive',action='append',required=True,help='Repeat for each archive')
    build.add_argument('--output',required=True)
    selection=build.add_mutually_exclusive_group()
    selection.add_argument('--instances',type=int)
    selection.add_argument('--scene',help='JSON containing duration, instances, optional pivots')
    validate=commands.add_parser('validate'); validate.add_argument('path')
    audit=commands.add_parser('audit'); audit.add_argument('archive',nargs='+')
    options=parser.parse_args(argv)
    if options.command=='audit':
        result=[asdict(validate_archive(p)) for p in options.archive]
    else:
        if options.command=='publish':
            from pathlib import Path
            scene=strict_json(Path(options.scene).read_bytes()) if options.scene else None
            catalog=publish(options.archive,options.output,instances=options.instances,scene=scene)
        else: catalog=validate_publication(options.path)
        result={'revision':catalog.revision,'contents':list(catalog.contents),'instances':len(catalog.instances),'duration':catalog.duration}
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
