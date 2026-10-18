#!/usr/bin/env python3
"""Write an experimental progressive Gaussian MPD from prepared manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.mpd import export_mpd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Read JSON only; do not open or verify media/encoded/image payloads")
    parser.add_argument("--density-profile", action="append", default=[], metavar="OBJECT=RELATIVE_JSON",
                        help="Explicit density sidecar path relative to prepared-root; repeat per object")
    arguments = parser.parse_args(argv)
    densities = {}
    for value in arguments.density_profile:
        object_id, separator, path = value.partition("=")
        if not separator or not object_id or not path or object_id in densities:
            parser.error("--density-profile requires unique OBJECT=RELATIVE_JSON entries")
        densities[object_id] = path
    result = export_mpd(arguments.prepared_root, arguments.output, arguments.resume, arguments.overwrite,
                        metadata_only=arguments.metadata_only, density_profiles=densities)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
