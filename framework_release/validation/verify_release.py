"""Check packaged bytes and optional original/external inputs; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--release-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--original-root', type=Path)
    parser.add_argument('--asset', action='append', default=[], metavar='GROUP=ROOT')
    args = parser.parse_args()
    root = args.release_root.resolve()
    copied = json.loads((root / 'provenance/COPIED_FILES.json').read_text())['files']
    failures = []
    checked = 0

    def check(base, row):
        nonlocal checked
        path = base / row['path']
        checked += 1
        if not path.is_file() or sha256(path) != row['sha256']:
            failures.append(str(path))

    for row in copied:
        check(root, row)
        if args.original_root:
            check(args.original_root, row)
    assets = json.loads((root / 'provenance/EXTERNAL_ASSETS.json').read_text())
    for value in args.asset:
        group, separator, base = value.partition('=')
        if not separator or group not in assets['groups']:
            parser.error('Use GROUP=ROOT with a group from provenance/EXTERNAL_ASSETS.json')
        for row in assets['files']:
            if row['group'] == group:
                check(Path(base), row)
    print(json.dumps({'status': 'failed' if failures else 'passed', 'files_checked': checked,
                      'mismatches_or_missing': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
