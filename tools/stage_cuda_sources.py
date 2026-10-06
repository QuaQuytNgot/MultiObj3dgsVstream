#!/usr/bin/env python3
"""Stage clean pinned CUDA sources on the artifact mount before local builds."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(path, *arguments):
    return subprocess.check_output(['git', '-C', str(path), *arguments], text=True).strip()


def main():
    upstream = ROOT / 'third_party/dynamic-lapis-gs'
    sources = upstream / 'submodules'
    output = ROOT / 'output/build/cuda_sources'
    output.mkdir(parents=True, exist_ok=True)
    report = {'upstream_commit': git(upstream, 'rev-parse', 'HEAD'), 'extensions': {}}
    for name in ('simple-knn', 'diff-gaussian-rasterization'):
        source, target = sources / name, output / name
        if git(source, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError(f'Pinned source is modified: {source}')
        if target.exists():
            raise FileExistsError(f'Build source copy already exists: {target}; use its recorded provenance or a fresh output mount.')
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            '.git', 'build', 'dist', '*.egg-info', '__pycache__', '*.so', '*.o', '*.a'))
        files = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(target.rglob('*')) if path.is_file()}
        report['extensions'][name] = {
            'commit': git(source, 'rev-parse', 'HEAD'), 'source': str(source),
            'build_source': str(target.resolve()), 'source_file_sha256': files,
            'nested_submodules': git(source, 'submodule', 'status', '--recursive')}
    destination = ROOT / 'output/readiness/cuda_source_provenance.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Staged pinned sources: {output.resolve()}')
    print(f'Provenance: {destination.resolve()}')


if __name__ == '__main__':
    main()
