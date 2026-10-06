#!/usr/bin/env python3
"""Bounded GPU-0 kernel and installed-extension provenance verification."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/readiness/environment.json'))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Run through bash scripts/run_gpu0.sh; physical GPU 0 is required.')
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability(0) == (12, 0)
    prefix = Path(sys.prefix).resolve()
    extensions = {}
    for name in ('diff_gaussian_rasterization._C', 'simple_knn._C'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(prefix), f'Extension outside the new environment: {path}'
        extensions[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    a = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
    torch.testing.assert_close(a @ a.T, (a.cpu() @ a.cpu().T).cuda())
    from simple_knn._C import distCUDA2
    points = torch.randn(32, 3, device='cuda')
    distances = distCUDA2(points)
    expected = torch.cdist(points.cpu(), points.cpu()).square()
    expected.fill_diagonal_(float('inf'))
    torch.testing.assert_close(distances.cpu(), expected.topk(3, largest=False).values.mean(1), rtol=1e-4, atol=1e-5)
    torch.cuda.synchronize()
    def command(argv):
        return subprocess.check_output(argv, text=True).strip()
    props = torch.cuda.get_device_properties(0)
    driver_uuid = command(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'])
    normalize_uuid = lambda value: str(value).lower().replace('gpu-', '').replace('-', '')
    assert normalize_uuid(props.uuid) == normalize_uuid(driver_uuid), 'CUDA device differs from physical GPU 0'
    record = {
        'status': 'passed', 'physical_gpu': 0, 'python': sys.version.split()[0],
        'python_executable': sys.executable, 'torch': torch.__version__,
        'torch_cuda': torch.version.cuda, 'gpu': props.name,
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'gpu_uuid': driver_uuid,
        'vram_bytes': props.total_memory,
        'driver': command(['nvidia-smi', '-i', '0', '--query-gpu=driver_version', '--format=csv,noheader']),
        'nvcc': command(['nvcc', '--version']),
        'compiler': command(['/usr/bin/g++', '--version']).splitlines()[0],
        'cmake': command(['cmake', '--version']).splitlines()[0],
        'extensions': extensions,
        'checks': ['CUDA matrix multiply', 'simple-knn native CUDA vs CPU nearest-neighbor reference'],
        'renderer_gate': 'Run scripts/smoke_test.py and tools/benchmark_content_runtime.py separately.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    from tools.content_preparation.checkpoint import file_lock
    with file_lock(Path(tempfile.gettempdir()) / f'content-preparation-gpu-{os.getuid()}.lock'):
        main()
