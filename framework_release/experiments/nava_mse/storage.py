"""Hash-bound resumable raw image storage, with atomic completion markers."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import numpy as np
from .common import digest, read_json, sha_file, write_json


class ArtifactStore:
    def __init__(self, root, run_signature):
        self.root = Path(root)
        self.run_signature = run_signature

    def paths(self, job):
        if job.get('kind') not in ('object', 'joint'):
            raise ValueError('Invalid image job kind')
        key = digest({'run': self.run_signature, 'job': job})
        folder = self.root / 'images' / job['kind']
        return folder / (key + '.npz'), folder / (key + '.json')

    def load(self, job):
        arrays, marker = self.paths(job)
        if not marker.exists():
            return None
        metadata = read_json(marker)
        if metadata['run_signature'] != self.run_signature or metadata['job'] != job:
            raise ValueError('Cached image belongs to a different render context')
        if not arrays.is_file() or sha_file(arrays) != metadata['arrays_sha256']:
            raise ValueError(f'Incomplete or corrupted cached artifact: {arrays}')
        with np.load(arrays, allow_pickle=False) as archive:
            value = {name: archive[name].copy() for name in ('rgb', 'raw_rgb', 'alpha')}
        value['metadata'] = metadata['render_metadata']
        value['provenance'] = {'arrays_path': str(arrays), 'arrays_sha256': metadata['arrays_sha256'],
                               'metadata_path': str(marker)}
        return value

    def save(self, job, result):
        existing = self.load(job)
        if existing is not None:
            return existing
        arrays, marker = self.paths(job)
        arrays.parent.mkdir(parents=True, exist_ok=True)
        values = {name: np.asarray(getattr(result, name)) for name in ('rgb', 'raw_rgb', 'alpha')}
        rgb, raw, alpha = values['rgb'], values['raw_rgb'], values['alpha']
        if (rgb.ndim != 3 or rgb.shape[-1] != 3 or raw.shape != rgb.shape
            or alpha.shape != rgb.shape[:2] or any(not np.isfinite(v).all() for v in values.values())
            or (rgb < 0).any() or (rgb > 1).any() or (alpha < 0).any() or (alpha > 1).any()):
            raise ValueError('Invalid HWC RGB/alpha render result')
        with tempfile.NamedTemporaryFile(dir=arrays.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, **values)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, arrays)
        write_json(marker, dict(schema='nava-mse.image.v1', run_signature=self.run_signature,
                                job=job, arrays_sha256=sha_file(arrays), render_metadata=result.metadata))
        return self.load(job)


@contextmanager
def exclusive_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.writer.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'Run has another writer or stale lock: {lock}; remove only after confirming no writer is active') from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def save_preview(path, rgb):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
