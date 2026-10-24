"""Private inference namespace: never import upstream's generic scene/utils packages."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

PREFIX='_multiobj3dgs_streamstgs'


def vendor_root(explicit=None):
    if explicit is not None:
        root=Path(explicit)
    elif os.environ.get('MULTIOBJ3DGS_VENDOR_ROOT'):
        root=Path(os.environ['MULTIOBJ3DGS_VENDOR_ROOT'])
    else:
        root=Path(__file__).resolve().parents[4]/'third_party/streamstgs'
    if not (root/'manifest.json').is_file():
        raise RuntimeError('StreamSTGS source missing: run scripts/setup_streamstgs.py or set MULTIOBJ3DGS_VENDOR_ROOT')
    return root.resolve()


def verify_vendor(root=None):
    root=vendor_root(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Vendor source integrity mismatch: {name}')
    return manifest


def model_class(root=None):
    root=vendor_root(root)
    existing=sys.modules.get(PREFIX)
    if existing is not None and Path(existing.__path__[0]).resolve()!=root:
        raise RuntimeError('A different StreamSTGS vendor revision is already loaded')
    for suffix in ('','scene','utils'):
        key=PREFIX+('.'+suffix if suffix else '')
        if key not in sys.modules:
            module=types.ModuleType(key); module.__path__=[str(root/suffix)]
            module.__package__=key;sys.modules[key]=module
    return importlib.import_module(PREFIX+'.scene.streamoff_gaussian_model').GaussianModel
