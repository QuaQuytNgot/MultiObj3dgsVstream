"""Optional offline XSD validation using the repository's pinned MPEG resources."""
from importlib.resources import files, as_file
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess


def validate_mpd_schema(path):
    """Require xmllint explicitly; never access remote schemas or affect playback."""
    executable=shutil.which('xmllint')
    if executable is None: raise RuntimeError('XSD validation requires xmllint (libxml2-utils)')
    resources=files(__package__).joinpath('schemas')
    with as_file(resources) as directory:
        directory=Path(directory)
        # Hashes are embedded in the package, so a changed schema cannot silently
        # turn an offline consistency check into a different profile validator.
        expected=json.loads((directory/'checksums.json').read_text())
        for name,digest in expected.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Pinned schema checksum mismatch: {name}')
        env=dict(os.environ,XML_CATALOG_FILES=str(directory/'catalog.xml'))
        result=subprocess.run([executable,'--nonet','--noout','--schema',str(directory/'DASH-MPD.xsd'),str(Path(path).resolve())],env=env,capture_output=True,text=True)
        if result.returncode: raise ValueError('MPD XSD validation failed: '+result.stderr.strip())
    return True
