"""Separate project artifacts from pinned upstream source and installed binaries."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "dynamic-lapis-gs"


def require_upstream():
    if not (UPSTREAM_ROOT / "train.py").is_file():
        raise FileNotFoundError(
            f"Dynamic-LapisGS is not initialized at {UPSTREAM_ROOT}. "
            "Run git submodule update --init --recursive from the project root."
        )
    return UPSTREAM_ROOT


def configure_upstream_imports(extension_path=None):
    """Add upstream Python modules lazily; extension binaries stay environment-owned."""
    upstream = require_upstream()
    paths = [upstream]
    if extension_path:
        extension = Path(extension_path).expanduser()
        if not extension.is_absolute():
            extension = PROJECT_ROOT / extension
        if not extension.is_dir():
            raise FileNotFoundError(f"Configured extension_path does not exist: {extension}")
        paths.insert(0, extension.resolve())
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return upstream
