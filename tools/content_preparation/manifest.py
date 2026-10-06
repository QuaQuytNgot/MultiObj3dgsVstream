"""Machine-readable streaming index for one prepared volumetric object."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

MANIFEST_VERSION = "content-preparation.manifest.v1"
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def build_manifest(object_id: str, frames: list[dict], qualities: list,
                   package_index: dict, provenance: dict | None = None,
                   canonical_transform: dict | list | None = None,
                   proxy_path: str | None = None, quality_profile_path: str | None = None,
                   output_dir: Path | None = None) -> dict:
    """Assemble a deterministic manifest; hashes cover actual files, never estimates."""
    from .validation import safe_relative_path, validate_manifest
    if not frames:
        raise ValueError("A manifest requires a frame/time mapping")
    canonical = canonical_transform if canonical_transform is not None else {"matrix": IDENTITY, "scale": 1.0}
    assets = []
    for relative in (proxy_path, quality_profile_path):
        if relative:
            safe_relative_path(relative)
            if output_dir is not None:
                path = Path(output_dir) / relative
                assets.append({"path": relative, "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema_version": MANIFEST_VERSION,
                "object": {"id": object_id, "frames": copy.deepcopy(frames),
                           "frame_range": [frames[0]["frame"], frames[-1]["frame"]],
                           "time_range": [frames[0]["timestamp"], frames[-1]["timestamp"]],
                           "canonical_transform": copy.deepcopy(canonical),
                           "proxy_path": proxy_path, "quality_profile_path": quality_profile_path},
                "quality_order": [q["id"] if isinstance(q, dict) else q for q in qualities],
                "delivery_modes": package_index["delivery_modes"],
                "representations": copy.deepcopy(package_index["representations"]),
                "package": copy.deepcopy(package_index), "assets": assets,
                "provenance": copy.deepcopy(provenance or {}),
                "accounting": {"media_bytes": package_index["media_bytes"],
                               "definition": "actual complete requestable segment files, including container and codec headers",
                               "auxiliary_asset_bytes": sum(a["bytes"] for a in assets)},
                "quality_reference": "highest-quality decoded deliverable representation",
                "planner_distortion": "mse"}
    validate_manifest(manifest, output_dir)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    from .validation import validate_manifest
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf8")
    os.replace(temporary, path)


def load_manifest(path: Path, validate_files: bool = False) -> dict:
    from .validation import validate_manifest
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf8"))
    validate_manifest(manifest, path.parent if validate_files else None)
    return manifest
