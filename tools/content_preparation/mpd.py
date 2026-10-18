"""Experimental Gaussian DASH description built from actual prepared containers.

CPSEG is a project container, not ISO-BMFF. The XML describes layer delivery for
a project decoder; it does not claim ordinary DASH video player compatibility.
Detailed access, codec, hashes and quality records remain in JSON sidecars.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .checkpoint import Journal, file_lock, sha256, write_json
from .manifest import load_manifest
from .validation import safe_relative_path

from src.content_mpd import (
    VERSION, NAMESPACE, PROFILE, PROPERTY, UINT32_MAX,
    EXTENDED_BANDWIDTH_PROPERTY, EXTENDED_BANDWIDTH_SCHEME, INDEX_SCHEMA,
    build_mpd, _prefix,
)
from src.content_metadata import (
    MAX_DOCUMENT_BYTES, density_reference, digest, json_bytes, profile_reference,
    relative_path, validate_object_metadata,
)

SCHEMAS = Path(__file__).with_name("schemas")


def _root_path(root: Path, relative: str) -> Path:
    safe_relative_path(relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping prepared asset: {relative}")
    return path


def _metadata_file(root: Path, relative: str) -> tuple[Path, bytes, dict | list]:
    relative_path(relative)
    if Path(relative).suffix.lower() != ".json":
        raise ValueError("Only JSON metadata may be read by metadata export")
    path = _root_path(root, relative)
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Metadata document exceeds limit: {relative}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    return path, raw, json_bytes(raw)


def load_prepared(root: Path, *, metadata_only=False,
                  density_profiles: dict[str, str] | None = None) -> tuple[list[dict], list[Path]]:
    """Check JSON sources; full mode additionally verifies actual payload files.

    Metadata-only mode never reads media, encoded streams, images or NPZ files.
    Density publication is explicit, so a discovered subset is never promoted
    automatically to a complete planner profile.
    """
    root = Path(root).resolve()
    total_read = 0

    def read(relative):
        nonlocal total_read
        relative_path(relative)
        candidate = _root_path(root, relative)
        if total_read + candidate.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Metadata bundle exceeds total size limit")
        result = _metadata_file(root, relative)
        total_read += len(result[1])
        if total_read > 128 * 1024 * 1024:
            raise ValueError("Metadata bundle exceeds total size limit")
        return result

    server_path, _, server = read("manifest.json")
    density_profiles = dict(density_profiles or {})
    if server.get("schema_version") != "content-preparation.server.v1" or not server.get("objects"):
        raise ValueError("Expected a nonempty content-preparation server index")
    objects, inputs, ids = [], [server_path], set()
    for record in server["objects"]:
        path, raw, manifest = read(record["manifest_path"])
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Server index object manifest bytes/hash mismatch")
        if not metadata_only:
            load_manifest(path, validate_files=True)
        if manifest["object"]["id"] != record["id"] or record["id"] in ids:
            raise ValueError("Server object identity is duplicated or inconsistent")
        if manifest["delivery_modes"] != ["progressive"]:
            raise ValueError("MPD export currently requires progressive-only preparation")
        if manifest["package"]["temporal_prediction"]:
            raise ValueError("This MPD exporter only describes independently initialized temporal frames")
        ids.add(record["id"])
        directory = path.parent.relative_to(root).as_posix()
        package_relative = _prefix(directory, "package_index.json")
        package_path, package_raw, package = read(package_relative)
        validate_object_metadata(manifest, package)
        inputs += [path, package_path]
        if not metadata_only:
            for segment in manifest["package"]["segments"]:
                inputs.append(_root_path(root, _prefix(directory, segment["path"])))
        for asset in manifest.get("assets", []):
            relative = _prefix(directory, asset["path"])
            if Path(relative).suffix.lower() == ".json":
                asset_path, asset_raw, _ = read(relative)
                if len(asset_raw) != asset["bytes"] or digest(asset_raw) != asset["sha256"]:
                    raise ValueError("Metadata asset bytes/hash mismatch")
                inputs.append(asset_path)
            elif not metadata_only:
                inputs.append(_root_path(root, relative))
        loaded = {"manifest": manifest, "manifest_path": record["manifest_path"],
                        "manifest_sha256": record["sha256"], "manifest_bytes": record["bytes"],
                        "directory": directory, "package_path": package_relative,
                        "package_sha256": digest(package_raw), "package_bytes": len(package_raw)}
        if manifest["object"].get("quality_profile_path"):
            relative = _prefix(directory, manifest["object"]["quality_profile_path"])
            profile_path, profile_raw, profile = read(relative)
            assets = {a["path"]: a for a in manifest.get("assets", [])}
            asset = assets.get(manifest["object"]["quality_profile_path"])
            if not asset or asset["bytes"] != len(profile_raw) or asset["sha256"] != digest(profile_raw):
                raise ValueError("Quality profile lacks matching manifest asset hash")
            loaded["quality_profile"] = profile_reference(relative, profile_raw, profile, manifest)
            inputs.append(profile_path)
        if record["id"] in density_profiles:
            if "quality_profile" not in loaded:
                raise ValueError("Density requires an authoritative source MSE profile")
            relative = density_profiles[record["id"]]
            density_path, density_raw, density = read(relative)
            loaded["density_profile"] = density_reference(relative, density_raw, density, profile,
                                                         loaded["quality_profile"], manifest)
            inputs.append(density_path)
        objects.append(loaded)
    if set(density_profiles) - ids:
        raise ValueError("Density profile references an unknown object")
    return objects, sorted(set(inputs))


def validate_mpd_schema(path: Path, required=True) -> dict:
    """Validate with pinned schemas offline; never fetch XML imports at runtime."""
    sources = json.loads((SCHEMAS / "sources.json").read_text())
    for descriptor in sources["files"]:
        if sha256(SCHEMAS / descriptor["path"]) != descriptor["sha256"]:
            raise ValueError("Pinned MPD validation schema hash mismatch")
    executable = shutil.which("xmllint")
    if executable is None:
        if required:
            raise RuntimeError("MPD schema validation requires xmllint; install libxml2 tools or expose the conda binary")
        return {"validated": False, "reason": "xmllint unavailable"}
    env = dict(os.environ, XML_CATALOG_FILES=str(SCHEMAS / "catalog.xml"))
    process = subprocess.run([executable, "--nonet", "--noout", "--schema", str(SCHEMAS / "DASH-MPD.xsd"),
                              str(path)], capture_output=True, text=True, env=env)
    if process.returncode:
        raise ValueError("MPD schema validation failed: " + process.stderr.strip())
    return {"validated": True, "validator": "xmllint", "offline": True,
            "mpeg_dash_commit": sources["mpeg_dash_commit"], "schema_sha256": sha256(SCHEMAS / "DASH-MPD.xsd")}


def _atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        name = temporary.name
    os.replace(name, path)


def validate_mpd(path: Path, prepared_root: Path | None = None, *, metadata_only=False,
                 density_profiles: dict[str, str] | None = None) -> dict:
    """Validate XML schema and agreement with actual source files and sidecar."""
    path = Path(path).resolve()
    root = Path(prepared_root).resolve() if prepared_root is not None else path.parent
    if path.parent != root:
        raise ValueError("MPD must be a direct child of the prepared root")
    objects, _ = load_prepared(root, metadata_only=metadata_only, density_profiles=density_profiles)
    expected_xml, expected_index = build_mpd(objects)
    if path.read_bytes() != expected_xml:
        raise ValueError("MPD URLs/timing/dependencies/bandwidth disagree with prepared manifests")
    expected_index["mpd"] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    index = json.loads((root / "content_index.json").read_text())
    if index != expected_index:
        raise ValueError("Content index disagrees with source manifests or actual MPD")
    report = validate_mpd_schema(path)
    return {**report, "passed": True, "object_count": len(objects),
            "media_bytes": sum(obj["manifest"]["accounting"]["media_bytes"] for obj in objects),
            "temporal_prediction": False, "exact_files": not metadata_only,
            "payload_integrity": "NOT_RUN" if metadata_only else "verified"}


def export_mpd(prepared_root: Path, output: Path | None = None, resume=False, overwrite=False,
               *, metadata_only=False, density_profiles: dict[str, str] | None = None) -> dict:
    """Publish XML and sidecars under the pipeline's lock, with a separate journal."""
    root = Path(prepared_root).resolve()
    output = Path(output).resolve() if output is not None else root / "manifest.mpd"
    if output.parent != root:
        raise ValueError("MPD output must be a direct child of the prepared root so URLs remain root-relative")
    if output.name in {"manifest.json", "content_index.json", "mpd_validation.json"}:
        raise ValueError("MPD output collides with an existing input or reserved sidecar")
    content_path, validation_path = root / "content_index.json", root / "mpd_validation.json"
    journal_root = root / ".preparation" / "mpd"
    owner_path = journal_root / "owner.json"
    identity = {"schema_version": "content-preparation.mpd-owner.v1", "output": output.name,
                "content_index": content_path.name, "validation": validation_path.name,
                "metadata_only": bool(metadata_only), "density_profiles": dict(density_profiles or {})}
    if not root.is_dir():
        raise FileNotFoundError(root)
    with file_lock(root / ".preparation" / "run.lock"):
        if owner_path.exists():
            previous_owner = json.loads(owner_path.read_text())
            if previous_owner != identity:
                owned_paths = ("schema_version", "output", "content_index", "validation")
                if not overwrite or any(previous_owner.get(k) != identity[k] for k in owned_paths):
                    raise FileExistsError("MPD export ownership/options differ; use --overwrite only for owned outputs")
        elif any(path.exists() for path in (output, content_path, validation_path)):
            raise FileExistsError("Refusing to overwrite unowned MPD outputs")
        objects, inputs = load_prepared(root, metadata_only=metadata_only, density_profiles=density_profiles)
        write_json(owner_path, identity)
        journal = Journal(root, resume=resume, overwrite=overwrite)
        journal.path = journal_root / "journal.json"
        journal.data = json.loads(journal.path.read_text()) if journal.path.exists() else {"version": 1, "tasks": {}, "stages": {}}
        implementation = [Path(__file__), Path(__file__).parents[2] / "src/content_mpd.py",
                          Path(__file__).parents[2] / "src/content_metadata.py", *sorted(SCHEMAS.glob("*"))]
        inputs += implementation

        def action():
            encoded, index = build_mpd(objects)
            _atomic_bytes(output, encoded)
            validation = validate_mpd_schema(output)
            index["mpd"] = {"path": output.name, "bytes": output.stat().st_size, "sha256": sha256(output)}
            write_json(content_path, index)
            validation.update(passed=True, mpd=index["mpd"],
                              content_index={"path": content_path.name, "bytes": content_path.stat().st_size, "sha256": sha256(content_path)},
                              media_bytes=sum(obj["manifest"]["accounting"]["media_bytes"] for obj in objects),
                              object_count=len(objects), decoded_reference="highest-quality decoded deliverable",
                              exact_files=not metadata_only,
                              payload_integrity="NOT_RUN" if metadata_only else "verified",
                              metadata_only=bool(metadata_only),
                              temporal_prediction=False, writer_version=VERSION)
            write_json(validation_path, validation)
            result = {"mpd": output.name, "content_index": content_path.name,
                      "validation": validation_path.name, "object_count": len(objects), "schema_validated": True}
            return result, [output, content_path, validation_path]

        result = journal.run("mpd/export", identity, inputs, action,
                             command="export_content_mpd.py", version=VERSION)
        return {**result, "tasks_executed": journal.executed, "tasks_skipped": journal.skipped}
