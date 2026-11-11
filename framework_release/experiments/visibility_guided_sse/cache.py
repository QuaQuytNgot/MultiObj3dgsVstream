"""Content-addressed references to compatible artifacts; never copies image arrays.

Importing this module cannot create a renderer.  Completion markers and payloads
are hashed before reuse.  The native adapter delegates scene/frame validation to
the existing NAVA validators rather than reinterpreting historical identities.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping
import math

from experiments.nava_mse.common import digest, read_json, sha_file, write_json


IDENTITY_SCHEMA = "visibility-guided-sse.snapshot.v1"
REGISTRY_SCHEMA = "visibility-guided-sse.registry.v1"
QUALITIES = (20, 28, 36, 40)


def _present(value: Any, name: str) -> None:
    if value is None or value == {} or value == "":
        raise ValueError(f"Missing {name} identity")
    # Strict canonical JSON also rejects NaN/Infinity and unsupported values.
    digest(value)


def snapshot_identity(sample: Mapping[str, Any], allocation: Mapping[str, int], *,
                      kind: str, renderer_identity: Any, content_identity: Any,
                      gop: int = 0, proxy_protocol: Any = None) -> dict[str, Any]:
    """Describe semantics, excluding trace labels so identical snapshots dedup.

    ``kind='joint'`` is shared by reference and selected-allocation jobs.
    ``kind='solo_alpha'`` may reference an existing CRF40 solo RGB artifact,
    because its independently acquired native alpha is the desired quantity.
    No geometry-only trace may be promoted to native validity by this factory;
    the caller must select native-valid samples before constructing GPU jobs.
    """
    if kind not in ("joint", "object", "solo_alpha", "proxy"):
        raise ValueError("Unsupported snapshot kind")
    if isinstance(gop, bool) or not isinstance(gop, int) or gop < 0:
        raise ValueError("GOP must be a nonnegative integer")
    timestamp = sample.get("time", sample.get("timestamp_s", sample.get("timestamp")))
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("An exact finite nonnegative sample time is required")
    _present(renderer_identity, "renderer")
    _present(content_identity, "content")
    camera = deepcopy(sample["camera"])
    if not {"width", "height", "K", "world_to_camera", "near", "far"}.issubset(camera):
        raise ValueError("Complete camera intrinsics/extrinsics/clipping identity required")
    instances = []
    for row in sample["instances"]:
        required = ("instance_id", "object_id", "object_to_world", "source_camera")
        if any(k not in row for k in required):
            raise ValueError("Complete instance/source-camera/transform identity required")
        item = {key: deepcopy(row[key]) for key in required}
        item["conditioning_mode"] = row.get("conditioning_mode", "exact_pose")
        instances.append(item)
    # Concatenation order is part of native Gaussian rasterization semantics:
    # equal-depth ties can depend on the declared order.  Never sort instances.
    ids = [row["instance_id"] for row in instances]
    if not ids or len(ids) != len(set(ids)) or set(allocation) != set(ids):
        raise ValueError("Allocation must contain every independent instance exactly once")
    if any(isinstance(q, bool) or not isinstance(q, int) or q not in QUALITIES for q in allocation.values()):
        raise ValueError("Unknown CRF representation")
    if kind in ("object", "solo_alpha") and len(ids) != 1:
        raise ValueError("A solo snapshot must contain exactly one instance")
    if kind == "proxy":
        _present(proxy_protocol, "independent proxy protocol")
        if set(allocation.values()) != {40}:
            raise ValueError("Visibility proxy must be independently fixed at CRF40")
    elif proxy_protocol is not None:
        raise ValueError("Proxy protocol cannot label a target RGB snapshot")
    result = dict(schema=IDENTITY_SCHEMA, kind=kind, gop=gop,
                  timestamp_s=float(timestamp), camera=camera, instances=instances,
                  allocation=dict(sorted(allocation.items())),
                  renderer_identity=deepcopy(renderer_identity),
                  content_identity=deepcopy(content_identity),
                  proxy_protocol=deepcopy(proxy_protocol))
    digest(result)
    return result


def validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("Unsupported snapshot identity")
    rebuilt = snapshot_identity(identity, identity["allocation"], kind=identity["kind"],
        renderer_identity=identity["renderer_identity"], content_identity=identity["content_identity"],
        gop=identity["gop"], proxy_protocol=identity.get("proxy_protocol"))
    if rebuilt != identity:
        raise ValueError("Snapshot identity is incomplete or noncanonical")
    return rebuilt


def content_identity_from_frozen(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Hash-bound source and representation bank shared by all policy jobs.

    Physical paths and budgets are deliberately absent; all source packages,
    representation manifests, component hashes and timeline samples are bound.
    """
    inputs = frozen["inputs"]
    sources = {oid: {key: value for key, value in source.items()
                     if key in ("archive_sha256", "package_manifest_sha256")}
               for oid, source in inputs["sources"].items()}
    representations = {key: {field: deepcopy(value[field]) for field in
                              ("manifest_sha256", "components", "samples")}
                       for key, value in inputs["representations"].items()}
    result = dict(schema="visibility-guided-sse.content-bank.v1", sources=sources,
                  representations=representations)
    digest(result)
    return result


def renderer_identity_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    runtime = metadata.get("runtime", {})
    if not metadata.get("renderer") or not runtime.get("rasterizer_files_sha256"):
        raise ValueError("Native renderer and rasterizer binary hashes are required")
    return dict(renderer=metadata["renderer"],
                rasterizer_files_sha256=deepcopy(runtime["rasterizer_files_sha256"]),
                torch=runtime.get("torch"), cuda=runtime.get("cuda"),
                gpu=runtime.get("gpu"), matmul_allow_tf32=runtime.get("matmul_allow_tf32"),
                cudnn_allow_tf32=runtime.get("cudnn_allow_tf32"),
                normalization=metadata.get("normalization"), data_range=metadata.get("data_range"),
                background=deepcopy(metadata.get("background")), layout=metadata.get("layout"))


class CacheRegistry:
    """A local registry of external immutable artifacts, with fail-closed reuse.

    Each lookup hashes both marker and payload.  Inventories can count historical
    markers without pretending those unvalidated files are registered/reusable.
    The surrounding invocation lock serializes mutations.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "cache_registry.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return dict(schema=REGISTRY_SCHEMA, entries={})
        result = read_json(self.path)
        if result.get("schema") != REGISTRY_SCHEMA or not isinstance(result.get("entries"), dict):
            raise ValueError("Invalid cache registry")
        return result

    def lookup(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        canonical = validate_identity(identity)
        entry = self._read()["entries"].get(digest(canonical))
        if entry is None:
            return None
        if entry.get("identity") != canonical:
            raise ValueError("Registry semantic identity mismatch")
        for descriptor in entry["files"]:
            path = Path(descriptor["path"])
            if (not path.is_file() or path.stat().st_size != descriptor["bytes"]
                    or sha_file(path) != descriptor["sha256"]):
                raise ValueError(f"Missing or changed registered artifact: {path}")
        return deepcopy(entry)

    def register(self, identity: Mapping[str, Any], marker_path: str | Path,
                 arrays_path: str | Path | None = None, *,
                 validator: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
        canonical = validate_identity(identity)
        marker_path = Path(marker_path).resolve()
        arrays_path = Path(arrays_path).resolve() if arrays_path is not None else marker_path.with_suffix(".npz")
        marker = read_json(marker_path)
        if validator is None:
            if marker.get("snapshot_identity") != canonical:
                raise ValueError("Completion marker must bind the exact semantic snapshot identity")
        else:
            validator(marker)
        arrays_hash = sha_file(arrays_path)
        if marker.get("arrays_sha256") != arrays_hash:
            raise ValueError("Completion marker does not match payload hash")
        files = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha_file(path))
                 for path in (marker_path, arrays_path)]
        result = dict(identity=canonical, marker_path=str(marker_path), arrays_path=str(arrays_path),
                      files=files, status="COMPLETE", arrays_sha256=arrays_hash)
        saved = self._read()
        key = digest(canonical)
        existing = saved["entries"].get(key)
        if existing is not None:
            # An exact identity may have multiple equivalent physical stores;
            # retain the first verified artifact rather than move old results.
            self.lookup(canonical)
            if existing["arrays_sha256"] != arrays_hash:
                raise ValueError("Conflicting payload for an existing snapshot identity")
            return existing
        saved["entries"][key] = result
        write_json(self.path, saved)
        return deepcopy(result)

    def register_native_artifact(self, identity: Mapping[str, Any], marker_path: str | Path,
                                 *, frozen: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, Any]:
        """Reuse NAVA/occlusion native markers through their original validators."""
        from experiments.nava_mse.rendering import validate_render_job
        from experiments.nava_mse.workflow import validate_frozen_image
        from experiments.occlusion_joint_sse.rendering import image_job as occlusion_job

        canonical = validate_identity(identity)
        if canonical["content_identity"] != content_identity_from_frozen(frozen):
            raise ValueError("Requested content bank differs from frozen native inputs")

        def check(marker: Mapping[str, Any]) -> None:
            if marker.get("schema") == "nava-mse.image.v1":
                if canonical["kind"] == "proxy":
                    raise ValueError("A joint RGB artifact cannot supply independent visibility-proxy maps")
                if marker.get("run_signature") != frozen["freeze_sha256"]:
                    raise ValueError("Native marker belongs to a different frozen run")
                job, metadata = marker["job"], marker["render_metadata"]
            elif marker.get("schema") == "occlusion.arrays.v1":
                if canonical["kind"] != "proxy" or marker.get("kind") != "proxy":
                    raise ValueError("Proxy array may only fulfill a visibility-proxy job")
                if marker.get("run_identity") != frozen["freeze_sha256"]:
                    raise ValueError("Proxy belongs to a different frozen run")
                saved = marker["metadata"]
                if saved.get("quality") != 40 or saved.get("purpose") != "LOW_COST_OCCLUSION_PROXY":
                    raise ValueError("Proxy acquisition is not independent CRF40")
                metadata = saved["render_metadata"]
                job = occlusion_job(scene, canonical["timestamp_s"], canonical["allocation"])
                if canonical["proxy_protocol"].get("backend") != "occlusion_joint_sse":
                    raise ValueError("Old proxy protocol differs from requested analyzer protocol")
            else:
                raise ValueError("Unsupported historical native completion marker")
            expected_instances = canonical["instances"]
            requested_ids = {row["instance_id"] for row in expected_instances}
            declared_order = []
            for row in scene["instances"]:
                if row["instance_id"] in requested_ids:
                    declared_order.append({key: deepcopy(row[key]) for key in
                        ("instance_id", "object_id", "object_to_world", "source_camera")})
                    declared_order[-1]["conditioning_mode"] = row.get("conditioning_mode", "exact_pose")
            if declared_order != expected_instances:
                raise ValueError("Requested native concatenation order differs from the frozen source scene")
            if canonical["camera"] != scene["camera"]:
                raise ValueError("Requested camera differs from reuse scene")
            if (job["allocation"] != canonical["allocation"]
                    or job["timestamp_s"] != canonical["timestamp_s"]):
                raise ValueError("Requested quality/time differs from native marker")
            if canonical["kind"] in ("object", "solo_alpha") and job["kind"] != "object":
                raise ValueError("Solo-alpha acquisition requires a real solo artifact")
            if canonical["kind"] == "joint" and job["kind"] != "joint":
                raise ValueError("Joint evaluation requires a true joint artifact")
            if renderer_identity_from_metadata(metadata) != canonical["renderer_identity"]:
                raise ValueError("Requested native renderer differs from cached artifact")
            validate_render_job(metadata, job, canonical["gop"], expected_camera=scene["camera"],
                                expected_instances=expected_instances)
            validate_frozen_image(frozen, dict(metadata=metadata))
            if metadata.get("interactive_viewer_used") is not False:
                raise ValueError("Native headless renderer provenance is required")

        return self.register(canonical, marker_path, validator=check)

    def inventory(self) -> dict[str, Any]:
        saved = self._read()
        entries = list(saved["entries"].values())
        return dict(registered_snapshots=len(entries),
                    registered_payload_bytes=sum(row["files"][1]["bytes"] for row in entries),
                    copied_image_payloads=0,
                    entries={key: dict(kind=row["identity"]["kind"], marker_path=row["marker_path"])
                             for key, row in saved["entries"].items()})


def bounded_marker_inventory(roots: list[str | Path], *, max_markers: int = 10000) -> dict[str, Any]:
    """Metadata-only availability inventory, never a cache-validity declaration."""
    if isinstance(max_markers, bool) or not isinstance(max_markers, int) or max_markers <= 0:
        raise ValueError("Finite positive marker inventory limit required")
    rows, observed = [], 0
    for root in roots:
        root = Path(root)
        kinds: dict[str, int] = {}
        size = 0
        truncated = False
        for folder, schema in ((root / "images" / "object", "nava-mse.image.v1"),
                               (root / "images" / "joint", "nava-mse.image.v1"),
                               (root / "arrays" / "proxy", "occlusion.arrays.v1")):
            if not folder.is_dir():
                continue
            for marker in sorted(folder.glob("*.json")):
                if observed >= max_markers:
                    truncated = True
                    break
                observed += 1
                value = read_json(marker)
                if value.get("schema") != schema:
                    raise ValueError(f"Unrecognized artifact marker: {marker}")
                kind = "proxy" if schema == "occlusion.arrays.v1" else value["job"]["kind"]
                kinds[kind] = kinds.get(kind, 0) + 1
                arrays = marker.with_suffix(".npz")
                if arrays.is_file():
                    size += arrays.stat().st_size
            if truncated:
                break
        rows.append(dict(root=str(root), observed_markers=sum(kinds.values()), kinds=kinds,
                         observed_payload_bytes=size, truncated=truncated,
                         validation="AVAILABILITY_ONLY_NOT_REGISTERED"))
    return dict(roots=rows, observed_markers=observed, payloads_hashed=0,
                copied_image_payloads=0, native_jobs_launched=0)
