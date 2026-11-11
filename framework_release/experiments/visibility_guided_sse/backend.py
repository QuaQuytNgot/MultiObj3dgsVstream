"""Lazy Task-2 native rendering and independent Q0 visibility acquisition.

Task 1 imports/tests this module with injected CPU fixtures only.  Native work
starts exclusively when an explicitly supplied executor is called.  Unsupported
target viewpoints may receive geometry/visibility analysis of CRF40 states
activated at the declared recorded source eye; they never receive native joint
quality rendering.  The acquisition and analysis cameras remain distinct.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from experiments.nava_mse.common import PROJECT, digest, read_json, sha_file, write_json
from experiments.nava_mse.geometry import validate_source_conditioning
from experiments.nava_mse.rendering import StreamSTGSRenderer, _GPUBackend, validate_render_job
from experiments.nava_mse.workflow import validate_frozen_image
from experiments.occlusion_joint_sse.scenes import camera_from_metadata
from .cache import (content_identity_from_frozen, renderer_identity_from_metadata,
                    validate_identity)
from .visibility import (ContributionMass, VisibilityAnalyzerAdapter, VisibilitySample,
                         VISIBILITY_PROXY)


COMPLETION_SCHEMA = "visibility-guided-sse.native-artifact.v1"
PROXY_ACTIVATION = "recorded_source_local_center_timeline"
_ACTIVATION_FIELDS = ("schema", "camera", "camera_pose_sha256", "timestamp_s", "gop",
    "objects", "allocation", "background", "normalization", "layout", "data_range",
    "renderer", "interactive_viewer_used", "runtime")


def native_cuda_smoke() -> dict[str, Any]:
    """Task 2 only: execute and synchronize a real native PyTorch CUDA kernel.

    Merely importing this function or constructing NativeExecutor does nothing.
    ``nvidia-smi`` is intentionally not accepted as CUDA execution evidence.
    """
    import torch
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Native PyTorch CUDA is unavailable; no rendering may start")
    value = torch.tensor([1., 2., 3.], device="cuda")
    result = (value * value).sum()
    torch.cuda.synchronize()
    if float(result.item()) != 14.:
        raise RuntimeError("Native CUDA arithmetic smoke test failed")
    return dict(status="NATIVE_CUDA_KERNEL_PASSED", gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, cuda=torch.version.cuda,
                observed_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                observed_cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                tf32_observation_scope="before_executor_applies_frozen_render_flags",
                native_kernel_result=14., wall_s=time.perf_counter() - started)


def _sample_from_dict(value: Mapping[str, Any]) -> VisibilitySample:
    return VisibilitySample(value["timestamp"],
        {key: ContributionMass(row["A_solo"], row["A_vis"], row.get("frustum_included"))
         for key, row in value["per_object"].items()},
        tuple(value["resolution"]), value["proxy_identity"], tuple(value["provenance"]),
        value["quality"], value["metadata"])


def _render_job(identity: Mapping[str, Any]) -> dict[str, Any]:
    sources = {row["source_camera"]["id"] for row in identity["instances"]}
    if len(sources) != 1:
        raise ValueError("Native request requires one declared source-camera ID")
    solo = identity["kind"] in ("object", "solo_alpha")
    return dict(kind="object" if solo else "joint", timestamp_s=identity["timestamp_s"],
                allocation=identity["allocation"], camera_id=next(iter(sources)),
                object_id=identity["instances"][0]["object_id"] if solo else None)


def _validate_request(identity: Mapping[str, Any], source_frozen: Mapping[str, Any]) -> dict[str, Any]:
    identity = validate_identity(identity)
    if identity["content_identity"] != content_identity_from_frozen(source_frozen):
        raise ValueError("Snapshot content bank differs from frozen physical source identities")
    if identity["kind"] not in ("joint", "proxy"):
        raise ValueError("NativeExecutor handles joint quality or independent visibility proxies only")
    expected_resolution = (source_frozen["config"]["width"], source_frozen["config"]["height"])
    camera = identity["camera"]
    if (camera["width"], camera["height"]) != expected_resolution:
        raise ValueError("Requested image resolution differs from the common offline profile")
    camera_from_metadata(camera)  # Validate target geometry even for visibility-only traces.
    if identity["kind"] == "proxy":
        protocol = identity["proxy_protocol"]
        if (not isinstance(protocol, dict) or protocol.get("backend") != "visibility_analyzer"
                or protocol.get("activation") != PROXY_ACTIVATION
                or protocol.get("quality") != 40 or protocol.get("area_units") != "alpha_weighted_pixels"
                or set(identity["allocation"].values()) != {40}):
            raise ValueError("Proxy requires the frozen independent recorded-center CRF40 activation protocol")
        acquisition = protocol["acquisition_camera"]
        if ((acquisition["width"], acquisition["height"]) != expected_resolution
                or acquisition["K"] != camera["K"]):
            raise ValueError("Proxy acquisition/analysis resolutions and fixed intrinsics must agree")
    else:
        acquisition = camera
    acquisition_camera = camera_from_metadata(acquisition)
    for instance in identity["instances"]:
        oid, iid = instance["object_id"], instance["instance_id"]
        source = source_frozen["inputs"]["sources"][oid]
        records = [row for row in source["cameras"] if row["id"] == instance["source_camera"]["id"]]
        if len(records) != 1 or records[0] != instance["source_camera"]:
            raise ValueError("Requested source camera differs from the frozen encoded calibration")
        validate_source_conditioning(acquisition_camera, instance["object_to_world"],
            instance["source_camera"], mode=instance["conditioning_mode"])
        representation = source_frozen["inputs"]["representations"][f"{oid}/{identity['gop']}/{identity['allocation'][iid]}"]
        frames = [row for row in representation["samples"] if row["seconds"] == identity["timestamp_s"]]
        if len(frames) != 1 or not math.isfinite(frames[0]["model_time"]):
            raise ValueError("Exact source timeline frame is absent or invalid")
    _render_job(identity)
    return deepcopy(acquisition)


class VisibilityBackend(_GPUBackend):
    """Existing native decoder plus the existing VisibilityAnalyzer, lazily used.

    Proxy analysis receives activated Q0 states and the actual target camera.
    Its zero RGB/alpha return is only a temporary StreamSTGSRenderer envelope;
    NativeExecutor discards those arrays and removes their synthetic hashes.
    """
    def __init__(self, *, visibility_adapter=None, **kwargs):
        super().__init__(**kwargs)
        self.visibility_adapter = visibility_adapter or VisibilityAnalyzerAdapter()
        self.identity = None
        self.last_visibility = None

    def configure(self, identity: Mapping[str, Any]) -> None:
        self.identity = deepcopy(identity)
        self.last_visibility = None

    def render(self, states, acquisition_camera):
        if self.identity is None:
            raise ValueError("Native backend has no exact frozen snapshot request")
        if self.identity["kind"] == "joint":
            return super().render(states, acquisition_camera)
        if self.identity["kind"] != "proxy" or set(states) != set(self.identity["allocation"]):
            raise ValueError("Proxy activation object identities differ from the frozen request")
        target_camera = camera_from_metadata(self.identity["camera"])
        self.last_visibility = self.visibility_adapter.analyze(states, target_camera,
            self.identity["timestamp_s"], proxy_identity=digest(self.identity), quality=40)
        # This makes StreamSTGSRenderer retain its source/frame metadata checks.
        # These arrays are not scientific outputs and are never persisted.
        return (np.zeros((acquisition_camera.height, acquisition_camera.width, 3), np.float32),
                np.zeros((acquisition_camera.height, acquisition_camera.width), np.float32))


def _save_arrays(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


class NativeExecutor:
    """Explicit Task-2 callable; construction is CPU-only and never imports torch.

    Injected ``renderer_factory`` and ``backend_factory`` are for deterministic
    CPU tests.  Production first runs the native CUDA kernel smoke test, applies
    the VRAM fraction, then creates the unchanged production renderer/decoder.
    """
    def __init__(self, source_frozen, packages_root, renderer_identity, output_root,
                 runtime=None, *, renderer_factory=None, backend_factory=None,
                 visibility_adapter=None):
        self.source_frozen = deepcopy(source_frozen)
        self.packages_root = Path(packages_root)
        self.renderer_identity = deepcopy(renderer_identity)
        self.output_root = Path(output_root)
        self.runtime = dict(runtime or {})
        self.max_vram_mb = self.runtime.get("max_gpu_memory_mb", 4500)
        if (isinstance(self.max_vram_mb, bool) or not isinstance(self.max_vram_mb, int)
                or not 0 < self.max_vram_mb <= 4500):
            raise ValueError("Explicit native VRAM limit must be between1 and4500MB")
        if any(type(self.runtime.get(name, 1)) is not int or self.runtime.get(name, 1) != 1
               for name in ("max_resident_models", "ffmpeg_threads")):
            raise ValueError("Minimal native execution is frozen to resident1 and decoder thread1")
        if (renderer_factory is None) != (backend_factory is None):
            raise ValueError("CPU test injection requires both renderer and backend factories")
        self._renderer_factory = renderer_factory or StreamSTGSRenderer
        self._backend_factory = backend_factory or VisibilityBackend
        self._injected = renderer_factory is not None
        self._adapter = visibility_adapter
        self.renderer = self.backend = None
        self.smoke = None

    def _ensure_renderer(self):
        if self.renderer is not None:
            return
        if not self._injected:
            self.smoke = native_cuda_smoke()
            import torch
            if self.smoke["gpu"] != self.renderer_identity.get("gpu"):
                raise ValueError("Native GPU model differs from the frozen source renderer identity")
            for field in ("matmul_allow_tf32", "cudnn_allow_tf32"):
                if type(self.renderer_identity.get(field)) is not bool:
                    raise ValueError("Frozen native TF32 runtime flags must be explicit booleans")
            torch.backends.cuda.matmul.allow_tf32 = self.renderer_identity["matmul_allow_tf32"]
            torch.backends.cudnn.allow_tf32 = self.renderer_identity["cudnn_allow_tf32"]
            self.smoke["post_setup_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            self.smoke["post_setup_cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., self.max_vram_mb * 1024**2 / total), 0)
        self.backend = self._backend_factory(device="cuda", ffmpeg_threads=1,
                                             visibility_adapter=self._adapter)
        c = self.source_frozen["config"]
        sources = self.source_frozen["inputs"]["sources"]
        packages = {(oid, c["gop"], quality): self.packages_root / oid /
                    f"gop_{c['gop']:03d}/encoded/crf_{quality}"
                    for oid in sources for quality in (20, 28, 36, 40)}
        self.renderer = self._renderer_factory(packages,
            metadata_root=PROJECT / c["metadata_root"], max_resident_models=1,
            bounds_source=PROJECT / c["bounds_source"], device="cuda", ffmpeg_threads=1,
            backend=self.backend)

    def __call__(self, job, *, remaining_seconds):
        if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                or not math.isfinite(remaining_seconds) or remaining_seconds <= 0):
            raise ValueError("A positive remaining global execution budget is required")
        identity = validate_identity(job["identity"])
        if job.get("job_id", digest(identity)) != digest(identity):
            raise ValueError("Requested job ID differs from its exact snapshot identity")
        if identity["renderer_identity"] != self.renderer_identity:
            raise ValueError("Requested renderer differs from the frozen native implementation")
        acquisition = _validate_request(identity, self.source_frozen)
        key = digest(identity)
        folder = self.output_root / "artifacts" / identity["kind"]
        arrays_path, marker_path = folder / f"{key}.npz", folder / f"{key}.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            validate_completion(job, marker, self.source_frozen)
            return dict(marker_path=str(marker_path), arrays_path=str(arrays_path))
        self._ensure_renderer()
        self.backend.configure(identity)
        started = time.perf_counter()
        result = self.renderer.render(camera_from_metadata(acquisition), identity["instances"],
                                      identity["timestamp_s"], identity["gop"], identity["allocation"])
        elapsed = time.perf_counter() - started
        metadata = result.metadata
        if renderer_identity_from_metadata(metadata) != self.renderer_identity:
            raise ValueError("Actual native rasterizer/runtime identity differs from the frozen renderer")
        validate_render_job(metadata, _render_job(identity), identity["gop"],
                            expected_camera=acquisition, expected_instances=identity["instances"])
        validate_frozen_image(self.source_frozen, dict(metadata=metadata))
        marker = dict(schema=COMPLETION_SCHEMA, snapshot_identity=identity,
            artifact_kind=identity["kind"], arrays_path=str(arrays_path.resolve()),
            target_camera=identity["camera"], execution_wall_s=elapsed,
            native_cuda_smoke=self.smoke,
            information_role=VISIBILITY_PROXY if identity["kind"] == "proxy" else "JOINT_IMAGE_GROUND_TRUTH")
        if hasattr(self.backend, "torch"):
            marker["peak_gpu_allocated_bytes"] = int(self.backend.torch.cuda.max_memory_allocated())
            marker["peak_gpu_reserved_bytes"] = int(self.backend.torch.cuda.max_memory_reserved())
        if identity["kind"] == "proxy":
            sample = self.backend.last_visibility
            if not isinstance(sample, VisibilitySample):
                raise ValueError("Native analyzer did not provide a validated independent proxy sample")
            sample = VisibilitySample(sample.timestamp, sample.per_object, sample.resolution,
                sample.proxy_identity, sample.provenance, sample.quality,
                {**sample.metadata, "proxy_acquisition_time_ms": max(0., elapsed * 1000 - sample.metadata.get("analysis_time_ms", 0.)),
                 "activation_camera_distinct_from_analysis": acquisition != identity["camera"],
                 "activation": PROXY_ACTIVATION})
            ordered = [row["instance_id"] for row in identity["instances"]]
            arrays = dict(A_solo=np.asarray([sample.per_object[iid].A_solo for iid in ordered], dtype=np.float64),
                          A_vis=np.asarray([sample.per_object[iid].A_vis for iid in ordered], dtype=np.float64),
                          instance_ids=np.asarray(ordered))
            activation = {field: deepcopy(metadata[field]) for field in _ACTIVATION_FIELDS if field in metadata}
            activation.update(envelope_kind="ACTIVATION_METADATA_ONLY", target_RGB_generated=False)
            marker.update(visibility_sample=sample.to_dict(), proxy_protocol=identity["proxy_protocol"],
                          acquisition_camera=acquisition, native_activation_metadata=activation,
                          zero_RGB_envelope_persisted=False,
                          actual_raster_passes=sample.metadata.get("joint_passes", 0) + sample.metadata.get("solo_passes", 0))
        else:
            arrays = dict(rgb=np.asarray(result.rgb), raw_rgb=np.asarray(result.raw_rgb), alpha=np.asarray(result.alpha))
            marker.update(render_metadata=deepcopy(metadata), actual_raster_passes=2)
            validate_frozen_image(self.source_frozen, dict(metadata=metadata, **arrays))
        _save_arrays(arrays_path, arrays)
        marker["arrays_sha256"] = sha_file(arrays_path)
        validate_completion(job, marker, self.source_frozen)
        write_json(marker_path, marker)
        return dict(marker_path=str(marker_path), arrays_path=str(arrays_path))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def validate_completion(job, marker, source_frozen):
    """Bind new artifact roles, exact target/acquisition identities and payloads."""
    identity = validate_identity(job["identity"])
    acquisition = _validate_request(identity, source_frozen)
    if (marker.get("schema") != COMPLETION_SCHEMA or marker.get("snapshot_identity") != identity
            or marker.get("artifact_kind") != identity["kind"] or marker.get("target_camera") != identity["camera"]):
        raise ValueError("Completion marker differs from the exact frozen snapshot request")
    arrays_path = Path(marker["arrays_path"])
    if not arrays_path.is_file() or sha_file(arrays_path) != marker.get("arrays_sha256"):
        raise ValueError("Native completion payload is missing or corrupt")
    if identity["kind"] == "proxy":
        if (marker.get("information_role") != VISIBILITY_PROXY or marker.get("proxy_protocol") != identity["proxy_protocol"]
                or marker.get("acquisition_camera") != acquisition
                or marker.get("zero_RGB_envelope_persisted") is not False or "render_metadata" in marker):
            raise ValueError("Proxy completion must retain independent activation/analysis camera roles")
        metadata = marker["native_activation_metadata"]
        if metadata.get("target_RGB_generated") is not False or metadata.get("envelope_kind") != "ACTIVATION_METADATA_ONLY":
            raise ValueError("A proxy envelope cannot be claimed as a native target image")
        if any("rgb" in key.lower() and key != "target_RGB_generated" for key in metadata):
            raise ValueError("Synthetic RGB envelope fields must not be persisted")
        sample = _sample_from_dict(marker["visibility_sample"])
        _validate_proxy_payload(identity, sample, arrays_path)
        passes = sample.metadata.get("joint_passes", 0) + sample.metadata.get("solo_passes", 0)
        if marker.get("actual_raster_passes") != passes or passes > len(identity["instances"]) + math.ceil(len(identity["instances"]) / 3):
            raise ValueError("Measured proxy raster calls differ from acquisition accounting")
    else:
        if marker.get("information_role") != "JOINT_IMAGE_GROUND_TRUTH" or "visibility_sample" in marker:
            raise ValueError("Joint images are evaluation-only, never policy visibility inputs")
        if marker.get("actual_raster_passes") != 2 or type(marker["actual_raster_passes"]) is not int:
            raise ValueError("Native joint RGB and alpha require exactly two measured raster calls")
        metadata = marker["render_metadata"]
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
            _validate_image_arrays(identity, arrays)
    if renderer_identity_from_metadata(metadata) != identity["renderer_identity"]:
        raise ValueError("Native completion renderer identity mismatch")
    validate_render_job(metadata, _render_job(identity), identity["gop"],
                        expected_camera=acquisition, expected_instances=identity["instances"])
    validate_frozen_image(source_frozen, dict(metadata=metadata))


def _validate_image_arrays(identity, arrays):
    if set(arrays) != {"rgb", "raw_rgb", "alpha"}:
        raise ValueError("Joint payload requires exact native RGB/raw-RGB/alpha arrays")
    shape = (identity["camera"]["height"], identity["camera"]["width"], 3)
    if (arrays["rgb"].shape != shape or arrays["raw_rgb"].shape != shape
            or arrays["alpha"].shape != shape[:2]
            or any(not np.isfinite(value).all() for value in arrays.values())
            or np.any(arrays["rgb"] < 0) or np.any(arrays["rgb"] > 1)
            or np.any(arrays["alpha"] < 0) or np.any(arrays["alpha"] > 1)
            or not np.array_equal(arrays["rgb"], np.clip(arrays["raw_rgb"], 0., 1.))):
        raise ValueError("Native joint arrays violate frozen resolution/RGB/alpha conventions")


def _validate_proxy_payload(identity, sample, arrays_path):
    expected_ids = [row["instance_id"] for row in identity["instances"]]
    if (sample.timestamp != identity["timestamp_s"] or sample.proxy_identity != digest(identity)
            or set(sample.per_object) != set(expected_ids)
            or sample.resolution != (identity["camera"]["width"], identity["camera"]["height"])):
        raise ValueError("Proxy sample differs from the exact analysis camera/time/object request")
    if (sample.metadata.get("backend") != "streamstgs_cuda_contribution"
            or sample.metadata.get("area_units") != "alpha_weighted_pixels"):
        raise ValueError("Proxy masses require the existing native analyzer and continuous alpha-pixel units")
    pixels = sample.resolution[0] * sample.resolution[1]
    if math.fsum(row.A_vis for row in sample.per_object.values()) > pixels + 1e-6 + 1e-5 * pixels:
        raise ValueError("Object visible contributions exceed the viewport alpha-mass bound")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"A_solo", "A_vis", "instance_ids"}:
            raise ValueError("Proxy payload must contain only actual contribution masses, never RGB envelopes")
        if archive["instance_ids"].tolist() != expected_ids:
            raise ValueError("Proxy payload object ordering mismatch")
        for field in ("A_solo", "A_vis"):
            actual = archive[field]
            expected = np.asarray([getattr(sample.per_object[iid], field) for iid in expected_ids])
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError("Proxy payload masses differ from validated visibility sample")


def _read_entry(entry):
    identity = validate_identity(entry["identity"])
    for descriptor in entry["files"]:
        path = Path(descriptor["path"])
        if (not path.is_file() or path.stat().st_size != descriptor["bytes"]
                or sha_file(path) != descriptor["sha256"]):
            raise ValueError("Registered artifact changed before loading")
    marker = read_json(entry["marker_path"])
    if sha_file(entry["arrays_path"]) != marker.get("arrays_sha256"):
        raise ValueError("Artifact marker/payload hash mismatch")
    if marker.get("schema") == COMPLETION_SCHEMA and marker.get("snapshot_identity") != identity:
        raise ValueError("Artifact marker semantic identity mismatch")
    return identity, marker, Path(entry["arrays_path"])


def load_image(entry):
    identity, marker, arrays_path = _read_entry(entry)
    if identity["kind"] != "joint" or marker.get("schema") not in (COMPLETION_SCHEMA, "nava-mse.image.v1"):
        raise ValueError("Quality evaluation requires a registered native joint image")
    if marker.get("schema") == COMPLETION_SCHEMA and marker.get("information_role") != "JOINT_IMAGE_GROUND_TRUTH":
        raise ValueError("Visibility proxy is not a joint-quality image")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _validate_image_arrays(identity, arrays)
    metadata = marker["render_metadata"]
    validate_render_job(metadata, _render_job(identity), identity["gop"],
                        expected_camera=identity["camera"], expected_instances=identity["instances"])
    if renderer_identity_from_metadata(metadata) != identity["renderer_identity"]:
        raise ValueError("Loaded native image renderer mismatch")
    return dict(**arrays, metadata=metadata, arrays_sha256=marker["arrays_sha256"],
                provenance=dict(arrays_path=str(arrays_path), arrays_sha256=marker["arrays_sha256"],
                                metadata_path=entry["marker_path"]))


def load_proxy(entry):
    identity, marker, arrays_path = _read_entry(entry)
    if (identity["kind"] != "proxy" or marker.get("schema") != COMPLETION_SCHEMA
            or marker.get("information_role") != VISIBILITY_PROXY
            or marker.get("zero_RGB_envelope_persisted") is not False):
        raise ValueError("Policy input requires an independently acquired native proxy artifact")
    protocol = identity["proxy_protocol"]
    if (marker.get("proxy_protocol") != protocol or marker.get("target_camera") != identity["camera"]
            or marker.get("acquisition_camera") != protocol.get("acquisition_camera")):
        raise ValueError("Loaded proxy analysis/acquisition identities differ from the frozen request")
    metadata = marker["native_activation_metadata"]
    if metadata.get("target_RGB_generated") is not False or metadata.get("envelope_kind") != "ACTIVATION_METADATA_ONLY":
        raise ValueError("Loaded proxy must retain activation-only provenance")
    validate_render_job(metadata, _render_job(identity), identity["gop"],
                        expected_camera=protocol["acquisition_camera"], expected_instances=identity["instances"])
    if renderer_identity_from_metadata(metadata) != identity["renderer_identity"]:
        raise ValueError("Loaded proxy native renderer identity mismatch")
    sample = _sample_from_dict(marker["visibility_sample"])
    _validate_proxy_payload(identity, sample, arrays_path)
    return sample
