"""Lazy headless StreamSTGS rendering; production decoder/rasterizer unchanged.

Importing this module never imports torch, initializes CUDA, decodes a package,
or renders an image. Task 2 explicitly calls render(). Images use HWC float32;
raw pre-clipping RGB and reference alpha are retained separately. Package hashes
bind object, representation, temporal identity, and source calibration before
the adapter is admitted. No historical pickle caches are deserialized.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from .geometry import (camera_metadata, camera_identity, camera_from_record,
                       relative_camera, validate_source_conditioning)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def exact_frame(reconstruction, timestamp_s, *, gop):
    """Only an exact encoded presentation frame is supported; no floor/nearest."""
    if isinstance(gop, bool) or not isinstance(gop, int) or gop < 0:
        raise ValueError("GOP must be a nonnegative integer")
    if isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError("Media timestamp must be finite and nonnegative")
    segment = reconstruction["segment"]
    if segment["gop_index"] != gop:
        raise ValueError("Encoded reconstruction GOP identity mismatch")
    rows = segment["timestamps"]
    if len({r["frame"] for r in rows}) != len(rows) or len({r["seconds"] for r in rows}) != len(rows):
        raise ValueError("Duplicate encoded presentation-frame identity")
    if [r["frame"] for r in rows] != segment["output_frames"]:
        raise ValueError("Encoded timestamps disagree with output frame identities")
    matches = [r for r in rows if r["seconds"] == timestamp_s]
    if len(matches) != 1:
        raise ValueError(f"Unavailable exact encoded timestamp {timestamp_s!r} in GOP{gop}")
    row = matches[0]
    fps = segment["fps"]
    expected_seconds = (row["frame"] - segment["video_origin_frame"]) / fps
    if row["seconds"] != expected_seconds or not math.isfinite(row["model_time"]) or row["model_time"] < 0:
        raise ValueError("Encoded frame/media/model-time mapping is inconsistent")
    if row["frame"] < segment["gop_start_frame"] or row["frame"] > segment["gop_end_frame"]:
        raise ValueError("Presentation frame lies outside declared GOP")
    return dict(row)


@dataclass(frozen=True)
class RenderResult:
    rgb: np.ndarray
    raw_rgb: np.ndarray
    alpha: np.ndarray
    metadata: dict


def validate_render_alignment(reference_metadata, candidate_metadata):
    """Require identical rendering context except representation-specific data.

    The fixed reference must be all CRF20. RGB arrays alone cannot establish
    camera/source-frame alignment, so missing metadata is an explicit error.
    """
    common_fields = ("schema", "camera", "camera_pose_sha256", "timestamp_s", "gop",
                     "background", "normalization", "layout", "data_range", "renderer")
    object_fields = ("instance_id", "object_id", "gop", "frame_index", "timestamp_s", "model_time",
                     "training_identity", "checkpoint_sha256", "object_to_world", "object_transform_sha256",
                     "source_identity_sha256", "cameras_sha256", "source_camera_id", "source_camera_split",
                     "conditioning_mode", "source_camera_pose_sha256", "source_local_camera",
                     "source_local_camera_pose_sha256", "source_local_camera_center")
    try:
        if reference_metadata.get('runtime') != candidate_metadata.get('runtime'):
            raise ValueError('Reference/candidate GPU runtime provenance differs')
        for field in common_fields:
            if reference_metadata[field] != candidate_metadata[field]:
                raise ValueError(f"Reference/candidate rendering context differs: {field}")
        if reference_metadata["schema"] != "nava-mse-render.v1":
            raise ValueError("Unsupported renderer metadata schema")
        reference_objects, candidate_objects = reference_metadata["objects"], candidate_metadata["objects"]
        if not reference_objects or set(reference_objects) != set(candidate_objects):
            raise ValueError("Reference/candidate scene object identities differ")
        for iid, reference in reference_objects.items():
            if reference["crf"] != 20:
                raise ValueError("Compression-relative reference must be all CRF20")
            for field in object_fields:
                if reference[field] != candidate_objects[iid][field]:
                    raise ValueError(f"Reference/candidate object alignment differs: {iid}/{field}")
        for metadata in (reference_metadata, candidate_metadata):
            if metadata["allocation"] != {iid: obj["crf"] for iid, obj in metadata["objects"].items()}:
                raise ValueError("Rendered allocation differs from actual object representation identities")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing reference/candidate render alignment metadata") from exc


def validate_render_job(metadata, job, gop, *, expected_camera=None, expected_instances=None):
    """Bind an emitted/cached artifact to its requested exact scene job.

    expected_camera may be a production Camera or its camera_metadata dictionary;
    expected_instances are the same explicit instance records passed to render().
    This check belongs before both caching and scoring in the workflow.
    """
    try:
        if metadata["schema"] != "nava-mse-render.v1":
            raise ValueError("Unsupported rendered job metadata schema")
        if job["kind"] not in ("object", "joint"):
            raise ValueError("Unsupported render job kind")
        if metadata["gop"] != gop or metadata["timestamp_s"] != job["timestamp_s"]:
            raise ValueError("Rendered GOP/timestamp differs from requested exact job")
        if metadata["allocation"] != job["allocation"]:
            raise ValueError("Rendered allocation differs from requested job")
        objects = metadata["objects"]
        if not objects or set(objects) != set(job["allocation"]):
            raise ValueError("Rendered instances differ from requested allocation")
        for iid, obj in objects.items():
            if (obj["instance_id"] != iid or obj["gop"] != gop or obj["timestamp_s"] != job["timestamp_s"]
                    or obj["crf"] != job["allocation"][iid] or obj["source_camera_id"] != job["camera_id"]):
                raise ValueError("Rendered source/frame/quality identity differs from requested job")
        if job["kind"] == "object" and (len(objects) != 1 or
                next(iter(objects.values()))["object_id"] != job["object_id"]):
            raise ValueError("Solo rendered object identity differs from requested job")
        if expected_camera is not None:
            expected = expected_camera if isinstance(expected_camera, dict) else camera_metadata(expected_camera)
            if metadata["camera"] != expected or metadata["camera_pose_sha256"] != _digest(expected):
                raise ValueError("Rendered camera differs from requested exact scene camera")
        if expected_instances is not None:
            declared = {i["instance_id"]: i for i in expected_instances}
            if set(declared) != set(objects):
                raise ValueError("Rendered instance set differs from frozen scene")
            for iid, instance in declared.items():
                if (objects[iid]["object_id"] != instance["object_id"]
                        or objects[iid]["object_to_world"] != instance["object_to_world"]
                        or objects[iid]["object_transform_sha256"] != _digest(instance["object_to_world"])
                        or objects[iid]["source_camera_id"] != instance["source_camera"]["id"]
                        or objects[iid]["source_camera_split"] != instance["source_camera"]["split"]
                        or objects[iid]["conditioning_mode"] != instance.get("conditioning_mode", "exact_pose")):
                    raise ValueError("Rendered object transform/source conditioning differs from frozen scene")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing rendered job identity metadata") from exc


class _GPUBackend:
    """Small reusable wrapper over the verified production GPU interfaces."""
    def __init__(self, *, device="cuda", ffmpeg_threads=1):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; activate the documented GPU environment for Task 2")
        project = Path(__file__).resolve().parents[2]
        # Pinned vendor/adapter imports expect the production namespace.
        source = str(project / "src_final")
        if source not in sys.path:
            sys.path.insert(0, source)
        from multiobj3dgs.codecs.streamstgs import StreamSTGSAdapter
        from multiobj3dgs.codecs.streamstgs.vendor import verify_vendor
        from multiobj3dgs.client.visibility.analyzer import TensorContributionScene
        self.torch, self.scene_type = torch, TensorContributionScene
        root = project / 'third_party/streamstgs'
        verify_vendor(root)
        self.adapter = StreamSTGSAdapter(vendor_root=root, device=device, ffmpeg_threads=ffmpeg_threads)
        self._runtime = None

    def runtime_metadata(self):
        if self._runtime is None:
            import platform
            torch = self.torch
            module = sys.modules.get('diff_gaussian_rasterization')
            files = {}
            if module is not None:
                package = Path(module.__file__).parent
                for path in [Path(module.__file__), *package.glob('*.so')]:
                    files[path.name] = sha256_file(path)
            self._runtime = dict(python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), cuda_device=torch.cuda.current_device(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                rasterizer_files_sha256=files)
        return self._runtime

    def prepare_cpu(self, path):
        return self.adapter.prepare_cpu(path)

    def realize_gpu(self, prepared):
        return self.adapter.realize_gpu(prepared)

    def release(self, handle):
        return self.adapter.release(handle)

    def reconstruct(self, handle, model_time, camera):
        return self.adapter.reconstruct(handle, model_time, camera)

    def xyz(self, state):
        return state.xyz.detach().cpu().numpy()

    def render(self, states, camera):
        torch = self.torch
        with torch.inference_mode():
            scene = self.scene_type(states, camera)
            args = scene.joint.raster_args()
            raw = scene.rasterizer(**args)[0]
            args["colors_precomp"] = torch.ones_like(scene.joint.colors)
            alpha = scene.rasterizer(**args)[0][0]
            # Preserve actual RGB; the caller applies one identical clipping rule.
            return (raw.detach().cpu().numpy().transpose(1, 2, 0),
                    alpha.detach().cpu().numpy())


class StreamSTGSRenderer:
    """Bounded model LRU and one true joint raster pass for each allocation.

    packages[(object_id,gop,crf)] points to materialized, encoded package bytes.
    metadata_root is the audited per-object source-quality metadata tree. It is
    mandatory because reconstruction.json itself has no object-name field.
    backend is an injectable CPU mock for unit tests; omitted in real use.
    Only black background is supported by this verified production rasterizer.
    """
    def __init__(self, packages, *, metadata_root, max_resident_models=2,
                 bounds_source=None, device="cuda", ffmpeg_threads=1, backend=None):
        if isinstance(max_resident_models, bool) or not isinstance(max_resident_models, int) or max_resident_models < 1:
            raise ValueError("max_resident_models must be a positive integer")
        if isinstance(ffmpeg_threads, bool) or not isinstance(ffmpeg_threads, int) or ffmpeg_threads < 1:
            raise ValueError("ffmpeg_threads must be a positive integer")
        self.packages = {key: Path(value) for key, value in packages.items()}
        self.metadata_root = Path(metadata_root)
        self.max_resident_models = max_resident_models
        self.bounds_source = Path(bounds_source) if bounds_source is not None else None
        self.device, self.ffmpeg_threads = device, ffmpeg_threads
        self._backend = backend
        self._handles, self._metadata, self._bounds = OrderedDict(), {}, {}
        self._closed = False

    def _ensure_backend(self):
        if self._closed:
            raise RuntimeError("Renderer has been closed")
        if self._backend is None:
            self._backend = _GPUBackend(device=self.device, ffmpeg_threads=self.ffmpeg_threads)
        return self._backend

    def _package_metadata(self, key):
        if key in self._metadata:
            return self._metadata[key]
        oid, gop, crf = key
        if not isinstance(oid, str) or not oid or Path(oid).name != oid:
            raise ValueError("Invalid object identity")
        if isinstance(crf, bool) or not isinstance(crf, int) or crf not in (20, 28, 36, 40):
            raise ValueError("Unsupported CRF representation")
        if isinstance(gop, bool) or not isinstance(gop, int) or gop < 0:
            raise ValueError("Invalid GOP identity")
        if key not in self.packages:
            raise FileNotFoundError(f"No encoded package configured for {key}")
        package = self.packages[key]
        identity_root = self.metadata_root / oid / "content"
        source_root = identity_root / f"gop_{gop:03d}_crf_{crf}"
        identity = json.loads((identity_root / "stream_package_manifest.json").read_text())
        if identity["object"] != oid or crf not in identity["crfs"] or gop >= identity["gop_count"]:
            raise ValueError("Source package object/GOP/quality identity mismatch")
        hashes = {}
        for filename in ("manifest.json", "reconstruction.json"):
            hashes[filename] = sha256_file(package / filename)
            if hashes[filename] != sha256_file(source_root / filename):
                raise ValueError(f"Encoded {oid}/GOP{gop}/CRF{crf} {filename} differs from bound source metadata")
        reconstruction = json.loads((package / "reconstruction.json").read_text())
        manifest = json.loads((package / "manifest.json").read_text())
        if reconstruction["crf"] != crf or manifest["crf"] != crf or reconstruction["segment"]["gop_index"] != gop:
            raise ValueError("Encoded representation identity mismatch")
        # Verify every physical encoded component, including calibration/MLPs.
        total = 0
        for name, record in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe encoded component path")
            path = package / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing or linked encoded component: {name}")
            if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int) or record["bytes"] < 0:
                raise ValueError("Invalid encoded component bytes")
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Encoded component bytes/hash mismatch: {name}")
            total += record["bytes"]
        if total != manifest["actual_payload_bytes"]:
            raise ValueError("Encoded manifest physical payload total mismatch")
        cameras = json.loads((package / "cameras.json").read_text())
        source_cameras = json.loads((self.metadata_root / oid / "reference/cameras.json").read_text())
        if cameras != source_cameras:
            raise ValueError("Encoded source-camera calibration differs from bound source identity")
        value = dict(package=package, reconstruction=reconstruction, manifest=manifest,
                     cameras=cameras, reconstruction_sha256=hashes["reconstruction.json"],
                     manifest_sha256=hashes["manifest.json"],
                     source_identity_sha256=sha256_file(identity_root / "stream_package_manifest.json"),
                     cameras_sha256=sha256_file(package / "cameras.json"))
        self._metadata[key] = value
        return value

    def _handle(self, key, metadata):
        backend = self._ensure_backend()
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key]
        while len(self._handles) >= self.max_resident_models:
            _, handle = self._handles.popitem(last=False)
            backend.release(handle)
        prepared = backend.prepare_cpu(metadata["package"])
        if prepared.reconstruction != metadata["reconstruction"] or prepared.cameras != metadata["cameras"]:
            raise ValueError("Prepared StreamSTGS identity differs from verified encoded metadata")
        handle = backend.realize_gpu(prepared)
        if handle.reconstruction != metadata["reconstruction"] or handle.cameras != metadata["cameras"]:
            backend.release(handle)
            raise ValueError("Realized StreamSTGS identity differs from prepared package")
        self._handles[key] = handle
        return handle

    def render(self, camera, instances, timestamp_s, gop, allocation):
        if self._closed:
            raise RuntimeError("Renderer has been closed")
        instances = list(instances)
        ids = [i["instance_id"] for i in instances]
        if not ids or len(ids) != len(set(ids)) or set(allocation) != set(ids):
            raise ValueError("Exactly one CRF is required for each unique scene instance")
        requests = []
        for instance in instances:
            oid, iid = instance["object_id"], instance["instance_id"]
            crf = allocation[iid]
            key = oid, gop, crf
            metadata = self._package_metadata(key)
            source = instance["source_camera"]
            matches = [c for c in metadata["cameras"] if c["id"] == source["id"]]
            if len(matches) != 1 or matches[0] != source:
                raise ValueError("Instance source camera differs from its actual encoded calibration")
            conditioning = validate_source_conditioning(camera, instance["object_to_world"], source,
                         mode=instance.get("conditioning_mode", "exact_pose"))
            frame = exact_frame(metadata["reconstruction"], timestamp_s, gop=gop)
            requests.append((instance, key, metadata, conditioning, frame))
        # All CPU metadata validation precedes CUDA admission.
        backend = self._ensure_backend()
        states, provenance = {}, {}
        for instance, key, metadata, conditioning, frame in requests:
            iid, oid = instance["instance_id"], instance["object_id"]
            local = relative_camera(camera, instance["object_to_world"])
            handle = self._handle(key, metadata)
            state = backend.reconstruct(handle, frame["model_time"], local)
            xyz = np.asarray(backend.xyz(state))
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError("Nonfinite or malformed decoded Gaussian positions")
            states[iid] = state.transformed(np.asarray(instance["object_to_world"]))
            provenance[iid] = dict(instance_id=iid, object_id=oid, gop=gop, crf=key[2],
                frame_index=frame["frame"], timestamp_s=frame["seconds"], model_time=frame["model_time"],
                training_identity=metadata["reconstruction"]["training_identity"],
                checkpoint_sha256=metadata["reconstruction"]["checkpoint_sha256"],
                object_to_world=instance["object_to_world"],
                object_transform_sha256=_digest(instance["object_to_world"]),
                reconstruction_sha256=metadata["reconstruction_sha256"],
                manifest_sha256=metadata["manifest_sha256"],
                source_identity_sha256=metadata["source_identity_sha256"],
                cameras_sha256=metadata["cameras_sha256"],
                encoded_package=str(metadata["package"].resolve()),
                decoded_local_xyz_sha256=hashlib.sha256(np.ascontiguousarray(xyz, dtype="<f4").tobytes()).hexdigest(),
                gaussians=len(xyz), **conditioning)
        raw, alpha = backend.render(states, camera)
        raw, alpha = np.asarray(raw, dtype=np.float32), np.asarray(alpha, dtype=np.float32)
        if raw.shape != (camera.height, camera.width, 3) or alpha.shape != (camera.height, camera.width):
            raise ValueError("GPU backend returned unexpected RGB/alpha dimensions")
        if not np.isfinite(raw).all() or not np.isfinite(alpha).all():
            raise ValueError("Nonfinite renderer RGB/alpha")
        # Numerical alpha tolerance only; large excursions indicate a backend error.
        raw_alpha_min, raw_alpha_max = float(alpha.min()), float(alpha.max())
        if raw_alpha_min < -1e-6 or raw_alpha_max > 1.+1e-5:
            raise ValueError("Renderer alpha lies outside [0,1]")
        # Normalize only accepted numerical epsilon excursions, retaining extrema
        # and a correction count. Hashes/masks use this exact persisted alpha.
        alpha_clipped_pixels = int(((alpha < 0.) | (alpha > 1.)).sum())
        alpha = np.clip(alpha, 0., 1.)
        rgb = np.clip(raw, 0., 1.)
        clipped = (raw < 0.) | (raw > 1.)
        foreground = alpha > .01
        metadata = dict(schema="nava-mse-render.v1", camera=camera_metadata(camera),
            camera_pose_sha256=camera_identity(camera), timestamp_s=timestamp_s, gop=gop,
            objects=provenance, allocation=dict(allocation), background=[0., 0., 0.],
            normalization="clip", layout="HWC", data_range=1.,
            renderer="production StreamSTGSAdapter + TensorContributionScene CUDA GaussianRasterizer",
            interactive_viewer_used=False, raw_rgb_min=float(raw.min()), raw_rgb_max=float(raw.max()),
            raw_alpha_min=raw_alpha_min, raw_alpha_max=raw_alpha_max,
            alpha_numerical_clipped_pixels=alpha_clipped_pixels,
            clipping_diagnostic_alpha_threshold=0.01,
            clipped_channels=int(clipped.sum()), clipped_channel_fraction=float(clipped.mean()),
            foreground_clipped_channel_fraction=float(clipped[foreground].mean()) if foreground.any() else None,
            rgb_float32_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            raw_rgb_float32_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            alpha_float32_sha256=hashlib.sha256(np.ascontiguousarray(alpha).tobytes()).hexdigest())
        if hasattr(backend, 'runtime_metadata'):
            metadata['runtime'] = backend.runtime_metadata()
        return RenderResult(rgb, raw, alpha, metadata)

    def full_gop_bounds(self, object_id, gop, reference_crf=20):
        """Reuse provenance-bound full decoded CRF20 means, or explicitly decode.

        Bounds encompass all presentation frames, all Gaussian means, and no
        opacity/visibility threshold. They are not inferred from the four image
        samples. Legacy saved bounds are checked against their copied immutable
        source metadata and the current package before reuse.
        """
        if reference_crf != 20:
            raise ValueError("Frozen full-GOP bounds must use the CRF20 reference")
        key = object_id, gop, reference_crf
        if key in self._bounds:
            return self._bounds[key]
        metadata = self._package_metadata(key)
        if self.bounds_source is not None:
            if not self.bounds_source.is_file():
                raise FileNotFoundError(f"Configured verified bounds missing: {self.bounds_source}")
            data = json.loads(self.bounds_source.read_text())
            if data["reference_crf"] != 20 or data["coverage"] != "all presentation frames of all five GOPs per content":
                raise ValueError("Unsupported saved full-GOP bounds definition")
            record = data["contents"][object_id]["gops"][str(gop)]
            for name, expected in (("manifest", metadata["manifest_sha256"]),
                                   ("reconstruction", metadata["reconstruction_sha256"])):
                source = (self.bounds_source.parent / record[name]).resolve()
                if not source.is_relative_to(self.bounds_source.parent.resolve()) or sha256_file(source) != expected:
                    raise ValueError("Saved decoded bounds do not match current source package")
            expected_frames = metadata["reconstruction"]["segment"]["timestamps"]
            actual_frames = record["frames"]
            if [(r["frame"], r["seconds"], r["model_time"]) for r in actual_frames] != [
                    (r["frame"], r["seconds"], r["model_time"]) for r in expected_frames]:
                raise ValueError("Saved bounds do not cover exact complete presentation frames")
            per_frame = np.asarray([r["bounds"] for r in actual_frames], dtype=np.float64)
            union = np.stack([per_frame[:, 0].min(0), per_frame[:, 1].max(0)])
            if not np.array_equal(union, np.asarray(record["bounds"])):
                raise ValueError("Saved full-GOP bound differs from its per-frame union")
            result = dict(bounds=union.tolist(), frames=len(actual_frames),
                          definition=data["bounds_definition"], reference_crf=20,
                          source=str(self.bounds_source.resolve()), source_sha256=sha256_file(self.bounds_source),
                          reconstruction_sha256=metadata["reconstruction_sha256"],
                          manifest_sha256=metadata["manifest_sha256"])
        else:
            backend = self._ensure_backend()
            handle = self._handle(key, metadata)
            camera = camera_from_record(metadata["cameras"][0])
            bounds = []
            for row in metadata["reconstruction"]["segment"]["timestamps"]:
                frame = exact_frame(metadata["reconstruction"], row["seconds"], gop=gop)
                state = backend.reconstruct(handle, frame["model_time"], camera)
                xyz = np.asarray(backend.xyz(state))
                if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
                    raise ValueError("Invalid decoded means for full-GOP bounds")
                bounds.append([xyz.min(0), xyz.max(0)])
            bounds = np.asarray(bounds)
            result = dict(bounds=[bounds[:, 0].min(0).tolist(), bounds[:, 1].max(0).tolist()],
                          frames=len(bounds), reference_crf=20,
                          definition="all decoded Gaussian means over every presentation frame; no opacity filtering",
                          reconstruction_sha256=metadata["reconstruction_sha256"],
                          manifest_sha256=metadata["manifest_sha256"])
        checked = np.asarray(result["bounds"])
        if checked.shape != (2, 3) or not np.isfinite(checked).all() or np.any(checked[0] >= checked[1]):
            raise ValueError("Invalid full-GOP AABB")
        self._bounds[key] = result
        return result

    def close(self):
        if not self._closed:
            if self._backend is not None:
                for handle in self._handles.values():
                    self._backend.release(handle)
            self._handles.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
