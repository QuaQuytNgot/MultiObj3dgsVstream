"""Snapshot preparation and calls to the unchanged Dynamic-LapisGS renderer.

Imports of PyTorch and the environment-owned CUDA extension are lazy. This
module transforms geometry, but never implements projection or compositing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tools.content_preparation.assets import GaussianState
from tools.content_preparation.paths import PROJECT_ROOT, configure_upstream_imports
from tools.content_preparation.renderer_adapter import RendererUnavailable
from tools.content_preparation.spatial_rendering import validate_camera


@dataclass(frozen=True, slots=True)
class ProxySnapshot:
    object_id: str
    proxy: GaussianState
    timestamp: float


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _rotation_quaternion(rotation):
    """Return a unit wxyz quaternion for a validated proper rotation."""
    trace = float(np.trace(rotation))
    if trace > 0:
        factor = 2 * math.sqrt(trace + 1)
        quaternion = [factor / 4,
                      (rotation[2, 1] - rotation[1, 2]) / factor,
                      (rotation[0, 2] - rotation[2, 0]) / factor,
                      (rotation[1, 0] - rotation[0, 1]) / factor]
    else:
        axis = int(np.argmax(np.diag(rotation)))
        other, last = (axis + 1) % 3, (axis + 2) % 3
        factor = 2 * math.sqrt(max(0., 1 + rotation[axis, axis]
                                  - rotation[other, other] - rotation[last, last]))
        quaternion = np.zeros(4)
        quaternion[0] = (rotation[last, other] - rotation[other, last]) / factor
        quaternion[axis + 1] = factor / 4
        quaternion[other + 1] = (rotation[other, axis] + rotation[axis, other]) / factor
        quaternion[last + 1] = (rotation[last, axis] + rotation[axis, last]) / factor
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _transformed_state(state, transform):
    matrix = np.array(transform, dtype=np.float64, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("object_to_world must be a finite nontransposed 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
        raise ValueError("object_to_world must be an affine matrix")
    linear = matrix[:3, :3]
    # Column norms identify the common positive scale without assuming an axis.
    scale = float(np.linalg.norm(linear, axis=0).mean())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Object transforms require positive uniform scale")
    rotation = linear / scale
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-5)):
        raise ValueError("Object transforms support rotation and positive uniform scale; shear, reflection and nonuniform scale are unsupported")
    arrays = {name: value.copy() for name, value in state.arrays.items()}
    if not np.array_equal(matrix, np.eye(4)):
        arrays["xyz"] = (arrays["xyz"].astype(np.float64) @ linear.T + matrix[:3, 3]).astype(np.float32)
        arrays["scale"] = (arrays["scale"].astype(np.float64) + math.log(scale)).astype(np.float32)
        left = _rotation_quaternion(rotation)
        right = arrays["rotation"].astype(np.float64)
        result = np.empty_like(right)
        result[:, 0] = left[0] * right[:, 0] - right[:, 1:] @ left[1:]
        result[:, 1:] = (left[0] * right[:, 1:] + right[:, :1] * left[1:]
                         + np.cross(left[1:], right[:, 1:]))
        arrays["rotation"] = result.astype(np.float32)
    return GaussianState(arrays, copy.deepcopy(state.metadata), state.ids.copy())


def validate_and_snapshot(objects: Sequence, camera: Mapping[str, Any],
                          timestamp: float, settings: Mapping[str, Any] | None = None):
    """Copy timestamp-resolved proxies and transform them once for all passes."""
    timestamp = _finite_number(timestamp, "timestamp")
    validated_camera = validate_camera(camera)
    for field in ("near", "far"):
        _finite_number(validated_camera[field], f"camera {field}")
    options = dict(settings or {})
    unknown = set(options) - {"backend", "extension_path", "debug", "scaling_modifier"}
    if unknown:
        raise ValueError(f"Unsupported analysis renderer settings: {sorted(unknown)}")
    if options.get("backend", "upstream_cuda") != "upstream_cuda":
        raise ValueError("Visibility contribution measurement requires upstream_cuda")
    modifier = _finite_number(options.get("scaling_modifier", 1.), "scaling_modifier")
    if modifier <= 0:
        raise ValueError("scaling_modifier must be positive")
    if not isinstance(options.get("debug", False), (bool, np.bool_)):
        raise ValueError("debug must be a boolean")
    options.update(backend="upstream_cuda", debug=bool(options.get("debug", False)),
                   scaling_modifier=modifier)
    snapshots, seen = [], set()
    for obj in objects:
        object_id = getattr(obj, "object_id", None)
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("Each visibility object requires a nonempty string object_id")
        if object_id in seen:
            raise ValueError(f"Duplicate object_id: {object_id}")
        seen.add(object_id)
        object_time = _finite_number(getattr(obj, "timestamp", None), f"{object_id} timestamp")
        if abs(object_time - timestamp) > 1e-9:
            raise ValueError(f"{object_id}: proxy timestamp does not match requested timestamp")
        state = getattr(obj, "proxy", None)
        if not isinstance(state, GaussianState):
            raise ValueError(f"{object_id}: proxy must be a resolved GaussianState")
        state.validate()
        transformed = _transformed_state(state, obj.object_to_world)
        snapshots.append(ProxySnapshot(object_id, transformed, object_time))
    return tuple(snapshots), validated_camera, options


def _load_runtime(settings):
    try:
        import torch
    except ImportError as exc:
        raise RendererUnavailable("Visibility analysis requires the configured PyTorch/CUDA environment") from exc
    if not torch.cuda.is_available():
        raise RendererUnavailable("Visibility analysis requires accessible CUDA; no fallback rasterizer is used")
    extension = settings.get("extension_path")
    if extension is not None:
        extension = Path(extension).expanduser()
        extension = (extension if extension.is_absolute() else PROJECT_ROOT / extension).resolve()
    try:
        upstream = configure_upstream_imports(extension)
        import gaussian_renderer
        import diff_gaussian_rasterization
        from diff_gaussian_rasterization import _C
        from scene.gaussian_model import GaussianModel
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
    except (ImportError, OSError, FileNotFoundError) as exc:
        raise RendererUnavailable("Cannot import the pinned Dynamic-LapisGS renderer and CUDA extension") from exc
    renderer_path = Path(gaussian_renderer.__file__).resolve()
    rasterizer_path = Path(diff_gaussian_rasterization.__file__).resolve()
    if not renderer_path.is_relative_to(upstream.resolve()):
        raise RendererUnavailable(f"Gaussian renderer was imported outside the pinned checkout: {renderer_path}")
    if extension is not None and not rasterizer_path.is_relative_to(extension):
        raise RendererUnavailable(f"Rasterizer was imported outside extension_path: {rasterizer_path}")
    binary_path = Path(_C.__file__).resolve()
    if extension is not None and not binary_path.is_relative_to(extension):
        raise RendererUnavailable(f"CUDA rasterizer binary was imported outside extension_path: {binary_path}")
    # Generic scene/utils package names may already exist in sys.modules. Path
    # setup alone cannot replace those cached dependencies from another checkout.
    component_paths = {name: Path(inspect.getfile(component)).resolve()
                       for name, component in (("model_module", GaussianModel),
                                               ("camera_module", MiniCam),
                                               ("projection_module", getProjectionMatrix))}
    for name, path in component_paths.items():
        if not path.is_relative_to(upstream.resolve()):
            raise RendererUnavailable(f"{name} was imported outside the pinned checkout: {path}")
    metadata = {"backend": "upstream_cuda", "renderer": "gaussian_renderer.render",
                "renderer_module": str(renderer_path), "rasterizer_module": str(rasterizer_path),
                "rasterizer_binary": str(binary_path),
                **{name: str(path) for name, path in component_paths.items()}}
    return torch, gaussian_renderer.render, GaussianModel, MiniCam, getProjectionMatrix, metadata


@dataclass(slots=True)
class PreparedScene:
    torch: Any
    original_render: Any
    joint_model: Any
    object_models: tuple
    object_ids: tuple[str, ...]
    object_slots: Any
    camera: Any
    pipe: Any
    black: Any
    scaling_modifier: float
    metadata: dict[str, Any]

    def render(self, model, colors=None, background=None):
        return self.original_render(self.camera, model, self.pipe,
                                    self.black if background is None else background,
                                    scaling_modifier=self.scaling_modifier, override_color=colors)

    def joint_colors(self, group_start):
        colors = self.torch.zeros((len(self.object_slots), 3), dtype=self.torch.float32,
                                  device=self.black.device)
        for slot in range(group_start, min(group_start + 3, len(self.object_ids))):
            colors[self.object_slots == slot, slot - group_start] = 1.
        return colors

    def solo_colors(self, index):
        count = len(self.object_models[index].get_xyz)
        colors = self.torch.zeros((count, 3), dtype=self.torch.float32, device=self.black.device)
        colors[:, 0] = 1.
        return colors


def prepare_scene(snapshots, camera, settings):
    """Upload a validated scene. Caller owns the GPU guard and no-grad context."""
    torch, render, model_type, camera_type, projection_fn, metadata = _load_runtime(settings)
    device = torch.device("cuda")
    degree = max((snapshot.proxy.sh_degree for snapshot in snapshots), default=0)
    models = []
    for snapshot in snapshots:
        arrays = snapshot.proxy.arrays
        model = model_type(degree)
        model.active_sh_degree = degree
        model._xyz = torch.as_tensor(arrays["xyz"], dtype=torch.float32, device=device)
        model._rotation = torch.as_tensor(arrays["rotation"], dtype=torch.float32, device=device)
        model._scaling = torch.as_tensor(arrays["scale"], dtype=torch.float32, device=device)
        model._opacity = torch.as_tensor(arrays["opacity"], dtype=torch.float32, device=device)
        # Zero padding preserves each object's SH when scene degrees differ.
        features = torch.zeros((snapshot.proxy.count, (degree + 1)**2, 3),
                               dtype=torch.float32, device=device)
        features[:, :arrays["sh"].shape[2]] = torch.as_tensor(
            arrays["sh"].transpose(0, 2, 1).copy(), dtype=torch.float32, device=device)
        model._features_dc, model._features_rest = features[:, :1], features[:, 1:]
        models.append(model)
    joint = model_type(degree)
    joint.active_sh_degree = degree
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
        setattr(joint, name, torch.cat([getattr(model, name) for model in models], dim=0))
    counts = [snapshot.proxy.count for snapshot in snapshots]
    slots = torch.repeat_interleave(torch.arange(len(snapshots), device=device),
                                    torch.tensor(counts, device=device))
    world = torch.as_tensor(np.asarray(camera["world_view"], dtype=np.float32).T.copy(), device=device)
    projection = projection_fn(camera["near"], camera["far"], camera["fovx"], camera["fovy"]).transpose(0, 1).to(device=device, dtype=torch.float32)
    viewpoint = camera_type(camera["width"], camera["height"], camera["fovy"], camera["fovx"],
                            camera["near"], camera["far"], world, world @ projection)
    metadata.update(gaussian_count=sum(counts), object_count=len(snapshots),
                    per_object_gaussian_counts=dict(zip((s.object_id for s in snapshots), counts)),
                    sh_degree=degree, alpha_convention="accepted_splat_alpha_compositing",
                    depth_order="tile_sorted_gaussian_center_camera_z")
    return PreparedScene(torch, render, joint, tuple(models), tuple(s.object_id for s in snapshots),
                         slots, viewpoint, SimpleNamespace(debug=settings["debug"],
                         compute_cov3D_python=False, convert_SHs_python=False),
                         torch.zeros(3, dtype=torch.float32, device=device), settings["scaling_modifier"], metadata)
