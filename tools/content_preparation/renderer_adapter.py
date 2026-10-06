"""Decoded-state bridge to the unmodified upstream Gaussian renderer.

``upstream_cuda`` is the production backend. ``cpu_fixture`` is an explicitly
selected degree-zero analytic test backend; it is never selected on GPU failure.
Alpha and coverage-conditioned expected depth are obtained using two additional
sequential calls to the same upstream renderer with overridden colors.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .paths import PROJECT_ROOT, configure_upstream_imports

ROOT = PROJECT_ROOT
_GPU_LOCK = threading.Lock()
_GPU_STATS = {"active": 0, "peak_active": 0, "views": 0}


class RendererUnavailable(RuntimeError):
    """The explicitly requested backend cannot run in this environment."""


class GPUExecutionGuard:
    """Reject concurrent GPU models; a whole view is one sequential work unit."""
    def __enter__(self):
        if not _GPU_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent GPU work is forbidden: render one object/quality/view at a time.")
        _GPU_STATS["active"] += 1
        _GPU_STATS["peak_active"] = max(_GPU_STATS["peak_active"], _GPU_STATS["active"])
        return self

    def __exit__(self, exc_type, exc, tb):
        _GPU_STATS["active"] -= 1
        _GPU_STATS["views"] += 1
        _GPU_LOCK.release()

    @staticmethod
    def statistics():
        return dict(_GPU_STATS)


def cleanup_gpu():
    """Release Python references before CUDA allocator cleanup; safe without torch."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _arrays(state):
    arrays = state.arrays if hasattr(state, "arrays") else state["arrays"]
    return {name: np.asarray(value) for name, value in arrays.items()}


def camera_parameters(sample: Mapping[str, Any], settings: Mapping[str, Any]):
    """Canonical orbit camera; every quality receives the same sample/settings.

    With ``up_axis=y`` (default), azimuth zero lies on +Z and elevation points
    toward +Y. With ``up_axis=z``, azimuth zero lies on +Y and elevation points
    toward +Z. Positive azimuth rotates toward +X in both conventions. Camera
    +Z points forward and image +Y points downward.
    ``scale`` is a projected-size multiplier implemented by distance / scale.
    """
    width, height = int(settings.get("width", 256)), int(settings.get("height", 256))
    if width <= 0 or height <= 0:
        raise ValueError("Render width and height must be positive.")
    az = math.radians(float(sample.get("azimuth", 0)))
    el = math.radians(float(sample.get("elevation", 0)))
    distance = float(sample.get("distance", settings.get("distance", 3)))
    scale = float(sample.get("scale", 1))
    if distance <= 0 or scale <= 0:
        raise ValueError("Camera distance and projected scale must be positive.")
    target = np.asarray(sample.get("target", settings.get("center", [0, 0, 0])), dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Camera target/center must be a finite three-vector.")
    up_axis = settings.get("up_axis", "y")
    if up_axis not in ("y", "z"):
        raise ValueError("renderer.up_axis must be 'y' or 'z'.")
    if up_axis == "y":
        direction = np.array([math.sin(az)*math.cos(el), math.sin(el), math.cos(az)*math.cos(el)])
        up = np.array([0., 1., 0.])
        alternate_up = np.array([0., 0., 1.])
    else:
        direction = np.array([math.sin(az)*math.cos(el), math.cos(az)*math.cos(el), math.sin(el)])
        up = np.array([0., 0., 1.])
        alternate_up = np.array([0., 1., 0.])
    eye = target + distance / scale * direction
    forward = (target-eye) / np.linalg.norm(target-eye)
    if abs(float(forward @ up)) > .999:
        up = alternate_up
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    world_view = np.eye(4, dtype=np.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = -rotation @ eye
    fovy = math.radians(float(settings.get("fov_degrees", 60)))
    if not 0 < fovy < math.pi:
        raise ValueError("fov_degrees must lie strictly between 0 and 180.")
    fovx = 2*math.atan(math.tan(fovy/2)*width/height)
    near, far = float(settings.get("near", .01)), float(settings.get("far", 100))
    if not 0 < near < far:
        raise ValueError("Render clipping planes require 0 < near < far.")
    return SimpleNamespace(width=width, height=height, eye=eye, rotation=rotation,
                           world_view=world_view, fovy=fovy, fovx=fovx, near=near, far=far,
                           distance=distance/scale, up_axis=up_axis)


def _upstream_render(state, sample, settings):
    try:
        import torch
    except ImportError as exc:
        raise RendererUnavailable("upstream_cuda requires the Dynamic-LapisGS PyTorch environment.") from exc
    if not torch.cuda.is_available():
        raise RendererUnavailable("upstream_cuda requires accessible CUDA. Activate the project environment and check GPU access; explicitly select cpu_fixture only for analytic fixtures.")
    try:
        configure_upstream_imports(settings.get("vendor_path"))
    except FileNotFoundError as exc:
        raise RendererUnavailable(str(exc)) from exc
    try:
        from gaussian_renderer import render as original_render
        from scene.gaussian_model import GaussianModel
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
    except (ImportError, OSError) as exc:
        raise RendererUnavailable("Cannot import original CUDA renderer/extensions. Build the repo extensions or configure vendor_path to the verified upstream extension.") from exc
    arrays = _arrays(state)
    k = arrays["sh"].shape[2]
    degree = int(round(math.sqrt(k)-1))
    if (degree+1)**2 != k:
        raise ValueError("SH coefficients must have shape N x 3 x (degree+1)^2.")
    camera = camera_parameters(sample, settings)
    background_values = settings.get("background", [0., 0., 0.])
    if len(background_values) != 3:
        raise ValueError("Renderer background requires three RGB values.")
    model = viewpoint = result = rgb_tensor = alpha_tensor = depth_tensor = None
    background = black = colors = depth_colors = camera_depth = world = projection = None
    with GPUExecutionGuard():
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model = GaussianModel(degree)
                model.active_sh_degree = degree
                model._xyz = torch.as_tensor(arrays["xyz"], dtype=torch.float32, device="cuda")
                model._rotation = torch.as_tensor(arrays["rotation"], dtype=torch.float32, device="cuda")
                model._scaling = torch.as_tensor(arrays["scale"], dtype=torch.float32, device="cuda")
                model._opacity = torch.as_tensor(arrays["opacity"], dtype=torch.float32, device="cuda")
                features = torch.as_tensor(arrays["sh"], dtype=torch.float32, device="cuda").transpose(1, 2).contiguous()
                model._features_dc, model._features_rest = features[:, :1], features[:, 1:]
                world = torch.as_tensor(camera.world_view.T.copy(), device="cuda")
                projection = getProjectionMatrix(camera.near, camera.far, camera.fovx, camera.fovy).transpose(0, 1).cuda()
                viewpoint = MiniCam(camera.width, camera.height, camera.fovy, camera.fovx,
                                    camera.near, camera.far, world, world @ projection)
                pipe = SimpleNamespace(debug=bool(settings.get("debug", False)),
                                       compute_cov3D_python=False, convert_SHs_python=False)
                background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
                black = torch.zeros(3, device="cuda")
                result = original_render(viewpoint, model, pipe, background)
                rgb_tensor = result["render"]
                rgb = rgb_tensor.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy().copy()
                result = rgb_tensor = None
                colors = torch.ones((len(arrays["xyz"]), 3), device="cuda")
                result = original_render(viewpoint, model, pipe, black, override_color=colors)
                alpha_tensor = result["render"][0]
                alpha = alpha_tensor.detach().clamp(0, 1).cpu().numpy().copy()
                result = alpha_tensor = colors = None
                camera_depth = model.get_xyz @ world[:3, 2] + world[3, 2]
                depth_colors = (camera_depth/camera.far)[:, None].expand(-1, 3).contiguous()
                result = original_render(viewpoint, model, pipe, black, override_color=depth_colors)
                depth_tensor = result["render"][0]
                depth_numerator = depth_tensor.detach().cpu().numpy().copy()*camera.far
                depth = np.divide(depth_numerator, alpha, out=np.zeros_like(alpha), where=alpha > 1e-8)
                return {"rgb": rgb, "alpha": alpha, "depth": depth,
                        "metadata": {"backend": "upstream_cuda", "renderer": "gaussian_renderer.render",
                                     "up_axis": camera.up_axis,
                                     "depth_convention": "coverage_conditioned_expected_camera_z",
                                     "alpha_convention": "one_minus_transmittance", "sequential_passes": 3,
                                     "gaussian_count": len(arrays["xyz"]),
                                     "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())}}
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA out of memory while rendering one decoded state/view. Target is 6 GB: reduce configured render width/height or Gaussian count; LPIPS must remain batch_size=1. No settings were changed automatically.") from exc
        finally:
            model = viewpoint = result = rgb_tensor = alpha_tensor = depth_tensor = None
            background = black = colors = depth_colors = camera_depth = world = projection = None
            if "features" in locals():
                del features
            cleanup_gpu()


def _quaternion_rotation(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Gaussian quaternion must be nonzero.")
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _cpu_fixture_render(state, sample, settings):
    arrays = _arrays(state)
    if arrays["sh"].shape[2] != 1:
        raise ValueError("cpu_fixture supports SH degree zero only; use upstream_cuda for real content.")
    camera = camera_parameters(sample, settings)
    xyz = np.asarray(arrays["xyz"], dtype=np.float64)
    xyz_cam = xyz @ camera.rotation.T + camera.world_view[:3, 3]
    fx, fy = camera.width/(2*math.tan(camera.fovx/2)), camera.height/(2*math.tan(camera.fovy/2))
    transmittance = np.ones((camera.height, camera.width), dtype=np.float64)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.float64)
    depth_num = np.zeros((camera.height, camera.width), dtype=np.float64)
    colors = np.maximum(0, .28209479177387814*arrays["sh"][:, :, 0] + .5)
    # Stable sort gives deterministic order when several fixture points share Z.
    for i in np.argsort(xyz_cam[:, 2], kind="stable"):
        x, y, z = xyz_cam[i]
        if z <= camera.near or z >= camera.far:
            continue
        u, v = fx*x/z + camera.width/2-.5, fy*y/z + camera.height/2-.5
        rotation = _quaternion_rotation(arrays["rotation"][i])
        cov = rotation @ np.diag(np.exp(2*arrays["scale"][i])) @ rotation.T
        cov_cam = camera.rotation @ cov @ camera.rotation.T
        jacobian = np.array([[fx/z, 0, -fx*x/z**2], [0, fy/z, -fy*y/z**2]])
        cov_pixel = jacobian @ cov_cam @ jacobian.T + .3*np.eye(2)
        radius = 3*math.sqrt(float(np.linalg.eigvalsh(cov_pixel).max()))
        left, right = max(0, int(math.floor(u-radius))), min(camera.width, int(math.ceil(u+radius+1)))
        top, bottom = max(0, int(math.floor(v-radius))), min(camera.height, int(math.ceil(v+radius+1)))
        if left >= right or top >= bottom:
            continue
        yy, xx = np.mgrid[top:bottom, left:right]
        delta = np.stack([xx-u, yy-v], axis=-1)
        exponent = -.5*np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov_pixel), delta)
        opacity = 1/(1+np.exp(-float(np.clip(arrays["opacity"][i, 0], -80, 80))))
        alpha = np.minimum(.99, opacity*np.exp(exponent))
        contribution = transmittance[top:bottom, left:right]*alpha
        image[top:bottom, left:right] += contribution[:, :, None]*colors[i]
        depth_num[top:bottom, left:right] += contribution*z
        transmittance[top:bottom, left:right] *= 1-alpha
    alpha = 1-transmittance
    image += transmittance[:, :, None]*np.asarray(settings.get("background", [0, 0, 0]))
    depth = np.divide(depth_num, alpha, out=np.zeros_like(alpha), where=alpha > 1e-8)
    return {"rgb": np.clip(image, 0, 1).astype(np.float32), "alpha": alpha.astype(np.float32),
            "depth": depth.astype(np.float32),
            "metadata": {"backend": "cpu_fixture", "renderer": "analytic_degree_zero_fixture",
                         "up_axis": camera.up_axis,
                         "depth_convention": "coverage_conditioned_expected_camera_z",
                         "alpha_convention": "one_minus_transmittance", "production_renderer": False}}


def render(state, sample: Mapping[str, Any], settings: Mapping[str, Any] | None = None):
    """Render one decoded state and return CPU numpy RGB/alpha/depth arrays."""
    settings = dict(settings or {})
    backend = settings.get("backend", "upstream_cuda")
    if backend == "upstream_cuda":
        return _upstream_render(state, sample, settings)
    if backend == "cpu_fixture":
        return _cpu_fixture_render(state, sample, settings)
    raise ValueError(f"Unknown renderer backend {backend!r}; choose upstream_cuda or cpu_fixture.")
