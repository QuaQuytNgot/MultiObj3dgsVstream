"""Compatibility with current Open3D/NumPy while leaving submodule source intact.

Original function bodies run with isolated compatibility namespaces. Open3D's
obsolete headless=False keyword is removed and NeRF image RGB uses uint8. The
launcher installs a process-local scene callback; source files remain untouched.
"""
import importlib
from pathlib import Path
from types import FunctionType

import numpy as np

from .paths import configure_upstream_imports


def render_2d_image(*args, **kwargs):
    configure_upstream_imports()
    module = importlib.import_module("dataset_prepare")
    rendering = module.rendering

    class RenderingAPI:
        def __getattr__(self, name):
            return getattr(rendering, name)

        def OffscreenRenderer(self, *dimensions, **options):
            options.pop("headless", None)
            return rendering.OffscreenRenderer(*dimensions, **options)

    original = module.render_2d_image
    namespace = dict(original.__globals__, rendering=RenderingAPI())
    function = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    return function(*args, **kwargs)


def rescale_image(*args, **kwargs):
    configure_upstream_imports()
    return importlib.import_module("dataset_prepare").rescale_image(*args, **kwargs)


def store_point_cloud(path, xyz, rgb):
    """Write the upstream initialization PLY schema without signed-byte RGB."""
    from plyfile import PlyData, PlyElement

    xyz, rgb = np.asarray(xyz), np.asarray(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("Point coordinates and RGB must both have shape N x 3")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError("Point coordinates must be finite and RGB must be in [0, 255]")
    data = np.zeros(len(xyz), dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
                    + [(name, "u1") for name in ("red", "green", "blue")])
    for index, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        data[name] = rgb[:, index]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def compatible_nerf_loader(readers):
    """Clone the original NeRF loader with the local workspace's uint8 image fix."""
    original = readers.readCamerasFromTransforms
    numpy_api, image_api = original.__globals__["np"], original.__globals__["Image"]

    class NumpyAPI:
        byte = numpy_api.uint8

        def __getattr__(self, name):
            return getattr(numpy_api, name)

    class ImageAPI:
        def __getattr__(self, name):
            return getattr(image_api, name)

        def fromarray(self, array, mode=None):
            if mode == "RGB" and array.dtype == numpy_api.uint8:
                return image_api.fromarray(array)
            return image_api.fromarray(array, mode)

    cameras = FunctionType(
        original.__code__, dict(original.__globals__, np=NumpyAPI(), Image=ImageAPI()),
        original.__name__, original.__defaults__, original.__closure__,
    )
    loader = readers.readNerfSyntheticInfo
    return FunctionType(
        loader.__code__, dict(loader.__globals__, readCamerasFromTransforms=cameras),
        loader.__name__, loader.__defaults__, loader.__closure__,
    )


def configure_training_compatibility():
    """Install only the NeRF dataset callback for this trainer/render subprocess."""
    configure_upstream_imports()
    scene = importlib.import_module("scene")
    readers = importlib.import_module("scene.dataset_readers")
    scene.sceneLoadTypeCallbacks = dict(
        scene.sceneLoadTypeCallbacks, Blender=compatible_nerf_loader(readers)
    )
