"""Object contributions accumulated by the existing CUDA compositing loop.

RGB basis colors on black turn the renderer's three color channels into three
object contribution channels. Every object remains an occluder in every joint
group. Only reductions, never alpha compositing, happen outside the rasterizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tools.content_preparation.assets import GaussianState
from tools.content_preparation.renderer_adapter import GPUExecutionGuard
from .upstream import prepare_scene, validate_and_snapshot


@dataclass(frozen=True, slots=True)
class VisibilityObject:
    object_id: str
    proxy: GaussianState
    timestamp: float
    object_to_world: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass(frozen=True, slots=True)
class ObjectContribution:
    object_id: str
    A_visible: float
    A_solo: float


@dataclass(frozen=True, slots=True)
class ContributionDiagnostics:
    joint_maps: dict[str, np.ndarray]
    solo_maps: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class ContributionMeasurements:
    timestamp: float
    resolution: tuple[int, int]
    per_object: dict[str, ObjectContribution]
    diagnostics: ContributionDiagnostics | None
    metadata: dict[str, Any]


def measure_contributions(
    objects: Sequence[VisibilityObject],
    camera: Mapping[str, Any],
    timestamp: float,
    *,
    diagnostics: bool = False,
    renderer_settings: Mapping[str, Any] | None = None,
) -> ContributionMeasurements:
    """Render joint and solo snapshots, returning alpha-weighted pixel areas.

Resolution is (width, height); optional maps are CPU-owned float32 (height,
width) arrays. Without diagnostics, only the compact area array leaves CUDA.
Empty scenes/objects do not require CUDA. Nonempty input never falls back to a
CPU rasterizer. The caller must not mutate snapshots during this call.
"""
    if not isinstance(diagnostics, (bool, np.bool_)):
        raise ValueError("diagnostics must be a boolean")
    snapshots, camera, settings = validate_and_snapshot(objects, camera, timestamp, renderer_settings)
    object_ids = tuple(snapshot.object_id for snapshot in snapshots)
    counts = [snapshot.proxy.count for snapshot in snapshots]
    width, height = camera["width"], camera["height"]
    maps = ContributionDiagnostics({}, {}) if diagnostics else None
    metadata = {"backend": "upstream_cuda", "renderer": "gaussian_renderer.render",
                "object_count": len(snapshots), "gaussian_count": sum(counts),
                "per_object_gaussian_counts": dict(zip(object_ids, counts)),
                "joint_passes": 0, "solo_passes": 0, "area_units": "alpha_weighted_pixels",
                "accumulation_dtype": "float64", "diagnostics": bool(diagnostics)}
    if not sum(counts):
        if maps is not None:
            for object_id in object_ids:
                maps.joint_maps[object_id] = np.zeros((height, width), dtype=np.float32)
                maps.solo_maps[object_id] = np.zeros((height, width), dtype=np.float32)
        return ContributionMeasurements(float(timestamp), (width, height),
                                        {key: ObjectContribution(key, 0., 0.) for key in object_ids}, maps, metadata)

    scene = model = areas = colors = result = image = selected = None
    with GPUExecutionGuard():
        try:
            # prepare_scene checks CUDA availability before importing extensions.
            scene = prepare_scene(snapshots, camera, settings)
            torch = scene.torch
            with torch.no_grad():
                areas = torch.zeros((len(snapshots), 2), dtype=torch.float64, device=scene.black.device)
                for start in range(0, len(snapshots), 3):
                    colors = scene.joint_colors(start)
                    result = scene.render(scene.joint_model, colors)
                    image = result["render"]
                    stop = min(start + 3, len(snapshots))
                    selected = image[:stop - start]
                    areas[start:stop, 0] = selected.sum(dim=(1, 2), dtype=torch.float64)
                    if maps is not None:
                        exported = selected.detach().cpu().numpy().copy()
                        for channel, slot in enumerate(range(start, stop)):
                            maps.joint_maps[object_ids[slot]] = exported[channel].copy()
                    metadata["joint_passes"] += 1
                    colors = result = image = selected = None
                for slot, model in enumerate(scene.object_models):
                    if counts[slot]:
                        colors = scene.solo_colors(slot)
                        result = scene.render(model, colors)
                        image = result["render"][0]
                        areas[slot, 1] = image.sum(dtype=torch.float64)
                        if maps is not None:
                            maps.solo_maps[object_ids[slot]] = image.detach().cpu().numpy().copy()
                        metadata["solo_passes"] += 1
                        colors = result = image = None
                    elif maps is not None:
                        maps.solo_maps[object_ids[slot]] = np.zeros((height, width), dtype=np.float32)
                reduced = areas.cpu().numpy()
            metadata.update(scene.metadata)
            per_object = {key: ObjectContribution(key, float(reduced[slot, 0]), float(reduced[slot, 1]))
                          for slot, key in enumerate(object_ids)}
            return ContributionMeasurements(float(timestamp), (width, height), per_object, maps, metadata)
        finally:
            # Drop session references; do not synchronize/empty the global CUDA
            # allocator after each pass. Scalar/map transfers above finish reads.
            scene = model = areas = colors = result = image = selected = None
