"""Object contributions accumulated by the existing CUDA compositing loop.

RGB basis colors on black turn the renderer's three color channels into three
object contribution channels. Every object remains an occluder in every joint
group. Only reductions, never alpha compositing, happen outside the rasterizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import copy
from time import perf_counter
from typing import Any

import numpy as np

from tools.content_preparation.assets import GaussianState
from tools.content_preparation.renderer_adapter import GPUExecutionGuard
from .upstream import _load_runtime, prepare_scene, validate_and_snapshot


@dataclass(frozen=True, slots=True, init=False)
class VisibilityObject:
    """A resolved state; source provenance never controls rasterization.

    ``proxy`` remains a constructor/property alias for existing demo callers.
    Asset selection, loading and byte accounting belong outside the renderer.
    """
    object_id: str
    gaussian_state: GaussianState
    timestamp: float
    object_to_world: np.ndarray = field(default_factory=lambda: np.eye(4))
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, object_id, gaussian_state=None, timestamp=None,
                 object_to_world=None, source_metadata=None, *, proxy=None):
        if gaussian_state is not None and proxy is not None:
            raise ValueError("Supply gaussian_state or its legacy proxy alias, not both")
        if source_metadata is not None and not isinstance(source_metadata, Mapping):
            raise ValueError("source_metadata must be a mapping")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "gaussian_state", gaussian_state if proxy is None else proxy)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "object_to_world", np.eye(4) if object_to_world is None else object_to_world)
        object.__setattr__(self, "source_metadata", copy.deepcopy(dict(source_metadata or {})))

    @property
    def proxy(self):
        return self.gaussian_state


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
    profile: bool = False,
) -> ContributionMeasurements:
    """Render joint and solo snapshots, returning alpha-weighted pixel areas.

Resolution is (width, height); optional maps are CPU-owned float32 (height,
width) arrays. Without diagnostics, only the compact area array leaves CUDA.
Empty scenes/objects do not require CUDA. Nonempty input never falls back to a
CPU rasterizer. The caller must not mutate snapshots during this call.
"""
    wall_start = perf_counter()
    if not isinstance(profile, (bool, np.bool_)):
        raise ValueError("profile must be a boolean")
    if not isinstance(diagnostics, (bool, np.bool_)):
        raise ValueError("diagnostics must be a boolean")
    snapshots, camera, settings = validate_and_snapshot(objects, camera, timestamp, renderer_settings)
    object_ids = tuple(snapshot.object_id for snapshot in snapshots)
    counts = [snapshot.gaussian_state.count for snapshot in snapshots]
    width, height = camera["width"], camera["height"]
    maps = ContributionDiagnostics({}, {}) if diagnostics else None
    metadata = {"backend": "upstream_cuda", "renderer": "gaussian_renderer.render",
                "object_count": len(snapshots), "gaussian_count": sum(counts),
                "per_object_gaussian_counts": dict(zip(object_ids, counts)),
                "joint_passes": 0, "solo_passes": 0, "area_units": "alpha_weighted_pixels",
                "accumulation_dtype": "float64", "diagnostics": bool(diagnostics)}
    timings = None
    if profile:
        timings = {key: 0. for key in (
            "preparation_upload_ms", "joint_render_ms", "solo_render_ms",
            "gpu_reduction_ms", "scalar_transfer_ms", "identity_colors_ms",
            "diagnostic_transfer_ms",
        )}
        timings.update({key: 0 for key in (
            "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
            "baseline_cuda_allocated_bytes", "baseline_cuda_reserved_bytes",
            "incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes",
        )})
        timings["validation_snapshot_ms"] = (perf_counter() - wall_start) * 1000
        metadata["profile"] = timings
    if not sum(counts):
        if maps is not None:
            for object_id in object_ids:
                maps.joint_maps[object_id] = np.zeros((height, width), dtype=np.float32)
                maps.solo_maps[object_id] = np.zeros((height, width), dtype=np.float32)
        if timings is not None:
            timings["renderer_wall_ms"] = (perf_counter() - wall_start) * 1000
        return ContributionMeasurements(float(timestamp), (width, height),
                                        {key: ObjectContribution(key, 0., 0.) for key in object_ids}, maps, metadata)

    scene = model = areas = colors = result = image = selected = None
    with GPUExecutionGuard():
        try:
            # prepare_scene checks CUDA availability before importing extensions.
            events = {key: [] for key in (
                "joint_render_ms", "solo_render_ms", "gpu_reduction_ms", "identity_colors_ms")}
            if timings is not None:
                torch = _load_runtime(settings)[0]
                torch.cuda.synchronize()
                timings["baseline_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
                timings["baseline_cuda_reserved_bytes"] = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                prepare_start = perf_counter()
            scene = prepare_scene(snapshots, camera, settings)
            torch = scene.torch
            if timings is not None:
                torch.cuda.synchronize()
                timings["preparation_upload_ms"] = (perf_counter() - prepare_start) * 1000

            def begin_event(key):
                if timings is None:
                    return None
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[key].append(pair)
                pair[0].record()
                return pair

            def end_event(pair):
                if pair is not None:
                    pair[1].record()

            with torch.no_grad():
                areas = torch.zeros((len(snapshots), 2), dtype=torch.float64, device=scene.black.device)
                for start in range(0, len(snapshots), 3):
                    pair = begin_event("identity_colors_ms")
                    colors = scene.joint_colors(start)
                    end_event(pair)
                    pair = begin_event("joint_render_ms")
                    result = scene.render(scene.joint_model, colors)
                    end_event(pair)
                    image = result["render"]
                    stop = min(start + 3, len(snapshots))
                    selected = image[:stop - start]
                    pair = begin_event("gpu_reduction_ms")
                    areas[start:stop, 0] = selected.sum(dim=(1, 2), dtype=torch.float64)
                    end_event(pair)
                    if maps is not None:
                        export_start = perf_counter()
                        exported = selected.detach().cpu().numpy().copy()
                        for channel, slot in enumerate(range(start, stop)):
                            maps.joint_maps[object_ids[slot]] = exported[channel].copy()
                        if timings is not None:
                            timings["diagnostic_transfer_ms"] += (perf_counter() - export_start) * 1000
                    metadata["joint_passes"] += 1
                    colors = result = image = selected = None
                for slot, model in enumerate(scene.object_models):
                    if counts[slot]:
                        pair = begin_event("identity_colors_ms")
                        colors = scene.solo_colors(slot)
                        end_event(pair)
                        pair = begin_event("solo_render_ms")
                        result = scene.render(model, colors)
                        end_event(pair)
                        image = result["render"][0]
                        pair = begin_event("gpu_reduction_ms")
                        areas[slot, 1] = image.sum(dtype=torch.float64)
                        end_event(pair)
                        if maps is not None:
                            export_start = perf_counter()
                            maps.solo_maps[object_ids[slot]] = image.detach().cpu().numpy().copy()
                            if timings is not None:
                                timings["diagnostic_transfer_ms"] += (perf_counter() - export_start) * 1000
                        metadata["solo_passes"] += 1
                        colors = result = image = None
                    elif maps is not None:
                        maps.solo_maps[object_ids[slot]] = np.zeros((height, width), dtype=np.float32)
                if timings is not None:
                    torch.cuda.synchronize()
                    transfer_start = perf_counter()
                reduced = areas.cpu().numpy()
                if timings is not None:
                    timings["scalar_transfer_ms"] = (perf_counter() - transfer_start) * 1000
                    for key, pairs in events.items():
                        timings[key] = sum(first.elapsed_time(last) for first, last in pairs)
                    for kind in ("allocated", "reserved"):
                        peak = getattr(torch.cuda, f"max_memory_{kind}")()
                        timings[f"peak_cuda_{kind}_bytes"] = peak
                        timings[f"incremental_peak_{kind}_bytes"] = max(
                            0, peak - timings[f"baseline_cuda_{kind}_bytes"])
            metadata.update(scene.metadata)
            per_object = {key: ObjectContribution(key, float(reduced[slot, 0]), float(reduced[slot, 1]))
                          for slot, key in enumerate(object_ids)}
            if timings is not None:
                timings["renderer_wall_ms"] = (perf_counter() - wall_start) * 1000
            return ContributionMeasurements(float(timestamp), (width, height), per_object, maps, metadata)
        finally:
            # Drop session references; do not synchronize/empty the global CUDA
            # allocator after each pass. Scalar/map transfers above finish reads.
            scene = model = areas = colors = result = image = selected = None
