"""Frozen Gaussian-subset proxy and silhouette/depth/alpha validation.

Selection is made once from the highest delivered decoded representation, then
tracked by stable IDs through media time. It does not change when ABR quality
changes. This is a baseline visibility proxy, not a final display representation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .assets import file_hash, load_state, save_state, state_hash
from .quality_profile import _write_json
from .renderer_adapter import render


def stable_subset(state, ids: Sequence[int]):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) != len(np.unique(ids)):
        raise ValueError("Proxy selection IDs must be unique.")
    locations = {int(value): index for index, value in enumerate(state.ids)}
    missing = [int(value) for value in ids if int(value) not in locations]
    if missing:
        raise ValueError(f"Frozen proxy IDs disappeared in this frame: {missing[:8]}. Provide a persistent-ID lineage or a separately tracked geometry proxy; automatic reselection would make the proxy unstable.")
    subset = state.subset(np.array([locations[int(value)] for value in ids], dtype=np.int64))
    subset.metadata.update({"asset_role": "quality_independent_visibility_proxy",
                            "proxy_backend": "fixed_gaussian_subset", "proxy_selection_frozen": True})
    return subset


def generate_proxy(state, proxy_config: Mapping | None, output_dir, frame=None):
    """Generate/update one frame of a frozen proxy asset, returning its record.

    ``state`` must be the highest-quality *decoded* state. Subsequent calls to
    the same output directory reuse the original selection, including its order.
    Paths inside proxy.json are relative to that descriptor's directory.
    """
    config = dict(proxy_config or {})
    backend = config.get("backend", "fixed_gaussian_subset")
    if backend not in {"fixed_gaussian_subset", "gaussian_subset"}:
        raise ValueError(f"Unsupported proxy backend {backend!r}; implement a ProxyBackend adapter for another backend.")
    count = int(config.get("count", config.get("max_gaussians", 256)))
    if count <= 0:
        raise ValueError("Proxy Gaussian count must be positive.")
    if state.count == 0:
        raise ValueError("Cannot generate a visibility proxy from an empty state.")
    frame = int(state.metadata.get("frame", 0) if frame is None else frame)
    output_dir = Path(output_dir)
    descriptor_path = output_dir/"proxy.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if descriptor_path.exists():
        descriptor = json.loads(descriptor_path.read_text())
        if descriptor.get("schema") != "dynamic-lapisgs.proxy.v1":
            raise ValueError("Unsupported proxy descriptor schema.")
        if descriptor["configured_count"] != count or descriptor["backend"] != "fixed_gaussian_subset":
            raise ValueError("Existing proxy selection uses another configuration; choose a new output directory.")
        if config.get("require_stable_ids", True) and not state.metadata.get("stable_ids", False) and frame != descriptor["selection_frame"]:
            raise ValueError("Tracking a frozen proxy across frames requires verified persistent Gaussian IDs.")
        ids = np.asarray(descriptor["selected_ids"], dtype=np.int64)
    else:
        # Highest opacity first; stable IDs break ties deterministically. Select
        # IDs only once; all subsequent states use this exact frozen selection.
        ranking = np.lexsort((state.ids, -state.arrays["opacity"][:, 0]))
        ids = np.sort(state.ids[ranking[:min(count, state.count)]])
        descriptor = {"schema": "dynamic-lapisgs.proxy.v1", "backend": "fixed_gaussian_subset",
                      "quality_independent": True, "display_asset": False,
                      "selection_policy": "first_highest_decoded_frame_opacity_descending_ID_tiebreak",
                      "selection_frame": frame, "configured_count": count, "selected_count": len(ids),
                      "selected_ids": [int(value) for value in ids],
                      "selected_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                      "selection_source_state_sha256": state_hash(state),
                      "depth_convention": "coverage_conditioned_expected_camera_z", "frames": []}
    proxy_state = stable_subset(state, ids)
    record = save_state(output_dir/f"frame_{frame:06d}.npz", proxy_state)
    record.update({"frame": frame, "path": Path(record["path"]).name,
                   "source_decoded_state_sha256": state_hash(state), "gaussian_count": proxy_state.count})
    existing = next((r for r in descriptor["frames"] if r["frame"] == frame), None)
    if existing is not None:
        descriptor["frames"].remove(existing)
    descriptor["frames"].append(record)
    descriptor["frames"].sort(key=lambda r: r["frame"])
    _write_json(descriptor_path, descriptor)
    return {**record, "descriptor_path": str(descriptor_path),
            "descriptor_sha256": file_hash(descriptor_path), "quality_independent": True,
            "backend": "fixed_gaussian_subset"}


def load_proxy(path, frame=None):
    path = Path(path)
    if path.suffix == ".json":
        descriptor = json.loads(path.read_text())
        if descriptor.get("schema") != "dynamic-lapisgs.proxy.v1" or descriptor.get("quality_independent") is not True:
            raise ValueError("Invalid proxy descriptor.")
        if not descriptor["frames"]:
            raise ValueError("Proxy descriptor has no frames.")
        candidates = descriptor["frames"] if frame is None else [r for r in descriptor["frames"] if r["frame"] == frame]
        if not candidates:
            raise KeyError(f"Proxy frame {frame} is unavailable.")
        record = candidates[0]
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Proxy payload paths must remain relative to the descriptor.")
        payload = path.parent/relative
        if payload.stat().st_size != record["bytes"] or file_hash(payload) != record["sha256"]:
            raise ValueError("Proxy payload byte size/checksum mismatch.")
        state = load_state(payload)
        if state_hash(state) != record["state_sha256"] or list(state.ids) != descriptor["selected_ids"]:
            raise ValueError("Proxy numeric state/selection checksum mismatch.")
        return state
    state = load_state(path)
    if state.metadata.get("asset_role") != "quality_independent_visibility_proxy":
        raise ValueError("The given Gaussian asset is not a proxy.")
    return state


def render_proxy(state_or_path, sample, settings=None, frame=None):
    state = load_proxy(state_or_path, frame) if isinstance(state_or_path, (str, Path)) else state_or_path
    return render(state, sample, settings)


def validate_proxy_render(proxy_render, reference_render, threshold=.01):
    """Compare proxy geometry channels against the highest decoded quality."""
    if not 0 <= threshold <= 1:
        raise ValueError("Silhouette threshold must be in [0,1].")
    proxy_alpha, reference_alpha = (np.asarray(value["alpha"], dtype=np.float64) for value in (proxy_render, reference_render))
    proxy_depth, reference_depth = (np.asarray(value["depth"], dtype=np.float64) for value in (proxy_render, reference_render))
    if any(value.shape != reference_alpha.shape for value in (proxy_alpha, proxy_depth, reference_depth)):
        raise ValueError("Proxy/reference alpha/depth dimensions must match.")
    if any(not np.all(np.isfinite(value)) for value in (proxy_alpha, reference_alpha, proxy_depth, reference_depth)):
        raise ValueError("Proxy/reference alpha/depth arrays must be finite.")
    proxy_mask, reference_mask = proxy_alpha > threshold, reference_alpha > threshold
    intersection, union = np.logical_and(proxy_mask, reference_mask), np.logical_or(proxy_mask, reference_mask)
    alpha_error = proxy_alpha-reference_alpha
    valid_depth = intersection & (proxy_depth > 0) & (reference_depth > 0)
    depth_error = proxy_depth[valid_depth]-reference_depth[valid_depth]
    return {"reference": "highest_quality_decoded_delivered_representation", "silhouette_alpha_threshold": float(threshold),
            "silhouette_iou": 1. if not union.any() else float(intersection.sum()/union.sum()),
            "silhouette_false_positive_pixels": int(np.logical_and(proxy_mask, ~reference_mask).sum()),
            "silhouette_false_negative_pixels": int(np.logical_and(~proxy_mask, reference_mask).sum()),
            "alpha_mae": float(np.abs(alpha_error).mean()), "alpha_rmse": float(np.sqrt(np.mean(alpha_error**2))),
            "depth_valid_pixels": int(valid_depth.sum()),
            "depth_mae": None if not len(depth_error) else float(np.abs(depth_error).mean()),
            "depth_rmse": None if not len(depth_error) else float(np.sqrt(np.mean(depth_error**2))),
            "depth_mean_relative_error": None if not len(depth_error) else float(np.mean(np.abs(depth_error)/reference_depth[valid_depth])),
            "depth_mask": "intersection_of_thresholded_silhouettes_positive_camera_depth"}


class ProxyBackend:
    """Interchangeable backend interface for future coarse geometry proxies."""
    def generate(self, state, config, output_dir, frame=None):
        raise NotImplementedError

    def load(self, path, frame=None):
        raise NotImplementedError

    def render(self, state_or_path, sample, settings=None, frame=None):
        raise NotImplementedError


class FixedGaussianSubsetProxy(ProxyBackend):
    def generate(self, state, config, output_dir, frame=None):
        return generate_proxy(state, config, output_dir, frame)

    def load(self, path, frame=None):
        return load_proxy(path, frame)

    def render(self, state_or_path, sample, settings=None, frame=None):
        return render_proxy(state_or_path, sample, settings, frame)
