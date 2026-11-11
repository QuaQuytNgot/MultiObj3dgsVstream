"""Foreground RGB distortion with explicit normalization and correct pooling.

All means include every RGB channel.  The mask always comes from the reference
alpha; candidate alpha is deliberately absent from the API.  Raw normalized RGB
is the default because the StreamSTGS renderer can emit values outside [0, 1].
Clipping is an explicit alternate measurement protocol, never an implicit fix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class MetricStats:
    sse: float
    foreground_pixels: int
    channels: int = 3
    data_range: float = 1.0
    normalization: str = "raw"

    def __post_init__(self) -> None:
        if not math.isfinite(self.sse) or self.sse < 0:
            raise ValueError("SSE must be finite and nonnegative")
        if isinstance(self.foreground_pixels, bool) or not isinstance(self.foreground_pixels, (int, np.integer)) or self.foreground_pixels <= 0:
            raise ValueError("A metric requires a positive integer reference foreground count")
        if self.channels != 3:
            raise ValueError("This experiment measures three RGB channels")
        if not math.isfinite(self.data_range) or self.data_range <= 0:
            raise ValueError("data_range must be positive and finite")
        if self.normalization not in {"raw", "clip"}:
            raise ValueError("normalization must be 'raw' or 'clip'")

    @property
    def sample_count(self) -> int:
        """Number of scalar RGB observations, not the number of images."""
        return self.foreground_pixels * self.channels

    @property
    def mse(self) -> float:
        return self.sse / self.sample_count

    @property
    def psnr(self) -> float:
        return math.inf if self.sse == 0 else 10.0 * math.log10(self.data_range**2 / self.mse)

    @property
    def foreground_pixel_count(self) -> int:
        return self.foreground_pixels

    def to_dict(self) -> dict[str, Any]:
        # JSON has no infinity. Preserve the zero-error distinction explicitly.
        return {
            "sse": float(self.sse), "foreground_pixels": int(self.foreground_pixels),
            "channels": self.channels, "data_range": self.data_range,
            "normalization": self.normalization, "mse": self.mse,
            "psnr_db": None if math.isinf(self.psnr) else self.psnr,
            "psnr_infinite": math.isinf(self.psnr),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricStats":
        result = cls(sse=float(value["sse"]), foreground_pixels=value["foreground_pixels"],
                     channels=value.get("channels", 3), data_range=float(value.get("data_range", 1.0)),
                     normalization=value.get("normalization", "raw"))
        if "mse" in value and not math.isclose(float(value["mse"]), result.mse, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Serialized MSE does not match SSE / RGB count")
        if "psnr_infinite" in value and bool(value["psnr_infinite"]) != math.isinf(result.psnr):
            raise ValueError("Serialized PSNR infinity flag is inconsistent")
        if value.get("psnr_db") is not None and not math.isclose(float(value["psnr_db"]), result.psnr, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Serialized PSNR does not match SSE / RGB count")
        return result


def normalize_rgb(rgb: Any, *, rgb_divisor: float = 1.0, normalization: str = "raw", data_range: float = 1.0) -> np.ndarray:
    """Convert to float64 with a declared divisor (e.g. 255 for uint8 PNGs)."""
    if not math.isfinite(rgb_divisor) or rgb_divisor <= 0:
        raise ValueError("rgb_divisor must be positive and finite")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be positive and finite")
    if normalization not in {"raw", "clip"}:
        raise ValueError("normalization must be 'raw' or 'clip'")
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if not np.isfinite(array).all():
        raise ValueError("RGB image contains nonfinite values")
    array = array / rgb_divisor
    return np.clip(array, 0.0, data_range) if normalization == "clip" else array


def reference_foreground_mask(reference_alpha: Any, image_shape: tuple[int, int], *, alpha_threshold: float = 0.01) -> np.ndarray:
    if not math.isfinite(alpha_threshold) or not 0 <= alpha_threshold < 1:
        raise ValueError("alpha_threshold must be in [0, 1)")
    alpha = np.asarray(reference_alpha, dtype=np.float64)
    if alpha.ndim == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    if alpha.shape != image_shape:
        raise ValueError("Reference alpha and RGB must have identical image dimensions")
    if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Reference alpha must be finite and in [0, 1]")
    mask = alpha > alpha_threshold
    if not mask.any():
        raise ValueError("Reference foreground is empty; distortion is unavailable")
    return mask


def foreground_metrics(reference_rgb: Any, candidate_rgb: Any, reference_alpha: Any, *, alpha_threshold: float = 0.01,
                       normalization: str = "raw", data_range: float = 1.0, rgb_divisor: float = 1.0) -> MetricStats:
    reference = normalize_rgb(reference_rgb, rgb_divisor=rgb_divisor, normalization=normalization, data_range=data_range)
    candidate = normalize_rgb(candidate_rgb, rgb_divisor=rgb_divisor, normalization=normalization, data_range=data_range)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate RGB shapes differ")
    mask = reference_foreground_mask(reference_alpha, reference.shape[:2], alpha_threshold=alpha_threshold)
    difference = candidate[mask] - reference[mask]
    sse = float(np.sum(difference * difference, dtype=np.float64))
    return MetricStats(sse, int(np.count_nonzero(mask)), data_range=data_range, normalization=normalization)


def pool_metrics(metrics: Iterable[MetricStats]) -> MetricStats:
    values = tuple(metrics)
    if not values:
        raise ValueError("Cannot pool an empty metric collection")
    protocol = (values[0].channels, values[0].data_range, values[0].normalization)
    if any((v.channels, v.data_range, v.normalization) != protocol for v in values):
        raise ValueError("Cannot pool measurements with different normalization protocols")
    return MetricStats(math.fsum(v.sse for v in values), sum(v.foreground_pixels for v in values),
                       channels=protocol[0], data_range=protocol[1], normalization=protocol[2])


def rgb_range_statistics(rgb: Any, *, rgb_divisor: float = 1.0, data_range: float = 1.0) -> dict[str, Any]:
    """Statistics before clipping; fractions count RGB scalar values."""
    array = normalize_rgb(rgb, rgb_divisor=rgb_divisor, data_range=data_range)
    below = int(np.count_nonzero(array < 0.0))
    above = int(np.count_nonzero(array > data_range))
    return {"minimum": float(array.min()), "maximum": float(array.max()), "rgb_scalar_count": int(array.size),
            "below_range_count": below, "above_range_count": above,
            "clipped_fraction": (below + above) / int(array.size), "rgb_divisor": rgb_divisor, "data_range": data_range}


def foreground_diagnostics(reference_rgb: Any, candidate_rgb: Any, reference_alpha: Any, **kwargs: Any) -> dict[str, Any]:
    stats = foreground_metrics(reference_rgb, candidate_rgb, reference_alpha, **kwargs)
    range_args = {key: kwargs[key] for key in ("rgb_divisor", "data_range") if key in kwargs}
    return {"metrics": stats.to_dict(), "projected_object_area_pixels": stats.foreground_pixels,
            "mask_source": "CRF20_reference_alpha", "alpha_threshold": kwargs.get("alpha_threshold", 0.01),
            "reference_rgb_range": rgb_range_statistics(reference_rgb, **range_args),
            "candidate_rgb_range": rgb_range_statistics(candidate_rgb, **range_args)}


def finite_image_quality_proxy(mse: float, *, reference_identity: str, data_range: float = 1.0,
                               mse_floor: float = 1e-10) -> dict[str, Any]:
    """Explicit finite image-PSNR adaptation, separate from measured PSNR.

    Prefer original-PLY render measurements. A CRF20-relative proxy, if used,
    must be declared by the caller in reference_identity and in the report.
    """
    if not reference_identity or not isinstance(reference_identity, str):
        raise ValueError("A finite quality proxy requires its measured reference identity")
    if not math.isfinite(mse) or mse < 0 or not math.isfinite(mse_floor) or mse_floor <= 0:
        raise ValueError("MSE must be nonnegative and the proxy floor positive and finite")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be positive and finite")
    effective_mse = max(mse, mse_floor)
    return {"quality_db": 10.0 * math.log10(data_range**2 / effective_mse),
            "proxy": "finite_image_psnr", "reference_identity": reference_identity,
            "measured_mse": mse, "mse_floor": mse_floor, "floor_applied": mse < mse_floor,
            "is_geometric_psnr": False}


def distortion_reduction(candidate: MetricStats, crf40: MetricStats) -> float:
    """Normalized RGB-MSE benefit; negative benefits remain negative."""
    if (candidate.channels, candidate.data_range, candidate.normalization) != (crf40.channels, crf40.data_range, crf40.normalization):
        raise ValueError("Distortion reduction requires the same metric protocol")
    if candidate.foreground_pixels != crf40.foreground_pixels:
        raise ValueError("Distortion reduction requires the same reference foreground mask")
    return crf40.mse - candidate.mse
