"""Fixed-reference RGB SSE and independent joint marginal measurements.

RGB is clipped to [0, 1] before float64 subtraction. A reference mask is made
once from its CRF20 alpha and is never a function of candidate RGB or alpha.
The saved pixel error map sums RGB channels, so its sum is SSE, not MSE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.nava_mse.metrics import MetricStats, normalize_rgb, pool_metrics


def _readonly(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReferenceFrame:
    rgb: np.ndarray
    alpha: np.ndarray
    mask: np.ndarray
    identity: str
    alpha_threshold: float


@dataclass(frozen=True)
class FrameMeasurement:
    foreground: MetricStats | None
    full_frame: MetricStats
    error_map: np.ndarray
    reference_identity: str
    foreground_pixels: int

    @property
    def sse(self) -> float:
        return 0.0 if self.foreground is None else self.foreground.sse

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreground": None if self.foreground is None else self.foreground.to_dict(),
            "foreground_sse": self.sse,
            "foreground_pixels": self.foreground_pixels,
            "foreground_defined": self.foreground is not None,
            "full_frame": self.full_frame.to_dict(),
            "reference_identity": self.reference_identity,
            "mask_source": "fixed_CRF20_reference_alpha",
        }


def prepare_reference(rgb: Any, alpha: Any, *, identity: str,
                      alpha_threshold: float = 0.01) -> ReferenceFrame:
    if not isinstance(identity, str) or not identity:
        raise ValueError("A nonempty CRF20 reference identity is required")
    if not math.isfinite(alpha_threshold) or not 0 <= alpha_threshold < 1:
        raise ValueError("alpha_threshold must be in [0, 1)")
    rgb64 = normalize_rgb(rgb, normalization="clip")
    alpha64 = np.asarray(alpha, dtype=np.float64)
    if alpha64.ndim == 3 and alpha64.shape[-1] == 1:
        alpha64 = alpha64[..., 0]
    if alpha64.shape != rgb64.shape[:2]:
        raise ValueError("Reference RGB and alpha dimensions differ")
    if not np.isfinite(alpha64).all() or np.any(alpha64 < 0) or np.any(alpha64 > 1):
        raise ValueError("Reference alpha must be finite and in [0,1]")
    # An entirely hidden/empty object is valid. SSE=0 is recorded, but PSNR
    # and MSE are unavailable rather than invented from a zero denominator.
    return ReferenceFrame(_readonly(rgb64), _readonly(alpha64),
                          _readonly(alpha64 > alpha_threshold, dtype=bool),
                          identity, float(alpha_threshold))


def measure_frame(reference: ReferenceFrame, candidate_rgb: Any) -> FrameMeasurement:
    if not isinstance(reference, ReferenceFrame):
        raise TypeError("measure_frame requires a frozen ReferenceFrame")
    candidate = normalize_rgb(candidate_rgb, normalization="clip")
    if candidate.shape != reference.rgb.shape:
        raise ValueError("Reference and candidate RGB dimensions differ")
    difference = candidate - reference.rgb
    full_errors = np.sum(difference * difference, axis=-1, dtype=np.float64)
    masked = np.where(reference.mask, full_errors, 0.0)
    count = int(np.count_nonzero(reference.mask))
    foreground = None if not count else MetricStats(
        float(np.sum(masked, dtype=np.float64)), count, normalization="clip")
    full_frame = MetricStats(float(np.sum(full_errors, dtype=np.float64)),
                             int(full_errors.size), normalization="clip")
    return FrameMeasurement(foreground, full_frame, _readonly(masked), reference.identity, count)


def pool_frames(frames: Iterable[FrameMeasurement]) -> dict[str, Any]:
    """Pool temporal SSE and RGB observations, never average frame PSNR."""
    values = tuple(frames)
    if not values:
        raise ValueError("Cannot pool an empty GOP")
    foregrounds = tuple(v.foreground for v in values if v.foreground is not None)
    fg = pool_metrics(foregrounds) if foregrounds else None
    full = pool_metrics(v.full_frame for v in values)
    return {"sse": 0.0 if fg is None else fg.sse,
            "foreground_pixels": sum(v.foreground_pixels for v in values),
            "foreground": None if fg is None else fg.to_dict(),
            "full_frame": full.to_dict(), "sample_count": len(values),
            "temporal_aggregation": "sum_SSE_and_RGB_sample_counts",
            "reference_identities": [v.reference_identity for v in values]}


def marginal_gain(old_sse: float, new_sse: float) -> float:
    """Actual or solo SSE reduction. Negative upgrades remain negative."""
    if not all(math.isfinite(x) and x >= 0 for x in (old_sse, new_sse)):
        raise ValueError("Marginals require finite nonnegative old/new SSE")
    return float(old_sse - new_sse)


def mixed_finite_difference(e00: float, e10: float, e01: float, e11: float) -> float:
    """E(new_i,new_j)-E(new_i,old_j)-E(old_i,new_j)+E(old_i,old_j)."""
    if not all(math.isfinite(x) and x >= 0 for x in (e00, e10, e01, e11)):
        raise ValueError("Mixed differences require finite nonnegative SSE")
    return math.fsum((e11, -e10, -e01, e00))


def compare_gains(predicted: float, actual: float, *, near_zero: float = 1e-10) -> dict[str, Any]:
    if not all(math.isfinite(x) for x in (predicted, actual, near_zero)) or near_zero < 0:
        raise ValueError("Gain comparison requires finite values and nonnegative tolerance")
    error = predicted - actual
    defined = abs(actual) > near_zero
    return {"predicted_gain": float(predicted), "actual_joint_gain": float(actual),
            "signed_error": float(error), "absolute_error": abs(float(error)),
            "relative_absolute_error": abs(float(error)) / abs(actual) if defined else None,
            "gain_ratio": predicted / actual if defined else None,
            "relative_denominator_defined": defined, "near_zero_tolerance": near_zero,
            "joint_gain_negative": actual < -near_zero,
            "joint_gain_near_zero": abs(actual) <= near_zero}


def _ranks(values: np.ndarray) -> np.ndarray:
    # Stable average ranks, including ties. Spearman is Pearson of these ranks.
    order = np.argsort(values, kind="stable")
    out = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return out


def ranking_summary(predicted: Sequence[float], actual: Sequence[float], *,
                    top_k: int = 3, near_zero: float = 1e-10) -> dict[str, Any]:
    """Compare one matched upgrade group; mathematically undefined ranks use None.

    Pairwise ranking excludes actual ties and scores predicted ties as one half.
    Top-k reports coverage of the actual top-k set with boundary ties included;
    predicted ties use that same inclusive rule rather than arbitrary row order.
    """
    p, a = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if p.ndim != 1 or a.shape != p.shape or not p.size or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Ranking requires nonempty aligned finite gain vectors")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(near_zero) or near_zero < 0:
        raise ValueError("near_zero must be nonnegative and finite")
    r_p, r_a = _ranks(p), _ranks(a)
    spearman = None
    if p.size > 1 and np.ptp(r_p) > 0 and np.ptp(r_a) > 0:
        spearman = float(np.corrcoef(r_p, r_a)[0, 1])
    scores = []
    for i in range(p.size):
        for j in range(i):
            if abs(a[i] - a[j]) <= near_zero:
                continue
            scores.append(0.5 if abs(p[i] - p[j]) <= near_zero else
                          float((p[i] - p[j]) * (a[i] - a[j]) > 0))
    best_actual = set(np.flatnonzero(a >= a.max() - near_zero).tolist())
    best_predicted = set(np.flatnonzero(p >= p.max() - near_zero).tolist())
    k = min(top_k, p.size)
    cut_a, cut_p = np.sort(a)[-k], np.sort(p)[-k]
    actual_top = set(np.flatnonzero(a >= cut_a - near_zero).tolist())
    predicted_top = set(np.flatnonzero(p >= cut_p - near_zero).tolist())
    error = p - a
    return {"count": int(p.size), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))), "signed_bias": float(np.mean(error)),
            "spearman": spearman, "pairwise_ranking_accuracy": None if not scores else float(np.mean(scores)),
            "non_tied_actual_pair_count": len(scores),
            "top1_hit": bool(best_actual & best_predicted),
            "top1_actual_set": sorted(best_actual), "top1_predicted_set": sorted(best_predicted),
            "top_k": int(k), "top_k_recall_including_boundary_ties": len(actual_top & predicted_top) / len(actual_top),
            "top_k_actual_set": sorted(actual_top), "top_k_predicted_set": sorted(predicted_top),
            "negative_joint_gain_fraction": float(np.mean(a < -near_zero)),
            "near_zero_joint_gain_fraction": float(np.mean(np.abs(a) <= near_zero)),
            "near_zero_tolerance": near_zero}
