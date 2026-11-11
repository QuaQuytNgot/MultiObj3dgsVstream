"""Occlusion diagnostics from a separately rendered low-quality proxy.

For unchanged CRF40 Gaussians, the native backend renders object i as white
and every other object as black, retaining all original opacities and depth
ordering. The resulting image is i's alpha-weighted contribution. Dividing it
by i's solo alpha defines an *effective* object/pixel attenuation. It retains
the rasterizer's Gaussian order, but is not a claim that a whole object is a
single depth layer, nor that attenuation stays fixed as candidate CRFs change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .metrics import _readonly


LOW_COST_OCCLUSION_PROXY = "LOW_COST_OCCLUSION_PROXY"
NATIVE_PROXY_METHOD = "CRF40_native_Gaussian_order_object_contribution_over_solo_alpha"


@dataclass(frozen=True)
class ProxyFrame:
    solo_alpha: np.ndarray
    contributions: np.ndarray
    joint_alpha: np.ndarray
    transmittance: np.ndarray
    visible_fraction: np.ndarray
    measured_occlusion: tuple[float | None, ...]
    source_identity: str
    quality: int
    method: str
    statistics: dict[str, Any]

    @property
    def provenance(self) -> tuple[str, ...]:
        return (LOW_COST_OCCLUSION_PROXY,)

    def to_dict(self) -> dict[str, Any]:
        return {"source_identity": self.source_identity, "quality": self.quality,
                "method": self.method, "provenance": list(self.provenance),
                "visible_fraction": self.visible_fraction.tolist(),
                "measured_occlusion": list(self.measured_occlusion),
                "statistics": self.statistics}


def build_proxy(solo_alpha: Any, contributions: Any, joint_alpha: Any, *,
                source_identity: str, quality: int = 40,
                alpha_threshold: float = 0.01,
                numerical_tolerance: float = 1e-4,
                method: str = NATIVE_PROXY_METHOD) -> ProxyFrame:
    """Build frozen effective T without candidate joint RGB, SSE or masks.

    V_i = sum_p contribution_i / sum_p solo_alpha_i. Its denominator is
    object i's own continuous alpha mass, never another object's image area.
    Zero-mass objects get V=0 for optimization and an undefined (None) measured
    occlusion with an explicit denominator-defined flag. Numerical excursions
    no larger than the declared absolute alpha tolerance are clipped and
    recorded; material nonphysical contributions raise rather than hide them.
    """
    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("A separate low-quality proxy identity is required")
    if quality != 40 or isinstance(quality, bool):
        raise ValueError("This frozen proxy protocol requires CRF40")
    if not isinstance(method, str) or not method:
        raise ValueError("Proxy approximation method must be recorded")
    if not math.isfinite(alpha_threshold) or not 0 <= alpha_threshold < 1:
        raise ValueError("alpha_threshold must be in [0,1)")
    if not math.isfinite(numerical_tolerance) or numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be nonnegative and finite")
    alpha = np.asarray(solo_alpha, dtype=np.float64)
    visible = np.asarray(contributions, dtype=np.float64)
    joint = np.asarray(joint_alpha, dtype=np.float64)
    if alpha.ndim != 3 or min(alpha.shape) <= 0 or visible.shape != alpha.shape or joint.shape != alpha.shape[1:]:
        raise ValueError("Expected solo/contribution [N,H,W] and joint alpha [H,W]")
    if not all(np.isfinite(x).all() for x in (alpha, visible, joint)):
        raise ValueError("Proxy arrays must be finite")
    # Original values remain in the diagnostic statistics, then tiny backend
    # rounding excursions are normalized before safe division.
    low = min(float(alpha.min()), float(visible.min()), float(joint.min()))
    high = max(float(alpha.max()), float(joint.max()))
    excess = visible - alpha
    if low < -numerical_tolerance or high > 1 + numerical_tolerance or float(excess.max()) > numerical_tolerance:
        raise ValueError("Material nonphysical alpha/contribution in proxy")
    raw_excess_max = float(excess.max())
    clipped_count = int(np.count_nonzero(alpha < 0) + np.count_nonzero(alpha > 1) +
                        np.count_nonzero(joint < 0) + np.count_nonzero(joint > 1) +
                        np.count_nonzero(visible < 0) + np.count_nonzero(visible > alpha))
    alpha = np.clip(alpha, 0, 1)
    joint = np.clip(joint, 0, 1)
    visible = np.clip(visible, 0, alpha)
    positive = alpha > 0
    transmittance = np.divide(visible, alpha, out=np.zeros_like(alpha), where=positive)
    potential_mass = np.sum(alpha, axis=(1, 2), dtype=np.float64)
    visible_mass = np.sum(visible, axis=(1, 2), dtype=np.float64)
    defined = potential_mass > 0
    fraction = np.divide(visible_mass, potential_mass, out=np.zeros_like(potential_mass), where=defined)
    # Continuous alpha weighting avoids dropping low-alpha Gaussian tails;
    # alpha_threshold only defines separately reported pixel-count diagnostics.
    foreground = alpha > alpha_threshold
    overlap = np.count_nonzero(foreground, axis=0) >= 2
    interior = (transmittance > alpha_threshold) & (transmittance < 1 - alpha_threshold) & positive
    per_object = []
    for i in range(alpha.shape[0]):
        fg = foreground[i]
        fg_count = int(np.count_nonzero(fg))
        mid_count = int(np.count_nonzero(interior[i] & fg))
        alpha_mass_mid = float(np.sum(alpha[i][interior[i]], dtype=np.float64))
        per_object.append({
            "object_index": i, "potential_alpha_mass": float(potential_mass[i]),
            "visible_alpha_mass": float(visible_mass[i]),
            "own_potential_denominator_defined": bool(defined[i]),
            "visible_fraction": float(fraction[i]),
            "measured_occlusion": float(1 - fraction[i]) if defined[i] else None,
            "potential_foreground_pixels": fg_count,
            "proxy_overlap_pixels": int(np.count_nonzero(fg & overlap)),
            "intermediate_effective_transmittance_pixels": mid_count,
            "intermediate_effective_transmittance_foreground_fraction": mid_count / fg_count if fg_count else None,
            "intermediate_effective_transmittance_alpha_mass_fraction": alpha_mass_mid / potential_mass[i] if defined[i] else None,
            "effective_transmittance_min_on_potential": float(transmittance[i][positive[i]].min()) if defined[i] else None,
            "effective_transmittance_max_on_potential": float(transmittance[i][positive[i]].max()) if defined[i] else None,
            "effective_transmittance_mean_on_foreground": float(np.mean(transmittance[i][fg])) if fg_count else None,
            "own_alpha_intermediate_pixels": int(np.count_nonzero((alpha[i] > alpha_threshold) & (alpha[i] < 1 - alpha_threshold))),
        })
    statistics = {
        "denominator": "sum_pixel_own_CRF40_solo_alpha_continuous_mass",
        "occlusion_definition": "1-sum_pixel_object_joint_alpha_weighted_contribution/sum_pixel_own_solo_alpha",
        "transmittance_definition": "object_contribution/own_solo_alpha; zero when own_alpha=0",
        "attenuation_approximation": "alpha_mass_weighted_effective_per_object_pixel; native_per_Gaussian_depth_order",
        "candidate_independent": True, "candidate_joint_RGB_or_SSE_used": False,
        "quality": quality, "alpha_threshold_for_pixel_diagnostics": alpha_threshold,
        "numerical_absolute_alpha_tolerance": numerical_tolerance,
        "numerical_clipped_scalar_count": clipped_count,
        "raw_minimum_alpha_or_contribution": low,
        "raw_maximum_alpha": high,
        "raw_maximum_contribution_minus_own_alpha": raw_excess_max,
        "contribution_partition_max_abs_error_vs_joint_alpha": float(np.max(np.abs(np.sum(visible, axis=0) - joint))),
        "contribution_partition_sum_abs_error_vs_joint_alpha": float(np.sum(np.abs(np.sum(visible, axis=0) - joint), dtype=np.float64)),
        "objects": per_object,
        "potential_overlap_pixels": int(np.count_nonzero(overlap)),
        "joint_foreground_pixels": int(np.count_nonzero(joint > alpha_threshold)),
        "joint_intermediate_alpha_pixels": int(np.count_nonzero((joint > alpha_threshold) & (joint < 1 - alpha_threshold))),
        "scene_category_or_target_occlusion_confirmed_by_layout": False,
    }
    return ProxyFrame(_readonly(alpha), _readonly(visible), _readonly(joint),
                      _readonly(transmittance), _readonly(fraction),
                      tuple(float(1 - fraction[i]) if defined[i] else None for i in range(alpha.shape[0])),
                      source_identity, quality, method, statistics)


def composite_layers(rgb: Any, alpha: Any, *, background: Any = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic front-to-back premultiplied compositing fixture, not native data.

    Inputs are unpremultiplied [N,H,W,3] colors and [N,H,W] alpha in supplied
    front-to-back order. Returns RGB, alpha and preceding-layer T. This helper
    is explicitly synthetic and is never a replacement for native Gaussian
    contributions or a Scene D achievement claim.
    """
    colors, opacity = np.asarray(rgb, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if colors.ndim != 4 or colors.shape[-1] != 3 or opacity.shape != colors.shape[:3] or bg.shape != (3,):
        raise ValueError("Synthetic layer fixture dimensions differ")
    if not all(np.isfinite(v).all() for v in (colors, opacity, bg)) or np.any(opacity < 0) or np.any(opacity > 1):
        raise ValueError("Synthetic layer alpha must be in [0,1] and inputs finite")
    preceding = np.ones(opacity.shape[1:], dtype=np.float64)
    total = np.zeros(colors.shape[1:], dtype=np.float64)
    layers = []
    for color, a in zip(colors, opacity):
        layers.append(preceding.copy())
        total += preceding[..., None] * a[..., None] * color
        preceding *= 1 - a
    return total + preceding[..., None] * bg, 1 - preceding, np.asarray(layers)
