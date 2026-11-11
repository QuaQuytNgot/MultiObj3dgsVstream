"""Historical adaptation objectives over one frozen set of representations.

This module constructs utilities only. Physical costs, mandatory bases and the
optimizer are supplied by the common allocation adapter. In particular, the
NAVA and historical AABB-MSE projected-box conventions remain ``paper_discard``.
Replacing this with clipping would change the audited historical objective.
"""
from __future__ import annotations

import math

from experiments.nava_mse.geometry import aabb_area
from experiments.nava_mse.policies import projected_aabb_importance
from experiments.visibility_weighted_sse.common import finite, require
from experiments.visibility_weighted_sse.profiles import build_utilities

NAVA = "NAVA"
VISIBILITY_PSNR = "VISIBILITY_PSNR"
VISIBILITY_MSE = "VISIBILITY_MSE"
SOLO_SSE = "SOLO_SSE"
VISIBILITY_WEIGHTED_SSE = "VISIBILITY_WEIGHTED_SSE"
AABB_MSE_HISTORICAL = "AABB_MSE_HISTORICAL"
METHODS = (NAVA, VISIBILITY_PSNR, VISIBILITY_MSE, SOLO_SSE, VISIBILITY_WEIGHTED_SSE)
SUPPORTED_METHODS = (*METHODS, AABB_MSE_HISTORICAL)


def _temporal_inputs(trajectory, aggregated, ids):
    poses = tuple(trajectory)
    require(bool(poses), "A complete predicted GOP trajectory is required")
    require(set(aggregated["per_object"]) == set(ids), "Visibility/profile instance coverage mismatch")
    timestamps = [pose.timestamp for pose in poses]
    require(timestamps == list(aggregated["timestamps"]), "Trajectory/visibility timestamp mismatch")
    require(all(a < b for a, b in zip(timestamps, timestamps[1:])), "Trajectory must be chronological")
    weights = tuple(finite(w, "temporal weight") for w in aggregated["temporal_weights"])
    require(len(weights) == len(poses) and all(w >= 0 for w in weights)
            and math.isclose(math.fsum(weights), 1., abs_tol=1e-12, rel_tol=0),
            "Complete normalized GOP temporal weights are required")
    # Match the historical policy's normalization and multiplication order.
    total = math.fsum(weights)
    weights = tuple(w / total for w in weights)
    for pose in poses:
        require({row["instance_id"] for row in pose.instances} == set(ids),
                "Every trajectory pose must cover all selected instances")
    return poses, weights


def _visible_samples(poses, aggregated, ids):
    columns = []
    for iid in ids:
        series = aggregated["per_object"][iid]["time_series"]
        require(len(series) == len(poses)
                and [row["timestamp"] for row in series] == [pose.timestamp for pose in poses],
                "Full per-sample visibility is required")
        values = tuple(finite(row["A_visible"], "visible contribution") for row in series)
        require(all(v >= 0 for v in values), "Visible contributions must be nonnegative")
        columns.append(values)
    return tuple(tuple(column[t] for column in columns) for t in range(len(poses)))


def _aabb_samples(poses, rows, metadata):
    areas = []
    for pose in poses:
        instances = {row["instance_id"]: row for row in pose.instances}
        areas.append(tuple(aabb_area(metadata["objects"][row.instance_id, row.gop_id]["bounds"],
                                    pose.camera, instances[row.instance_id]["object_to_world"],
                                    method="paper_discard") for row in rows))
    return tuple(areas)


def construct_utilities(method, profiles, gop_id, trajectory, aggregated, metadata):
    """Return existing allocation-compatible utility rows without solving.

    ``trajectory`` contains every ``TracePose`` in the requested GOP and
    ``aggregated`` is the existing cached Visibility Analyzer aggregation.
    Metadata is the common loader's validated per-instance/GOP quality mapping.
    The first quality in each existing profile is its declared mandatory Q0.
    """
    require(method in SUPPORTED_METHODS, f"Unknown adaptation method: {method}")
    rows = profiles.for_gop(gop_id)
    ids = tuple(row.instance_id for row in rows)
    poses, temporal = _temporal_inputs(trajectory, aggregated, ids)
    references = metadata.get("quality_references", {}) if metadata else {}
    common = {"method": method, "temporal_weights": list(temporal),
              "temporal_integration": aggregated.get("temporal_integration"),
              "joint_quality_inputs_used": False,
              "prediction_assumption": "ORACLE_CAMERA_TRACE"}

    if method in (SOLO_SSE, VISIBILITY_WEIGHTED_SSE):
        # Delegate both signed SSE differences and the proposed scale ratio to
        # their sole existing implementation. Solo supplies its own denominator
        # as numerator, yielding exactly one without modifying any profile.
        utilities = {row.instance_id: build_utilities(
            row, row.offline_mean_solo_contribution if method == SOLO_SSE
            else aggregated["per_object"][row.instance_id]["A_visible"]) for row in rows}
        diagnostics = {**common,
            "formula": ("D_i(Q0)-D_i(q)" if method == SOLO_SSE
                        else "[A_visible_GOP_i/A_solo_offline_mean_i] * [D_i(Q0)-D_i(q)]"),
            "utility_units": "solo_SSE_reduction",
            "visibility_source": ("none" if method == SOLO_SSE
                                  else "candidate_independent_alpha_transmittance_contribution"),
            "quality_reference": references.get("sse", "fixed_own_CRF20_reference"),
            "utility_implementation": "experiments.visibility_weighted_sse.profiles.build_utilities",
            "cross_object_normalization": False,
            "negative_gains_preserved": True}
        return {"per_object": utilities, "diagnostics": diagnostics}

    visible = _visible_samples(poses, aggregated, ids)
    alpha_weighted = method in (VISIBILITY_PSNR, VISIBILITY_MSE)
    if alpha_weighted:
        areas = visible
        source = "normalized_alpha_transmittance_A_visible"
    else:
        areas = _aabb_samples(poses, rows, metadata)
        source = "normalized_projected_AABB_paper_discard"
    importance = projected_aabb_importance(areas)
    weights = tuple(math.fsum(w * sample[i] for w, sample in zip(temporal, importance))
                    for i in range(len(rows)))
    utilities = {}
    for i, profile in enumerate(rows):
        quality_metadata = metadata["objects"][profile.instance_id, profile.gop_id]["qualities"]
        baseline = quality_metadata[profile.qualities[0].quality_id]
        qualities = []
        for quality in profile.qualities:
            measured = quality_metadata[quality.quality_id]
            require(measured["representation_id"] == quality.representation_id,
                    "Quality metadata/profile representation mismatch")
            if method in (VISIBILITY_MSE, AABB_MSE_HISTORICAL):
                distortion = finite(measured["mse"], "offline MSE")
                gain = finite(baseline["mse"], "base offline MSE") - distortion
                value = math.fsum(w * sample[i] * gain for w, sample in zip(temporal, importance))
                extra = {"offline_mse": distortion, "mse_improvement": gain}
            else:
                quality_db = finite(measured["psnr_db"], "offline PSNR")
                value = math.fsum(w * sample[i] * quality_db for w, sample in zip(temporal, importance))
                extra = {"offline_psnr_db": quality_db}
            qualities.append({"quality_id": quality.quality_id,
                "representation_id": quality.representation_id,
                "weighted_utility": finite(value, "weighted utility"),
                "dependencies": list(quality.dependencies), "resource_ids": list(quality.resource_ids), **extra})
        utilities[profile.instance_id] = {"object_id": profile.object_id,
            "instance_id": profile.instance_id, "gop_id": profile.gop_id,
            "weight": weights[i], "qualities": qualities}
    mse = method in (VISIBILITY_MSE, AABB_MSE_HISTORICAL)
    diagnostics = {**common,
        "formula": ("sum_t tau_t * I_ti * [MSE_i(Q0)-MSE_i(q)]" if mse
                    else "sum_t tau_t * I_ti * PSNR_i(q)"),
        "utility_units": "foreground_MSE_reduction" if mse else "weighted_metadata_PSNR_dB",
        "visibility_source": source,
        "quality_reference": references.get("mse" if mse else "psnr"),
        "raw_areas_per_sample": [list(row) for row in areas],
        "importance_per_sample": [list(row) for row in importance],
        "instance_order": list(ids), "cross_object_normalization": True,
        "negative_gains_preserved": True,
        "geometry_method": None if alpha_weighted else "paper_discard",
        "original_geometry_PSNR_replaced_by_image_proxy": method == NAVA,
        "historical_aabb_zero_with_visible_alpha": [
            {"timestamp": pose.timestamp, "instance_id": ids[i]}
            for t, pose in enumerate(poses) for i in range(len(ids))
            if not alpha_weighted and areas[t][i] == 0 and visible[t][i] > 0],
        "historical_geometry_limitation": (None if alpha_weighted else
            "Exterior-vertex triangle discard can assign zero AABB importance to visibly rendered objects; retained unchanged")}
    return {"per_object": utilities, "diagnostics": diagnostics}
