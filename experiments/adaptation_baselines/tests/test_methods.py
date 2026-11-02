"""Historical-objective recovery and delegation, using CPU metadata only."""
from copy import deepcopy
import math

import numpy as np
import pytest

from experiments.adaptation_baselines import methods
from experiments.nava_mse.geometry import aabb_area
from experiments.nava_mse.policies import (
    AVERAGE_MSE, NAVA_STREAMSTGS_ADAPTED, PolicyInputs, allocate,
)
from experiments.visibility_weighted_sse.profiles import build_utilities, profiles_from_dict
from experiments.visibility_weighted_sse.tests.test_profiles_allocation import fixture_payload
from experiments.visibility_weighted_sse.trace import COORDINATE_CONVENTION, TraceCameraProvider


def objective_inputs():
    profiles = profiles_from_dict(fixture_payload(n=2))
    camera = {"width": 100, "height": 100,
              "K": [[80., 0., 50.], [0., 80., 50.], [0., 0., 1.]],
              "world_to_camera": np.eye(4).tolist()}
    samples = []
    times = (0., .2, 1.4)
    for t in times:
        instances = []
        for i in range(2):
            transform = np.eye(4)
            transform[0, 3] = (-.5 if i == 0 else .7) + .1 * t
            instances.append({"instance_id": f"i{i}", "object_id": f"object{i}",
                "object_to_world": transform.tolist(), "source_camera_id": "recorded:0",
                "source_state_id": f"content/frame{t}"})
        samples.append({"timestamp": t, "camera": deepcopy(camera), "instances": instances})
    provider = TraceCameraProvider({"schema": "visibility-weighted-sse.trace.v1",
        "trace_id": "unit_trace", "coordinate_convention": COORDINATE_CONVENTION,
        "prediction_assumption": "ORACLE_CAMERA_TRACE", "predictor_enabled": False,
        "gops": [{"gop_id": "g0", "start_s": 0., "end_s": 2., "samples": samples}]})
    temporal = provider.temporal_weights("g0")
    alpha = ((1., 3.), (90., 10.), (0., 0.))
    aggregated = {"timestamps": list(times), "temporal_weights": list(temporal),
        "temporal_integration": "left_hold_over_half_open_GOP", "per_object": {}}
    for i in range(2):
        aggregated["per_object"][f"i{i}"] = {"object_id": f"object{i}", "instance_id": f"i{i}",
            "A_visible": math.fsum(w * row[i] for w, row in zip(temporal, alpha)),
            "time_series": [{"timestamp": t, "A_visible": row[i]}
                            for t, row in zip(times, alpha)]}
    metadata = {"quality_references": {
        "psnr": {"kind": "original_ply_synthetic_rgb", "aggregation": "arithmetic_mean_dB"},
        "mse": {"kind": "own_CRF20", "aggregation": "pooled_sse_over_RGB_count"}}, "objects": {}}
    for i, profile in enumerate(profiles.for_gop("g0")):
        qualities = {}
        for k, q in enumerate(profile.qualities):
            qualities[q.quality_id] = {"representation_id": q.representation_id,
                "psnr_db": (25., 29., 34., 23.)[k] + i,
                "mse": (0.1, 0.06, 0.01, 0.12)[k] * (1 + i)}
        metadata["objects"][profile.instance_id, profile.gop_id] = {
            "bounds": [[-.3, -.5, 3.], [.3, .5, 4.]], "qualities": qualities}
    return profiles, provider.get_trajectory("g0"), aggregated, metadata


def utilities(result):
    return tuple(tuple(q["weighted_utility"] for q in row["qualities"])
                 for row in result["per_object"].values())


@pytest.mark.parametrize("method,legacy", [
    (methods.NAVA, NAVA_STREAMSTGS_ADAPTED), (methods.AABB_MSE_HISTORICAL, AVERAGE_MSE)])
def test_recovers_historical_nava_and_mse_utilities(method, legacy):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    rows = profiles.for_gop("g0")
    areas = tuple(tuple(aabb_area(metadata["objects"][r.instance_id, "g0"]["bounds"], pose.camera,
                                  next(i for i in pose.instances if i["instance_id"] == r.instance_id)["object_to_world"],
                                  method="paper_discard") for r in rows) for pose in trajectory)
    quality_rows = [metadata["objects"][r.instance_id, "g0"]["qualities"] for r in rows]
    old = allocate(legacy, PolicyInputs(tuple(r.object_id for r in rows), (40, 36, 28, 20),
        ((10, 20, 30, 40), (10, 20, 30, 40)), 100, areas, tuple(aggregated["temporal_weights"]),
        average_mse=tuple(tuple(q[r.quality_id]["mse"] for r in p.qualities)
                          for p, q in zip(rows, quality_rows)),
        nava_psnr=tuple(tuple(q[r.quality_id]["psnr_db"] for r in p.qualities)
                        for p, q in zip(rows, quality_rows))))
    result = methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)
    assert utilities(result) == old.utilities
    assert result["diagnostics"]["importance_per_sample"] == [list(row) for row in old.importance]
    assert result["diagnostics"]["geometry_method"] == "paper_discard"


def test_visibility_psnr_normalizes_each_sample_before_temporal_aggregation():
    profiles, trajectory, aggregated, metadata = objective_inputs()
    result = methods.construct_utilities(methods.VISIBILITY_PSNR, profiles, "g0", trajectory, aggregated, metadata)
    # Averages normalized masses, including a fully invisible interval. This is
    # deliberately different from normalizing already GOP-averaged alpha.
    weights = [.1 * .25 + .6 * .9, .1 * .75 + .6 * .1]
    for i, row in enumerate(result["per_object"].values()):
        assert row["weight"] == pytest.approx(weights[i])
        assert row["weight"] != pytest.approx(aggregated["per_object"][f"i{i}"]["A_visible"] /
            sum(r["A_visible"] for r in aggregated["per_object"].values()))
        assert [q["weighted_utility"] for q in row["qualities"]] == pytest.approx(
            [weights[i] * (db + i) for db in (25., 29., 34., 23.)])
    assert result["diagnostics"]["importance_per_sample"][-1] == [0., 0.]
    assert result["diagnostics"]["quality_reference"] == metadata["quality_references"]["psnr"]


@pytest.mark.parametrize("method", [methods.SOLO_SSE, methods.VISIBILITY_WEIGHTED_SSE])
def test_sse_methods_delegate_to_unchanged_existing_builder(monkeypatch, method):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    calls = []

    def spy(profile, visible):
        calls.append((profile, visible))
        return build_utilities(profile, visible)

    monkeypatch.setattr(methods, "build_utilities", spy)
    result = methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)
    for profile, (called_profile, visible) in zip(profiles.for_gop("g0"), calls):
        assert called_profile is profile
        expected = (profile.offline_mean_solo_contribution if method == methods.SOLO_SSE
                    else aggregated["per_object"][profile.instance_id]["A_visible"])
        assert visible == expected
        assert result["per_object"][profile.instance_id] == build_utilities(profile, expected)
    assert len(calls) == 2


def test_solo_signed_gains_do_not_receive_visibility_or_other_weights():
    profiles, trajectory, aggregated, metadata = objective_inputs()
    for row in aggregated["per_object"].values():
        row["A_visible"] = 0
    result = methods.construct_utilities(methods.SOLO_SSE, profiles, "g0", trajectory, aggregated, metadata)
    assert all(row["weight"] == 1 for row in result["per_object"].values())
    assert utilities(result) == ((0, 50, 90, -10),) * 2
    assert result["diagnostics"]["visibility_source"] == "none"


def test_mse_measured_negative_gains_are_retained():
    profiles, trajectory, aggregated, metadata = objective_inputs()
    result = methods.construct_utilities(methods.VISIBILITY_MSE, profiles, "g0", trajectory, aggregated, metadata)
    assert all(row["qualities"][0]["weighted_utility"] == 0 for row in result["per_object"].values())
    assert all(row["qualities"][-1]["weighted_utility"] < 0 for row in result["per_object"].values())


@pytest.mark.parametrize("method", [methods.NAVA, methods.AABB_MSE_HISTORICAL])
def test_paper_discard_false_zero_is_preserved_and_reported(method):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    for record in metadata["objects"].values():
        # Every front triangle touches an exterior vertex, while cached alpha
        # confirms a visible contribution. Historical behavior remains zero.
        record["bounds"] = [[-100., -100., 3.], [100., 100., 4.]]
    result = methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)
    assert utilities(result) == ((0.,) * 4,) * 2
    assert len(result["diagnostics"]["historical_aabb_zero_with_visible_alpha"]) == 4


@pytest.mark.parametrize("method", methods.METHODS)
def test_existing_one_to_four_quality_subsets_keep_identity_order(method):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    from dataclasses import replace
    one = {key: replace(row, qualities=(row.qualities[0], row.qualities[2]))
           for key, row in profiles.objects.items()}
    profiles = replace(profiles, objects=one)
    result = methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)
    assert all([q["quality_id"] for q in row["qualities"]] == ["Q0", "Q2"]
               for row in result["per_object"].values())


@pytest.mark.parametrize("issue", ["timestamp", "missing_pose", "weight", "visibility"])
def test_rejects_incomplete_or_misaligned_temporal_inputs(issue):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    if issue == "timestamp":
        aggregated["timestamps"][0] = .01
    elif issue == "missing_pose":
        trajectory = trajectory[:-1]
    elif issue == "weight":
        aggregated["temporal_weights"] = [1., 1., 1.]
    else:
        aggregated["per_object"]["i0"]["time_series"].pop()
    with pytest.raises(ValueError):
        methods.construct_utilities(methods.VISIBILITY_PSNR, profiles, "g0", trajectory, aggregated, metadata)


@pytest.mark.parametrize("method", [methods.NAVA, methods.VISIBILITY_PSNR, methods.VISIBILITY_MSE])
def test_nonfinite_quality_is_not_capped_or_substituted(method):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    field = "mse" if method == methods.VISIBILITY_MSE else "psnr_db"
    metadata["objects"]["i0", "g0"]["qualities"]["Q0"][field] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)


def test_unknown_method_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown"):
        methods.construct_utilities("invented", None, "g0", (), {}, {})


def test_primary_psnr_and_mse_share_alpha_weights_without_aabb_metadata():
    profiles, trajectory, aggregated, metadata = objective_inputs()
    for record in metadata["objects"].values():
        record.pop("bounds")
    psnr = methods.construct_utilities(methods.VISIBILITY_PSNR, profiles, "g0", trajectory, aggregated, metadata)
    mse = methods.construct_utilities(methods.VISIBILITY_MSE, profiles, "g0", trajectory, aggregated, metadata)
    for field in ("importance_per_sample", "raw_areas_per_sample", "visibility_source", "geometry_method"):
        assert mse["diagnostics"][field] == psnr["diagnostics"][field]
    for iid, row in mse["per_object"].items():
        weight = psnr["per_object"][iid]["weight"]
        assert row["weight"] == weight
        for quality in row["qualities"]:
            assert quality["weighted_utility"] == pytest.approx(weight * quality["mse_improvement"])
    assert mse["diagnostics"]["historical_aabb_zero_with_visible_alpha"] == []


@pytest.mark.parametrize("method", [methods.VISIBILITY_PSNR, methods.VISIBILITY_MSE])
def test_alpha_methods_fully_invisible_interval_has_zero_utility(method):
    profiles, trajectory, aggregated, metadata = objective_inputs()
    for row in aggregated["per_object"].values():
        for sample in row["time_series"]:
            sample["A_visible"] = 0.
    result = methods.construct_utilities(method, profiles, "g0", trajectory, aggregated, metadata)
    assert utilities(result) == ((0.,)*4,)*2


def test_historical_method_does_not_change_five_primary_method_ids():
    assert len(methods.METHODS) == 5
    assert methods.AABB_MSE_HISTORICAL not in methods.METHODS
    assert methods.AABB_MSE_HISTORICAL in methods.SUPPORTED_METHODS
