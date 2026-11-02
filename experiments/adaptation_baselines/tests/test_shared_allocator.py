"""The policy seam shares physical costs without altering baseline utilities."""
from copy import deepcopy
import hashlib
import itertools
import random

import pytest

from experiments.visibility_weighted_sse.allocation import allocate as shared_allocate, allocate_utilities as shared_allocate_utilities
from experiments.visibility_weighted_sse.profiles import build_utilities, profiles_from_dict


def allocate(*args, solver_limits=None, **kwargs):
    return shared_allocate(*args, solver_limits={"discretization": "exact_bytes", **(solver_limits or {})}, **kwargs)


def allocate_utilities(*args, solver_limits=None, **kwargs):
    return shared_allocate_utilities(*args, solver_limits={"discretization": "exact_bytes", **(solver_limits or {})}, **kwargs)


def payload(family="standalone", counts=(4, 4)):
    data = {
        "schema": "visibility-weighted-sse.profiles.v1", "content_revision": "content-v1",
        "profile_revision": "profile-v1", "representation_family": family,
        "compatibility": {
            "resolution": [100, 100], "coordinate_convention": "opencv_positive_z_column_vectors",
            "contribution_convention": "continuous_alpha_weighted_pixels",
            "distribution_id": "offline-two-cameras", "time_aggregation": "normalized_camera_time_mean",
            "proxy_representation": "Q0", "transform_convention": "positive_similarity_object_to_world"},
        "provenance": {"information_role": "OFFLINE_AVERAGE_SOLO_SSE", "joint_quality_used": False},
        "resources": {}, "objects": []}
    for i, count in enumerate(counts):
        qualities = []
        for q in range(count):
            key = f"r{i}_{q}"
            data["resources"][key] = {"bytes": (q+1)*10,
                                      "sha256": hashlib.sha256(key.encode()).hexdigest()}
            qualities.append({"quality_id": f"Q{q}", "representation_id": f"rep{q}",
                              "average_solo_sse": 100-q*20, "resource_ids": [key],
                              "dependencies": ["Q0"] if family == "progressive" and q else []})
        data["objects"].append({"object_id": f"object{i}", "instance_id": f"i{i}", "gops": [{
            "gop_id": "g0", "offline_mean_solo_contribution": 100,
            "offline_sample_ids": ["offline-camera0", "offline-camera1"],
            "offline_sample_weights": [.5, .5], "offline_sse": qualities}]})
    return data


def utilities(profiles, values=None, weight=1):
    return {row.instance_id: {
        "object_id": row.object_id, "instance_id": row.instance_id, "gop_id": row.gop_id,
        "weight": weight, "qualities": [
            {"quality_id": q.quality_id, "representation_id": q.representation_id,
             "weighted_utility": values[i][k] if values is not None else k}
            for k, q in enumerate(row.qualities)]}
        for i, row in enumerate(profiles.for_gop("g0"))}


def oracle(profiles, values, budget, cached=(), base_only=()):
    rows = profiles.for_gop("g0")
    cached_hashes = {profiles.resources[key].sha256 for key in cached}
    candidates = [(0,) if row.instance_id in base_only else range(len(row.qualities)) for row in rows]
    feasible = []
    for selection in itertools.product(*candidates):
        resources = {profiles.resources[key].sha256: profiles.resources[key].bytes
                     for row, index in zip(rows, selection) for key in row.qualities[index].resource_ids
                     if profiles.resources[key].sha256 not in cached_hashes}
        cost = sum(resources.values())
        utility = sum(values[row.instance_id]["qualities"][index]["weighted_utility"]
                      for row, index in zip(rows, selection))
        if cost <= budget:
            feasible.append((utility, tuple(reversed(selection)), selection, cost))
    return min(feasible, key=lambda item: (-item[0], item[1])) if feasible else None


@pytest.mark.parametrize("counts", [(1,), (2, 3), (4, 1, 2, 3)])
def test_ragged_quality_counts_share_native_solver(counts):
    profiles = profiles_from_dict(payload(counts=counts))
    values = utilities(profiles)
    result = allocate_utilities(profiles, "g0", 1000, values)
    assert result["status"] == "optimal"
    assert [row["selected_quality"] for row in result["allocations"]] == [f"Q{k-1}" for k in counts]
    assert result["total_selected_bytes"] == sum(k*10 for k in counts)
    assert result["diagnostics"]["algorithm"] == "src_final.ProposedDPSolver"


def test_signed_utilities_are_preserved_and_weight_metadata_does_not_restrict_candidates():
    profiles = profiles_from_dict(payload(counts=(4,)))
    values = utilities(profiles, ((-10, -2, -5, -1),), weight=0)
    result = allocate_utilities(profiles, "g0", 40, values)
    assert result["allocations"][0]["selected_quality"] == "Q3"
    assert result["allocations"][0]["weighted_utility"] == -1
    assert result["selected_utility"] == -1
    restricted = allocate_utilities(profiles, "g0", 40, values, base_only_instance_ids=("i0",))
    assert restricted["allocations"][0]["selected_quality"] == "Q0"
    assert restricted["selected_utility"] == -10


def test_existing_proposed_wrapper_keeps_formula_and_base_only_contract():
    profiles = profiles_from_dict(payload())
    visible = {"i0": 0, "i1": 200}
    values = {row.instance_id: build_utilities(row, visible[row.instance_id])
              for row in profiles.for_gop("g0")}
    proposed = allocate(profiles, "g0", visible, 50, utilities=values)
    generic = allocate_utilities(profiles, "g0", 50, values, base_only_instance_ids=("i0",))
    assert proposed["allocations"] == generic["allocations"]
    assert proposed["total_selected_bytes"] == generic["total_selected_bytes"]
    assert proposed["allocations"][0]["selected_quality"] == "Q0"
    values["i1"]["qualities"][1]["weighted_utility"] += 1
    with pytest.raises(ValueError, match="formula"):
        allocate(profiles, "g0", visible, 50, utilities=values)


@pytest.mark.parametrize("selected", [("Q0",), ("Q2", "Q0"), ("Q0", "Q1", "Q3"),
                                      ("Q0", "Q1", "Q2", "Q3")])
def test_configured_quality_subsets_preserve_profile_order_and_proposed_compatibility(selected):
    profiles = profiles_from_dict(payload(counts=(4,)))
    allowed = {"i0": list(selected)}
    values = utilities(profiles)
    generic = allocate_utilities(profiles, "g0", 100, values, allowed_quality_ids=allowed)
    proposed = allocate(profiles, "g0", {"i0": 100}, 100, allowed_quality_ids=allowed)
    best = max(int(qid[1:]) for qid in selected)
    assert generic["allocations"][0]["selected_quality"] == f"Q{best}"
    assert proposed["allocations"][0]["selected_quality"] == f"Q{best}"
    assert generic["allowed_quality_ids"] == {"i0": sorted(selected)}
    assert "allowed_quality_ids" not in allocate_utilities(profiles, "g0", 100, values)


def test_quality_subset_preserves_progressive_dependencies_that_are_not_candidates():
    data = payload("progressive", counts=(4,))
    data["objects"][0]["gops"][0]["offline_sse"][2]["dependencies"] = ["Q1"]
    profiles = profiles_from_dict(data)
    allowed = {"i0": ["Q0", "Q2"]}
    result = allocate_utilities(profiles, "g0", 60, utilities(profiles), allowed_quality_ids=allowed)
    choice = result["allocations"][0]
    assert choice["selected_quality"] == "Q2"
    assert choice["dependencies"] == ["Q0", "Q1"]
    assert choice["required_representations"] == ["rep0", "rep1", "rep2"]
    assert set(choice["required_resource_ids"]) == {"r0_0", "r0_1", "r0_2"}
    assert result["total_selected_bytes"] == 60
    assert result["enhancement_bytes"] == 50
    restricted = allocate_utilities(profiles, "g0", 60, utilities(profiles),
                                    allowed_quality_ids=allowed, base_only_instance_ids=("i0",))
    assert restricted["allocations"][0]["selected_quality"] == "Q0"


@pytest.mark.parametrize("allowed", [{}, {"i0": ["Q0"]},
    {"i0": ["Q0"], "i1": []}, {"i0": ["Q0"], "i1": ["Q1"]},
    {"i0": ["Q0"], "i1": ["Q0", "unknown"]}, {"i0": ["Q0"], "i1": ["Q0", "Q0"]},
    {"i0": ["Q0"], "i1": "Q0"}, {"i0": ["Q0"], "i1": ["Q0", 1]}])
def test_invalid_allowed_quality_subsets_are_explicit_errors(allowed):
    profiles = profiles_from_dict(payload())
    with pytest.raises(ValueError):
        allocate_utilities(profiles, "g0", 80, utilities(profiles), allowed_quality_ids=allowed)
    with pytest.raises(ValueError):
        allocate(profiles, "g0", {"i0": 100, "i1": 100}, 80, allowed_quality_ids=allowed)


@pytest.mark.parametrize("family", ["standalone", "progressive"])
def test_generic_physical_resource_allocations_match_independent_exhaustive_search(family):
    rng = random.Random(51821)
    for case in range(20):
        data = payload(family, counts=(3, 4))
        for record in data["resources"].values():
            record["bytes"] = rng.randrange(1, 16)
        if case % 2 == 0:
            data["objects"][1]["gops"][0]["offline_sse"][2]["resource_ids"] = ["r0_2"]
        profiles = profiles_from_dict(data)
        values = utilities(profiles, tuple(tuple(rng.randrange(-15, 30) for _ in row.qualities)
                                           for row in profiles.for_gop("g0")), weight=0)
        cached = ("r0_2",) if case % 3 == 0 else ()
        base_only = ("i0",) if case % 5 == 0 else ()
        base_hashes = {profiles.resources[key].sha256: profiles.resources[key].bytes
                       for row in profiles.for_gop("g0") for key in row.qualities[0].resource_ids
                       if key not in cached}
        budget = sum(base_hashes.values()) + rng.randrange(0, 40)
        result = allocate_utilities(profiles, "g0", budget, values,
                                    cached_resource_ids=cached, base_only_instance_ids=base_only)
        expected = oracle(profiles, values, budget, cached, base_only)
        assert result["status"] == "optimal"
        assert result["selected_utility"] == expected[0]
        assert result["total_selected_bytes"] == expected[3]
        assert [row["selected_quality"] for row in result["allocations"]] == [f"Q{k}" for k in expected[2]]
        assert sum(row["physical_byte_cost"] for row in result["allocations"]) == expected[3]
        assert result["total_selected_bytes"] <= budget


def test_infeasible_base_and_resource_limit_never_expose_a_selection():
    profiles = profiles_from_dict(payload())
    values = utilities(profiles)
    for budget, limits, expected in ((19, None, "infeasible_base"),
                                      (80, {"max_dp_cells": 1}, "incomplete_skipped")):
        result = allocate_utilities(profiles, "g0", budget, values, solver_limits=limits)
        assert result["status"] == expected
        assert result["allocations"] == []
        assert result["total_selected_bytes"] is None
        assert result["budget_feasible"] is False


@pytest.mark.parametrize("change", ["missing_instance", "extra_instance", "weight", "count", "quality",
                                    "representation", "instance", "object", "gop", "nonfinite", "bool",
                                    "missing_utility", "not_mapping", "quality_not_mapping"])
def test_generic_rejects_invalid_utility_inputs(change):
    profiles = profiles_from_dict(payload())
    values = deepcopy(utilities(profiles))
    row = values["i0"]
    if change == "missing_instance":
        values.pop("i1")
    elif change == "extra_instance":
        values["extra"] = deepcopy(row)
    elif change == "weight":
        row["weight"] = float("inf")
    elif change == "count":
        row["qualities"].pop()
    elif change == "quality":
        row["qualities"][0]["quality_id"] = "wrong"
    elif change == "representation":
        row["qualities"][0]["representation_id"] = "wrong"
    elif change in ("instance", "object", "gop"):
        row[change + "_id"] = "wrong"
    elif change == "nonfinite":
        row["qualities"][0]["weighted_utility"] = float("nan")
    elif change == "bool":
        row["qualities"][0]["weighted_utility"] = True
    elif change == "missing_utility":
        del row["qualities"][0]["weighted_utility"]
    elif change == "not_mapping":
        values["i0"] = None
    else:
        row["qualities"][0] = None
    with pytest.raises(ValueError):
        allocate_utilities(profiles, "g0", 80, values)


@pytest.mark.parametrize("base_only", ["i0", ("unknown",), ("i0", "i0")])
def test_generic_rejects_invalid_base_only_instance_ids(base_only):
    profiles = profiles_from_dict(payload())
    with pytest.raises(ValueError):
        allocate_utilities(profiles, "g0", 80, utilities(profiles), base_only_instance_ids=base_only)
