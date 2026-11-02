"""CPU-only metadata, signed utility and exact physical-closure regressions."""
from copy import deepcopy
import hashlib
import itertools
import random

import pytest

from experiments.visibility_weighted_sse.allocation import allocate as shared_allocate
from experiments.visibility_weighted_sse.common import write_json
from experiments.visibility_weighted_sse.profiles import (
    build_utilities, load_profiles, mean_contribution_from_artifact, profiles_from_dict,
)
from src_final.multiobj3dgs.shared.dash import build_mpd
from src_final.multiobj3dgs.shared.metadata.serialization import (
    SCHEMA as PUBLICATION_SCHEMA, PROFILE, json_bytes, revision,
)


def allocate(*args, solver_limits=None, **kwargs):
    """These exhaustive regressions explicitly exercise the exact reference."""
    return shared_allocate(*args, solver_limits={"discretization": "exact_bytes", **(solver_limits or {})}, **kwargs)


def fixture_payload(family="standalone", n=2):
    data = {"schema": "visibility-weighted-sse.profiles.v1", "content_revision": "content-v1",
            "profile_revision": "profile-v1", "representation_family": family,
            "compatibility": {"resolution": [100, 100],
                              "coordinate_convention": "opencv_positive_z_column_vectors",
                              "contribution_convention": "continuous_alpha_weighted_pixels",
                              "distribution_id": "fixed-two-offline-cameras",
                              "time_aggregation": "normalized_camera_time_mean",
                              "proxy_representation": "Q0",
                              "transform_convention": "positive_similarity_object_to_world"},
            "provenance": {"information_role": "OFFLINE_AVERAGE_SOLO_SSE", "joint_quality_used": False},
            "resources": {}, "objects": []}
    for i in range(n):
        qualities = []
        for q, size in enumerate((10, 20, 30, 40)):
            key = f"r{i}_{q}"
            data["resources"][key] = {"bytes": size, "sha256": hashlib.sha256(key.encode()).hexdigest()}
            qualities.append({"quality_id": f"Q{q}", "representation_id": f"rep{q}",
                              "average_solo_sse": (100, 50, 10, 110)[q], "resource_ids": [key],
                              "dependencies": ["Q0"] if family == "progressive" and q else []})
        data["objects"].append({"object_id": f"object{i}", "instance_id": f"i{i}", "gops": [
            {"gop_id": "g0", "offline_mean_solo_contribution": 100,
             "offline_sample_ids": ["offline-cam0-t0", "offline-cam1-t0"],
             "offline_sample_weights": [0.5, 0.5], "offline_sse": qualities}]})
    return data


def test_weight_is_not_clamped_or_normalized_and_signed_gain_is_preserved():
    profiles = profiles_from_dict(fixture_payload())
    value = build_utilities(profiles.objects["i0", "g0"], 250)
    assert value["weight"] == 2.5
    assert [row["sse_improvement"] for row in value["qualities"]] == [0, 50, 90, -10]
    assert [row["weighted_utility"] for row in value["qualities"]] == [0, 125, 225, -25]
    values = {row.instance_id: build_utilities(row, 250) for row in profiles.for_gop("g0")}
    assert sum(value["weight"] for value in values.values()) == 5


@pytest.mark.parametrize("denominator", [0, -1, None, 10001, float("nan")])
def test_invalid_denominator_is_a_profile_error(denominator):
    data = fixture_payload()
    data["objects"][0]["gops"][0]["offline_mean_solo_contribution"] = denominator
    with pytest.raises(ValueError):
        profiles_from_dict(data)


@pytest.mark.parametrize("change", ["distribution", "weights", "joint", "quality_distribution", "family", "missing_gop"])
def test_incompatible_metadata_is_explicit(change):
    data = fixture_payload()
    gop = data["objects"][0]["gops"][0]
    if change == "distribution":
        gop["sse_metadata"] = deepcopy(data["compatibility"])
        gop["denominator_metadata"] = {**data["compatibility"], "distribution_id": "unrelated"}
    elif change == "weights":
        gop["offline_sample_weights"] = [.2, .2]
    elif change == "joint":
        data["provenance"]["joint_quality_used"] = True
    elif change == "quality_distribution":
        gop["offline_sse"][0]["offline_sample_ids"] = ["unrelated"]
    elif change == "family":
        data["representation_family"] = "mixed"
    elif change == "missing_gop":
        data["objects"][0]["gops"] = []
    with pytest.raises(ValueError):
        profiles_from_dict(data)


def test_frozen_loading_and_bounded_offline_mean_construction(tmp_path):
    data = fixture_payload()
    path = tmp_path / "profiles.json"
    write_json(path, data)
    a, b = load_profiles(path), load_profiles(path)
    assert a == b
    with pytest.raises(TypeError):
        a.objects["x"] = None
    artifact = {"schema": "visibility-weighted-sse.offline-contribution.v1",
                "compatibility": data["compatibility"],
                "provenance": {"validated": True, "joint_quality_used": False},
                "samples": [{"sample_id": "a", "weight": .25, "solo_contribution": 100},
                            {"sample_id": "b", "weight": .75, "solo_contribution": 200}]}
    artifact_path = tmp_path / "contribution.json"
    write_json(artifact_path, artifact)
    built = mean_contribution_from_artifact(artifact_path, a.compatibility)
    assert built["offline_mean_solo_contribution"] == 175
    with pytest.raises(ValueError, match="bound"):
        mean_contribution_from_artifact(artifact_path, a.compatibility, max_samples=1)
    with pytest.raises(ValueError, match="byte bound"):
        mean_contribution_from_artifact(artifact_path, a.compatibility, max_bytes=1)
    with pytest.raises(ValueError, match="incompatible"):
        mean_contribution_from_artifact(artifact_path, {**a.compatibility, "distribution_id": "other"})


def test_missing_gop_and_utility_tampering_are_errors():
    profiles = profiles_from_dict(fixture_payload())
    with pytest.raises(ValueError, match="Missing"):
        allocate(profiles, "g1", {"i0": 100, "i1": 100}, 50)
    utilities = {row.instance_id: build_utilities(row, 100) for row in profiles.for_gop("g0")}
    utilities["i0"]["qualities"][1]["weighted_utility"] *= 2
    with pytest.raises(ValueError, match="formula"):
        allocate(profiles, "g0", {"i0": 100, "i1": 100}, 50, utilities=utilities)


def test_progressive_transitive_dependencies_and_physical_costs():
    data = fixture_payload("progressive", 1)
    data["objects"][0]["gops"][0]["offline_sse"][2]["dependencies"] = ["Q1"]
    profiles = profiles_from_dict(data)
    q2 = profiles.objects["i0", "g0"].qualities[2]
    assert q2.dependencies == ("Q0", "Q1")
    assert set(q2.resource_ids) == {"r0_0", "r0_1", "r0_2"}
    result = allocate(profiles, "g0", {"i0": 100}, 60)
    assert result["allocations"][0]["selected_quality"] == "Q2"
    assert result["allocations"][0]["required_representations"] == ["rep0", "rep1", "rep2"]
    assert (result["base_bytes"], result["enhancement_bytes"], result["total_selected_bytes"]) == (10, 50, 60)


@pytest.mark.parametrize("kind", ["cycle", "unknown", "no_base", "standalone_dependencies", "unknown_resource", "conflicting_hash"])
def test_invalid_dependency_and_resource_metadata(kind):
    data = fixture_payload("progressive", 1)
    qualities = data["objects"][0]["gops"][0]["offline_sse"]
    if kind == "cycle":
        qualities[0]["dependencies"] = ["Q1"]
    elif kind == "unknown":
        qualities[1]["dependencies"] = ["Q9"]
    elif kind == "no_base":
        qualities[1]["dependencies"] = []
    elif kind == "standalone_dependencies":
        data["representation_family"] = "standalone"
    elif kind == "unknown_resource":
        qualities[1]["resource_ids"] = ["missing"]
    else:
        data["resources"]["r0_1"]["sha256"] = data["resources"]["r0_0"]["sha256"]
    with pytest.raises(ValueError):
        profiles_from_dict(data)


def test_no_skip_and_base_infeasible_even_if_higher_quality_is_cheaper():
    data = fixture_payload(n=1)
    data["resources"]["r0_1"]["bytes"] = 1
    profiles = profiles_from_dict(data)
    result = allocate(profiles, "g0", {"i0": 100}, 9)
    assert result["status"] == "infeasible_base"
    assert result["allocations"] == []
    assert not result["budget_feasible"]


def test_invisible_object_is_base_and_dominant_object_gets_budget():
    profiles = profiles_from_dict(fixture_payload())
    result = allocate(profiles, "g0", {"i0": 0, "i1": 500}, 40)
    assert [row["selected_quality"] for row in result["allocations"]] == ["Q0", "Q2"]
    assert result["total_selected_bytes"] == 40


def test_shared_optional_resources_and_cached_aliases_are_not_double_counted():
    data = fixture_payload()
    for obj in data["objects"]:
        qualities = obj["gops"][0]["offline_sse"]
        qualities[1]["resource_ids"] = ["r0_1"]
        qualities[2]["resource_ids"] = ["r0_2"]
    profiles = profiles_from_dict(data)
    result = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 30)
    assert [row["selected_quality"] for row in result["allocations"]] == ["Q2", "Q2"]
    assert result["total_selected_bytes"] == 30
    assert sum(row["physical_byte_cost"] for row in result["allocations"]) == 30
    assert result["diagnostics"]["shared_activation_group_count"] == 2
    cached = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 20, cached_resource_ids=["r0_2"])
    assert cached["total_selected_bytes"] == 0
    capped = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 30,
                      solver_limits={"max_shared_resources": 0})
    assert capped["status"] == "incomplete_shared_limit" and not capped["allocations"]


def test_shared_group_with_multiple_hashes_uses_one_activation_bit():
    data = fixture_payload()
    for obj in data["objects"]:
        obj["gops"][0]["offline_sse"][2]["resource_ids"] = ["r0_2", "r1_2"]
    profiles = profiles_from_dict(data)
    result = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 60)
    assert result["total_selected_bytes"] == 60
    assert result["selected_utility"] == 180
    assert result["diagnostics"]["shared_activation_group_count"] == 1


def _brute(profiles, visibility, budget):
    rows = profiles.for_gop("g0")
    candidates = [range(len(row.qualities)) if visibility[row.instance_id] else (0,) for row in rows]
    best = None
    for selection in itertools.product(*candidates):
        resource_keys = {profiles.resources[key].sha256: profiles.resources[key].bytes
                         for row, q in zip(rows, selection) for key in row.qualities[q].resource_ids}
        cost = sum(resource_keys.values())
        utility = sum(build_utilities(row, visibility[row.instance_id])["qualities"][q]["weighted_utility"]
                      for row, q in zip(rows, selection))
        if cost <= budget and (best is None or utility > best):
            best = utility
    return best


@pytest.mark.parametrize("family", ["standalone", "progressive"])
def test_existing_solver_matches_brute_force_small_random_physical_closures(family):
    rng = random.Random(7189)
    for case in range(20):
        data = fixture_payload(family)
        for resource in data["resources"].values():
            resource["bytes"] = rng.randrange(1, 16)
        for obj in data["objects"]:
            for q in obj["gops"][0]["offline_sse"][1:]:
                q["average_solo_sse"] = rng.randrange(0, 150)
        if case % 3 == 0:
            data["objects"][1]["gops"][0]["offline_sse"][1]["resource_ids"] = ["r0_1"]
        profiles = profiles_from_dict(data)
        visibility = {"i0": rng.choice((0, 50, 200)), "i1": rng.choice((20, 100, 300))}
        base = sum(data["resources"][f"r{i}_0"]["bytes"] for i in range(2))
        budget = base+rng.randrange(0, 41)
        result = allocate(profiles, "g0", visibility, budget)
        assert result["status"] == "optimal"
        assert result["selected_utility"] == pytest.approx(_brute(profiles, visibility, budget))
        assert result["total_selected_bytes"] <= budget


def test_ties_are_repeatable_and_limits_return_no_vector():
    data = fixture_payload()
    for obj in data["objects"]:
        for q in obj["gops"][0]["offline_sse"]:
            q["average_solo_sse"] = 100
    profiles = profiles_from_dict(data)
    a = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 80)
    b = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 80)
    assert a["allocations"] == b["allocations"]
    assert [r["selected_quality"] for r in a["allocations"]] == ["Q0", "Q0"]
    capped = allocate(profiles, "g0", {"i0": 100, "i1": 100}, 80,
                      solver_limits={"max_dp_cells": 1})
    assert capped["status"] == "incomplete_skipped" and not capped["allocations"]
    with pytest.raises(ValueError, match="deadlines"):
        allocate(profiles, "g0", {"i0": 100, "i1": 100}, 80,
                 solver_limits={"timeout_seconds": .001})


def test_exact_gcd_budget_reduction_keeps_original_byte_constraint():
    data = fixture_payload(n=1)
    for q, size in enumerate((10000001, 10000007, 10000013, 10000019)):
        data["resources"][f"r0_{q}"]["bytes"] = size
    profiles = profiles_from_dict(data)
    a = allocate(profiles, "g0", {"i0": 100}, 10000012)
    assert a["allocations"][0]["selected_quality"] == "Q1"
    assert a["total_selected_bytes"] == 10000007
    assert a["diagnostics"]["native_budget_states_max"] == 2


def test_existing_local_mpd_parser_validates_actual_physical_publication_closures(tmp_path):
    data = fixture_payload(n=1)
    gop = data["objects"][0]["gops"][0]
    gop["source_gop_index"] = 0
    representations = []
    for q, quality in enumerate(gop["offline_sse"]):
        key = quality["resource_ids"][0]
        resource = {"id": key, "url": key+".cpseg", **data["resources"][key]}
        segment = {"id": f"segment-{q}", "gop_index": 0, "start": 0, "duration": 1,
                   "resource": resource, "members": {"reconstruction.json": ""},
                   "shared_assets": [], "frames": [{"source_frame": 0, "seconds": 0, "model_time": 0}]}
        representations.append({"id": quality["representation_id"], "crf": (40, 36, 28, 20)[q],
                                "rank": q, "bandwidth": resource["bytes"]*8, "segments": [segment]})
    index = {"schema": PUBLICATION_SCHEMA, "profile": PROFILE, "duration": 1,
             "contents": [{"id": "object0", "fps": 1, "duration": 1, "representations": representations}],
             "instances": [{"id": "i0", "content_id": "object0"}]}
    index["revision"] = revision(index)
    encoded = json_bytes(index)
    (tmp_path / "index.json").write_bytes(encoded)
    (tmp_path / "manifest.mpd").write_bytes(build_mpd(index, index_sha256=hashlib.sha256(encoded).hexdigest()))
    data["content_revision"] = index["revision"]
    data["publication_mpd"] = "manifest.mpd"
    path = tmp_path / "profiles.json"
    write_json(path, data)
    profiles = load_profiles(path)
    assert profiles.content_revision == index["revision"]
    data["resources"]["r0_1"]["bytes"] += 1
    write_json(path, data)
    with pytest.raises(ValueError, match="physical closure"):
        load_profiles(path)
