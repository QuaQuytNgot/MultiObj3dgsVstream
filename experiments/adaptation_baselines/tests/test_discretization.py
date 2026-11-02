"""Conservative bitrate grid, physical sharing and native resource accounting."""
import itertools
import math
import random
from types import SimpleNamespace

import pytest

from experiments.visibility_weighted_sse.allocation import allocate_utilities
from experiments.visibility_weighted_sse import allocation
from experiments.visibility_weighted_sse.profiles import profiles_from_dict
from .test_shared_allocator import payload, utilities


def scaled_profiles(sizes=(100000, 162501, 225000, 287499), n=2):
    data = payload(counts=(4,)*n)
    for i in range(n):
        for q, size in enumerate(sizes):
            data["resources"][f"r{i}_{q}"]["bytes"] = size
    return profiles_from_dict(data)


def test_default_grid_is_production_025_mbps_and_conservative_at_boundary():
    profiles = scaled_profiles()
    values = utilities(profiles, ((0, 10, 11, 12),)*2)
    budget = 325002
    exact = allocate_utilities(profiles, "g0", budget, values,
                               solver_limits={"discretization": "exact_bytes"})
    grid = allocate_utilities(profiles, "g0", budget, values)
    assert exact["selected_utility"] == 20
    assert grid["selected_utility"] == 11
    assert grid["total_selected_bytes"] == 325000 <= budget
    assert grid["diagnostics"]["delta_bps"] == 250000
    assert grid["diagnostics"]["gop_duration_seconds"] == 2
    assert grid["diagnostics"]["native_delta_units"] == 250000
    assert grid["diagnostics"]["optimality_scope"] == "conservatively_discretized_residual_problem"
    assert grid["diagnostics"]["exact_byte_feasible"] is True


@pytest.mark.parametrize("delta", [50000, 250000, 1000000])
def test_grid_matches_independent_exhaustive_quantized_problem(delta):
    rng = random.Random(84494)
    for _ in range(10):
        sizes = (100000, *(100000+rng.randrange(1, 300000) for _ in range(3)))
        profiles = scaled_profiles(sizes, n=3)
        values = utilities(profiles, tuple(tuple(rng.randrange(-5, 20) for _ in range(4)) for _ in range(3)))
        budget = 300000+rng.randrange(0, 700000)
        solved = allocate_utilities(profiles, "g0", budget, values, solver_limits={"delta_bps": delta})
        candidates = []
        for selection in itertools.product(range(4), repeat=3):
            cells = sum(math.ceil((sizes[q]-100000)*4/delta) for q in selection)
            if cells <= (budget-300000)*4//delta:
                gain = sum(values[f"i{i}"]["qualities"][q]["weighted_utility"] for i, q in enumerate(selection))
                candidates.append((gain, tuple(reversed(selection)), selection))
        expected = min(candidates, key=lambda row: (-row[0], row[1]))
        assert solved["selected_utility"] == expected[0]
        assert [row["selected_quality"] for row in solved["allocations"]] == [f"Q{q}" for q in expected[2]]
        assert solved["total_selected_bytes"] <= budget


def test_mandatory_offsets_do_not_exclude_base_on_a_coarse_grid():
    profiles = scaled_profiles(sizes=(100003, 150001, 200001, 250001))
    result = allocate_utilities(profiles, "g0", 200006, utilities(profiles), solver_limits={"delta_bps": 1000000})
    assert result["status"] == "optimal"
    assert result["total_selected_bytes"] == 200006
    assert [row["selected_quality"] for row in result["allocations"]] == ["Q0", "Q0"]
    assert result["diagnostics"]["native_budget_states_max"] == 1


def test_shared_activation_is_charged_once_exactly_and_cache_alias_is_respected():
    data = payload()
    data["resources"]["r0_2"]["bytes"] = 125001
    for obj in data["objects"]:
        obj["gops"][0]["offline_sse"][2]["resource_ids"] = ["r0_2"]
    profiles = profiles_from_dict(data)
    result = allocate_utilities(profiles, "g0", 125001, utilities(profiles))
    assert [row["selected_quality"] for row in result["allocations"]] == ["Q2", "Q2"]
    assert result["total_selected_bytes"] == 125001
    assert sum(row["physical_byte_cost"] for row in result["allocations"]) == 125001
    cached = allocate_utilities(profiles, "g0", 20, utilities(profiles), cached_resource_ids=("r0_2",))
    assert cached["selected_utility"] == 4 and cached["total_selected_bytes"] == 0


def test_duration_changes_bitrate_grid_and_fractional_duration_boundaries_are_exact():
    profiles = scaled_profiles((100000, 131250, 200000, 250000), n=1)
    values = utilities(profiles)
    one_second = allocate_utilities(profiles, "g0", 131250, values,
                                    solver_limits={"gop_duration_seconds": 1})
    two_seconds = allocate_utilities(profiles, "g0", 131250, values)
    assert one_second["allocations"][0]["selected_quality"] == "Q1"
    assert two_seconds["allocations"][0]["selected_quality"] == "Q0"
    fractional = scaled_profiles((100000, 109375, 200000, 250000), n=1)
    exact_boundary = allocate_utilities(fractional, "g0", 109375, utilities(fractional),
                                        solver_limits={"gop_duration_seconds": .3})
    assert exact_boundary["allocations"][0]["selected_quality"] == "Q1"


def test_actual_native_cells_and_working_memory_are_reported_and_limits_are_enforced():
    profiles = scaled_profiles()
    values = utilities(profiles)
    result = allocate_utilities(profiles, "g0", 325002, values)
    diag = result["diagnostics"]
    assert diag["native_budget_states_max"] == 3
    assert diag["native_dp_cells_max"] == diag["native_dp_cells_total"] == 6
    assert diag["native_estimated_working_set_bytes_max"] == 6+16*3
    assert diag["process_peak_rss_kib"] > 0
    assert diag["solver_wall_ms"] >= diag["solver_elapsed_ms"] > 0
    limited = allocate_utilities(profiles, "g0", 325002, values, solver_limits={"max_dp_cells": 1})
    assert limited["status"] == "incomplete_skipped" and limited["allocations"] == []
    assert limited["diagnostics"]["native_requested_dp_cells_max"] == 6
    assert limited["diagnostics"]["native_dp_cells_total"] == 0
    assert limited["diagnostics"]["exact_byte_feasible"] is False


@pytest.mark.parametrize("limits", [{"discretization": "unknown"}, {"delta_bps": 0}, {"delta_bps": True},
                                    {"gop_duration_seconds": 0}, {"gop_duration_seconds": float("nan")}])
def test_invalid_grid_configuration_fails_explicitly(limits):
    profiles = scaled_profiles()
    with pytest.raises(ValueError):
        allocate_utilities(profiles, "g0", 325002, utilities(profiles), solver_limits=limits)


def test_peak_rss_prefers_current_executable_linux_high_water(monkeypatch):
    monkeypatch.setattr(allocation.Path, "read_text", lambda *args, **kwargs: "VmRSS:\t1000 kB\nVmHWM:\t4321 kB\n")
    monkeypatch.setattr(allocation.resource, "getrusage", lambda *_: SimpleNamespace(ru_maxrss=999999))
    value, scope = allocation._process_peak_rss()
    assert value == 4321
    assert scope == "current_executable_VmHWM_including_native_allocations"


def test_peak_rss_fallback_reports_its_scope_and_converts_macos_bytes(monkeypatch):
    def inaccessible(*args, **kwargs):
        raise PermissionError("proc unavailable")
    monkeypatch.setattr(allocation.Path, "read_text", inaccessible)
    monkeypatch.setattr(allocation.resource, "getrusage", lambda *_: SimpleNamespace(ru_maxrss=1025))
    monkeypatch.setattr(allocation.sys, "platform", "darwin")
    value, scope = allocation._process_peak_rss()
    assert value == 2
    assert scope == "getrusage_process_lifetime_high_water_including_native_allocations"
