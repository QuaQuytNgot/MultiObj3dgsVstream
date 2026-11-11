"""Physical-resource adapter around the unchanged production native DP.

The default conservatively discretizes incremental bitrate at 0.25 Mbps. Fixed
mandatory, shared-activation and row-minimum bytes are charged exactly before
discretization. ``solver_limits.discretization=exact_bytes`` retains the earlier
GCD-reduced exact-byte reference for bounded comparisons. Optional inter-object
sharing still uses the same bounded activation enumeration and MCKP objective.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from fractions import Fraction
import math
from pathlib import Path
import resource
import sys
import time

from ..solver import ProposedDPSolver, INT64_MAX

from .common import finite, integer, require
from .profiles import build_utilities


@lru_cache(maxsize=16)
def _solver(max_dp_cells, timeout_seconds, delta_bps=1):
    return ProposedDPSolver(delta_bps=delta_bps, max_dp_cells=max_dp_cells,
                            timeout_seconds=timeout_seconds)


def solver_settings(solver_limits=None):
    """Validate the shared experiments-only grid and resource-limit contract."""
    limits = dict(solver_limits or {})
    require(set(limits) <= {"max_dp_cells", "max_shared_resources", "max_shared_patterns", "timeout_seconds",
                            "discretization", "delta_bps", "gop_duration_seconds"}, "unknown solver limit")
    max_cells = integer(limits.get("max_dp_cells", 50_000_000), "max_dp_cells", 1)
    max_shared = integer(limits.get("max_shared_resources", 12), "max_shared_resources")
    max_patterns = integer(limits.get("max_shared_patterns", 4096), "max_shared_patterns", 1)
    timeout = limits.get("timeout_seconds", 0)
    require(isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and math.isfinite(timeout) and timeout >= 0,
            "timeout_seconds must be finite and nonnegative")
    require(timeout == 0, "wall-clock deadlines are disabled for deterministic experiments; use DP cell/state limits")
    mode = limits.get("discretization", "bitrate")
    require(mode in ("bitrate", "exact_bytes"), "discretization must be bitrate or exact_bytes")
    delta = integer(limits.get("delta_bps", 250_000), "delta_bps", 1)
    require(delta <= INT64_MAX, "delta_bps exceeds native int64 range")
    duration = limits.get("gop_duration_seconds", 2.)
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration > 0,
            "gop_duration_seconds must be finite and positive")
    return dict(max_dp_cells=max_cells, max_shared_resources=max_shared, max_shared_patterns=max_patterns,
                timeout_seconds=float(timeout), discretization=mode, delta_bps=delta,
                gop_duration_seconds=float(duration))


def _ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def _process_peak_rss():
    """Measure this executable's Linux high-water RSS, including native memory.

    Linux getrusage can retain a launcher's high-water mark across exec. VmHWM
    starts with the current executable and avoids attributing parent memory to
    fresh benchmark workers. Other platforms retain a clearly named fallback.
    """
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                _, value, unit = line.split()
                if unit == "kB" and int(value) > 0:
                    return int(value), "current_executable_VmHWM_including_native_allocations"
    except (OSError, ValueError):
        pass
    measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix hosts report KiB.
    kib = math.ceil(measured / 1024) if sys.platform == "darwin" else int(measured)
    return kib, "getrusage_process_lifetime_high_water_including_native_allocations"


def allocate(profiles, gop_id, visible_contributions, budget_bytes, *,
             cached_resource_ids=(), solver_limits=None, utilities=None,
             allowed_quality_ids=None):
    """Visibility-Weighted SSE policy, retaining its strict formula contract."""
    started = time.perf_counter()
    integer(budget_bytes, "GOP byte budget")
    require(budget_bytes <= INT64_MAX, "GOP budget exceeds native int64 range")
    rows = profiles.for_gop(gop_id)
    ids = tuple(row.instance_id for row in rows)
    require(set(visible_contributions) == set(ids), "visibility/profile instance coverage mismatch")
    expected_utilities = {row.instance_id: build_utilities(row, visible_contributions[row.instance_id]) for row in rows}
    if utilities is not None:
        require(utilities == expected_utilities, "supplied utilities differ from Visibility-Weighted SSE formula")
    else:
        utilities = expected_utilities
    # The proposed policy keeps invisible instances at their mandatory Q0.
    # Other policies pass their candidate restrictions explicitly to the shared
    # allocator; its cost model must not interpret a policy's weight metadata.
    base_only = tuple(iid for iid in ids if utilities[iid]["weight"] == 0)
    output = allocate_utilities(profiles, gop_id, budget_bytes, utilities,
                               cached_resource_ids=cached_resource_ids,
                               solver_limits=solver_limits,
                               base_only_instance_ids=base_only,
                               allowed_quality_ids=allowed_quality_ids)
    output["diagnostics"]["allocation_adapter_elapsed_ms"] = (time.perf_counter()-started)*1000
    return output


def allocate_utilities(profiles, gop_id, budget_bytes, utilities, *,
                       cached_resource_ids=(), solver_limits=None,
                       base_only_instance_ids=(), allowed_quality_ids=None):
    """Allocate supplied utilities with exact physical costs and a bounded grid.

    Each utility row binds the profile's quality and representation identities.
    Values may be signed and are never transformed. ``weight`` is finite policy
    metadata only; candidate restrictions use ``base_only_instance_ids``.
    ``allowed_quality_ids`` may select a Q0-containing subset for every instance;
    dependency and physical-resource metadata always retain the complete profile.
    Cache IDs declare verified resident physical resources in this frozen input.
    No cache admission, network request, renderer or decoder is invoked.
    """
    started = time.perf_counter()
    integer(budget_bytes, "GOP byte budget")
    require(budget_bytes <= INT64_MAX, "GOP budget exceeds native int64 range")
    settings = solver_settings(solver_limits)
    mode, delta = settings["discretization"], settings["delta_bps"]
    rate_per_byte = Fraction(8, 1) / Fraction(str(settings["gop_duration_seconds"]))
    rows = profiles.for_gop(gop_id)
    ids = tuple(row.instance_id for row in rows)
    require(isinstance(utilities, Mapping) and set(utilities) == set(ids),
            "utility/profile instance coverage mismatch")
    for row in rows:
        utility = utilities[row.instance_id]
        require(isinstance(utility, Mapping), "each utility row must be a mapping")
        finite(utility.get("weight"), "utility weight metadata")
        for field, expected in (("object_id", row.object_id), ("instance_id", row.instance_id),
                                ("gop_id", row.gop_id)):
            require(field not in utility or utility[field] == expected,
                    f"utility/profile {field} mismatch")
        qualities = utility.get("qualities")
        require(isinstance(qualities, (list, tuple)) and len(qualities) == len(row.qualities),
                "utility/profile quality count mismatch")
        for supplied, quality in zip(qualities, row.qualities):
            require(isinstance(supplied, Mapping), "each quality utility must be a mapping")
            require(supplied.get("quality_id") == quality.quality_id and
                    supplied.get("representation_id") == quality.representation_id,
                    "utility/profile quality or representation identity mismatch")
            finite(supplied.get("weighted_utility"), "quality utility")
    require(not isinstance(base_only_instance_ids, (str, bytes)),
            "base-only instances must be an ID sequence")
    base_only_ids = tuple(base_only_instance_ids)
    require(len(set(base_only_ids)) == len(base_only_ids), "duplicate base-only instance ID")
    require(set(base_only_ids) <= set(ids), "unknown base-only instance ID")
    base_only = set(base_only_ids)
    allowed_indices = {}
    if allowed_quality_ids is not None:
        require(isinstance(allowed_quality_ids, Mapping) and set(allowed_quality_ids) == set(ids),
                "allowed quality/profile instance coverage mismatch")
        for row in rows:
            selected = allowed_quality_ids[row.instance_id]
            require(isinstance(selected, (list, tuple)) and bool(selected),
                    "allowed quality IDs must be a nonempty sequence")
            require(all(isinstance(qid, str) for qid in selected),
                    "allowed quality IDs must be strings")
            require(len(set(selected)) == len(selected), "duplicate allowed quality ID")
            known = tuple(q.quality_id for q in row.qualities)
            require(set(selected) <= set(known), "unknown allowed quality ID")
            require(known[0] in selected, "allowed quality IDs must include mandatory base/Q0")
            # Preserve profile order and existing native tie semantics even if
            # the configuration lists the eligible candidates in another order.
            allowed_indices[row.instance_id] = tuple(k for k, qid in enumerate(known) if qid in selected)
    require(not isinstance(cached_resource_ids, (str, bytes)), "cached resources must be an ID sequence")
    cached_ids = tuple(cached_resource_ids)
    require(len(set(cached_ids)) == len(cached_ids), "duplicate cached resource ID")
    require(all(key in profiles.resources for key in cached_ids), "unknown cached physical resource")
    cached = {_resource_key(profiles.resources[key]) for key in cached_ids}
    resource_by_hash = {_resource_key(r): r for r in profiles.resources.values()}
    sizes = {key: ref.bytes for key, ref in resource_by_hash.items()}

    def missing(quality):
        return frozenset(_resource_key(profiles.resources[key]) for key in quality.resource_ids) - cached

    closures = tuple(tuple(missing(q) for q in row.qualities) for row in rows)
    base_union = frozenset().union(*(row[0] for row in closures))
    base_bytes = sum(sizes[key] for key in base_union)
    diagnostics = {"algorithm": "src_final.ProposedDPSolver",
                   "cost_units": ("conservative_bitrate_grid" if mode == "bitrate" else
                                  "exact_integer_bytes_with_GCD_reduction"),
                   "discretization": mode, "delta_bps": delta if mode == "bitrate" else None,
                   "optimality_scope": ("conservatively_discretized_residual_problem" if mode == "bitrate" else
                                        "original_physical_byte_problem"),
                   "gop_duration_seconds": settings["gop_duration_seconds"],
                   "native_delta_units": delta if mode == "bitrate" else 1, "native_calls": 0,
                   "shared_activation_patterns": 0, "solver_elapsed_ms": 0.,
                   "solver_wall_ms": 0., "native_dp_cells_total": 0, "native_dp_cells_max": 0,
                   "native_requested_dp_cells_total": 0, "native_requested_dp_cells_max": 0,
                   "native_estimated_working_set_bytes_max": 0, "native_subproblems": [],
                   "native_budget_states_max": 0, "native_visited": 0,
                   "cached_input_execution": True,
                   "tie_policy": "native strict first-index ties; deterministic reversed-index ordering across sharing subproblems"}

    def result(status, *, selection=None, detail=None):
        output = {"gop_id": gop_id, "solver_status": status, "status": status,
                  "budget_feasible": selection is not None, "feasible": selection is not None,
                  "available_budget_bytes": budget_bytes, "base_bytes": base_bytes,
                  "total_selected_bytes": None, "enhancement_bytes": None,
                  "selected_utility": None, "allocations": [],
                  "cached_resource_ids": list(cached_ids), "diagnostics": dict(diagnostics)}
        if detail is not None:
            output["detail"] = detail
        if allowed_quality_ids is not None:
            output["allowed_quality_ids"] = {
                row.instance_id: [row.qualities[k].quality_id for k in allowed_indices[row.instance_id]]
                for row in rows}
        if selection is not None:
            chosen = [row[index] for row, index in zip(closures, selection)]
            union = frozenset().union(*chosen)
            total = sum(sizes[key] for key in union)
            require(total <= budget_bytes, "selected physical resource union exceeds budget")
            accounted = set()
            allocations = []
            for row, index, resources in zip(rows, selection, chosen):
                quality = row.qualities[index]
                charged = resources - accounted
                accounted.update(resources)
                representations = {q.quality_id: q.representation_id for q in row.qualities}
                allocations.append({"object_id": row.object_id, "instance_id": row.instance_id,
                                    "selected_quality": quality.quality_id,
                                    "representation_id": quality.representation_id,
                                    "dependencies": list(quality.dependencies),
                                    "required_representations": [representations[q] for q in (*quality.dependencies, quality.quality_id)],
                                    "required_resource_ids": list(quality.resource_ids),
                                    "physical_byte_cost": sum(sizes[key] for key in charged),
                                    "standalone_missing_bytes": sum(sizes[key] for key in resources),
                                    "charged_resource_sha256": sorted(charged),
                                    "weighted_utility": utilities[row.instance_id]["qualities"][index]["weighted_utility"]})
            output.update(total_selected_bytes=total, enhancement_bytes=total-base_bytes,
                          selected_utility=sum(item["weighted_utility"] for item in allocations),
                          allocations=allocations, selected_resource_sha256=sorted(union))
        output["diagnostics"]["exact_byte_feasible"] = selection is not None
        peak_kib, peak_scope = _process_peak_rss()
        output["diagnostics"]["process_peak_rss_kib"] = peak_kib
        output["diagnostics"]["process_peak_rss_scope"] = peak_scope
        output["diagnostics"]["allocation_adapter_elapsed_ms"] = (time.perf_counter()-started)*1000
        return output

    if base_bytes > budget_bytes:
        return result("infeasible_base", detail="mandatory Q0/base physical union exceeds GOP byte budget")
    max_cells = settings["max_dp_cells"]
    max_shared = settings["max_shared_resources"]
    max_patterns = settings["max_shared_patterns"]
    timeout = settings["timeout_seconds"]

    candidates = tuple((0,) if row.instance_id in base_only else
                       allowed_indices.get(row.instance_id, tuple(range(len(row.qualities))))
                       for row in rows)
    allowed_closures = tuple(tuple(closure[k] for k in indices) for closure, indices in zip(closures, candidates))
    mandatory = frozenset().union(*(frozenset.intersection(*row) for row in allowed_closures))
    optional = tuple(tuple(closure-mandatory for closure in row) for row in allowed_closures)
    occurrences = {}
    for row_index, row in enumerate(optional):
        for key in frozenset().union(*row):
            occurrences.setdefault(key, set()).add(row_index)
    shared = tuple(sorted(key for key, owners in occurrences.items() if len(owners) > 1))
    # Resources with the same candidate incidence always activate together.
    groups_by_signature = {}
    for key in shared:
        signature = tuple((i, tuple(j for j, closure in enumerate(row) if key in closure))
                          for i, row in enumerate(optional) if any(key in closure for closure in row))
        groups_by_signature.setdefault(signature, []).append(key)
    groups = tuple(tuple(group) for signature, group in sorted(groups_by_signature.items()))
    diagnostics.update(optional_shared_resource_count=len(shared), shared_activation_group_count=len(groups))
    if len(groups) > max_shared or (1 << len(groups)) > max_patterns:
        return result("incomplete_shared_limit", detail="optional shared-resource activation bound exceeded; no allocation certified")
    group_masks = {key: 1 << i for i, group in enumerate(groups) for key in group}
    # Each group may contain several hashes; use bitwise OR rather than adding
    # repeated bits for resources with identical activation signatures.
    required_masks = tuple(tuple(_mask(closure, group_masks) for closure in row) for row in optional)
    shared_set = set(shared)
    unique_costs = tuple(tuple(sum(sizes[key] for key in closure-shared_set) for closure in row) for row in optional)
    mandatory_bytes = sum(sizes[key] for key in mandatory)
    group_bytes = tuple(sum(sizes[key] for key in group) for group in groups)
    solver = None
    best = None
    failures = []
    for activation in range(1 << len(groups)):
        diagnostics["shared_activation_patterns"] += 1
        fixed = mandatory_bytes + sum(cost for i, cost in enumerate(group_bytes) if activation & (1 << i))
        if fixed > budget_bytes:
            continue
        indices = tuple(tuple(j for j, requirement in enumerate(row) if not requirement & ~activation)
                        for row in required_masks)
        if any(not row for row in indices):
            continue
        costs = [[unique_costs[i][j] for j in row] for i, row in enumerate(indices)]
        values = [[utilities[ids[i]]["qualities"][candidates[i][j]]["weighted_utility"] for j in row]
                  for i, row in enumerate(indices)]
        offsets = [min(row) for row in costs]
        fixed += sum(offsets)
        if fixed > budget_bytes:
            continue
        reduced = [[cost-offset for cost in row] for row, offset in zip(costs, offsets)]
        if mode == "exact_bytes":
            divisor = math.gcd(*(value for row in reduced for value in row)) or 1
            reduced = [[cost//divisor for cost in row] for row in reduced]
            reduced_budget = min((budget_bytes-fixed)//divisor, sum(max(row) for row in reduced))
            native_delta = 1
        else:
            # Integer ceil of a rational rate cannot lose physical bytes. The
            # unchanged C++ solver applies ceil(rate/delta), floor(budget/delta).
            # Fixed costs and row offsets remain exact; zero residual minima
            # keep the mandatory cheapest vector available on every grid.
            reduced = [[_ceil_fraction(cost * rate_per_byte) for cost in row] for row in reduced]
            residual_rate_budget = ((budget_bytes-fixed) * rate_per_byte).__floor__()
            max_needed_units = sum(max(cost//delta + bool(cost % delta) for cost in row) for row in reduced)
            reduced_budget = min(residual_rate_budget, max_needed_units*delta)
            native_delta = delta
        require(reduced_budget <= INT64_MAX, "native reduced budget exceeds int64")
        if solver is None:
            solver = _solver(max_cells, float(timeout), native_delta)
        padded_costs = [row + [row[-1]]*(4-len(row)) for row in reduced]
        padded_values = [row + [row[-1]]*(4-len(row)) for row in values]
        call_started = time.perf_counter()
        native = solver.solve(padded_costs, padded_values, reduced_budget)
        diagnostics["solver_wall_ms"] += (time.perf_counter()-call_started)*1000
        diagnostics["native_calls"] += 1
        diagnostics["solver_elapsed_ms"] += native.elapsed_ms
        diagnostics["native_budget_states_max"] = max(diagnostics["native_budget_states_max"], native.budget_states)
        diagnostics["native_visited"] += native.visited
        requested_cells = len(rows)*native.budget_states
        allocated_cells = requested_cells if native.status == "optimal" or native.visited else 0
        diagnostics["native_requested_dp_cells_total"] += requested_cells
        diagnostics["native_requested_dp_cells_max"] = max(diagnostics["native_requested_dp_cells_max"], requested_cells)
        diagnostics["native_dp_cells_total"] += allocated_cells
        diagnostics["native_dp_cells_max"] = max(diagnostics["native_dp_cells_max"], allocated_cells)
        working_bytes = allocated_cells + 16*native.budget_states if allocated_cells else 0
        diagnostics["native_estimated_working_set_bytes_max"] = max(
            diagnostics["native_estimated_working_set_bytes_max"], working_bytes)
        diagnostics["native_subproblems"].append(dict(activation=activation, status=native.status,
            fixed_exact_bytes=fixed, budget_states=native.budget_states,
            requested_dp_cells=requested_cells, allocated_dp_cells=allocated_cells,
            estimated_working_set_bytes=working_bytes, visited=native.visited))
        if native.status != "optimal":
            if native.status not in ("infeasible", "discretized_infeasible"):
                failures.append(native.status)
            continue
        selection = tuple(candidates[i][indices[i][min(k, len(indices[i])-1)]] for i, k in enumerate(native.selection))
        utility = sum(utilities[identifier]["qualities"][k]["weighted_utility"] for identifier, k in zip(ids, selection))
        tie = tuple(reversed(selection))
        if best is None or utility > best[0] or (utility == best[0] and tie < best[1]):
            best = utility, tie, selection
    if failures:
        return result("incomplete_" + sorted(set(failures))[0], detail="a native subproblem hit its explicit resource limit; no global allocation certified")
    if best is None:
        return result("infeasible", detail="no feasible physical-byte quality vector")
    return result("optimal", selection=best[2])


def _mask(resources, group_masks):
    value = 0
    for key in resources:
        value |= group_masks.get(key, 0)
    return value


def _resource_key(resource):
    """Admission ownership for costing; payload SHA remains the integrity hash."""
    return getattr(resource, "owner_key", resource.sha256)
